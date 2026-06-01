"""Amplification Validator — Phase 5.4 master scenario runner.

Runs the MOMENTUM_ROTATION strategy through all 5 filter scenarios
using identical assets, history, risk engine, slippage, commissions,
and walk-forward process.

ONLY the filter gate changes between scenarios.

How filters are applied:
  The FilteredBacktestEngine wraps OverlayBacktestEngine's trade-candidate
  loop.  At the point where a signal fires and the risk engine approves,
  the filter gate is consulted before the candidate enters the pending queue.
  This is identical in mechanism to the overlay gate in Phase 5.2, but
  applied via a composable filter chain rather than a RiskOverlay subclass.

Fairness guarantee (identical for ALL scenarios):
  - Same MomentumRotationStrategy (unchanged)
  - Same asset candles (same provider, same bars)
  - Same RiskEngine.from_config(EDGE_CONFIG)
  - Same TradeSimulator (same slippage, same commission)
  - Same Portfolio (same starting balance)
  - Same warmup period (strategy.min_candles)
  - Same EdgeStabilityAnalyzer windows (60% of history)
  - Only the filter gate differs

Promotion criteria (same as Phase 5.2):
  Trades >= 500, PF >= 1.50, Exp > 0, Max DD < 15%, Robustness = ROBUST

Research only. No execution. No broker code. No live trading.
No entry logic modifications. No signal engine changes.
"""

import logging
import math
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.backtest.equity_curve import EquityCurve
from src.backtest.models import BacktestResults, BacktestTrade
from src.backtest.portfolio import Portfolio
from src.backtest.performance_metrics import (
    average_loss, average_win, expectancy, largest_loss, largest_win,
    losing_trades, max_drawdown, profit_factor, sharpe_ratio,
    win_rate, winning_trades,
)
from src.backtest.trade_simulator import ActiveTrade, TradeSimulator
from src.data.models import Candle
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.models import RiskProfile, TradeCandidate
from src.risk.risk_engine import RiskEngine
from src.signals.models import SignalType
from src.amplification_validation.filter_profiles import FilterProfile
from src.amplification_validation.asset_exclusion_filter import AssetExclusionFilter
from src.amplification_validation.quality_band_filter import QualityBandFilter
from src.amplification_validation.volatility_filter import VolatilityFilter

logger = logging.getLogger(__name__)

# Promotion criteria — identical to Phase 5.2 / 5.3
PROMO_MIN_TRADES = 500
PROMO_MIN_PF     = 1.50
PROMO_MIN_EXP    = 0.0
PROMO_MAX_DD     = 15.0  # %
VALID_ROBUSTNESS = {"ROBUST"}


@dataclass
class ScenarioResult:
    """Complete validated result for one filter scenario."""

    scenario_name:  str
    description:    str
    trades:         int
    profit_factor:  float
    expectancy:     float
    win_rate:       float
    max_drawdown:   float
    robustness:     str        # ROBUST / MARGINAL / UNSTABLE
    history_bars:   int
    data_source:    str        # "LIVE_IBKR" or "SYNTHETIC"
    is_promoted:    bool
    pass_criteria:  List[str] = field(default_factory=list)
    fail_criteria:  List[str] = field(default_factory=list)

    # Delta vs baseline (filled by comparator after all scenarios run)
    delta_pf:       Optional[float] = None
    delta_trades:   Optional[int]   = None

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"

    @property
    def promotion_status(self) -> str:
        return "PROMOTION CANDIDATE" if self.is_promoted else "NOT READY"


class _FilteredBacktestEngine:
    """Bar-by-bar backtest with composable filter gate.

    Extends the OverlayBacktestEngine pattern from Phase 5.2 to support
    multi-dimensional filtering (asset + quality score + volatility regime).

    The filter gate is applied AFTER signal generation and risk approval,
    BEFORE the candidate is placed in the pending queue — identical timing
    to the overlay gate in OverlayBacktestEngine.

    No overlay logic is applied in Phase 5.4 (overlay=None equivalent).
    """

    def __init__(
        self,
        signal_engine:    StrategyInterface,
        risk_engine:      RiskEngine,
        portfolio:        Portfolio,
        simulator:        TradeSimulator,
        asset_filter:     AssetExclusionFilter,
        quality_filter:   QualityBandFilter,
        vol_filter:       VolatilityFilter,
        min_warmup:       int = 55,
        timeframe:        str = "D1",
    ) -> None:
        self._signal    = signal_engine
        self._risk      = risk_engine
        self._portfolio = portfolio
        self._simulator = simulator
        self._asset_f   = asset_filter
        self._quality_f = quality_filter
        self._vol_f     = vol_filter
        self._warmup    = min_warmup
        self._timeframe = timeframe

    def run(self, candles: List[Candle]) -> BacktestResults:
        """Execute a full bar-by-bar backtest with the filter gate."""
        self._portfolio.reset()

        symbol    = candles[0].symbol    if candles else ""
        timeframe = candles[0].timeframe if candles else self._timeframe

        # Fast-exit: asset filter rejects this symbol entirely
        if not self._asset_f.allows(symbol):
            logger.debug(
                "filtered_backtest: skip asset=%s (asset exclusion filter)", symbol
            )
            return self._compile_empty(symbol, timeframe)

        pending:      Optional[object]      = None
        active_trade: Optional[ActiveTrade] = None

        for i, bar in enumerate(candles):

            # Step 1: Execute pending entry at this bar's open
            if pending is not None and active_trade is None:
                active_trade = self._simulator.enter_trade(pending, bar, i)
                pending = None

            # Step 2: Check if open trade exits on this bar
            if active_trade is not None:
                closed = self._simulator.check_exit(active_trade, bar, i)
                if closed is not None:
                    self._portfolio.process_trade(closed)
                    active_trade = None

            # Step 3: Generate signal at end of this bar
            if (
                i >= self._warmup
                and active_trade is None
                and pending is None
            ):
                history = candles[: i + 1]
                signal  = self._signal.generate_signal(history)

                if signal.is_actionable:
                    candidate = self._risk.evaluate(signal, history)

                    if candidate.approved:
                        # --- Filter gate (Phase 5.4 addition) -----------
                        # Quality score filter
                        if not self._quality_f.allows(signal.strength_score):
                            logger.debug(
                                "filter_gate: QUALITY blocked bar=%d score=%.1f",
                                i, signal.strength_score,
                            )
                            continue

                        # Volatility regime filter
                        if not self._vol_f.allows(history):
                            logger.debug(
                                "filter_gate: VOL blocked bar=%d", i
                            )
                            continue

                        # All filters passed
                        if candidate.position_size > 0:
                            pending = candidate

        # Force-close any still-open position at end of window
        if active_trade is not None and candles:
            closed = self._simulator.force_close(
                active_trade, candles[-1], len(candles) - 1
            )
            self._portfolio.process_trade(closed)

        return self._compile(symbol, timeframe)

    def _compile(self, symbol: str, timeframe: str) -> BacktestResults:
        trades    = self._portfolio.trades
        eq_vals   = self._portfolio.equity_values
        start_bal = self._portfolio.starting_balance
        end_bal   = self._portfolio.balance
        curve     = EquityCurve(start_bal, trades)

        return BacktestResults(
            symbol           = symbol,
            timeframe        = timeframe,
            starting_balance = start_bal,
            ending_balance   = end_bal,
            net_profit       = end_bal - start_bal,
            total_trades     = len(trades),
            winning_trades   = winning_trades(trades),
            losing_trades    = losing_trades(trades),
            win_rate         = win_rate(trades),
            profit_factor    = profit_factor(trades),
            expectancy       = expectancy(trades),
            max_drawdown     = max_drawdown(eq_vals),
            sharpe_ratio     = sharpe_ratio(eq_vals),
            average_win      = average_win(trades),
            average_loss     = average_loss(trades),
            largest_win      = largest_win(trades),
            largest_loss     = largest_loss(trades),
            equity_curve     = curve.as_tuples(),
            trades           = trades,
        )

    def _compile_empty(self, symbol: str, timeframe: str) -> BacktestResults:
        """Return a zero-trade result for excluded assets."""
        start_bal = self._portfolio.starting_balance
        return BacktestResults(
            symbol           = symbol,
            timeframe        = timeframe,
            starting_balance = start_bal,
            ending_balance   = start_bal,
            net_profit       = 0.0,
            total_trades     = 0,
            winning_trades   = 0,
            losing_trades    = 0,
            win_rate         = 0.0,
            profit_factor    = 0.0,
            expectancy       = 0.0,
            max_drawdown     = 0.0,
            sharpe_ratio     = 0.0,
            average_win      = 0.0,
            average_loss     = 0.0,
            largest_win      = 0.0,
            largest_loss     = 0.0,
            equity_curve     = [],
            trades           = [],
        )


class AmplificationValidator:
    """Runs all 5 filter scenarios and produces ScenarioResult for each.

    Parameters
    ----------
    config      : EDGE_CONFIG dict (backtest + risk sections)
    data_source : "LIVE_IBKR" or "SYNTHETIC"
    """

    def __init__(self, config: dict, data_source: str = "SYNTHETIC") -> None:
        self._config     = config
        self._data_src   = data_source
        self._stability  = EdgeStabilityAnalyzer()

    def run_scenario(
        self,
        strategy:      StrategyInterface,
        profile:       FilterProfile,
        asset_candles: Dict[str, List[Candle]],
    ) -> ScenarioResult:
        """Run one scenario and return a ScenarioResult.

        Parameters
        ----------
        strategy      : MomentumRotationStrategy (unchanged)
        profile       : FilterProfile describing what to exclude
        asset_candles : {symbol: [Candle, ...]} — same for all scenarios
        """
        logger.info(
            "amplification_validator: scenario=%s filters=[%s]",
            profile.name, profile.filter_summary,
        )

        asset_f   = AssetExclusionFilter(profile.excluded_assets)
        quality_f = QualityBandFilter(profile.excluded_score_lo, profile.excluded_score_hi)
        vol_f     = VolatilityFilter()

        # --- Run backtest for each asset, aggregate ---
        all_trades: List[BacktestTrade] = []
        all_eq:     List[float]         = []
        worst_dd:   float               = 0.0
        max_bars:   int                 = 0

        for sym, candles in asset_candles.items():
            if not candles:
                continue
            engine = self._build_engine(strategy, asset_f, quality_f, vol_f)
            result = engine.run(candles)
            all_trades.extend(result.trades)
            all_eq.extend([eq for _, eq in result.equity_curve])
            if result.max_drawdown > worst_dd:
                worst_dd = result.max_drawdown
            if len(candles) > max_bars:
                max_bars = len(candles)

        # --- Aggregate metrics ---
        total = len(all_trades)
        if total > 0:
            wins   = [t for t in all_trades if t.is_win]
            losses = [t for t in all_trades if t.is_loss]
            gw = sum(t.pnl for t in wins)
            gl = abs(sum(t.pnl for t in losses))
            pf  = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
            exp_val = sum(t.pnl for t in all_trades) / total
            wr  = len(wins) / total
        else:
            pf, exp_val, wr = 0.0, 0.0, 0.0

        # --- Robustness via 3-window stability ---
        robustness = self._compute_robustness(
            strategy, profile, asset_candles
        )

        # --- Promotion check ---
        is_promoted, pass_c, fail_c = self._evaluate_promotion(
            profile.name, total, pf, exp_val, worst_dd, robustness
        )

        logger.info(
            "scenario=%s trades=%d pf=%.2f exp=%.2f dd=%.1f%% rob=%s promoted=%s",
            profile.name, total,
            pf if not math.isinf(pf) else 99,
            exp_val, worst_dd, robustness, is_promoted,
        )

        return ScenarioResult(
            scenario_name  = profile.name,
            description    = profile.description,
            trades         = total,
            profit_factor  = pf,
            expectancy     = exp_val,
            win_rate       = wr,
            max_drawdown   = worst_dd,
            robustness     = robustness,
            history_bars   = max_bars,
            data_source    = self._data_src,
            is_promoted    = is_promoted,
            pass_criteria  = pass_c,
            fail_criteria  = fail_c,
        )

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def _compute_robustness(
        self,
        strategy:      StrategyInterface,
        profile:       FilterProfile,
        asset_candles: Dict[str, List[Candle]],
    ) -> str:
        """ROBUST / MARGINAL / UNSTABLE via 3-window EdgeStabilityAnalyzer."""

        def build_fn(s: StrategyInterface) -> _FilteredBacktestEngine:
            asset_f   = AssetExclusionFilter(profile.excluded_assets)
            quality_f = QualityBandFilter(profile.excluded_score_lo, profile.excluded_score_hi)
            vol_f     = VolatilityFilter()
            return self._build_engine(s, asset_f, quality_f, vol_f)

        rating, _ = self._stability.analyze(strategy, asset_candles, build_fn)
        return rating

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_promotion(
        name:       str,
        trades:     int,
        pf:         float,
        exp_val:    float,
        dd:         float,
        robustness: str,
    ) -> Tuple[bool, List[str], List[str]]:
        safe_pf = pf if not math.isinf(pf) else 99.0
        checks = [
            (
                trades  >= PROMO_MIN_TRADES,
                f"Trades {trades} ≥ {PROMO_MIN_TRADES}",
                f"Trades {trades} < {PROMO_MIN_TRADES}",
            ),
            (
                safe_pf >= PROMO_MIN_PF,
                f"PF {safe_pf:.2f} ≥ {PROMO_MIN_PF}",
                f"PF {safe_pf:.2f} < {PROMO_MIN_PF}",
            ),
            (
                exp_val > PROMO_MIN_EXP,
                f"Expectancy ${exp_val:.2f} > $0",
                f"Expectancy ${exp_val:.2f} ≤ $0",
            ),
            (
                dd < PROMO_MAX_DD,
                f"Max DD {dd:.1f}% < {PROMO_MAX_DD}%",
                f"Max DD {dd:.1f}% ≥ {PROMO_MAX_DD}%",
            ),
            (
                robustness in VALID_ROBUSTNESS,
                f"Robustness {robustness}",
                f"Robustness {robustness} (must be ROBUST)",
            ),
        ]
        pass_c    = [msg for ok, msg, _ in checks if ok]
        fail_c    = [msg for ok, _, msg in checks if not ok]
        promoted  = all(ok for ok, _, _ in checks)
        return promoted, pass_c, fail_c

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _build_engine(
        self,
        strategy:  StrategyInterface,
        asset_f:   AssetExclusionFilter,
        quality_f: QualityBandFilter,
        vol_f:     VolatilityFilter,
    ) -> _FilteredBacktestEngine:
        """Build a _FilteredBacktestEngine with shared config."""
        bt_cfg = self._config.get("backtest", {})
        start  = float(bt_cfg.get("starting_balance", 10_000.0))

        return _FilteredBacktestEngine(
            signal_engine  = strategy,
            risk_engine    = RiskEngine.from_config(self._config),
            portfolio      = Portfolio(start),
            simulator      = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            asset_filter   = asset_f,
            quality_filter = quality_f,
            vol_filter     = vol_f,
            min_warmup     = strategy.min_candles,
            timeframe      = "D1",
        )
