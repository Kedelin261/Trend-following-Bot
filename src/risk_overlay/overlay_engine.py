"""Overlay Engine — Phase 5.2 master orchestrator.

Runs the MomentumRotationStrategy through each risk overlay variant using
the same synthetic data, same risk engine, same backtester, same assets.

Only the overlay gate differs between runs.
Fairness is guaranteed by the OverlayBacktestEngine.

Produces an OverlayResearchReport with per-overlay metrics and rankings.

Research only. No execution. No broker code. No live trading.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults, BacktestTrade
from src.backtest.portfolio import Portfolio
from src.backtest.performance_metrics import (
    expectancy, max_drawdown, profit_factor,
    win_rate, winning_trades, losing_trades,
    average_win, average_loss, largest_win, largest_loss, sharpe_ratio,
)
from src.backtest.trade_simulator import TradeSimulator
from src.backtest.equity_curve import EquityCurve
from src.data.models import Candle
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.risk_overlay.base_overlay import RiskOverlay
from src.risk_overlay.overlay_backtester import OverlayBacktestEngine

logger = logging.getLogger(__name__)

# Promotion criteria (same as Phase 5.1, Phase 5.2 spec)
PROMO_MIN_TRADES  = 500     # Phase 5.2 requirement (higher than 5.1's 100)
PROMO_MIN_PF      = 1.50
PROMO_MIN_EXP     = 0.0
PROMO_MAX_DD      = 15.0    # %
PROMO_MIN_BARS    = 3000
VALID_ROBUSTNESS  = {"ROBUST"}   # Phase 5.2 requires ROBUST (not just MARGINAL)


@dataclass
class OverlayResult:
    """Complete results for one overlay variant."""
    overlay_name:   str
    trades:         int
    profit_factor:  float
    expectancy:     float
    max_drawdown:   float
    robustness:     str          # ROBUST / MARGINAL / UNSTABLE
    history_bars:   int
    is_promoted:    bool
    pass_criteria:  List[str]    = field(default_factory=list)
    fail_criteria:  List[str]    = field(default_factory=list)

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"

    @property
    def status(self) -> str:
        return "PROMOTION CANDIDATE" if self.is_promoted else "NOT READY"


@dataclass
class OverlayResearchReport:
    """Top-level Phase 5.2 overlay research report."""
    overlay_results:   List[OverlayResult]
    best_candidate:    Optional[OverlayResult]
    recommendation:    str
    promotion_exists:  bool


class OverlayEngine:
    """Runs MomentumRotationStrategy through every risk overlay variant.

    Parameters
    ----------
    config : settings dict (backtest / risk sections)
    """

    def __init__(self, config: dict) -> None:
        self._config   = config
        self._stability = EdgeStabilityAnalyzer()

    def run(
        self,
        strategy:      StrategyInterface,
        overlays:      List[RiskOverlay],
        asset_candles: Dict[str, List[Candle]],
    ) -> OverlayResearchReport:
        """Evaluate every overlay and produce a research report."""
        results = []
        for overlay in overlays:
            logger.info("overlay_engine: evaluating %s ...", overlay.name)
            result = self._evaluate_overlay(strategy, overlay, asset_candles)
            results.append(result)

        # Find promotion candidates
        candidates = [r for r in results if r.is_promoted]

        if candidates:
            best = max(candidates, key=lambda r: r.expectancy)
            recommendation = f"PROMOTE TO PHASE 5.3"
            promotion_exists = True
        else:
            best = None
            promotion_exists = False
            # Find closest near-miss
            if results:
                near_miss = max(results, key=lambda r: (r.trades, r.expectancy))
                recommendation = "RETURN TO EDGE RESEARCH"
            else:
                recommendation = "RETURN TO EDGE RESEARCH"

        return OverlayResearchReport(
            overlay_results  = results,
            best_candidate   = best,
            recommendation   = recommendation,
            promotion_exists = promotion_exists,
        )

    def _evaluate_overlay(
        self,
        strategy:      StrategyInterface,
        overlay:       RiskOverlay,
        asset_candles: Dict[str, List[Candle]],
    ) -> OverlayResult:
        """Run one overlay across all assets and compute aggregate metrics."""
        all_trades: List[BacktestTrade] = []
        worst_dd: float = 0.0

        for sym, candles in asset_candles.items():
            if not candles:
                continue
            engine = self._build_engine(strategy, overlay)
            bt     = engine.run(candles)
            all_trades.extend(bt.trades)
            # Capture worst DD from the same run (no second pass needed)
            if bt.max_drawdown > worst_dd:
                worst_dd = bt.max_drawdown

        # Compute aggregate metrics from all trades
        total = len(all_trades)
        if total > 0:
            wins   = [t for t in all_trades if t.is_win]
            losses = [t for t in all_trades if t.is_loss]
            gw = sum(t.pnl for t in wins)
            gl = abs(sum(t.pnl for t in losses))
            pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
            exp_val = sum(t.pnl for t in all_trades) / total
        else:
            pf      = 0.0
            exp_val = 0.0

        # Robustness via stability windows (separate pass — required for 3-window analysis)
        robustness = self._compute_robustness(strategy, overlay, asset_candles)

        # History (max bars tested)
        max_bars = max((len(c) for c in asset_candles.values()), default=0)

        # Promotion evaluation
        is_promoted, pass_c, fail_c = self._evaluate_promotion(
            overlay.name, total, pf, exp_val, worst_dd, robustness, max_bars
        )

        logger.info(
            "overlay_engine: %s trades=%d pf=%.2f exp=%.2f dd=%.1f%% rob=%s promoted=%s",
            overlay.name, total,
            pf if not math.isinf(pf) else 99,
            exp_val, worst_dd, robustness, is_promoted,
        )

        return OverlayResult(
            overlay_name   = overlay.name,
            trades         = total,
            profit_factor  = pf,
            expectancy     = exp_val,
            max_drawdown   = worst_dd,
            robustness     = robustness,
            history_bars   = max_bars,
            is_promoted    = is_promoted,
            pass_criteria  = pass_c,
            fail_criteria  = fail_c,
        )

    def _compute_robustness(
        self,
        strategy:      StrategyInterface,
        overlay:       RiskOverlay,
        asset_candles: Dict[str, List[Candle]],
    ) -> str:
        """Compute ROBUST / MARGINAL / UNSTABLE via 3-window stability analysis."""

        def build_fn(s: StrategyInterface):
            return self._build_engine(s, overlay)

        rating, _windows = self._stability.analyze(strategy, asset_candles, build_fn)
        return rating

    @staticmethod
    def _evaluate_promotion(
        name:       str,
        trades:     int,
        pf:         float,
        exp_val:    float,
        dd:         float,
        robustness: str,
        bars:       int,
    ):
        """Check all promotion criteria for Phase 5.2."""
        safe_pf = pf if not math.isinf(pf) else 99.0
        checks = [
            (trades   >= PROMO_MIN_TRADES,
             f"Trades {trades} ≥ {PROMO_MIN_TRADES}",
             f"Trades {trades} < {PROMO_MIN_TRADES}"),
            (safe_pf  >= PROMO_MIN_PF,
             f"PF {safe_pf:.2f} ≥ {PROMO_MIN_PF}",
             f"PF {safe_pf:.2f} < {PROMO_MIN_PF}"),
            (exp_val  > PROMO_MIN_EXP,
             f"Expectancy ${exp_val:.2f} > $0",
             f"Expectancy ${exp_val:.2f} ≤ $0"),
            (dd       < PROMO_MAX_DD,
             f"Max DD {dd:.1f}% < {PROMO_MAX_DD}%",
             f"Max DD {dd:.1f}% ≥ {PROMO_MAX_DD}%"),
            (robustness in VALID_ROBUSTNESS,
             f"Robustness {robustness}",
             f"Robustness {robustness} (must be ROBUST)"),
            (bars     >= PROMO_MIN_BARS,
             f"History {bars:,} bars ≥ {PROMO_MIN_BARS:,}",
             f"History {bars:,} bars < {PROMO_MIN_BARS:,}"),
        ]
        pass_c = [msg for ok, msg, _ in checks if ok]
        fail_c = [msg for ok, _, msg in checks if not ok]
        promoted = all(ok for ok, _, _ in checks)
        return promoted, pass_c, fail_c

    def _build_engine(
        self,
        strategy: StrategyInterface,
        overlay:  Optional[RiskOverlay],
    ) -> OverlayBacktestEngine:
        """Build an OverlayBacktestEngine with shared config and injected overlay."""
        bt_cfg = self._config.get("backtest", {})
        start  = float(bt_cfg.get("starting_balance", 10_000.0))

        risk_engine = RiskEngine.from_config(self._config)

        return OverlayBacktestEngine(
            signal_engine = strategy,
            risk_engine   = risk_engine,
            portfolio     = Portfolio(start),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            overlay    = overlay,
            min_warmup = strategy.min_candles,
            symbol     = "",
            timeframe  = "D1",
        )
