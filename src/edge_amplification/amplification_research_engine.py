"""Amplification Research Engine — Phase 5.3 master orchestrator.

Runs MOMENTUM_ROTATION through all 8 assets to collect trades, then
dispatches to all 7 analysis modules in sequence.

Produces an AmplificationResearchResult dataclass that holds every
sub-report, plus amplification candidates derived from the research.

CANDIDATE RULES (anti-curve-fitting)
--------------------------------------
A candidate exclusion may be recommended ONLY IF:
  - PF improves vs baseline
  - Trades remain >= 500
  - Expectancy remains positive
  - Robustness does not worsen
  - The excluded segment has >= 50 trades (otherwise INSUFFICIENT SAMPLE)

No parameter sweeps. No ML. No optimization.
Research only. No execution. No broker code. No live trading.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest.performance_metrics import (
    profit_factor, expectancy, win_rate, max_drawdown, winning_trades, losing_trades,
    average_win, average_loss, largest_win, largest_loss, sharpe_ratio,
)
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.risk_engine import RiskEngine
from src.risk_overlay.overlay_backtester import OverlayBacktestEngine

from src.edge_amplification.asset_contribution_analyzer import (
    AssetContributionAnalyzer, AssetContribution,
)
from src.edge_amplification.regime_contribution_analyzer import (
    RegimeContributionAnalyzer, RegimeContribution,
)
from src.edge_amplification.volatility_contribution_analyzer import (
    VolatilityContributionAnalyzer, VolatilityContribution,
)
from src.edge_amplification.trade_quality_analyzer import (
    TradeQualityAnalyzer, QualityBucketResult,
)
from src.edge_amplification.holding_period_analyzer import (
    HoldingPeriodAnalyzer, HoldingPeriodResult,
)
from src.edge_amplification.profit_concentration_analyzer import (
    ProfitConcentrationAnalyzer, ProfitConcentrationReport,
)
from src.edge_amplification.loss_concentration_analyzer import (
    LossConcentrationAnalyzer, LossConcentrationReport,
)

logger = logging.getLogger(__name__)

# Promotion thresholds (from Phase 5.2 / overall project spec)
PROMO_MIN_TRADES = 500
PROMO_MIN_PF     = 1.50
PROMO_MIN_EXP    = 0.0
PROMO_MAX_DD     = 15.0


@dataclass
class BaselineMetrics:
    """Full-strategy baseline across all assets / all trades."""
    trades:        int
    profit_factor: float
    expectancy:    float
    win_rate:      float
    max_drawdown:  float
    gross_profit:  float
    gross_loss:    float

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


@dataclass
class AmplificationCandidate:
    """A single recommended exclusion / filter with projected impact."""
    candidate_id:         int
    description:          str      # what to exclude / filter
    dimension:            str      # ASSET / REGIME / VOLATILITY / QUALITY / HOLDING
    excluded_label:       str      # e.g. "IWM", "CRISIS", "EXTREME_VOL"
    excluded_trades:      int
    remaining_trades:     int
    projected_pf:         float
    projected_expectancy: float
    projected_dd:         float
    pf_improvement:       float    # projected_pf - baseline_pf
    trade_impact:         int      # change in trade count (negative = fewer trades)
    meets_min_trades:     bool
    meets_min_pf:         bool
    meets_min_exp:        bool
    risk_note:            str      # INSUFFICIENT SAMPLE / OUT-OF-SAMPLE REQUIRED / OK

    @property
    def projected_pf_str(self) -> str:
        return "∞" if math.isinf(self.projected_pf) else f"{self.projected_pf:.2f}"


@dataclass
class AmplificationResearchResult:
    """Complete Phase 5.3 edge amplification research result."""
    baseline:              BaselineMetrics
    all_trades:            List[BacktestTrade]

    asset_contributions:   List[AssetContribution]
    regime_contributions:  List[RegimeContribution]
    vol_contributions:     List[VolatilityContribution]
    quality_buckets:       List[QualityBucketResult]
    holding_periods:       List[HoldingPeriodResult]
    profit_concentration:  ProfitConcentrationReport
    loss_concentration:    LossConcentrationReport

    candidates:            List[AmplificationCandidate]  = field(default_factory=list)
    recommendation:        str                           = ""
    data_source:           str                           = "SYNTHETIC"
    history_bars:          int                           = 0


class AmplificationResearchEngine:
    """Orchestrates all Phase 5.3 analysis modules.

    Parameters
    ----------
    config : backtest/risk config dict (same structure as Phase 5.1/5.2)
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    def run(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
        data_source:   str = "SYNTHETIC",
    ) -> AmplificationResearchResult:
        """Execute full amplification research pipeline."""

        # ── Step 1: Run baseline backtest, collect all trades ─────────────
        logger.info("amplification_engine: running baseline backtest ...")
        all_trades: List[BacktestTrade] = []
        equity_by_asset: Dict[str, List[float]] = {}

        for sym, candles in asset_candles.items():
            if not candles:
                continue
            engine = self._build_engine(strategy)
            bt = engine.run(candles)
            all_trades.extend(bt.trades)
            equity_by_asset[sym] = [v for _, v in bt.equity_curve]
            logger.info(
                "  %s: %d trades, PF=%.2f",
                sym, bt.total_trades,
                bt.profit_factor if not math.isinf(bt.profit_factor) else 99.0,
            )

        baseline = self._compute_baseline(all_trades, equity_by_asset)
        logger.info(
            "amplification_engine: baseline  trades=%d  PF=%s  Exp=%.2f  DD=%.1f%%",
            baseline.trades, baseline.pf_str, baseline.expectancy, baseline.max_drawdown,
        )

        # ── Step 2: Asset contribution ────────────────────────────────────
        logger.info("amplification_engine: analyzing asset contributions ...")
        asset_results = AssetContributionAnalyzer(
            all_trades, equity_by_asset
        ).analyze()

        # ── Step 3: Regime contribution ───────────────────────────────────
        logger.info("amplification_engine: analyzing regime contributions ...")
        regime_results = RegimeContributionAnalyzer(
            all_trades, asset_candles
        ).analyze()

        # ── Step 4: Volatility contribution ───────────────────────────────
        logger.info("amplification_engine: analyzing volatility contributions ...")
        vol_results = VolatilityContributionAnalyzer(
            all_trades, asset_candles
        ).analyze()

        # ── Step 5: Trade quality ─────────────────────────────────────────
        logger.info("amplification_engine: analyzing trade quality ...")
        quality_results = TradeQualityAnalyzer(
            all_trades, asset_candles, strategy
        ).analyze()

        # ── Step 6: Holding period ────────────────────────────────────────
        logger.info("amplification_engine: analyzing holding periods ...")
        holding_results = HoldingPeriodAnalyzer(all_trades).analyze()

        # ── Step 7: Profit / Loss concentration ───────────────────────────
        logger.info("amplification_engine: analyzing profit concentration ...")
        profit_report = ProfitConcentrationAnalyzer(
            all_trades, asset_results, regime_results, vol_results, quality_results
        ).analyze()

        logger.info("amplification_engine: analyzing loss concentration ...")
        loss_report = LossConcentrationAnalyzer(
            all_trades, asset_results, regime_results, vol_results, quality_results
        ).analyze()

        # ── Step 8: Derive amplification candidates ────────────────────────
        logger.info("amplification_engine: deriving amplification candidates ...")
        candidates = self._derive_candidates(
            all_trades, baseline,
            asset_results, regime_results, vol_results, quality_results, holding_results,
        )

        recommendation = self._recommendation(baseline, candidates)

        max_bars = max((len(c) for c in asset_candles.values()), default=0)

        return AmplificationResearchResult(
            baseline             = baseline,
            all_trades           = all_trades,
            asset_contributions  = asset_results,
            regime_contributions = regime_results,
            vol_contributions    = vol_results,
            quality_buckets      = quality_results,
            holding_periods      = holding_results,
            profit_concentration = profit_report,
            loss_concentration   = loss_report,
            candidates           = candidates,
            recommendation       = recommendation,
            data_source          = data_source,
            history_bars         = max_bars,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_engine(self, strategy: StrategyInterface) -> OverlayBacktestEngine:
        bt_cfg = self._config.get("backtest", {})
        start  = float(bt_cfg.get("starting_balance", 10_000.0))
        return OverlayBacktestEngine(
            signal_engine = strategy,
            risk_engine   = RiskEngine.from_config(self._config),
            portfolio     = Portfolio(start),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            overlay    = None,
            min_warmup = strategy.min_candles,
            timeframe  = "D1",
        )

    @staticmethod
    def _compute_baseline(
        trades: List[BacktestTrade],
        equity_by_asset: Dict[str, List[float]],
    ) -> BaselineMetrics:
        if not trades:
            return BaselineMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        wins   = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]
        gp     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))

        pf_val = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        exp_val = sum(t.pnl for t in trades) / len(trades)
        wr_val  = win_rate(trades)

        # Use worst per-asset DD as baseline portfolio DD
        worst_dd = 0.0
        for eq in equity_by_asset.values():
            dd = max_drawdown(eq)
            if dd > worst_dd:
                worst_dd = dd

        return BaselineMetrics(
            trades        = len(trades),
            profit_factor = pf_val,
            expectancy    = exp_val,
            win_rate      = wr_val,
            max_drawdown  = worst_dd,
            gross_profit  = gp,
            gross_loss    = gl,
        )

    def _derive_candidates(
        self,
        all_trades:     List[BacktestTrade],
        baseline:       BaselineMetrics,
        asset_results,
        regime_results,
        vol_results,
        quality_results,
        holding_results,
    ) -> List[AmplificationCandidate]:
        """Generate candidate exclusions only where evidence is strong enough."""
        candidates: List[AmplificationCandidate] = []
        cid = 1

        # ── Asset candidates ─────────────────────────────────────────────
        for ar in asset_results:
            if ar.insufficient_sample:
                continue
            # Only recommend excluding assets with negative expectancy AND PF < 1.0
            if ar.expectancy < 0 and ar.profit_factor < 1.0:
                remaining = [t for t in all_trades if t.symbol != ar.symbol]
                if not remaining:
                    continue
                c = self._project_candidate(
                    cid        = cid,
                    description = f"Exclude asset: {ar.symbol} (PF={ar.pf_str}, Exp={ar.expectancy:+.2f})",
                    dimension  = "ASSET",
                    label      = ar.symbol,
                    excluded   = [t for t in all_trades if t.symbol == ar.symbol],
                    remaining  = remaining,
                    baseline   = baseline,
                )
                if c is not None:
                    candidates.append(c)
                    cid += 1

        # ── Regime candidates ────────────────────────────────────────────
        for rr in regime_results:
            if rr.insufficient_sample:
                continue
            if rr.expectancy < 0 and rr.profit_factor < 1.0:
                # We'd need to know which trades fall in this regime; use loss_pct > profit_pct
                # as a proxy filter — regime destroys more than it creates
                if rr.loss_pct > rr.profit_pct + 10.0:  # material loss dominance
                    # Approximate excluded trades by regime (regime field not on trade;
                    # we use the ratio instead of exact membership)
                    excluded_est = int(rr.trades)
                    remaining_est = baseline.trades - excluded_est
                    if remaining_est < PROMO_MIN_TRADES:
                        risk = f"Would drop trades to {remaining_est} (< {PROMO_MIN_TRADES} minimum)"
                    else:
                        risk = "OUT-OF-SAMPLE REQUIRED before implementation"

                    candidates.append(AmplificationCandidate(
                        candidate_id     = cid,
                        description      = (
                            f"Filter regime: {rr.regime} "
                            f"(PF={rr.pf_str}, Exp={rr.expectancy:+.2f}, "
                            f"Loss%={rr.loss_pct:.1f}% > Profit%={rr.profit_pct:.1f}%)"
                        ),
                        dimension        = "REGIME",
                        excluded_label   = rr.regime,
                        excluded_trades  = excluded_est,
                        remaining_trades = remaining_est,
                        projected_pf     = 0.0,   # requires re-run — noted in risk
                        projected_expectancy = 0.0,
                        projected_dd     = 0.0,
                        pf_improvement   = 0.0,
                        trade_impact     = -excluded_est,
                        meets_min_trades = remaining_est >= PROMO_MIN_TRADES,
                        meets_min_pf     = False,   # unknown without re-run
                        meets_min_exp    = False,
                        risk_note        = risk,
                    ))
                    cid += 1

        # ── Volatility candidates ────────────────────────────────────────
        for vr in vol_results:
            if vr.insufficient_sample:
                continue
            if vr.expectancy < 0 and vr.profit_factor < 1.0:
                excluded_est = int(vr.trades)
                remaining_est = baseline.trades - excluded_est
                risk = (
                    f"Would drop trades to {remaining_est} (< {PROMO_MIN_TRADES} minimum)"
                    if remaining_est < PROMO_MIN_TRADES
                    else "OUT-OF-SAMPLE REQUIRED before implementation"
                )
                candidates.append(AmplificationCandidate(
                    candidate_id     = cid,
                    description      = (
                        f"Filter volatility regime: {vr.regime} "
                        f"(PF={vr.pf_str}, Exp={vr.expectancy:+.2f})"
                    ),
                    dimension        = "VOLATILITY",
                    excluded_label   = vr.regime,
                    excluded_trades  = excluded_est,
                    remaining_trades = remaining_est,
                    projected_pf     = 0.0,
                    projected_expectancy = 0.0,
                    projected_dd     = 0.0,
                    pf_improvement   = 0.0,
                    trade_impact     = -excluded_est,
                    meets_min_trades = remaining_est >= PROMO_MIN_TRADES,
                    meets_min_pf     = False,
                    meets_min_exp    = False,
                    risk_note        = risk,
                ))
                cid += 1

        # ── Quality candidates ───────────────────────────────────────────
        for qr in quality_results:
            if qr.insufficient_sample:
                continue
            if not qr.has_edge and qr.trades >= 50:
                remaining = [t for t in all_trades]  # approximation
                excluded_est = qr.trades
                remaining_est = baseline.trades - excluded_est
                risk = (
                    f"Would drop trades to {remaining_est} (< {PROMO_MIN_TRADES} minimum)"
                    if remaining_est < PROMO_MIN_TRADES
                    else "OUT-OF-SAMPLE REQUIRED before implementation"
                )
                candidates.append(AmplificationCandidate(
                    candidate_id     = cid,
                    description      = (
                        f"Raise minimum_signal_score to exclude band {qr.band} "
                        f"(PF={qr.pf_str}, Exp={qr.expectancy:+.2f}, no edge)"
                    ),
                    dimension        = "QUALITY",
                    excluded_label   = qr.band,
                    excluded_trades  = excluded_est,
                    remaining_trades = remaining_est,
                    projected_pf     = 0.0,
                    projected_expectancy = 0.0,
                    projected_dd     = 0.0,
                    pf_improvement   = 0.0,
                    trade_impact     = -excluded_est,
                    meets_min_trades = remaining_est >= PROMO_MIN_TRADES,
                    meets_min_pf     = False,
                    meets_min_exp    = False,
                    risk_note        = risk,
                ))
                cid += 1

        return candidates

    def _project_candidate(
        self,
        cid:         int,
        description: str,
        dimension:   str,
        label:       str,
        excluded:    List[BacktestTrade],
        remaining:   List[BacktestTrade],
        baseline:    BaselineMetrics,
    ) -> Optional[AmplificationCandidate]:
        """Project metrics for a concrete exclusion of specific trades."""
        if len(excluded) < 50:
            return None  # INSUFFICIENT SAMPLE — never recommend

        n_remaining = len(remaining)
        if n_remaining == 0:
            return None

        wins   = [t for t in remaining if t.is_win]
        losses = [t for t in remaining if t.is_loss]
        gp     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))

        proj_pf  = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        proj_exp = sum(t.pnl for t in remaining) / n_remaining

        safe_proj = proj_pf if not math.isinf(proj_pf) else 99.0
        safe_base = baseline.profit_factor if not math.isinf(baseline.profit_factor) else 99.0
        pf_delta  = safe_proj - safe_base

        # Only recommend if PF actually improves
        if proj_pf <= baseline.profit_factor:
            return None

        meets_trades = n_remaining >= PROMO_MIN_TRADES
        meets_pf     = proj_pf >= PROMO_MIN_PF
        meets_exp    = proj_exp > PROMO_MIN_EXP

        risk = "OUT-OF-SAMPLE REQUIRED before implementation"
        if not meets_trades:
            risk = f"Would drop trades to {n_remaining} (< {PROMO_MIN_TRADES} minimum)"

        return AmplificationCandidate(
            candidate_id         = cid,
            description          = description,
            dimension            = dimension,
            excluded_label       = label,
            excluded_trades      = len(excluded),
            remaining_trades     = n_remaining,
            projected_pf         = proj_pf,
            projected_expectancy = proj_exp,
            projected_dd         = baseline.max_drawdown,   # DD requires full re-run
            pf_improvement       = pf_delta,
            trade_impact         = -(len(excluded)),
            meets_min_trades     = meets_trades,
            meets_min_pf         = meets_pf,
            meets_min_exp        = meets_exp,
            risk_note            = risk,
        )

    @staticmethod
    def _recommendation(
        baseline:   BaselineMetrics,
        candidates: List[AmplificationCandidate],
    ) -> str:
        """Determine whether to proceed to Phase 5.4 or continue research."""
        viable = [
            c for c in candidates
            if c.meets_min_trades and c.pf_improvement > 0
        ]
        if viable:
            return "PROCEED TO PHASE 5.4"
        return "RETURN TO STRATEGY RESEARCH"
