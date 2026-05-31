"""Strategy Analyzer — synthesises all research findings into a ResearchReport.

Aggregates results from regime analysis, ADX filtering, volatility analysis,
breakout research, parameter sweep, multi-asset testing, and benchmarking
into a single structured report with actionable research conclusions.

SAFEGUARDS:
  - Never recommends based on profit alone
  - Flags insufficient sample sizes
  - Separates observations from recommendations

No broker code. No API calls. Pure data synthesis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.research.adx_filter import ADXFilterResult
from src.research.benchmark_engine import BenchmarkResult
from src.research.breakout_quality import BreakoutQualityResult
from src.research.market_regime import MarketRegime, RegimePerformance
from src.research.multi_asset_runner import AssetResearchResult
from src.research.parameter_sweep import SweepResult
from src.research.volatility_filter import VolatilityCategory, VolatilityPerformance

logger = logging.getLogger(__name__)

OVERALL_PASS_EXPECTANCY    = 0.0
OVERALL_PASS_PROFIT_FACTOR = 1.5
OVERALL_PASS_MIN_ASSETS    = 2


@dataclass
class ResearchReport:
    """Complete research findings from Phase 4.5 strategy analysis.

    Fields are populated by StrategyAnalyzer.  Downstream phases (paper
    trading, live) should read this report and only proceed when
    edge_confirmed is True and sufficient assets are recommended.
    """

    # Multi-asset findings
    asset_results:       List[AssetResearchResult]
    recommended_assets:  List[str]
    avoid_assets:        List[str]
    best_asset:          Optional[str]
    worst_asset:         Optional[str]

    # Regime findings
    regime_performance:  Dict[MarketRegime, RegimePerformance]
    best_regime:         Optional[MarketRegime]
    regime_filter_recommended: bool

    # ADX findings
    adx_results:         List[ADXFilterResult]
    best_adx_threshold:  Optional[float]
    adx_filter_recommended: bool

    # Volatility findings
    volatility_results:  Dict[VolatilityCategory, VolatilityPerformance]
    best_volatility_category: Optional[VolatilityCategory]

    # Breakout quality findings
    breakout_results:    List[BreakoutQualityResult]
    best_breakout_threshold: Optional[float]

    # Parameter sweep findings
    sweep_results:       List[SweepResult]

    # Benchmark
    benchmark_results:   Dict[str, BenchmarkResult]

    # Overall assessment
    edge_confirmed:      bool     # True only when multiple conditions pass
    expected_expectancy: float
    expected_profit_factor: float
    expected_max_drawdown: float
    warnings:            List[str] = field(default_factory=list)
    notes:               List[str] = field(default_factory=list)


class StrategyAnalyzer:
    """Synthesises all Phase 4.5 research modules into a ResearchReport."""

    def __init__(self, min_trades_per_asset: int = 30) -> None:
        self.min_trades = min_trades_per_asset

    def analyze(
        self,
        asset_results:       List[AssetResearchResult],
        regime_performance:  Dict[MarketRegime, RegimePerformance],
        adx_results:         List[ADXFilterResult],
        volatility_results:  Dict[VolatilityCategory, VolatilityPerformance],
        breakout_results:    List[BreakoutQualityResult],
        sweep_results:       List[SweepResult],
        benchmark_results:   Dict[str, BenchmarkResult],
    ) -> ResearchReport:
        """Synthesise all research findings into a ResearchReport."""

        # ---- Asset findings ------------------------------------------
        recommended = [r.symbol for r in asset_results if r.recommended]
        avoid       = [
            r.symbol for r in asset_results
            if r.sufficient and r.results.expectancy < 0
        ]
        best_ar   = max((r for r in asset_results if r.sufficient),
                        key=lambda r: r.expectancy, default=None)
        worst_ar  = min((r for r in asset_results if r.sufficient),
                        key=lambda r: r.expectancy, default=None)

        # ---- Regime findings -----------------------------------------
        sufficient_regimes = {
            r: p for r, p in regime_performance.items() if p.sufficient
        }
        best_regime = (
            max(sufficient_regimes, key=lambda r: sufficient_regimes[r].expectancy)
            if sufficient_regimes else None
        )
        worst_regime_val = (
            min(sufficient_regimes.values(), key=lambda p: p.expectancy)
            if sufficient_regimes else None
        )
        # Recommend regime filter if BULL >> BEAR/SIDEWAYS
        regime_filter_rec = (
            best_regime == MarketRegime.BULL
            and worst_regime_val is not None
            and worst_regime_val.expectancy < 0
        )

        # ---- ADX findings --------------------------------------------
        sufficient_adx = [r for r in adx_results if r.sufficient]
        best_adx = (
            max(sufficient_adx, key=lambda r: r.expectancy, default=None)
        )
        best_adx_thresh = best_adx.threshold if best_adx else None
        adx_rec = (
            best_adx is not None
            and best_adx.threshold > 0
            and best_adx.expectancy > (adx_results[0].expectancy if adx_results else 0)
        )

        # ---- Volatility findings -------------------------------------
        best_vol = (
            max(
                (p for p in volatility_results.values() if p.sufficient),
                key=lambda p: p.expectancy,
                default=None,
            )
        )

        # ---- Breakout findings ---------------------------------------
        sufficient_bq = [r for r in breakout_results if r.sufficient]
        best_bq_thresh = (
            max(sufficient_bq, key=lambda r: r.results.expectancy).config.breakout_threshold
            if sufficient_bq else None
        )

        # ---- Expected metrics ----------------------------------------
        # Use the median of sufficient asset results as the expected value
        sufficient_assets = [r for r in asset_results if r.sufficient]
        exp_expectancy = (
            sum(r.results.expectancy for r in sufficient_assets) / len(sufficient_assets)
            if sufficient_assets else 0.0
        )
        exp_pf = (
            sum(r.results.profit_factor for r in sufficient_assets) / len(sufficient_assets)
            if sufficient_assets else 0.0
        )
        exp_dd = (
            sum(r.results.max_drawdown for r in sufficient_assets) / len(sufficient_assets)
            if sufficient_assets else 0.0
        )

        # ---- Overall edge confirmation --------------------------------
        # Edge is confirmed only when multiple conditions align
        edge = (
            len(recommended) >= OVERALL_PASS_MIN_ASSETS
            and exp_expectancy > OVERALL_PASS_EXPECTANCY
            and exp_pf >= OVERALL_PASS_PROFIT_FACTOR
        )

        # ---- Warnings ------------------------------------------------
        warnings = []
        insufficient = [r.symbol for r in asset_results if not r.sufficient]
        if insufficient:
            warnings.append(
                f"Insufficient trades on: {', '.join(insufficient)} — "
                "results not statistically reliable"
            )
        if not recommended:
            warnings.append(
                "No assets pass the strategy health check — "
                "do not proceed to paper trading"
            )
        if not edge:
            warnings.append(
                "EDGE NOT CONFIRMED — strategy has not demonstrated a "
                "consistent edge across sufficient assets and regimes"
            )

        # ---- Notes ---------------------------------------------------
        notes = []
        if best_regime:
            notes.append(f"Best performing regime: {best_regime.value}")
        if best_adx_thresh:
            notes.append(f"ADX filter ≥ {best_adx_thresh:.0f} improves selectivity")
        if best_bq_thresh:
            notes.append(
                f"Breakout threshold {best_bq_thresh * 100:.2f}% shows research promise"
            )

        report = ResearchReport(
            asset_results              = asset_results,
            recommended_assets         = recommended,
            avoid_assets               = avoid,
            best_asset                 = best_ar.symbol if best_ar else None,
            worst_asset                = worst_ar.symbol if worst_ar else None,
            regime_performance         = regime_performance,
            best_regime                = best_regime,
            regime_filter_recommended  = regime_filter_rec,
            adx_results                = adx_results,
            best_adx_threshold         = best_adx_thresh,
            adx_filter_recommended     = adx_rec,
            volatility_results         = volatility_results,
            best_volatility_category   = best_vol.category if best_vol else None,
            breakout_results           = breakout_results,
            best_breakout_threshold    = best_bq_thresh,
            sweep_results              = sweep_results,
            benchmark_results          = benchmark_results,
            edge_confirmed             = edge,
            expected_expectancy        = round(exp_expectancy, 2),
            expected_profit_factor     = round(exp_pf, 2),
            expected_max_drawdown      = round(exp_dd, 1),
            warnings                   = warnings,
            notes                      = notes,
        )

        logger.info(
            "strategy_analysis: edge_confirmed=%s recommended=%s",
            edge, recommended,
        )
        return report
