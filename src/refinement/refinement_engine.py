"""Refinement Engine — Phase 4.6 master orchestrator.

Runs V1 (current) and V2 (refined) strategies on each asset,
compares results, and produces a RefinementReport with a clear
promotion recommendation.

SUCCESS CRITERIA (all must pass for V2 to be promoted):
  Profit Factor ≥ 1.5
  Expectancy > $0
  Max Drawdown < 15 %
  Trades ≥ 50

If criteria are not met the system explicitly recommends returning to
research rather than proceeding to paper trading.

No broker code. No API calls. No execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults, StrategyHealth
from src.data.models import Candle
from src.refinement.asset_selector import AssetSelector
from src.refinement.strategy_comparator import ComparisonResult, StrategyComparator
from src.refinement.strategy_v2 import StrategyProfile, V1_PROFILE, V2_PROFILE

logger = logging.getLogger(__name__)

OVERALL_MIN_TRADES = 50   # aggregate across all assets


@dataclass
class RefinementReport:
    """Complete refinement analysis result."""

    # Per-asset comparisons
    comparisons:      List[ComparisonResult]

    # Aggregate V1 / V2 health
    v1_health:        StrategyHealth
    v2_health:        StrategyHealth

    # V2 aggregate metrics (across all assets)
    v2_total_trades:  int
    v2_avg_pf:        float
    v2_avg_expectancy: float
    v2_avg_drawdown:  float

    # Promotion decision
    v2_promoted:      bool
    promotion_reason: str        # "PROMOTED" or rejection explanation

    # Guidance
    recommendation:   str
    notes:            List[str] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)


class RefinementEngine:
    """Runs V1 and V2 backtests on multiple assets and generates RefinementReport.

    Parameters
    ----------
    config       : settings dict (backtest, risk sections)
    v1_profile   : current strategy profile (default = V1_PROFILE)
    v2_profile   : refined strategy profile (default = V2_PROFILE)
    asset_selector : which assets to include
    """

    def __init__(
        self,
        config:         dict,
        v1_profile:     StrategyProfile = None,
        v2_profile:     StrategyProfile = None,
        asset_selector: Optional[AssetSelector] = None,
    ) -> None:
        self._config     = config
        self._v1         = v1_profile or V1_PROFILE
        self._v2         = v2_profile or V2_PROFILE
        self._selector   = asset_selector or AssetSelector.default()
        self._comparator = StrategyComparator()

    def run(
        self,
        asset_candles: Dict[str, List[Candle]],
    ) -> RefinementReport:
        """Run both strategies on all assets and produce a RefinementReport."""
        if not asset_candles:
            raise ValueError("asset_candles must not be empty")

        comparisons: List[ComparisonResult] = []

        for symbol, candles in asset_candles.items():
            if not self._selector.is_tradeable(symbol):
                logger.info("refinement: skipping excluded asset %s", symbol)
                continue
            if not candles:
                logger.warning("refinement: no candles for %s — skipping", symbol)
                continue

            logger.info("refinement: running %s (%d candles)", symbol, len(candles))
            v1_bt = self._v1.run_backtest(candles, self._config)
            v2_bt = self._v2.run_backtest(candles, self._config)

            comparison = self._comparator.compare(
                symbol, self._v1, self._v2, v1_bt, v2_bt
            )
            comparisons.append(comparison)

        if not comparisons:
            return self._empty_report()

        return self._compile(comparisons)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compile(self, comparisons: List[ComparisonResult]) -> RefinementReport:
        # Aggregate V1 and V2 results for health check
        v1_total_trades = sum(c.v1_results.total_trades for c in comparisons)
        v2_total_trades = sum(c.v2_results.total_trades for c in comparisons)

        v2_avg_pf  = (
            sum(c.v2_results.profit_factor for c in comparisons
                if not __import__("math").isinf(c.v2_results.profit_factor))
            / max(1, sum(1 for c in comparisons
                         if not __import__("math").isinf(c.v2_results.profit_factor)))
        )
        v2_avg_exp = sum(c.v2_results.expectancy for c in comparisons) / len(comparisons)
        v2_avg_dd  = sum(c.v2_results.max_drawdown for c in comparisons) / len(comparisons)

        # Build aggregate health using aggregate metrics
        v1_health = self._aggregate_health(
            [c.v1_results for c in comparisons], v1_total_trades
        )
        v2_health = self._aggregate_health(
            [c.v2_results for c in comparisons], v2_total_trades
        )

        # Promotion: V2 must pass on ALL assets AND have ≥ OVERALL_MIN_TRADES
        all_promoted = all(c.promote_v2 for c in comparisons)
        sufficient   = v2_total_trades >= OVERALL_MIN_TRADES

        if not sufficient:
            promoted = False
            reason   = (
                f"Insufficient total trades across all assets: "
                f"{v2_total_trades} < {OVERALL_MIN_TRADES}. "
                "Fetch more historical data or widen the asset universe."
            )
        elif all_promoted:
            promoted = True
            reason   = "V2 passes all criteria across all tested assets"
        else:
            promoted = False
            failures = [c.rejection_reason for c in comparisons
                        if not c.promote_v2 and c.rejection_reason]
            reason   = " | ".join(failures[:3]) if failures else "One or more assets failed"

        # Recommendation
        if promoted:
            recommendation = (
                "PROMOTE V2 TO CANDIDATE STRATEGY — "
                "Proceed to paper trading (Phase 6) with V2 settings."
            )
        else:
            recommendation = (
                "DO NOT PROMOTE — Return to research. "
                "V2 does not yet meet all success criteria."
            )

        # Warnings
        warnings = []
        if v2_total_trades < OVERALL_MIN_TRADES:
            warnings.append(
                f"V2 only generated {v2_total_trades} total trades across "
                f"all assets (need ≥ {OVERALL_MIN_TRADES}). "
                "Results are not statistically reliable."
            )
        for c in comparisons:
            if c.trade_count_change < -10:
                warnings.append(
                    f"{c.symbol}: V2 trades ({c.v2_results.total_trades}) "
                    f"significantly fewer than V1 ({c.v1_results.total_trades})"
                )

        notes = []
        for c in comparisons:
            if c.v2_improved_pf and c.v2_improved_expectancy:
                notes.append(
                    f"{c.symbol}: V2 improves PF "
                    f"({c.v1_results.profit_factor:.2f} → "
                    f"{c.v2_results.profit_factor:.2f}) and expectancy "
                    f"(${c.v1_results.expectancy:.2f} → "
                    f"${c.v2_results.expectancy:.2f})"
                )

        logger.info(
            "refinement_engine: promoted=%s v2_trades=%d v2_pf=%.2f v2_exp=%.2f",
            promoted, v2_total_trades, v2_avg_pf, v2_avg_exp,
        )

        return RefinementReport(
            comparisons       = comparisons,
            v1_health         = v1_health,
            v2_health         = v2_health,
            v2_total_trades   = v2_total_trades,
            v2_avg_pf         = v2_avg_pf,
            v2_avg_expectancy = v2_avg_exp,
            v2_avg_drawdown   = v2_avg_dd,
            v2_promoted       = promoted,
            promotion_reason  = reason,
            recommendation    = recommendation,
            notes             = notes,
            warnings          = warnings,
        )

    @staticmethod
    def _aggregate_health(
        results: List[BacktestResults],
        total_trades: int,
    ) -> StrategyHealth:
        if not results:
            return StrategyHealth.evaluate(results[0]) if results else StrategyHealth(
                passed=False, expectancy_ok=False, profit_factor_ok=False,
                drawdown_ok=False, min_trades_ok=False,
                reasons_passed=[], reasons_failed=["No results"],
            )
        # Use the worst-performing asset's health as the aggregate health
        # (conservative: the weakest link determines overall health)
        worst = min(results, key=lambda r: r.expectancy)
        return StrategyHealth.evaluate(worst, min_trades=OVERALL_MIN_TRADES)

    def _empty_report(self) -> RefinementReport:
        empty_health = StrategyHealth(
            passed=False, expectancy_ok=False, profit_factor_ok=False,
            drawdown_ok=False, min_trades_ok=False,
            reasons_passed=[], reasons_failed=["No assets processed"],
        )
        return RefinementReport(
            comparisons       = [],
            v1_health         = empty_health,
            v2_health         = empty_health,
            v2_total_trades   = 0,
            v2_avg_pf         = 0.0,
            v2_avg_expectancy = 0.0,
            v2_avg_drawdown   = 0.0,
            v2_promoted       = False,
            promotion_reason  = "No assets processed",
            recommendation    = "No data available for comparison",
            warnings          = ["No tradeable assets with candle data"],
        )
