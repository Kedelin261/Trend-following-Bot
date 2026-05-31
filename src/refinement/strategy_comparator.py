"""Strategy Comparator — side-by-side evaluation of V1 vs V2.

Produces a ComparisonResult with improvement metrics and a promotion
recommendation based on statistical thresholds.

SAFEGUARDS:
  - V2 must have ≥ min_trades to be considered
  - V2 must meet ALL success criteria (PF, expectancy, drawdown)
  - If V2 has fewer trades than V1 but better per-trade metrics, both are
    reported and the human analyst decides whether the quality/quantity
    trade-off is acceptable

No broker code. No API calls.
"""

import math
import logging
from dataclasses import dataclass
from typing import List, Optional

from src.backtest.models import BacktestResults, StrategyHealth
from src.refinement.strategy_v2 import StrategyProfile

logger = logging.getLogger(__name__)

PROMOTION_MIN_TRADES      = 30
PROMOTION_MIN_PF          = 1.5
PROMOTION_MIN_EXPECTANCY  = 0.0
PROMOTION_MAX_DRAWDOWN    = 15.0


@dataclass
class ComparisonResult:
    """Side-by-side performance comparison of two strategy versions."""

    symbol:        str
    v1_profile:    StrategyProfile
    v2_profile:    StrategyProfile
    v1_results:    BacktestResults
    v2_results:    BacktestResults

    # Change metrics (positive = V2 better for profit metrics)
    pf_change:          float    # V2 PF - V1 PF
    pf_change_pct:      float    # (V2 PF - V1 PF) / V1 PF × 100
    expectancy_change:  float    # V2 exp - V1 exp
    drawdown_change:    float    # V1 DD - V2 DD  (positive = V2 has less drawdown)
    trade_count_change: int      # V2 trades - V1 trades (negative = fewer trades)
    net_profit_change:  float    # V2 net - V1 net

    # Promotion decision
    promote_v2:         bool
    rejection_reason:   Optional[str]

    @property
    def v2_sufficient(self) -> bool:
        return self.v2_results.total_trades >= PROMOTION_MIN_TRADES

    @property
    def v2_improved_pf(self) -> bool:
        return self.pf_change > 0

    @property
    def v2_improved_expectancy(self) -> bool:
        return self.expectancy_change > 0


class StrategyComparator:
    """Evaluates whether V2 represents a statistically significant improvement.

    Parameters
    ----------
    min_trades    : minimum V2 trades for a valid comparison
    min_pf        : profit factor threshold for promotion
    min_expectancy: expectancy threshold for promotion
    max_drawdown  : maximum acceptable drawdown for promotion
    """

    def __init__(
        self,
        min_trades:     int   = PROMOTION_MIN_TRADES,
        min_pf:         float = PROMOTION_MIN_PF,
        min_expectancy: float = PROMOTION_MIN_EXPECTANCY,
        max_drawdown:   float = PROMOTION_MAX_DRAWDOWN,
    ) -> None:
        self.min_trades     = min_trades
        self.min_pf         = min_pf
        self.min_expectancy = min_expectancy
        self.max_drawdown   = max_drawdown

    def compare(
        self,
        symbol:      str,
        v1_profile:  StrategyProfile,
        v2_profile:  StrategyProfile,
        v1_results:  BacktestResults,
        v2_results:  BacktestResults,
    ) -> ComparisonResult:
        """Produce a ComparisonResult and determine if V2 should be promoted."""
        v1_pf = v1_results.profit_factor
        v2_pf = v2_results.profit_factor

        pf_change     = (0 if math.isinf(v2_pf) else v2_pf) - (0 if math.isinf(v1_pf) else v1_pf)
        pf_change_pct = pf_change / v1_pf * 100 if v1_pf > 0 and not math.isinf(v1_pf) else 0.0
        exp_change    = v2_results.expectancy - v1_results.expectancy
        dd_change     = v1_results.max_drawdown - v2_results.max_drawdown
        trade_change  = v2_results.total_trades - v1_results.total_trades
        profit_change = v2_results.net_profit - v1_results.net_profit

        promote, reason = self._evaluate_promotion(v2_results)

        result = ComparisonResult(
            symbol             = symbol,
            v1_profile         = v1_profile,
            v2_profile         = v2_profile,
            v1_results         = v1_results,
            v2_results         = v2_results,
            pf_change          = pf_change,
            pf_change_pct      = pf_change_pct,
            expectancy_change  = exp_change,
            drawdown_change    = dd_change,
            trade_count_change = trade_change,
            net_profit_change  = profit_change,
            promote_v2         = promote,
            rejection_reason   = reason,
        )

        logger.info(
            "comparator: %s V1→V2 | pf %.2f→%.2f | exp %.2f→%.2f | "
            "dd %.1f%%→%.1f%% | promote=%s",
            symbol,
            v1_pf if not math.isinf(v1_pf) else float("inf"),
            v2_pf if not math.isinf(v2_pf) else float("inf"),
            v1_results.expectancy,
            v2_results.expectancy,
            v1_results.max_drawdown,
            v2_results.max_drawdown,
            promote,
        )
        return result

    def compare_multi(
        self,
        comparisons: List[ComparisonResult],
    ) -> bool:
        """Return True when V2 is promotable across ALL symbols."""
        return all(c.promote_v2 for c in comparisons)

    def _evaluate_promotion(
        self, v2: BacktestResults
    ) -> tuple:
        """Apply all promotion criteria. Returns (promote, reason_or_None)."""
        if v2.total_trades < self.min_trades:
            return False, (
                f"V2 insufficient trades: {v2.total_trades} < {self.min_trades} minimum. "
                "Increase candle history or relax filters."
            )

        if v2.profit_factor < self.min_pf:
            pf_str = f"{v2.profit_factor:.2f}" if not math.isinf(v2.profit_factor) else "∞"
            return False, (
                f"V2 profit factor {pf_str} < {self.min_pf} minimum"
            )

        if v2.expectancy <= self.min_expectancy:
            return False, (
                f"V2 expectancy ${v2.expectancy:.2f} ≤ ${self.min_expectancy:.2f} minimum"
            )

        if v2.max_drawdown > self.max_drawdown:
            return False, (
                f"V2 max drawdown {v2.max_drawdown:.1f}% > {self.max_drawdown:.1f}% limit"
            )

        return True, None
