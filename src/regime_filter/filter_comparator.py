"""Filter comparator — side-by-side evaluation of filtered vs unfiltered strategy.

Computes improvement metrics:
  PF improvement         : filtered_pf - baseline_pf
  Expectancy improvement : filtered_exp - baseline_exp
  Drawdown reduction     : baseline_dd - filtered_dd  (positive = better)
  Trade reduction        : baseline_trades - filtered_trades

No broker code.  No API calls.  Pure analytics.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from src.regime_filter.filter_backtester import FilterBacktestResult


@dataclass
class FilterComparisonResult:
    """Side-by-side comparison of baseline vs one filtered strategy."""

    baseline:              FilterBacktestResult
    filtered:              FilterBacktestResult

    pf_improvement:        float   # filtered.pf - baseline.pf
    pf_improvement_pct:    float   # % change in PF
    exp_improvement:       float   # filtered.exp - baseline.exp ($/trade)
    dd_reduction:          float   # baseline.dd - filtered.dd (positive = better)
    trade_reduction:       int     # trades removed (positive = fewer trades)

    baseline_meets_quality: bool
    filtered_meets_quality: bool
    is_improvement:         bool   # True when quality improved

    @property
    def pf_direction(self) -> str:
        return "↑" if self.pf_improvement > 0 else ("↓" if self.pf_improvement < 0 else "→")

    @property
    def exp_direction(self) -> str:
        return "↑" if self.exp_improvement > 0 else ("↓" if self.exp_improvement < 0 else "→")


class FilterComparator:
    """Compares filtered strategy performance against the unfiltered baseline."""

    def compare(
        self,
        baseline: FilterBacktestResult,
        filtered: FilterBacktestResult,
    ) -> FilterComparisonResult:
        """Generate comparison metrics."""
        b_pf = baseline.profit_factor if not math.isinf(baseline.profit_factor) else 10.0
        f_pf = filtered.profit_factor if not math.isinf(filtered.profit_factor) else 10.0

        pf_imp     = f_pf - b_pf
        pf_imp_pct = (pf_imp / b_pf * 100) if b_pf > 0 else 0.0
        exp_imp    = filtered.expectancy - baseline.expectancy
        dd_red     = baseline.max_drawdown - filtered.max_drawdown
        trade_red  = baseline.total_trades_after - filtered.total_trades_after

        # Quality improved = filtered meets when baseline didn't,
        # OR filtered is materially better on key metrics
        quality_gained = (
            filtered.meets_quality and not baseline.meets_quality
        ) or (
            filtered.meets_quality
            and baseline.meets_quality
            and (pf_imp > 0.05 or exp_imp > 5.0)
        )

        return FilterComparisonResult(
            baseline             = baseline,
            filtered             = filtered,
            pf_improvement       = round(pf_imp, 3),
            pf_improvement_pct   = round(pf_imp_pct, 1),
            exp_improvement      = round(exp_imp, 2),
            dd_reduction         = round(dd_red, 1),
            trade_reduction      = trade_red,
            baseline_meets_quality = baseline.meets_quality,
            filtered_meets_quality = filtered.meets_quality,
            is_improvement       = quality_gained or (pf_imp > 0 and exp_imp > 0),
        )

    def compare_all(
        self,
        baseline: FilterBacktestResult,
        filtered_results: List[FilterBacktestResult],
    ) -> List[FilterComparisonResult]:
        """Compare baseline against every filtered result, sorted by exp improvement."""
        comparisons = [self.compare(baseline, f) for f in filtered_results
                       if f.filter_profile.name != baseline.filter_profile.name]
        return sorted(comparisons, key=lambda c: c.exp_improvement, reverse=True)

    def best_improvement(
        self, comparisons: List[FilterComparisonResult]
    ) -> Optional[FilterComparisonResult]:
        """Return the comparison with the best expectancy improvement."""
        improving = [c for c in comparisons if c.is_improvement and c.filtered.meets_quality]
        return max(improving, key=lambda c: c.exp_improvement) if improving else None
