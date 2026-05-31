"""Timeframe comparator — side-by-side evaluation of timeframe results.

Compares D1 vs H4, D1 vs H1, D1 vs W1 and highlights:
  - Trade count change
  - PF change
  - Expectancy change
  - Drawdown change

No broker code. No API calls. Pure analytics.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.timeframe.timeframe_backtester import TimeframeBacktestResult

QUALITY_MIN_PF  = 1.50
QUALITY_MIN_EXP = 0.0
QUALITY_MAX_DD  = 15.0


@dataclass
class TimeframeComparisonResult:
    """Side-by-side comparison of two timeframe backtest results."""

    symbol:            str
    baseline_tf:       str     # usually "D1"
    comparison_tf:     str     # "H4", "H1", or "W1"

    baseline_trades:   int
    comparison_trades: int
    trade_count_change: int    # comparison - baseline

    baseline_pf:       float
    comparison_pf:     float
    pf_change:         float   # comparison - baseline

    baseline_exp:      float
    comparison_exp:    float
    exp_change:        float   # comparison - baseline

    baseline_dd:       float
    comparison_dd:     float
    dd_change:         float   # baseline - comparison (positive = comparison lower)

    comparison_meets_quality: bool
    recommendation:    str     # "USE", "REJECT", "MONITOR"

    @property
    def adds_trades(self) -> bool:
        return self.trade_count_change > 0

    @property
    def preserves_quality(self) -> bool:
        return self.comparison_meets_quality

    @property
    def is_viable(self) -> bool:
        return self.adds_trades and self.preserves_quality


class TimeframeComparator:
    """Compares timeframe backtest results and recommends inclusion."""

    def compare(
        self,
        baseline:   TimeframeBacktestResult,
        comparison: TimeframeBacktestResult,
    ) -> TimeframeComparisonResult:
        """Generate a comparison between baseline and one other timeframe."""
        b, c = baseline.results, comparison.results

        b_pf = b.profit_factor if not math.isinf(b.profit_factor) else 99.0
        c_pf = c.profit_factor if not math.isinf(c.profit_factor) else 99.0

        trade_delta = c.total_trades - b.total_trades
        pf_delta    = c_pf - b_pf
        exp_delta   = c.expectancy - b.expectancy
        dd_delta    = b.max_drawdown - c.max_drawdown  # positive = comparison better

        meets = comparison.meets_quality

        if meets and trade_delta > 0:
            rec = "USE"
        elif meets and trade_delta == 0:
            rec = "MONITOR"
        else:
            rec = "REJECT"

        return TimeframeComparisonResult(
            symbol              = baseline.symbol,
            baseline_tf         = baseline.timeframe,
            comparison_tf       = comparison.timeframe,
            baseline_trades     = b.total_trades,
            comparison_trades   = c.total_trades,
            trade_count_change  = trade_delta,
            baseline_pf         = b_pf,
            comparison_pf       = c_pf,
            pf_change           = pf_delta,
            baseline_exp        = b.expectancy,
            comparison_exp      = c.expectancy,
            exp_change          = exp_delta,
            baseline_dd         = b.max_drawdown,
            comparison_dd       = c.max_drawdown,
            dd_change           = dd_delta,
            comparison_meets_quality = meets,
            recommendation      = rec,
        )

    def compare_all_to_baseline(
        self,
        baseline:    TimeframeBacktestResult,
        comparisons: List[TimeframeBacktestResult],
    ) -> List[TimeframeComparisonResult]:
        """Compare every timeframe against the baseline (usually D1)."""
        return [self.compare(baseline, c) for c in comparisons]

    def viable_timeframes(
        self, comparisons: List[TimeframeComparisonResult]
    ) -> List[str]:
        """Return timeframes recommended for inclusion in a combination."""
        return [c.comparison_tf for c in comparisons if c.is_viable]
