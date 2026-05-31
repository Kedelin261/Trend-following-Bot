"""Signal overlap analyzer — detects whether sub-daily signals are independent.

If H4 fires 4 times on the same day as a D1 signal, those 4 trades are
largely duplicative.  Artificially inflating trade count by counting
duplicates is not acceptable.

Overlap rule: two trades overlap when their entry timestamps share the
same calendar date.  This is conservative — some same-day entries on
different timeframes may be genuinely independent (morning vs afternoon).

PROMOTION GATING: overlap ≥ 50% causes the combination to be rejected
even if trade count and quality metrics pass.

No broker code. No API calls. Pure date arithmetic.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from src.backtest.models import BacktestTrade

logger = logging.getLogger(__name__)

MAX_ACCEPTABLE_OVERLAP = 0.50   # reject combination if overlap ≥ this


@dataclass
class OverlapResult:
    """Overlap analysis between two sets of backtest trades."""

    primary_timeframe:   str
    secondary_timeframe: str
    primary_trades:      int
    secondary_trades:    int
    overlapping_trades:  int
    unique_secondary:    int    # secondary trades not overlapping with primary
    overlap_pct:         float  # 0.0 – 1.0
    acceptable:          bool   # True when overlap < MAX_ACCEPTABLE_OVERLAP

    @property
    def effective_secondary_trades(self) -> int:
        """Conservative estimate: secondary trades after removing overlap."""
        return self.unique_secondary


@dataclass
class CombinedOverlapResult:
    """Aggregate overlap for a multi-timeframe combination."""

    timeframes:              List[str]
    total_raw_trades:        int
    estimated_unique_trades: int
    average_overlap_pct:     float
    acceptable:              bool

    @property
    def effective_trade_count(self) -> int:
        return self.estimated_unique_trades


class SignalOverlapAnalyzer:
    """Compares trade entry dates across timeframes to detect duplication.

    Parameters
    ----------
    max_overlap : maximum acceptable overlap fraction (default 0.50 = 50 %)
    """

    def __init__(self, max_overlap: float = MAX_ACCEPTABLE_OVERLAP) -> None:
        self.max_overlap = max_overlap

    # ------------------------------------------------------------------
    # Single pair comparison
    # ------------------------------------------------------------------

    def calculate_overlap(
        self,
        primary_trades:   List[BacktestTrade],
        secondary_trades: List[BacktestTrade],
        primary_tf:       str,
        secondary_tf:     str,
    ) -> OverlapResult:
        """Return overlap between primary and secondary trade sets.

        Overlap is measured by comparing entry dates.  If a secondary
        trade's entry date appears in the primary trade date set, it is
        counted as overlapping.
        """
        primary_dates = self._entry_dates(primary_trades)
        overlapping   = sum(
            1 for t in secondary_trades
            if self._entry_date(t) in primary_dates
        )
        unique = max(0, len(secondary_trades) - overlapping)
        overlap_pct = overlapping / len(secondary_trades) if secondary_trades else 0.0
        acceptable  = overlap_pct < self.max_overlap

        logger.debug(
            "overlap: %s(%d) vs %s(%d) → %.0f%% overlap acceptable=%s",
            primary_tf, len(primary_trades),
            secondary_tf, len(secondary_trades),
            overlap_pct * 100, acceptable,
        )

        return OverlapResult(
            primary_timeframe   = primary_tf,
            secondary_timeframe = secondary_tf,
            primary_trades      = len(primary_trades),
            secondary_trades    = len(secondary_trades),
            overlapping_trades  = overlapping,
            unique_secondary    = unique,
            overlap_pct         = overlap_pct,
            acceptable          = acceptable,
        )

    # ------------------------------------------------------------------
    # Multi-timeframe combination
    # ------------------------------------------------------------------

    def calculate_combined_overlap(
        self,
        results_by_tf: dict,  # {timeframe_str: List[BacktestTrade]}
    ) -> CombinedOverlapResult:
        """Estimate overlap across multiple timeframes.

        Uses the first (lowest frequency) timeframe as the primary and
        checks all others against it.  Returns combined stats.
        """
        if not results_by_tf:
            return CombinedOverlapResult([], 0, 0, 0.0, True)

        timeframes    = list(results_by_tf.keys())
        all_trades    = [t for tl in results_by_tf.values() for t in tl]
        total_raw     = len(all_trades)

        if len(timeframes) == 1:
            return CombinedOverlapResult(
                timeframes            = timeframes,
                total_raw_trades      = total_raw,
                estimated_unique_trades = total_raw,
                average_overlap_pct   = 0.0,
                acceptable            = True,
            )

        # Primary = first timeframe (lowest frequency)
        primary_tf     = timeframes[0]
        primary_trades = results_by_tf[primary_tf]
        primary_dates  = self._entry_dates(primary_trades)

        overlap_rates = []
        unique_total  = len(primary_trades)

        for tf in timeframes[1:]:
            sec_trades   = results_by_tf[tf]
            overlapping  = sum(
                1 for t in sec_trades
                if self._entry_date(t) in primary_dates
            )
            rate = overlapping / len(sec_trades) if sec_trades else 0.0
            overlap_rates.append(rate)
            unique_total += max(0, len(sec_trades) - overlapping)

        avg_overlap = sum(overlap_rates) / len(overlap_rates) if overlap_rates else 0.0
        acceptable  = avg_overlap < self.max_overlap

        return CombinedOverlapResult(
            timeframes              = timeframes,
            total_raw_trades        = total_raw,
            estimated_unique_trades = unique_total,
            average_overlap_pct     = avg_overlap,
            acceptable              = acceptable,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_date(trade: BacktestTrade) -> date:
        return trade.entry_time.date()

    @classmethod
    def _entry_dates(cls, trades: List[BacktestTrade]) -> set:
        return {cls._entry_date(t) for t in trades}
