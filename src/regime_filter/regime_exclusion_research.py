"""Regime exclusion research — tests individual exclusions and ranks by impact.

For each regime type, tests what happens when that environment is excluded:
  - Avoid STRONG_BULL
  - Avoid BEAR / STRONG_BEAR
  - Avoid EXTREME_VOL
  - Avoid CRISIS macro
  - Avoid CRASH drawdown
  - Avoid WEAK_TREND

Returns a ranked table of exclusions sorted by expectancy improvement.

No broker code.  No API calls.  Candle data only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.data.models import Candle
from src.regime_filter.filter_backtester import FilterBacktestResult, FilterBacktester
from src.regime_filter.filter_comparator import FilterComparator, FilterComparisonResult
from src.regime_filter.filter_profiles import (
    AVOID_BEAR,
    AVOID_CRISIS_AND_CRASH,
    AVOID_EXTREME_VOL,
    AVOID_STRONG_BULL,
    AVOID_STRONG_BULL_AND_EXTREME_VOL,
    AVOID_WEAK_TREND,
    BULL_ONLY,
    COMPREHENSIVE_FILTER,
    NO_FILTER,
    FilterProfile,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)

# Standard exclusion research set
EXCLUSION_PROFILES: List[FilterProfile] = [
    AVOID_STRONG_BULL,
    AVOID_BEAR,
    AVOID_EXTREME_VOL,
    AVOID_STRONG_BULL_AND_EXTREME_VOL,
    BULL_ONLY,
    AVOID_CRISIS_AND_CRASH,
    AVOID_WEAK_TREND,
    COMPREHENSIVE_FILTER,
]


@dataclass
class ExclusionResult:
    """Result of testing one regime exclusion."""

    profile:    FilterProfile
    baseline:   FilterBacktestResult
    filtered:   FilterBacktestResult
    comparison: FilterComparisonResult

    @property
    def is_beneficial(self) -> bool:
        return self.comparison.is_improvement and self.filtered.meets_quality

    @property
    def trades_retained_pct(self) -> float:
        return self.filtered.retention_pct


class RegimeExclusionResearcher:
    """Systematically tests individual regime exclusions.

    Uses a single backtest run and applies all filters post-hoc for speed.
    """

    def __init__(
        self,
        config:           dict,
        strategy_profile: StrategyProfile = None,
    ) -> None:
        self._backtester = FilterBacktester(config, strategy_profile)
        self._comparator = FilterComparator()

    def research(
        self,
        asset_candles: Dict[str, List[Candle]],
        profiles:      List[FilterProfile] = None,
    ) -> List[ExclusionResult]:
        """Test all exclusion profiles and return results ranked by expectancy."""
        test_profiles = [NO_FILTER] + (profiles or EXCLUSION_PROFILES)

        logger.info(
            "exclusion_research: running %d profiles on %d assets",
            len(test_profiles) - 1, len(asset_candles),
        )

        # Single backtest run, multiple filter applications
        all_results = self._backtester.backtest_all_profiles(asset_candles, test_profiles)
        baseline    = next(r for r in all_results if r.filter_profile.name == "NO_FILTER")

        exclusion_results = [
            ExclusionResult(
                profile    = r.filter_profile,
                baseline   = baseline,
                filtered   = r,
                comparison = self._comparator.compare(baseline, r),
            )
            for r in all_results
            if r.filter_profile.name != "NO_FILTER"
        ]

        # Sort: beneficial exclusions first, then by expectancy improvement
        return sorted(
            exclusion_results,
            key=lambda e: (int(e.is_beneficial), e.comparison.exp_improvement),
            reverse=True,
        )

    def best_exclusion(self, results: List[ExclusionResult]) -> "ExclusionResult | None":
        """Return the exclusion with the best expectancy improvement."""
        beneficial = [r for r in results if r.is_beneficial]
        return max(beneficial, key=lambda r: r.comparison.exp_improvement) if beneficial else None

    def beneficial_exclusions(self, results: List[ExclusionResult]) -> List[str]:
        """Return names of all exclusions that improve quality."""
        return [r.profile.name for r in results if r.is_beneficial]
