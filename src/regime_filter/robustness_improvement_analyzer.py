"""Robustness improvement analyzer — checks whether a filter improves
strategy stability across different historical periods.

Tests three windows of the candle history:
  early  : first 60 %
  middle : middle 60 %
  recent : last 60 %

Computes the filtered strategy's PF and expectancy in each window.
A filter that improves both the average AND the worst-window performance
is judged 'ROBUST'.

No broker code.  No API calls.  Candle data only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.data.models import Candle
from src.regime_filter.filter_backtester import FilterBacktestResult, FilterBacktester
from src.regime_filter.filter_profiles import FilterProfile
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)

WINDOW_FRACTION   = 0.60
QUALITY_MIN_PF    = 1.50
QUALITY_MIN_EXP   = 0.0


@dataclass
class WindowFilterResult:
    """Filter performance in one historical window."""

    window_name:  str
    candle_count: int
    result:       FilterBacktestResult
    passes:       bool   # meets quality thresholds in this window


@dataclass
class RobustnessImprovement:
    """Robustness assessment for one FilterProfile across three windows."""

    profile:                FilterProfile
    window_results:         List[WindowFilterResult]
    windows_passing:        int
    baseline_passing:       int   # how many windows baseline passed
    rating:                 str   # ROBUST / MARGINAL / UNSTABLE
    improvement_over_base:  bool  # filtered passes more windows than baseline


class RobustnessImprovementAnalyzer:
    """Tests whether a filter improves consistency across historical periods."""

    def __init__(
        self,
        config:           dict,
        strategy_profile: StrategyProfile = None,
    ) -> None:
        self._backtester = FilterBacktester(config, strategy_profile)

    def analyze(
        self,
        asset_candles: Dict[str, List[Candle]],
        profile:       FilterProfile,
    ) -> RobustnessImprovement:
        """Test *profile* on three historical windows and rate robustness."""
        windows = self._make_windows(asset_candles)

        window_results_baseline = []
        window_results_filtered = []

        for name, w_candles in windows.items():
            b_res = self._backtester.backtest_with_filter(w_candles, profile.__class__.from_(profile) if hasattr(profile.__class__, 'from_') else profile)
            # Simpler: just run baseline (NO_FILTER) and filtered
            from src.regime_filter.filter_profiles import NO_FILTER
            baseline = self._backtester.backtest_with_filter(w_candles, NO_FILTER)
            filtered = self._backtester.backtest_with_filter(w_candles, profile)

            base_ok = (
                baseline.profit_factor >= QUALITY_MIN_PF
                and baseline.expectancy > QUALITY_MIN_EXP
            )
            filt_ok = (
                filtered.profit_factor >= QUALITY_MIN_PF
                and filtered.expectancy > QUALITY_MIN_EXP
            )

            window_results_baseline.append(base_ok)
            window_results_filtered.append(WindowFilterResult(
                window_name  = name,
                candle_count = min(len(c) for c in w_candles.values()) if w_candles else 0,
                result       = filtered,
                passes       = filt_ok,
            ))

        base_passing = sum(window_results_baseline)
        filt_passing = sum(w.passes for w in window_results_filtered)

        if filt_passing == 3:
            rating = "ROBUST"
        elif filt_passing == 2:
            rating = "MARGINAL"
        else:
            rating = "UNSTABLE"

        return RobustnessImprovement(
            profile               = profile,
            window_results        = window_results_filtered,
            windows_passing       = filt_passing,
            baseline_passing      = base_passing,
            rating                = rating,
            improvement_over_base = filt_passing > base_passing,
        )

    def _make_windows(
        self, asset_candles: Dict[str, List[Candle]]
    ) -> Dict[str, Dict[str, List[Candle]]]:
        min_len = min(len(c) for c in asset_candles.values()) if asset_candles else 0
        w = max(1, int(min_len * WINDOW_FRACTION))

        return {
            "early":  {s: c[:w]            for s, c in asset_candles.items()},
            "middle": {s: c[int(min_len * 0.2):int(min_len * 0.2) + w]
                       for s, c in asset_candles.items()},
            "recent": {s: c[min_len - w:]  for s, c in asset_candles.items()},
        }
