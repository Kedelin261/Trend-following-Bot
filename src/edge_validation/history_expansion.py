"""History expansion — tests whether edge quality persists across increasing sample sizes.

Reuses EdgeEngine (Phase 5.0) and EdgeStabilityAnalyzer.
No duplicate logic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.data.models import Candle
from src.edge_lab.edge_engine import EdgeEngine
from src.edge_lab.edge_profile import EdgeProfile
from src.edge_lab.strategy_interface import StrategyInterface

CANDLE_COUNTS = [750, 1500, 3000, 4000, 5000]


@dataclass
class HistorySliceResult:
    candle_count: int
    actual_bars:  int          # may be < requested if data is shorter
    profile:      EdgeProfile

    @property
    def trades(self) -> int:       return self.profile.total_trades
    @property
    def pf(self) -> float:         return self.profile.profit_factor
    @property
    def expectancy(self) -> float: return self.profile.expectancy
    @property
    def drawdown(self) -> float:   return self.profile.max_drawdown
    @property
    def robustness(self) -> str:   return self.profile.robustness_rating


class HistoryExpansionResearcher:
    """Slices the maximum-available candle series to increasing window sizes
    and re-runs the strategy on each slice.
    """

    def __init__(self, config: dict, counts: List[int] = None) -> None:
        self._engine = EdgeEngine(config)
        self._counts = counts or CANDLE_COUNTS

    def research(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
    ) -> List[HistorySliceResult]:
        max_bars = min(len(c) for c in asset_candles.values()) if asset_candles else 0
        results  = []

        for count in self._counts:
            actual = min(count, max_bars)
            sliced = {s: c[-actual:] for s, c in asset_candles.items()}
            [profile] = self._engine.evaluate_all([strategy], sliced)
            results.append(HistorySliceResult(count, actual, profile))

        # Also run on full available history if it differs from last explicit count
        if max_bars > max(self._counts, default=0):
            [profile] = self._engine.evaluate_all([strategy], asset_candles)
            results.append(HistorySliceResult(max_bars, max_bars, profile))

        return results
