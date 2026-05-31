"""Support and resistance level detection via swing-point analysis.

Identifies swing highs and lows, clusters nearby levels, and returns
a ranked list of support / resistance zones.

No hardcoded prices. Works equally well on SPY (400–500), BTC (20k–70k),
EURUSD (1.05–1.15), or any other liquid instrument.
"""

import logging
from typing import List, Optional

from src.data.models import Candle

logger = logging.getLogger(__name__)


class SupportResistanceDetector:
    """Detects price-based support and resistance levels automatically.

    Algorithm
    ---------
    1. Swing highs: bar *i* is a swing high if ``high[i]`` is the maximum
       over the window ``[i − lookback, i + lookback]``.
    2. Swing lows: symmetric — ``low[i]`` is the minimum over the window.
    3. Cluster: merge swing levels that are within ``cluster_tolerance``
       (as a fraction of price) of each other; use the centroid of each
       cluster as the canonical level.

    Parameters
    ----------
    lookback          : bars to check on each side for swing detection
    cluster_tolerance : fractional price distance to merge nearby levels
                        (default 0.5 % → 0.005)
    max_levels        : keep only the strongest N levels per side
    """

    def __init__(
        self,
        lookback: int = 5,
        cluster_tolerance: float = 0.005,
        max_levels: int = 10,
    ) -> None:
        self.lookback = lookback
        self.cluster_tolerance = cluster_tolerance
        self.max_levels = max_levels

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def find_swing_highs(self, candles: List[Candle]) -> List[float]:
        """Return swing-high price levels from the candle series."""
        if len(candles) < 2 * self.lookback + 1:
            return []

        highs = [c.high for c in candles]
        swing_highs: List[float] = []

        for i in range(self.lookback, len(highs) - self.lookback):
            window_left  = highs[i - self.lookback : i]
            window_right = highs[i + 1 : i + self.lookback + 1]
            if all(highs[i] >= h for h in window_left) and \
               all(highs[i] >= h for h in window_right):
                swing_highs.append(highs[i])

        return swing_highs

    def find_swing_lows(self, candles: List[Candle]) -> List[float]:
        """Return swing-low price levels from the candle series."""
        if len(candles) < 2 * self.lookback + 1:
            return []

        lows = [c.low for c in candles]
        swing_lows: List[float] = []

        for i in range(self.lookback, len(lows) - self.lookback):
            window_left  = lows[i - self.lookback : i]
            window_right = lows[i + 1 : i + self.lookback + 1]
            if all(lows[i] <= l for l in window_left) and \
               all(lows[i] <= l for l in window_right):
                swing_lows.append(lows[i])

        return swing_lows

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_levels(self, levels: List[float]) -> List[float]:
        """Merge price levels within ``cluster_tolerance`` of each other.

        Returns the centroid of each cluster, sorted ascending.
        """
        if not levels:
            return []

        sorted_lvls = sorted(levels)
        clusters: List[List[float]] = [[sorted_lvls[0]]]

        for level in sorted_lvls[1:]:
            ref = clusters[-1][0]
            if ref > 0 and abs(level - ref) / ref <= self.cluster_tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])

        centroids = [sum(c) / len(c) for c in clusters]
        return sorted(centroids)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_support_levels(self, candles: List[Candle]) -> List[float]:
        """Return clustered support levels, sorted ascending.

        Returns up to ``max_levels`` levels. Levels below current price
        are relevant as support; callers may filter further.
        """
        raw = self.find_swing_lows(candles)
        clustered = self.cluster_levels(raw)
        return clustered[-self.max_levels :]

    def get_resistance_levels(self, candles: List[Candle]) -> List[float]:
        """Return clustered resistance levels, sorted ascending.

        Returns up to ``max_levels`` levels.
        """
        raw = self.find_swing_highs(candles)
        clustered = self.cluster_levels(raw)
        return clustered[-self.max_levels :]

    def nearest_support(
        self, candles: List[Candle], current_price: float
    ) -> Optional[float]:
        """Return the closest support level at or below current price."""
        levels = [s for s in self.get_support_levels(candles) if s <= current_price]
        return max(levels) if levels else None

    def nearest_resistance(
        self, candles: List[Candle], current_price: float
    ) -> Optional[float]:
        """Return the closest resistance level at or above current price."""
        levels = [r for r in self.get_resistance_levels(candles) if r >= current_price]
        return min(levels) if levels else None
