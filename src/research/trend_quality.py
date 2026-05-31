"""Trend quality analysis — measures EMA slope, separation, and coherence.

Research use: determine which trend structures produce the best strategy
results by scoring trends on strength, slope, and price alignment.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)


class TrendQualityCategory(str, Enum):
    STRONG   = "STRONG"
    MODERATE = "MODERATE"
    WEAK     = "WEAK"
    UNKNOWN  = "UNKNOWN"


@dataclass
class TrendQuality:
    """Quantified trend quality metrics at a point in time."""

    ema_fast:              Optional[float]   # current EMA50 value
    ema_slow:              Optional[float]   # current EMA200 value
    ema_separation_pct:    float             # |EMA50 - EMA200| / EMA200 × 100
    price_above_ema_fast:  bool              # close > EMA50
    price_above_ema_slow:  bool              # close > EMA200
    ema_fast_slope:        float             # (EMA50_now - EMA50_N_bars_ago) / EMA50_N_bars_ago
    ema_slow_slope:        float             # same for EMA200
    quality_score:         float             # 0–100 composite
    category:              TrendQualityCategory


class TrendQualityAnalyzer:
    """Scores the quality of the current trend using EMA structure.

    Quality score weights:
      - EMA separation (aligned direction): 40 %
      - EMA fast slope (positive for bull): 30 %
      - Price above EMA fast alignment:     30 %
    """

    def __init__(
        self,
        fast_period:       int = 50,
        slow_period:       int = 200,
        slope_lookback:    int = 5,     # bars for slope calculation
    ) -> None:
        self.fast_period    = fast_period
        self.slow_period    = slow_period
        self.slope_lookback = slope_lookback
        self._detector      = TrendDetector(fast_period, slow_period)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, candles: List[Candle]) -> TrendQuality:
        """Return trend quality for the most recent bar in *candles*."""
        closes = [c.close for c in candles]
        current = closes[-1] if closes else 0.0

        ema_fast = self._detector.calculate_ema(closes, self.fast_period)
        ema_slow = self._detector.calculate_ema(closes, self.slow_period)

        if ema_fast is None or ema_slow is None or ema_slow == 0:
            return TrendQuality(
                ema_fast=ema_fast, ema_slow=ema_slow,
                ema_separation_pct=0.0,
                price_above_ema_fast=False, price_above_ema_slow=False,
                ema_fast_slope=0.0, ema_slow_slope=0.0,
                quality_score=0.0, category=TrendQualityCategory.UNKNOWN,
            )

        sep_pct = abs(ema_fast - ema_slow) / ema_slow * 100.0

        # Slope: (now - N_bars_ago) / N_bars_ago
        fast_slope = self._ema_slope(closes, self.fast_period)
        slow_slope = self._ema_slope(closes, self.slow_period)

        above_fast = current > ema_fast
        above_slow = current > ema_slow

        score = self._score(sep_pct, fast_slope, above_fast, ema_fast, ema_slow)
        cat   = self._categorize(score)

        return TrendQuality(
            ema_fast              = ema_fast,
            ema_slow              = ema_slow,
            ema_separation_pct    = sep_pct,
            price_above_ema_fast  = above_fast,
            price_above_ema_slow  = above_slow,
            ema_fast_slope        = fast_slope,
            ema_slow_slope        = slow_slope,
            quality_score         = round(score, 1),
            category              = cat,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ema_slope(self, closes: List[float], period: int) -> float:
        """Return fractional slope of EMA over the last *slope_lookback* bars."""
        n = self.slope_lookback
        if len(closes) < period + n:
            return 0.0
        ema_now  = self._detector.calculate_ema(closes, period)
        ema_prev = self._detector.calculate_ema(closes[:-n], period)
        if ema_now is None or ema_prev is None or ema_prev == 0:
            return 0.0
        return (ema_now - ema_prev) / ema_prev

    def _score(
        self,
        sep_pct:    float,
        slope:      float,
        above_fast: bool,
        ema_fast:   float,
        ema_slow:   float,
    ) -> float:
        # EMA separation component (0–40)
        sep_score = min(40.0, sep_pct * 800.0)

        # Slope component (0–30): positive slope → full 30; flat/negative → 0
        slope_score = min(30.0, max(0.0, slope * 3000.0))

        # Price alignment component (0–30)
        align_score = 30.0 if above_fast and ema_fast > ema_slow else (
            15.0 if above_fast or ema_fast > ema_slow else 0.0
        )

        return sep_score + slope_score + align_score

    @staticmethod
    def _categorize(score: float) -> TrendQualityCategory:
        if score >= 60:
            return TrendQualityCategory.STRONG
        if score >= 30:
            return TrendQualityCategory.MODERATE
        return TrendQualityCategory.WEAK
