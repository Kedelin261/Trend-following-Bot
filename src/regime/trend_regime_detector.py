"""Trend regime detector — STRONG_TREND / MODERATE_TREND / WEAK_TREND.

Measures the quality of the current trend using:
  1. EMA50 slope over the last *slope_lookback* bars
  2. Fractional separation between EMA20 and EMA50

Strong and persistent trends produce the highest quality signals.
Weak or decelerating trends are associated with more noise.

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)

MODERATE_SLOPE   = 0.005   # 0.5 % / lookback period
STRONG_SLOPE     = 0.020   # 2.0 % / lookback period
MODERATE_SEP     = 0.010   # 1.0 % EMA separation
STRONG_SEP       = 0.030   # 3.0 % EMA separation


class TrendRegime(str, Enum):
    STRONG_TREND   = "STRONG_TREND"
    MODERATE_TREND = "MODERATE_TREND"
    WEAK_TREND     = "WEAK_TREND"
    UNKNOWN        = "UNKNOWN"


class TrendRegimeDetector:
    """Classifies trend quality for each bar in a candle series.

    Parameters
    ----------
    fast_period    : fast EMA for separation (default 20)
    slow_period    : slow EMA for slope measurement (default 50)
    slope_lookback : bars over which EMA slope is computed (default 10)
    """

    def __init__(
        self,
        fast_period:    int = 20,
        slow_period:    int = 50,
        slope_lookback: int = 10,
    ) -> None:
        self.fast_period    = fast_period
        self.slow_period    = slow_period
        self.slope_lookback = slope_lookback
        self._det = TrendDetector(fast_period, slow_period)

    def classify(self, candles: List[Candle]) -> TrendRegime:
        """Return the trend regime at the most recent bar."""
        needed = self.slow_period + self.slope_lookback
        if len(candles) < needed:
            return TrendRegime.UNKNOWN

        closes = [c.close for c in candles]

        ema_slow_now  = self._det.calculate_ema(closes, self.slow_period)
        ema_slow_prev = self._det.calculate_ema(closes[: -self.slope_lookback], self.slow_period)
        ema_fast_now  = self._det.calculate_ema(closes, self.fast_period)

        if any(e is None or e == 0
               for e in (ema_slow_now, ema_slow_prev, ema_fast_now)):
            return TrendRegime.UNKNOWN

        slope = abs((ema_slow_now - ema_slow_prev) / ema_slow_prev)
        sep   = abs((ema_fast_now - ema_slow_now) / ema_slow_now)

        if slope >= STRONG_SLOPE or sep >= STRONG_SEP:
            return TrendRegime.STRONG_TREND
        if slope >= MODERATE_SLOPE or sep >= MODERATE_SEP:
            return TrendRegime.MODERATE_TREND
        return TrendRegime.WEAK_TREND

    def classify_at_timestamp(
        self, candles: List[Candle], timestamp
    ) -> TrendRegime:
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.classify(history) if history else TrendRegime.UNKNOWN
