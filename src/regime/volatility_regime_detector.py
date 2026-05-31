"""Volatility regime detector — LOW_VOL / NORMAL_VOL / HIGH_VOL / EXTREME_VOL.

Uses ATR as a percentage of price (ATR%) to classify the current
volatility environment.  Different volatility regimes produce
significantly different strategy outcomes.

Default thresholds (ATR%):
  LOW_VOL    : < 0.5 %
  NORMAL_VOL : 0.5 % – 2.0 %
  HIGH_VOL   : 2.0 % – 4.0 %
  EXTREME_VOL: > 4.0 %

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.risk.atr_calculator import ATRCalculator

logger = logging.getLogger(__name__)


class VolatilityRegime(str, Enum):
    LOW_VOL     = "LOW_VOL"
    NORMAL_VOL  = "NORMAL_VOL"
    HIGH_VOL    = "HIGH_VOL"
    EXTREME_VOL = "EXTREME_VOL"
    UNKNOWN     = "UNKNOWN"


class VolatilityRegimeDetector:
    """Classifies volatility state using ATR-as-percentage-of-price.

    Parameters
    ----------
    period     : ATR lookback (default 14)
    low_thresh : ATR% upper bound for LOW_VOL
    high_thresh: ATR% lower bound for HIGH_VOL (NORMAL is between low and high)
    extreme_thresh: ATR% lower bound for EXTREME_VOL
    """

    def __init__(
        self,
        period:         int   = 14,
        low_thresh:     float = 0.5,
        high_thresh:    float = 2.0,
        extreme_thresh: float = 4.0,
    ) -> None:
        self.low_thresh     = low_thresh
        self.high_thresh    = high_thresh
        self.extreme_thresh = extreme_thresh
        self._atr = ATRCalculator(period=period)

    def atr_pct(self, candles: List[Candle]) -> Optional[float]:
        """Return current ATR as a percentage of last close."""
        atr = self._atr.calculate_atr(candles)
        if atr is None:
            return None
        close = candles[-1].close
        return (atr / close * 100.0) if close > 0 else None

    def classify(self, candles: List[Candle]) -> VolatilityRegime:
        """Return the volatility regime at the most recent bar."""
        pct = self.atr_pct(candles)
        if pct is None:
            return VolatilityRegime.UNKNOWN
        if pct < self.low_thresh:
            return VolatilityRegime.LOW_VOL
        if pct < self.high_thresh:
            return VolatilityRegime.NORMAL_VOL
        if pct < self.extreme_thresh:
            return VolatilityRegime.HIGH_VOL
        return VolatilityRegime.EXTREME_VOL

    def classify_at_timestamp(
        self, candles: List[Candle], timestamp
    ) -> VolatilityRegime:
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.classify(history) if history else VolatilityRegime.UNKNOWN
