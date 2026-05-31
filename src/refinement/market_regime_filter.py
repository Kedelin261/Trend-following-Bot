"""Three-EMA market regime filter for Strategy V2.

BULL  : EMA_fast > EMA_mid > EMA_slow  (aligned bull stack)
BEAR  : EMA_fast < EMA_mid < EMA_slow  (aligned bear stack)
SIDEWAYS: all other configurations

Only BULL regime is permitted for V2 long entries.

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)


class RegimeLabel(str, Enum):
    BULL     = "BULL"
    BEAR     = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN  = "UNKNOWN"


class MarketRegimeFilter:
    """Three-EMA alignment filter.

    Default periods: EMA20 > EMA50 > EMA200 for bull confirmation.
    All three EMAs must be in descending order for the market to be
    classified as BULL and allowed to generate long entries.

    Parameters
    ----------
    fast_period : default 20
    mid_period  : default 50
    slow_period : default 200
    """

    def __init__(
        self,
        fast_period: int = 20,
        mid_period:  int = 50,
        slow_period: int = 200,
    ) -> None:
        if not (fast_period < mid_period < slow_period):
            raise ValueError(
                f"Periods must be strictly ascending: "
                f"{fast_period} < {mid_period} < {slow_period}"
            )
        self.fast_period = fast_period
        self.mid_period  = mid_period
        self.slow_period = slow_period
        self._det = TrendDetector(fast_period=fast_period, slow_period=slow_period)

    def classify(self, candles: List[Candle]) -> RegimeLabel:
        """Classify the 3-EMA regime at the most recent bar."""
        if len(candles) < self.slow_period:
            return RegimeLabel.UNKNOWN

        closes   = [c.close for c in candles]
        ema_fast = self._det.calculate_ema(closes, self.fast_period)
        ema_mid  = self._det.calculate_ema(closes, self.mid_period)
        ema_slow = self._det.calculate_ema(closes, self.slow_period)

        if any(e is None for e in (ema_fast, ema_mid, ema_slow)):
            return RegimeLabel.UNKNOWN

        if ema_fast > ema_mid > ema_slow:
            return RegimeLabel.BULL
        if ema_fast < ema_mid < ema_slow:
            return RegimeLabel.BEAR
        return RegimeLabel.SIDEWAYS

    def is_bull_regime(self, candles: List[Candle]) -> bool:
        """Return True only when market is in a confirmed BULL regime."""
        result = self.classify(candles) == RegimeLabel.BULL
        if not result:
            logger.debug(
                "regime_filter: BLOCKED — regime=%s (%s/%s)",
                self.classify(candles).value,
                candles[-1].symbol if candles else "?",
                candles[-1].timeframe if candles else "?",
            )
        return result

    def ema_values(
        self, candles: List[Candle]
    ) -> tuple:
        """Return (ema_fast, ema_mid, ema_slow) at current bar."""
        closes = [c.close for c in candles]
        return (
            self._det.calculate_ema(closes, self.fast_period),
            self._det.calculate_ema(closes, self.mid_period),
            self._det.calculate_ema(closes, self.slow_period),
        )
