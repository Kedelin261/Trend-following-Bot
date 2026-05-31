"""Five-state market regime classifier — STRONG_BULL / BULL / SIDEWAYS / BEAR / STRONG_BEAR.

Classification uses three-EMA alignment (EMA20, EMA50, EMA200) combined
with EMA separation percentage.  STRONG states require ≥ STRONG_THRESHOLD
separation between both adjacent EMA pairs.

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)

STRONG_THRESHOLD = 0.03   # 3 % separation to qualify as STRONG


class MarketRegime(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL        = "BULL"
    SIDEWAYS    = "SIDEWAYS"
    BEAR        = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    UNKNOWN     = "UNKNOWN"


class MarketRegimeClassifier:
    """Classifies market state into five regimes using EMA20/50/200 alignment.

    Parameters
    ----------
    ema_fast     : fast EMA period (default 20)
    ema_mid      : mid EMA period (default 50)
    ema_slow     : slow EMA period (default 200)
    strong_threshold : fractional EMA separation required for STRONG state
    """

    def __init__(
        self,
        ema_fast:         int   = 20,
        ema_mid:          int   = 50,
        ema_slow:         int   = 200,
        strong_threshold: float = STRONG_THRESHOLD,
    ) -> None:
        self.ema_fast        = ema_fast
        self.ema_mid         = ema_mid
        self.ema_slow        = ema_slow
        self.strong_threshold = strong_threshold
        self._det = TrendDetector(ema_fast, ema_slow)

    def classify(self, candles: List[Candle]) -> MarketRegime:
        """Return the market regime at the most recent bar."""
        if len(candles) < self.ema_slow:
            return MarketRegime.UNKNOWN

        closes = [c.close for c in candles]
        ema_f  = self._det.calculate_ema(closes, self.ema_fast)
        ema_m  = self._det.calculate_ema(closes, self.ema_mid)
        ema_s  = self._det.calculate_ema(closes, self.ema_slow)

        if any(e is None or e == 0 for e in (ema_f, ema_m, ema_s)):
            return MarketRegime.UNKNOWN

        sep_fm = (ema_f - ema_m) / ema_m   # positive in bull
        sep_ms = (ema_m - ema_s) / ema_s   # positive in bull

        if ema_f > ema_m > ema_s:
            if sep_fm > self.strong_threshold and sep_ms > self.strong_threshold:
                return MarketRegime.STRONG_BULL
            return MarketRegime.BULL

        if ema_f < ema_m < ema_s:
            if sep_fm < -self.strong_threshold and sep_ms < -self.strong_threshold:
                return MarketRegime.STRONG_BEAR
            return MarketRegime.BEAR

        return MarketRegime.SIDEWAYS

    def classify_at_timestamp(
        self, candles: List[Candle], timestamp
    ) -> MarketRegime:
        """Classify the regime using only candles up to *timestamp*."""
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.classify(history) if history else MarketRegime.UNKNOWN

    def classify_series(self, candles: List[Candle]) -> List[MarketRegime]:
        """Return regime at every bar (useful for visual inspection)."""
        return [self.classify(candles[: i + 1]) for i in range(len(candles))]
