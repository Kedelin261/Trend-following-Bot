"""Trend detection using EMA 50 / EMA 200 crossover logic.

No broker code. No API calls. Consumes a list of Candle objects only.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.signals.models import TrendDirection

logger = logging.getLogger(__name__)


class TrendDetector:
    """Classifies market direction using a dual-EMA system.

    Rules
    -----
    BULLISH : close > EMA50  AND  EMA50 > EMA200
    BEARISH : close < EMA50  AND  EMA50 < EMA200
    NEUTRAL : all other cases (including insufficient data)

    Trend strength (0–100) is derived from the relative separation between
    the two EMAs and the distance of price from EMA50.  Downstream scorers
    interpret anything below ~40 as a weak trend.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Return the current (most recent) EMA value for *period*.

        Returns None when the price series is shorter than *period*.
        Calculation: seed with SMA of the first *period* bars, then apply
        the standard exponential smoother for all subsequent bars.
        """
        if len(prices) < period or period <= 0:
            return None

        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = price * multiplier + ema * (1.0 - multiplier)

        return ema

    def calculate_ema_series(
        self, prices: List[float], period: int
    ) -> List[Optional[float]]:
        """Return the full EMA series aligned with *prices*.

        Index positions before the warmup period are None.
        Exposed for downstream charting and test verification.
        """
        if not prices or period <= 0:
            return []

        result: List[Optional[float]] = [None] * len(prices)
        if len(prices) < period:
            return result

        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        result[period - 1] = ema

        for i in range(period, len(prices)):
            ema = prices[i] * multiplier + ema * (1.0 - multiplier)
            result[i] = ema

        return result

    def get_trend_direction(self, candles: List[Candle]) -> TrendDirection:
        """Classify trend as BULLISH, BEARISH, or NEUTRAL."""
        if len(candles) < self.slow_period:
            logger.debug(
                "trend_detector: insufficient candles (%d < %d), returning NEUTRAL",
                len(candles),
                self.slow_period,
            )
            return TrendDirection.NEUTRAL

        closes = [c.close for c in candles]
        current = closes[-1]
        ema50 = self.calculate_ema(closes, self.fast_period)
        ema200 = self.calculate_ema(closes, self.slow_period)

        if ema50 is None or ema200 is None:
            return TrendDirection.NEUTRAL

        if current > ema50 and ema50 > ema200:
            return TrendDirection.BULLISH
        if current < ema50 and ema50 < ema200:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def get_trend_strength(self, candles: List[Candle]) -> float:
        """Return trend strength in the range [0, 100].

        Strength reflects how far the EMAs are separated and how decisively
        price sits above/below EMA50.  NEUTRAL trends return 25.0 (baseline).

        Calibration:
          ~5 % EMA separation  → strength ≈ 75
          ~2 % EMA separation  → strength ≈ 45
          ~0.5 % separation    → strength ≈ 20
        """
        if len(candles) < self.slow_period:
            return 0.0

        closes = [c.close for c in candles]
        ema50 = self.calculate_ema(closes, self.fast_period)
        ema200 = self.calculate_ema(closes, self.slow_period)

        if ema50 is None or ema200 is None or ema200 == 0 or ema50 == 0:
            return 0.0

        direction = self.get_trend_direction(candles)
        if direction == TrendDirection.NEUTRAL:
            return 25.0

        ema_sep_pct = abs(ema50 - ema200) / ema200        # e.g. 0.05 for 5 %
        price_sep_pct = abs(closes[-1] - ema50) / ema50   # distance of close from EMA50

        # Scale: 5 % EMA gap contributes 75 pts; price gap adds up to 25 pts
        raw = ema_sep_pct * 1500.0 + price_sep_pct * 500.0
        return round(min(100.0, max(0.0, raw)), 1)
