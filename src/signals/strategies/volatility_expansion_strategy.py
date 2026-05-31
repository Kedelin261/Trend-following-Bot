"""VOLATILITY_EXPANSION — enter when volatility expands after compression.

Entry logic:
  1. ATR compression: recent 5-bar ATR < 14-bar ATR × compression_ratio
  2. Volatility expansion: today's range > yesterday's range × expand_factor
  3. Bullish direction: close in upper 40% of today's range
  4. Close above prior 5-bar midpoint (positive momentum)

Rationale: after a quiet, contracting period the energy build-up tends
to release as a directional move.  Entering on the expansion bar catches
this move early.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.atr_calculator import ATRCalculator
from src.signals.models import Signal

logger = logging.getLogger(__name__)


class VolatilityExpansionStrategy(StrategyInterface):
    """Buy volatility expansion after ATR compression."""

    def __init__(
        self,
        atr_long:           int   = 14,
        atr_short:          int   = 5,
        compression_ratio:  float = 0.75,   # short ATR < long ATR × this
        expand_factor:      float = 1.20,   # today range > yesterday × this
        upper_band:         float = 0.60,   # close must be in top % of range
    ) -> None:
        self._long_period  = atr_long
        self._short_period = atr_short
        self._compress     = compression_ratio
        self._expand       = expand_factor
        self._upper        = upper_band
        self._atr_long     = ATRCalculator(atr_long)
        self._atr_short    = ATRCalculator(atr_short)

    @property
    def name(self) -> str:
        return "VOLATILITY_EXPANSION"

    @property
    def description(self) -> str:
        return (f"ATR{self._short_period} < ATR{self._long_period}×{self._compress} "
                f"then range expands ×{self._expand}")

    @property
    def min_candles(self) -> int:
        return self._long_period + 6

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        # Long-term ATR (14-bar)
        atr_long_val = self._atr_long.calculate_atr(candles)
        if atr_long_val is None:
            return self._none_signal(candles, "ATR-long not ready")

        # Short-term ATR (5-bar) on most recent 10 bars
        atr_short_val = self._atr_short.calculate_atr(candles[-self._short_period - 1:])
        if atr_short_val is None:
            return self._none_signal(candles, "ATR-short not ready")

        # 1. Compression: short ATR must be significantly below long ATR
        if atr_short_val >= atr_long_val * self._compress:
            return self._none_signal(candles, "no compression")

        today   = candles[-1]
        yest    = candles[-2]
        today_range = today.high - today.low
        yest_range  = yest.high  - yest.low

        # 2. Expansion: today's range > yesterday's range × factor
        if yest_range == 0 or today_range < yest_range * self._expand:
            return self._none_signal(candles, "range not expanding")

        # 3. Bullish direction: close in upper portion of range
        if today_range > 0:
            close_position = (today.close - today.low) / today_range
        else:
            close_position = 0.5
        if close_position < self._upper:
            return self._none_signal(candles, "close not in upper range band")

        # 4. Close above 5-bar midpoint
        mid5 = sum((c.high + c.low) / 2 for c in candles[-6:-1]) / 5
        if today.close <= mid5:
            return self._none_signal(candles, "close below recent midpoint")

        # Strength: how strong is the expansion
        expansion_ratio = today_range / (atr_long_val if atr_long_val > 0 else 1)
        strength = min(100.0, max(50.0, expansion_ratio * 60))

        vol = self._volume_ok(candles)
        sup = today.low
        res = max(c.high for c in candles[-20:])

        return self._long_signal(
            candles, strength, breakout=True,
            volume_conf=vol, support=sup, resistance=res,
            reasoning=[
                f"ATR compression: short={atr_short_val:.3f} < "
                f"long={atr_long_val:.3f}×{self._compress}",
                f"Range expansion: today={today_range:.3f} > "
                f"yesterday={yest_range:.3f}×{self._expand}",
                f"Bullish close position: {close_position:.1%}",
            ],
        )
