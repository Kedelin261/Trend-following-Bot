"""DONCHIAN_20 — classic 20-bar channel breakout.

Entry logic:
  Current close > highest high of the previous 20 bars.

One of the oldest and most tested trend-following rules, made famous
by the Turtle Traders.  No optimisation has been applied — the 20-bar
period is the canonical default.
"""

import logging
from typing import List

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)


class DonchianStrategy(StrategyInterface):
    """Breakout above the 20-bar Donchian channel high."""

    def __init__(self, period: int = 20) -> None:
        self._period = period

    @property
    def name(self) -> str:
        return "DONCHIAN_20"

    @property
    def description(self) -> str:
        return f"Close breaks {self._period}-bar highest high (Turtle-style channel breakout)"

    @property
    def min_candles(self) -> int:
        return self._period + 2

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        # 20-bar highest high EXCLUDING today (avoid lookahead)
        prev_highs = [c.high for c in candles[-(self._period + 1):-1]]
        channel_high = max(prev_highs)
        current_close = candles[-1].close

        if current_close <= channel_high:
            return self._none_signal(candles, f"close {current_close:.2f} ≤ channel {channel_high:.2f}")

        # Breakout confirmed
        breakout_pct = (current_close - channel_high) / channel_high * 100
        strength = min(100.0, max(50.0, breakout_pct * 500))

        vol  = self._volume_ok(candles)
        sup  = min(c.low for c in candles[-self._period:])
        ema50 = self._ema([c.close for c in candles], 50)

        return self._long_signal(
            candles, strength, breakout=True,
            volume_conf=vol, support=sup, resistance=channel_high,
            reasoning=[
                f"{self._period}-bar Donchian breakout",
                f"Close {current_close:.2f} > channel high {channel_high:.2f} "
                f"(+{breakout_pct:.2f}%)",
            ],
            ema50=ema50,
        )
