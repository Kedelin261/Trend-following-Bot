"""PULLBACK_CONTINUATION — buy the pullback in an established uptrend.

Entry logic:
  1. EMA20 > EMA50 (uptrend confirmed)
  2. Price pulled back: within last 5 bars, at least one close ≤ EMA20
  3. Resumption: current close > EMA20 × (1 + resume_threshold)
  4. Bullish bar: close > open

Rationale: capturing continuation moves after short-term consolidation
within a valid trend, rather than entering at trend highs.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)


class PullbackStrategy(StrategyInterface):
    """Buy pullback to EMA20 within an EMA20>EMA50 uptrend."""

    def __init__(
        self,
        fast_period:       int   = 20,
        slow_period:       int   = 50,
        pullback_window:   int   = 5,     # bars to look back for the pullback
        resume_threshold:  float = 0.001, # close must clear EMA20 by this %
    ) -> None:
        self._fast    = fast_period
        self._slow    = slow_period
        self._window  = pullback_window
        self._resume  = resume_threshold

    @property
    def name(self) -> str:
        return "PULLBACK_CONTINUATION"

    @property
    def description(self) -> str:
        return (f"EMA{self._fast}>{self._slow} uptrend, price pulls back "
                f"to EMA{self._fast} then resumes")

    @property
    def min_candles(self) -> int:
        return self._slow + self._window + 5

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        ema20 = self._ema(closes, self._fast)
        ema50 = self._ema(closes, self._slow)

        if ema20 is None or ema50 is None:
            return self._none_signal(candles, "EMA not ready")

        # 1. Uptrend: EMA20 > EMA50
        if ema20 <= ema50:
            return self._none_signal(candles, "no uptrend: EMA20 ≤ EMA50")

        # 2. Pullback occurred: recent bars touched or went below EMA20
        recent_closes = closes[-(self._window + 1):-1]
        recent_ema20s = [
            self._ema(closes[:-(self._window - i)], self._fast)
            for i in range(self._window)
        ]
        pullback_occurred = any(
            c <= (e or float("inf"))
            for c, e in zip(recent_closes, recent_ema20s)
            if e is not None
        )
        if not pullback_occurred:
            return self._none_signal(candles, "no recent pullback to EMA20")

        # 3. Resumption: current close cleared EMA20
        current = closes[-1]
        resume_level = ema20 * (1.0 + self._resume)
        if current < resume_level:
            return self._none_signal(candles, "no resumption above EMA20")

        # 4. Bullish bar
        if candles[-1].close <= candles[-1].open:
            return self._none_signal(candles, "not a bullish bar")

        # Strength: how far is price above EMA50 (trend conviction)
        sep_pct = (ema20 - ema50) / ema50 * 100
        strength = min(100.0, max(40.0, sep_pct * 800))

        vol = self._volume_ok(candles)
        sup = min(c.low for c in candles[-10:])
        res = max(c.high for c in candles[-20:])

        return self._long_signal(
            candles, strength, breakout=False,
            volume_conf=vol, support=sup, resistance=res,
            reasoning=[
                f"EMA{self._fast}({ema20:.2f}) > EMA{self._slow}({ema50:.2f})",
                "Price pulled back to EMA20 then resumed",
                f"Bullish close: {current:.2f}",
            ],
            ema50=ema20, ema200=ema50,
        )
