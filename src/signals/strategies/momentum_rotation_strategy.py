"""MOMENTUM_ROTATION — enter when short and medium-term momentum align.

Entry logic:
  1. 10-bar rate-of-change (ROC) > 0  (short momentum positive)
  2. 50-bar rate-of-change (ROC) > 0  (medium momentum positive)
  3. Close > EMA20                    (price above short-term average)
  4. EMA20 > EMA50                    (short average above medium average)

This strategy captures momentum persistence — assets with sustained
positive price change tend to continue outperforming.
"""

import logging
from typing import List

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)


class MomentumRotationStrategy(StrategyInterface):
    """Enter when short and medium momentum align positively."""

    def __init__(
        self,
        roc_short: int   = 10,
        roc_long:  int   = 50,
        ema_fast:  int   = 20,
        ema_slow:  int   = 50,
    ) -> None:
        self._roc_short = roc_short
        self._roc_long  = roc_long
        self._ema_fast  = ema_fast
        self._ema_slow  = ema_slow

    @property
    def name(self) -> str:
        return "MOMENTUM_ROTATION"

    @property
    def description(self) -> str:
        return (f"ROC{self._roc_short}>0 AND ROC{self._roc_long}>0 AND "
                f"EMA{self._ema_fast}>EMA{self._ema_slow}")

    @property
    def min_candles(self) -> int:
        return max(self._roc_long, self._ema_slow) + 5

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current = closes[-1]

        # Rates of change
        roc_s_base = closes[-self._roc_short - 1]
        roc_l_base = closes[-self._roc_long  - 1]

        if roc_s_base == 0 or roc_l_base == 0:
            return self._none_signal(candles, "zero base price")

        roc_short = (current - roc_s_base) / roc_s_base
        roc_long  = (current - roc_l_base) / roc_l_base

        # 1. Both momentum measures positive
        if roc_short <= 0:
            return self._none_signal(candles, f"short ROC negative ({roc_short:.2%})")
        if roc_long <= 0:
            return self._none_signal(candles, f"long ROC negative ({roc_long:.2%})")

        # 2. EMA alignment
        ema_fast = self._ema(closes, self._ema_fast)
        ema_slow = self._ema(closes, self._ema_slow)

        if ema_fast is None or ema_slow is None:
            return self._none_signal(candles, "EMA not ready")

        if ema_fast <= ema_slow:
            return self._none_signal(candles, f"EMA{self._ema_fast} ≤ EMA{self._ema_slow}")

        # 3. Close above EMA20
        if current <= ema_fast:
            return self._none_signal(candles, "close below EMA20")

        # Strength from momentum magnitude
        strength = min(100.0, max(40.0, (roc_short + roc_long) * 1000))

        vol = self._volume_ok(candles)
        sup = min(c.low for c in candles[-10:])
        res = max(c.high for c in candles[-20:])

        return self._long_signal(
            candles, strength, breakout=False,
            volume_conf=vol, support=sup, resistance=res,
            reasoning=[
                f"ROC{self._roc_short}: +{roc_short:.2%}",
                f"ROC{self._roc_long}:  +{roc_long:.2%}",
                f"EMA{self._ema_fast}({ema_fast:.2f}) > "
                f"EMA{self._ema_slow}({ema_slow:.2f})",
            ],
            ema50=ema_fast, ema200=ema_slow,
        )
