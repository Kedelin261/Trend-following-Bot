"""RELATIVE_STRENGTH — Family #3 Edge Discovery Research Strategy.

Research Hypothesis:
    Certain assets consistently outperform peers.
    Can rolling strength rank create persistent edge?

Single-asset signal design:
    The strategy ranks THIS asset relative to a peer universe by comparing
    its recent return to a stored per-symbol baseline.  When operating on
    a single asset's candle stream, the strategy uses an internal rolling
    return percentile approach:
        - Compute rolling 20-bar return for the current bar
        - Compare to the distribution of rolling 20-bar returns over the last
          60 bars (internal percentile rank)
        - Signal when rolling return is in the top 25th percentile of its
          own recent returns AND positive (relative strength within self)
        - Additionally: close > EMA50 (trend confirmation)

This avoids requiring a cross-asset context at signal time while still
capturing the relative strength concept through self-comparison.

Entry Logic (4 rules, all must pass):
    1. 20-bar return positive (asset gaining)
    2. Rolling 20-bar return in top 25th percentile vs own last 60 readings
       (self-relative strength)
    3. Close > EMA50 (trend filter)
    4. EMA20 > EMA50 (uptrend structure)

Strength Score:
    Base 60.
    Percentile rank bonus: (percentile - 0.75) × 160, max +40.
    Capped 100, floor 60.

min_candles: 50 (EMA) + 20 (return window) + 60 (distribution) + 5 = 135

No parameter optimization.  No machine learning.  No curve fitting.
"""

import logging
from typing import List

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)

_RETURN_WINDOW   = 20   # rolling return window
_DIST_BARS       = 60   # bars of rolling-return history for percentile rank
_PERCENTILE_MIN  = 0.75  # must be in top 25th percentile
_EMA_FAST        = 20
_EMA_SLOW        = 50
_STRENGTH_BASE   = 60.0


class RelativeStrengthStrategy(StrategyInterface):
    """Enter when self-relative strength is in top quartile with trend alignment."""

    @property
    def name(self) -> str:
        return "RELATIVE_STRENGTH"

    @property
    def description(self) -> str:
        return (
            f"Rolling {_RETURN_WINDOW}-bar return in top {int((1-_PERCENTILE_MIN)*100)}th "
            f"percentile (last {_DIST_BARS} bars) + EMA{_EMA_FAST}>EMA{_EMA_SLOW}"
        )

    @property
    def min_candles(self) -> int:
        return _EMA_SLOW + _RETURN_WINDOW + _DIST_BARS + 5

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current_close = closes[-1]

        # Rule 1: 20-bar return positive
        roc_base = closes[-_RETURN_WINDOW - 1]
        if roc_base <= 0:
            return self._none_signal(candles, "zero return base price")
        current_return = (current_close - roc_base) / roc_base
        if current_return <= 0:
            return self._none_signal(candles, f"20-bar return negative ({current_return:.3%})")

        # Rule 2: rolling return percentile (self-relative strength)
        rolling_returns = []
        for i in range(-_DIST_BARS, 0):
            idx = len(closes) + i
            if idx >= _RETURN_WINDOW + 1:
                base = closes[idx - _RETURN_WINDOW - 1]
                curr = closes[idx]
                if base > 0:
                    rolling_returns.append((curr - base) / base)

        if len(rolling_returns) < _DIST_BARS // 2:
            return self._none_signal(candles, "insufficient return distribution")

        # Percentile rank: fraction of past returns below current_return
        below = sum(1 for r in rolling_returns if r < current_return)
        percentile = below / len(rolling_returns)

        if percentile < _PERCENTILE_MIN:
            return self._none_signal(
                candles,
                f"return not in top quartile (percentile={percentile:.2f} < {_PERCENTILE_MIN})"
            )

        # Rule 3: close > EMA50
        ema50 = self._ema(closes, _EMA_SLOW)
        if ema50 is None:
            return self._none_signal(candles, "EMA50 not ready")
        if current_close < ema50:
            return self._none_signal(
                candles, f"close below EMA50 ({current_close:.2f} < {ema50:.2f})"
            )

        # Rule 4: EMA20 > EMA50 (uptrend structure)
        ema20 = self._ema(closes, _EMA_FAST)
        if ema20 is None:
            return self._none_signal(candles, "EMA20 not ready")
        if ema20 <= ema50:
            return self._none_signal(candles, f"EMA20 not above EMA50")

        # All rules pass — strength
        percentile_bonus = min(40.0, (percentile - _PERCENTILE_MIN) * 160)
        strength = min(100.0, _STRENGTH_BASE + percentile_bonus)

        return self._long_signal(
            candles,
            strength=strength,
            breakout=False,
            volume_conf=self._volume_ok(candles),
            reasoning=[
                f"20-bar return={current_return:.2%}",
                f"strength percentile={percentile:.2f}",
                f"EMA20={ema20:.2f} > EMA50={ema50:.2f}",
            ],
            ema50=ema50,
        )
