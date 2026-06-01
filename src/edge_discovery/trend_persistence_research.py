"""TREND_PERSISTENCE — Family #1 Edge Discovery Research Strategy.

Research Hypothesis:
    Winning trades tended to persist 20+ bars.
    Can trend persistence itself become the edge?

Entry Logic (4 rules, all must pass):
    1. ≥ 20 consecutive bars with close > EMA50 (trend persistence)
    2. Close within 1 × ATR14 of EMA50 (proximity pullback entry)
    3. EMA50 slope positive over 10 bars (trend still rising)
    4. 5-day ROC > 0 (short-term momentum not reversed)

Strength Score:
    Base 60.  Each bar beyond 20 consecutive adds 1.5 points.
    Capped at 100.  Floor at 60.

min_candles: 50 (EMA) + 20 (consecutive check) + 14 (ATR) + 5 (ROC) = 89
    Using 90 to ensure EMA50 is fully warmed up.

No parameter optimization.  No machine learning.  No curve fitting.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)

_EMA50_PERIOD   = 50
_ATR_PERIOD     = 14
_CONSEC_MIN     = 20   # minimum consecutive bars above EMA50
_SLOPE_BARS     = 10   # bars back for EMA50 slope check
_ROC_BARS       = 5    # short-term ROC window
_STRENGTH_BASE  = 60.0
_STRENGTH_STEP  = 1.5  # per bar beyond minimum


class TrendPersistenceStrategy(StrategyInterface):
    """Enter after sustained trend persistence with a proximity pullback."""

    @property
    def name(self) -> str:
        return "TREND_PERSISTENCE"

    @property
    def description(self) -> str:
        return (
            f"≥{_CONSEC_MIN} consecutive bars above EMA{_EMA50_PERIOD} "
            f"+ close within 1×ATR + positive slope + ROC{_ROC_BARS}>0"
        )

    @property
    def min_candles(self) -> int:
        # EMA50 warm-up + consecutive check + ATR + slope + ROC + buffer
        return _EMA50_PERIOD + _CONSEC_MIN + _ATR_PERIOD + _SLOPE_BARS + 5

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current_close = closes[-1]

        # Rule 1 & strength: count consecutive bars above EMA50 (index-based)
        ema50 = self._ema(closes, _EMA50_PERIOD)
        if ema50 is None:
            return self._none_signal(candles, "EMA50 not ready")

        consecutive = self._count_consecutive_above_ema(candles, closes)
        if consecutive < _CONSEC_MIN:
            return self._none_signal(
                candles, f"only {consecutive} consecutive bars above EMA50"
            )

        # Rule 2: proximity pullback — close within 1×ATR of EMA50
        atr = self._calc_atr(candles)
        if atr is None:
            return self._none_signal(candles, "ATR not ready")

        distance = abs(current_close - ema50)
        if distance > atr:
            return self._none_signal(
                candles,
                f"close {current_close:.2f} not within ATR={atr:.2f} of EMA50={ema50:.2f}",
            )

        # Rule 3: EMA50 slope positive over last _SLOPE_BARS bars
        if len(closes) < _EMA50_PERIOD + _SLOPE_BARS:
            return self._none_signal(candles, "insufficient history for slope")

        ema50_prev = self._ema(closes[: -_SLOPE_BARS], _EMA50_PERIOD)
        if ema50_prev is None:
            return self._none_signal(candles, "EMA50 prev not ready")
        if ema50 <= ema50_prev:
            return self._none_signal(candles, f"EMA50 slope flat/negative")

        # Rule 4: 5-day ROC > 0
        if len(closes) < _ROC_BARS + 1:
            return self._none_signal(candles, "insufficient history for ROC")
        roc_base = closes[-_ROC_BARS - 1]
        if roc_base <= 0:
            return self._none_signal(candles, "zero ROC base price")
        roc5 = (current_close - roc_base) / roc_base
        if roc5 <= 0:
            return self._none_signal(candles, f"ROC5 negative ({roc5:.3%})")

        # All 4 rules pass — compute strength
        strength = min(100.0, _STRENGTH_BASE + (consecutive - _CONSEC_MIN) * _STRENGTH_STEP)

        return self._long_signal(
            candles,
            strength=strength,
            breakout=False,
            volume_conf=self._volume_ok(candles),
            reasoning=[
                f"{consecutive} bars above EMA50",
                f"distance={distance:.2f} ATR={atr:.2f}",
                f"slope positive",
                f"ROC5={roc5:.3%}",
            ],
            ema50=ema50,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_consecutive_above_ema(
        candles: List[Candle], closes: List[float]
    ) -> int:
        """Count consecutive bars (from current bar backwards) above EMA50.

        Uses index-based iteration — O(n) correct implementation.
        Returns the count of how many recent bars all had close > EMA50.
        """
        count = 0
        n = len(closes)
        for idx in range(n - 1, -1, -1):
            # Compute EMA50 from beginning of history up to and including idx
            window = closes[: idx + 1]
            ema = StrategyInterface._ema(window, _EMA50_PERIOD)
            if ema is None:
                break
            if closes[idx] > ema:
                count += 1
            else:
                break
        return count

    def _calc_atr(self, candles: List[Candle]) -> Optional[float]:
        """Calculate ATR14 from the last _ATR_PERIOD+1 candles."""
        if len(candles) < _ATR_PERIOD + 1:
            return None
        window = candles[-(_ATR_PERIOD + 1) :]
        trs = []
        for i in range(1, len(window)):
            tr = max(
                window[i].high - window[i].low,
                abs(window[i].high - window[i - 1].close),
                abs(window[i].low  - window[i - 1].close),
            )
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)
