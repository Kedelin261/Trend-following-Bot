"""BREAKOUT_CONTINUATION — Family #2 Edge Discovery Research Strategy.

Research Hypothesis:
    Certain breakouts fail immediately; others continue for extended periods.
    Breakouts preceded by consolidation (ATR compression) may have higher
    continuation quality.

Entry Logic (4 rules, all must pass):
    1. ATR compression: current ATR14 < average ATR14 over prior 20 bars
       (compression ratio threshold: current < 0.85 × average = 15% below avg)
    2. Price range compression: 10-bar high/low range < 3% of current price
    3. Breakout bar: current close > 10-bar highest close (breaks out)
    4. Breakout with positive momentum: close above EMA20

Strength Score:
    Base 60.
    Compression depth bonus: (1 - ratio) × 100 points, max +25.
    Breakout magnitude bonus: ((close / range_high) - 1) × 1000, max +15.
    Total capped at 100, floor at 60.

min_candles: 20 (ATR avg) + 14 (ATR period) + 10 (range) + 20 (EMA) + 5 = 69

No parameter optimization.  No machine learning.  No curve fitting.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)

_ATR_PERIOD         = 14
_ATR_AVG_BARS       = 20   # how many prior ATR readings to average for baseline
_COMPRESS_RATIO     = 0.85  # current ATR must be < 85% of average
_RANGE_BARS         = 10   # bars for consolidation range check
_RANGE_PCT_THRESH   = 0.03  # range must be < 3% of price
_EMA_FAST           = 20
_STRENGTH_BASE      = 60.0


class BreakoutContinuationStrategy(StrategyInterface):
    """Enter breakouts preceded by ATR and price-range compression."""

    @property
    def name(self) -> str:
        return "BREAKOUT_CONTINUATION"

    @property
    def description(self) -> str:
        return (
            f"ATR compression (current < {_COMPRESS_RATIO:.0%} of avg{_ATR_AVG_BARS}) "
            f"+ range < {_RANGE_PCT_THRESH:.0%} of price "
            f"+ break above {_RANGE_BARS}-bar high + close > EMA{_EMA_FAST}"
        )

    @property
    def min_candles(self) -> int:
        return _ATR_AVG_BARS + _ATR_PERIOD + _RANGE_BARS + _EMA_FAST + 5

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current_close = closes[-1]

        # Rule 1: ATR compression
        current_atr = self._calc_atr(candles, _ATR_PERIOD)
        if current_atr is None:
            return self._none_signal(candles, "ATR not ready")

        avg_atr = self._calc_atr_avg(candles, _ATR_PERIOD, _ATR_AVG_BARS)
        if avg_atr is None or avg_atr <= 0:
            return self._none_signal(candles, "ATR average not ready")

        compression_ratio = current_atr / avg_atr
        if compression_ratio >= _COMPRESS_RATIO:
            return self._none_signal(
                candles,
                f"ATR not compressed ({compression_ratio:.2f} >= {_COMPRESS_RATIO})"
            )

        # Rule 2: Price range compression over last _RANGE_BARS
        window = candles[-_RANGE_BARS - 1 : -1]  # exclude current bar
        if len(window) < _RANGE_BARS:
            return self._none_signal(candles, "insufficient range window")

        range_high = max(c.close for c in window)
        range_low  = min(c.close for c in window)
        range_pct  = (range_high - range_low) / current_close if current_close > 0 else 1.0

        if range_pct >= _RANGE_PCT_THRESH:
            return self._none_signal(
                candles,
                f"range too wide ({range_pct:.2%} >= {_RANGE_PCT_THRESH:.0%})"
            )

        # Rule 3: Current close breaks above the range high
        if current_close <= range_high:
            return self._none_signal(
                candles,
                f"no breakout (close {current_close:.2f} <= range_high {range_high:.2f})"
            )

        # Rule 4: Close above EMA20 (momentum alignment)
        ema20 = self._ema(closes, _EMA_FAST)
        if ema20 is None:
            return self._none_signal(candles, "EMA20 not ready")
        if current_close < ema20:
            return self._none_signal(
                candles, f"close below EMA20 ({current_close:.2f} < {ema20:.2f})"
            )

        # All rules pass — compute strength
        compress_bonus = min(25.0, (1.0 - compression_ratio) * 100)
        breakout_mag   = (current_close / range_high - 1.0) * 1000 if range_high > 0 else 0.0
        breakout_bonus = min(15.0, breakout_mag)
        strength = min(100.0, _STRENGTH_BASE + compress_bonus + breakout_bonus)

        return self._long_signal(
            candles,
            strength=strength,
            breakout=True,
            volume_conf=self._volume_ok(candles),
            reasoning=[
                f"ATR compressed {compression_ratio:.2f}x vs avg",
                f"range={range_pct:.2%} of price",
                f"breakout above {range_high:.2f}",
                f"close > EMA20 ({ema20:.2f})",
            ],
            ema50=self._ema(closes, 50),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_atr(candles: List[Candle], period: int = 14) -> Optional[float]:
        """Simple ATR from last period+1 candles."""
        if len(candles) < period + 1:
            return None
        window = candles[-(period + 1):]
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

    @staticmethod
    def _calc_atr_avg(
        candles: List[Candle], atr_period: int, avg_bars: int
    ) -> Optional[float]:
        """Average of the last avg_bars ATR readings (each computed from prior bars).

        For each of the last avg_bars positions, compute the ATR ending at that
        bar and average them to get a baseline ATR level.
        """
        needed = atr_period + avg_bars + 1
        if len(candles) < needed:
            return None

        atrs = []
        # Compute ATR for positions ending at [-(avg_bars+1)...-2]  (exclude current)
        for offset in range(1, avg_bars + 1):
            end_idx = len(candles) - offset
            window  = candles[max(0, end_idx - atr_period - 1): end_idx]
            if len(window) < atr_period + 1:
                continue
            trs = []
            for i in range(1, len(window)):
                tr = max(
                    window[i].high - window[i].low,
                    abs(window[i].high - window[i - 1].close),
                    abs(window[i].low  - window[i - 1].close),
                )
                trs.append(tr)
            if trs:
                atrs.append(sum(trs) / len(trs))

        if not atrs:
            return None
        return sum(atrs) / len(atrs)
