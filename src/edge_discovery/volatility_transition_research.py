"""VOLATILITY_TRANSITION — Family #5 Edge Discovery Research Strategy.

Research Hypothesis:
    Expansion after compression is directional.
    Random expansion (without prior compression) is noise.
    Can volatility state transitions create edge?

Entry Logic (4 rules, all must pass):
    1. Confirmed compression state: ATR14 has been contracting for ≥5 bars
       (each bar's ATR < prior bar's ATR, measured as rolling 5-bar declining ATR)
    2. Expansion signal: current ATR14 > prior bar ATR by ≥10%
       (break out of compression — first expansion bar)
    3. Directional: close above open on expansion bar (bullish expansion)
    4. Close > EMA20 (minimal trend filter — we are in uptrend territory)

Compression Definition:
    Over the prior 5 bars (excluding current), the ATR must have been
    monotonically or substantially declining: at least 4 of 5 consecutive
    ATR readings below their predecessor.

Strength Score:
    Base 62.
    Expansion magnitude: ((current_atr / prior_atr) - 1.0) × 200, max +25.
    Compression depth: prior bars declining × 3, max +13.
    Capped 100, floor 62.

min_candles: 14 (ATR) + 5 (compression window) + 20 (EMA) + 10 (buffer) = 49
    Using 60 for safety.

No parameter optimization.  No machine learning.  No curve fitting.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)

_ATR_PERIOD       = 14
_COMPRESS_BARS    = 5    # bars of declining ATR required
_EXPAND_THRESH    = 0.10  # current ATR must be > prior ATR by this fraction
_MONOTONE_MIN     = 4    # minimum declining bars out of _COMPRESS_BARS
_EMA_FAST         = 20
_STRENGTH_BASE    = 62.0


class VolatilityTransitionStrategy(StrategyInterface):
    """Enter on confirmed compression-to-expansion volatility transition."""

    @property
    def name(self) -> str:
        return "VOLATILITY_TRANSITION"

    @property
    def description(self) -> str:
        return (
            f"ATR compression (≥{_MONOTONE_MIN}/{_COMPRESS_BARS} declining bars) "
            f"→ expansion (+{_EXPAND_THRESH:.0%}) + bullish bar + close>EMA{_EMA_FAST}"
        )

    @property
    def min_candles(self) -> int:
        return _ATR_PERIOD + _COMPRESS_BARS + _EMA_FAST + 10

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current_close = closes[-1]
        current_open  = candles[-1].open

        # Compute ATR series for the last _COMPRESS_BARS + 2 positions
        needed = _ATR_PERIOD + _COMPRESS_BARS + 2
        if len(candles) < needed:
            return self._none_signal(candles, "insufficient ATR history")

        # ATR at current bar and prior _COMPRESS_BARS + 1 bars
        atr_series = []
        for offset in range(_COMPRESS_BARS + 1, -1, -1):  # oldest first
            end_idx = len(candles) - offset
            atr_val = self._calc_atr_at(candles, end_idx, _ATR_PERIOD)
            if atr_val is None:
                return self._none_signal(candles, "ATR series incomplete")
            atr_series.append(atr_val)

        # atr_series[-1] = current, atr_series[-2] = prior bar
        current_atr = atr_series[-1]
        prior_atr   = atr_series[-2]
        compress_series = atr_series[:-1]  # _COMPRESS_BARS+1 values (prior bars)

        # Rule 1: Compression state — ATR declining in prior bars
        declining_count = 0
        for i in range(1, len(compress_series)):
            if compress_series[i] < compress_series[i - 1]:
                declining_count += 1

        if declining_count < _MONOTONE_MIN:
            return self._none_signal(
                candles,
                f"compression not confirmed ({declining_count}/{_COMPRESS_BARS} declining)"
            )

        # Rule 2: Expansion signal — current ATR significantly above prior
        if prior_atr <= 0:
            return self._none_signal(candles, "zero prior ATR")

        expand_ratio = (current_atr / prior_atr) - 1.0
        if expand_ratio < _EXPAND_THRESH:
            return self._none_signal(
                candles,
                f"expansion insufficient ({expand_ratio:.2%} < {_EXPAND_THRESH:.0%})"
            )

        # Rule 3: Directional — close above open (bullish expansion bar)
        if current_close <= current_open:
            return self._none_signal(candles, "expansion bar not bullish (close <= open)")

        # Rule 4: Close > EMA20
        ema20 = self._ema(closes, _EMA_FAST)
        if ema20 is None:
            return self._none_signal(candles, "EMA20 not ready")
        if current_close < ema20:
            return self._none_signal(
                candles, f"close below EMA20 ({current_close:.2f} < {ema20:.2f})"
            )

        # All rules pass
        expand_bonus   = min(25.0, expand_ratio * 200)
        compress_bonus = min(13.0, declining_count * 3.0)
        strength = min(100.0, _STRENGTH_BASE + expand_bonus + compress_bonus)

        return self._long_signal(
            candles,
            strength=strength,
            breakout=True,
            volume_conf=self._volume_ok(candles),
            reasoning=[
                f"compression {declining_count}/{_COMPRESS_BARS} declining ATR bars",
                f"expansion +{expand_ratio:.2%}",
                f"bullish bar close > open",
                f"close > EMA20={ema20:.2f}",
            ],
            ema50=self._ema(closes, 50),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_atr_at(
        candles: List[Candle], end_idx: int, period: int
    ) -> Optional[float]:
        """Compute ATR ending at end_idx (exclusive) using simple average of TRs."""
        start_idx = max(0, end_idx - period - 1)
        window    = candles[start_idx:end_idx]
        if len(window) < period + 1:
            return None
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
