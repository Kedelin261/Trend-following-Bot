"""MULTI_TIMEFRAME_ALIGNMENT — Family #6 Edge Discovery Research Strategy.

Research Hypothesis:
    Higher-timeframe trend alignment may improve trade outcome durability.
    Question: Does weekly/monthly/quarterly alignment create edge?

Single-asset daily-bar implementation:
    Since we operate on D1 candles only (no true multi-timeframe feed),
    we approximate higher-timeframe trends using rolling EMA windows that
    correspond roughly to trading-week, trading-month, and trading-quarter
    horizons:
        Weekly proxy  : EMA of last 5 bars  (≈ 1 trading week)
        Monthly proxy : EMA of last 21 bars (≈ 1 trading month)
        Quarterly proxy: EMA of last 63 bars (≈ 1 trading quarter)

    Alignment = all three proxy EMAs are in ascending order AND
                current close > all three EMAs.

Entry Logic (4 rules, all must pass):
    1. Close > EMA5 (weekly proxy bullish)
    2. Close > EMA21 (monthly proxy bullish)
    3. Close > EMA63 (quarterly proxy bullish)
    4. EMA5 > EMA21 > EMA63 (ascending alignment — all timeframes agree)

Strength Score:
    Base 62.
    Alignment spread bonus: normalized distance between EMA5 and EMA63
    relative to price: (ema5 - ema63) / price × 1000, max +25.
    Volume confirmation bonus: +13 if volume > average.
    Capped 100, floor 62.

min_candles: 63 (EMA63) + 21 (buffer) = 84

No parameter optimization.  No machine learning.  No curve fitting.
"""

import logging
from typing import List

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)

# Timeframe proxy EMA windows
_EMA_WEEKLY    = 5   # ~1 trading week
_EMA_MONTHLY   = 21  # ~1 trading month
_EMA_QUARTERLY = 63  # ~1 trading quarter

_STRENGTH_BASE = 62.0


class MultiTimeframeAlignmentStrategy(StrategyInterface):
    """Enter when weekly/monthly/quarterly EMA proxies all align bullishly."""

    @property
    def name(self) -> str:
        return "MULTI_TIMEFRAME_ALIGNMENT"

    @property
    def description(self) -> str:
        return (
            f"Close > EMA{_EMA_WEEKLY} > EMA{_EMA_MONTHLY} > EMA{_EMA_QUARTERLY} "
            f"(all {_EMA_WEEKLY}/{_EMA_MONTHLY}/{_EMA_QUARTERLY}-bar proxies aligned bullish)"
        )

    @property
    def min_candles(self) -> int:
        return _EMA_QUARTERLY + 21

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current_close = closes[-1]

        # Compute all three proxy EMAs
        ema_w = self._ema(closes, _EMA_WEEKLY)
        ema_m = self._ema(closes, _EMA_MONTHLY)
        ema_q = self._ema(closes, _EMA_QUARTERLY)

        if ema_w is None or ema_m is None or ema_q is None:
            return self._none_signal(candles, "EMA proxies not ready")

        # Rule 1: Close > weekly EMA proxy
        if current_close < ema_w:
            return self._none_signal(
                candles, f"close below EMA{_EMA_WEEKLY} ({current_close:.2f} < {ema_w:.2f})"
            )

        # Rule 2: Close > monthly EMA proxy
        if current_close < ema_m:
            return self._none_signal(
                candles, f"close below EMA{_EMA_MONTHLY} ({current_close:.2f} < {ema_m:.2f})"
            )

        # Rule 3: Close > quarterly EMA proxy
        if current_close < ema_q:
            return self._none_signal(
                candles, f"close below EMA{_EMA_QUARTERLY} ({current_close:.2f} < {ema_q:.2f})"
            )

        # Rule 4: Ascending alignment — EMA5 > EMA21 > EMA63
        if not (ema_w > ema_m > ema_q):
            return self._none_signal(
                candles,
                f"EMAs not fully aligned (EMA{_EMA_WEEKLY}={ema_w:.2f} "
                f"EMA{_EMA_MONTHLY}={ema_m:.2f} EMA{_EMA_QUARTERLY}={ema_q:.2f})"
            )

        # All rules pass — compute strength
        # Spread bonus: how far apart the EMAs are (normalized to price)
        spread_pct    = (ema_w - ema_q) / current_close if current_close > 0 else 0
        spread_bonus  = min(25.0, spread_pct * 1000)

        # Volume bonus
        vol_ok  = self._volume_ok(candles)
        vol_bonus = 13.0 if vol_ok else 0.0

        strength = min(100.0, _STRENGTH_BASE + spread_bonus + vol_bonus)

        return self._long_signal(
            candles,
            strength=strength,
            breakout=False,
            volume_conf=vol_ok,
            reasoning=[
                f"EMA{_EMA_WEEKLY}={ema_w:.2f} (weekly proxy)",
                f"EMA{_EMA_MONTHLY}={ema_m:.2f} (monthly proxy)",
                f"EMA{_EMA_QUARTERLY}={ema_q:.2f} (quarterly proxy)",
                f"all aligned ascending",
            ],
            ema50=self._ema(closes, 50),
        )
