"""MARKET_LEADERSHIP — Family #4 Edge Discovery Research Strategy.

Research Hypothesis:
    Leadership assets (those with ROC significantly above median) move first.
    Question: Can leadership predict future continuation?

Single-asset signal design:
    Market leadership is measured by comparing the asset's own recent ROC
    to a leadership threshold derived from the asset's own history.
    Leadership = current 10-bar ROC is above the 70th percentile of the
    asset's own 10-bar ROC distribution over the last 80 bars.

    Additional filters:
    - ROC must be strictly positive (absolute gain)
    - Recent acceleration: 10-bar ROC > 5-bar ROC baseline (momentum building)
    - Close > EMA50 (trend alignment)

Entry Logic (4 rules, all must pass):
    1. 10-bar ROC positive
    2. 10-bar ROC above 70th percentile of own 80-bar ROC distribution
       (leadership threshold)
    3. Short-term acceleration: 10-bar ROC > 3-bar average of recent ROC
       (momentum building, not fading)
    4. Close > EMA50

Strength Score:
    Base 65.
    Leadership intensity: (percentile - 0.70) × 116, max +30.
    Capped 100, floor 65.

min_candles: 50 (EMA) + 10 (ROC) + 80 (distribution) + 5 = 145

No parameter optimization.  No machine learning.  No curve fitting.
"""

import logging
from typing import List

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal

logger = logging.getLogger(__name__)

_ROC_WINDOW      = 10   # leadership ROC window
_DIST_BARS       = 80   # bars of ROC history for percentile
_LEAD_PERCENTILE = 0.70  # must be in top 30th percentile
_EMA_SLOW        = 50
_ACCEL_BARS      = 3    # recent ROC bars for acceleration check
_STRENGTH_BASE   = 65.0


class MarketLeadershipStrategy(StrategyInterface):
    """Enter when asset shows leadership ROC + trend alignment + acceleration."""

    @property
    def name(self) -> str:
        return "MARKET_LEADERSHIP"

    @property
    def description(self) -> str:
        return (
            f"{_ROC_WINDOW}-bar ROC in top {int((1-_LEAD_PERCENTILE)*100)}th "
            f"percentile (last {_DIST_BARS} bars) + acceleration + close>EMA{_EMA_SLOW}"
        )

    @property
    def min_candles(self) -> int:
        return _EMA_SLOW + _ROC_WINDOW + _DIST_BARS + _ACCEL_BARS + 5

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_signal(self, candles: List[Candle]) -> Signal:
        if len(candles) < self.min_candles:
            return self._none_signal(candles, "insufficient history")

        closes = [c.close for c in candles]
        current_close = closes[-1]

        # Rule 1: 10-bar ROC positive
        roc_base = closes[-_ROC_WINDOW - 1]
        if roc_base <= 0:
            return self._none_signal(candles, "zero ROC base")
        current_roc = (current_close - roc_base) / roc_base
        if current_roc <= 0:
            return self._none_signal(
                candles, f"10-bar ROC not positive ({current_roc:.3%})"
            )

        # Rule 2: Leadership threshold — top 30th percentile of own ROC history
        roc_history = []
        for i in range(-_DIST_BARS, 0):
            idx = len(closes) + i
            if idx >= _ROC_WINDOW + 1:
                base = closes[idx - _ROC_WINDOW - 1]
                curr = closes[idx]
                if base > 0:
                    roc_history.append((curr - base) / base)

        if len(roc_history) < _DIST_BARS // 2:
            return self._none_signal(candles, "insufficient ROC distribution")

        below = sum(1 for r in roc_history if r < current_roc)
        percentile = below / len(roc_history)

        if percentile < _LEAD_PERCENTILE:
            return self._none_signal(
                candles,
                f"ROC not leadership level (percentile={percentile:.2f} < {_LEAD_PERCENTILE})"
            )

        # Rule 3: Acceleration — current ROC > average of last _ACCEL_BARS ROC readings
        recent_rocs = []
        for i in range(_ACCEL_BARS, 0, -1):
            idx = len(closes) - i
            if idx >= _ROC_WINDOW + 1:
                base = closes[idx - _ROC_WINDOW - 1]
                curr = closes[idx]
                if base > 0:
                    recent_rocs.append((curr - base) / base)

        if len(recent_rocs) >= 2:
            avg_recent_roc = sum(recent_rocs) / len(recent_rocs)
            if current_roc <= avg_recent_roc:
                return self._none_signal(
                    candles,
                    f"ROC decelerating (current={current_roc:.3%} <= avg={avg_recent_roc:.3%})"
                )

        # Rule 4: Close > EMA50
        ema50 = self._ema(closes, _EMA_SLOW)
        if ema50 is None:
            return self._none_signal(candles, "EMA50 not ready")
        if current_close < ema50:
            return self._none_signal(
                candles, f"close below EMA50 ({current_close:.2f} < {ema50:.2f})"
            )

        # All rules pass
        intensity_bonus = min(30.0, (percentile - _LEAD_PERCENTILE) * 116)
        strength = min(100.0, _STRENGTH_BASE + intensity_bonus)

        return self._long_signal(
            candles,
            strength=strength,
            breakout=False,
            volume_conf=self._volume_ok(candles),
            reasoning=[
                f"10-bar ROC={current_roc:.2%}",
                f"leadership percentile={percentile:.2f}",
                f"accelerating ROC",
                f"close > EMA50={ema50:.2f}",
            ],
            ema50=ema50,
        )
