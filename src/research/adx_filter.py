"""ADX (Average Directional Index) calculator and trend-strength filter.

Implements J. Welles Wilder's original algorithm from
'New Concepts in Technical Trading Systems' (1978).

Research use: compare strategy results when filtered by ADX threshold vs
unfiltered to determine whether trend-strength gating improves edge.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.backtest.models import BacktestTrade
from src.backtest import performance_metrics as pm
from src.data.models import Candle

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = [0.0, 20.0, 25.0, 30.0]


@dataclass
class ADXFilterResult:
    """Strategy performance when trades are gated by ADX threshold."""

    threshold:     float
    trade_count:   int
    win_rate:      float
    expectancy:    float
    profit_factor: float
    trades_passed: int    # trades where ADX at entry >= threshold
    trades_blocked: int   # trades where ADX at entry < threshold
    sufficient:    bool   # trade_count >= min_trades


class ADXCalculator:
    """Computes ADX using Wilder's smoothed directional movement system.

    Requires at least 2 × period + 1 candles for a valid ADX reading.
    Returns None when insufficient data is available.
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"ADX period must be ≥ 1, got {period}")
        self.period = period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_adx(self, candles: List[Candle]) -> Optional[float]:
        """Return the current ADX value for the provided candle series."""
        adx, _, _ = self._calculate_adx_di(candles)
        return adx

    def calculate_plus_di(self, candles: List[Candle]) -> Optional[float]:
        _, plus_di, _ = self._calculate_adx_di(candles)
        return plus_di

    def calculate_minus_di(self, candles: List[Candle]) -> Optional[float]:
        _, _, minus_di = self._calculate_adx_di(candles)
        return minus_di

    def calculate_adx_series(
        self, candles: List[Candle]
    ) -> List[Optional[float]]:
        """Return ADX at every bar.  Index positions in the warmup period are None."""
        min_bars = 2 * self.period + 1
        result: List[Optional[float]] = [None] * len(candles)
        for i in range(min_bars - 1, len(candles)):
            result[i] = self.calculate_adx(candles[: i + 1])
        return result

    def adx_at_timestamp(
        self, candles: List[Candle], timestamp
    ) -> Optional[float]:
        """Return ADX at the bar whose timestamp matches or precedes *timestamp*."""
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.calculate_adx(history) if len(history) >= 2 * self.period + 1 else None

    # ------------------------------------------------------------------
    # Internal calculation
    # ------------------------------------------------------------------

    def _calculate_adx_di(
        self, candles: List[Candle]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (ADX, +DI14, -DI14) or (None, None, None) if insufficient."""
        n = len(candles)
        min_bars = 2 * self.period + 1
        if n < min_bars:
            return None, None, None

        # --- Raw TR, +DM, -DM starting from bar 1 ----------------------
        trs: List[float] = []
        pdms: List[float] = []
        mdms: List[float] = []

        for i in range(1, n):
            c, p = candles[i], candles[i - 1]
            trs.append(
                max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
            )
            up   = c.high - p.high
            down = p.low  - c.low
            pdms.append(up   if up > down and up > 0   else 0.0)
            mdms.append(down if down > up and down > 0 else 0.0)

        if len(trs) < self.period:
            return None, None, None

        # --- Wilder's sum-seeded smoothing for TR, +DM, -DM ------------
        # Seed: simple sum of first period values
        smt  = sum(trs  [: self.period])
        spdm = sum(pdms [: self.period])
        smdm = sum(mdms [: self.period])

        def _dx(smt_, spdm_, smdm_):
            if smt_ == 0:
                return 0.0
            pdi = 100.0 * spdm_ / smt_
            mdi = 100.0 * smdm_ / smt_
            denom = pdi + mdi
            return 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0

        dx_series = [_dx(smt, spdm, smdm)]

        for i in range(self.period, len(trs)):
            smt  = smt  - smt  / self.period + trs  [i]
            spdm = spdm - spdm / self.period + pdms [i]
            smdm = smdm - smdm / self.period + mdms [i]
            dx_series.append(_dx(smt, spdm, smdm))

        # --- ADX: SMA seed of first period DX values, then Wilder avg --
        if len(dx_series) < self.period:
            return None, None, None

        adx = sum(dx_series[: self.period]) / self.period
        for dx in dx_series[self.period :]:
            adx = (adx * (self.period - 1) + dx) / self.period

        # Final +DI/-DI from last smoothed values
        pdi_last = 100.0 * spdm / smt if smt > 0 else 0.0
        mdi_last = 100.0 * smdm / smt if smt > 0 else 0.0

        return adx, pdi_last, mdi_last


class ADXFilter:
    """Applies ADX threshold gating to backtest trades for research analysis.

    Each call to analyze_threshold() acts as a post-hoc filter:
    'what would have happened if we only took trades when ADX > X?'
    The underlying strategy is never modified.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._calc  = ADXCalculator(period)

    def analyze_threshold(
        self,
        trades:     List[BacktestTrade],
        candles:    List[Candle],
        threshold:  float,
        min_trades: int = 10,
    ) -> ADXFilterResult:
        """Return performance of trades whose entry ADX exceeded *threshold*."""
        passed: List[BacktestTrade] = []
        blocked = 0

        for trade in trades:
            adx = self._calc.adx_at_timestamp(candles, trade.entry_time)
            if adx is None or adx >= threshold:
                passed.append(trade)
            else:
                blocked += 1

        return ADXFilterResult(
            threshold     = threshold,
            trade_count   = len(passed),
            win_rate      = pm.win_rate(passed),
            expectancy    = pm.expectancy(passed),
            profit_factor = pm.profit_factor(passed),
            trades_passed = len(passed),
            trades_blocked = blocked,
            sufficient    = len(passed) >= min_trades,
        )

    def compare_thresholds(
        self,
        trades:     List[BacktestTrade],
        candles:    List[Candle],
        thresholds: List[float] = None,
        min_trades: int         = 10,
    ) -> List[ADXFilterResult]:
        """Compare performance across multiple ADX thresholds."""
        thresholds = thresholds or _DEFAULT_THRESHOLDS
        return [
            self.analyze_threshold(trades, candles, t, min_trades)
            for t in thresholds
        ]

    def best_threshold(
        self,
        results:    List[ADXFilterResult],
        metric:     str = "expectancy",
    ) -> Optional[ADXFilterResult]:
        """Return the threshold with best metric, restricted to sufficient results."""
        sufficient = [r for r in results if r.sufficient]
        if not sufficient:
            return None
        return max(sufficient, key=lambda r: getattr(r, metric))
