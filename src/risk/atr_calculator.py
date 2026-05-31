"""ATR (Average True Range) calculator using Wilder's smoothing method.

Industry-standard implementation: seed with SMA of the first *period* True
Ranges, then apply Wilder's exponential smoother:

    ATR_k = (ATR_{k-1} × (n − 1) + TR_k) / n

True Range:
    TR = max(High − Low, |High − Prev_Close|, |Low − Prev_Close|)

No broker code. No API calls. Candle data only.
"""

import logging
from typing import List, Optional

from src.data.models import Candle

logger = logging.getLogger(__name__)


class ATRCalculator:
    """Computes the Average True Range for a list of Candle objects.

    Parameters
    ----------
    period : lookback window for Wilder's ATR (default 14)
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"ATR period must be ≥ 1, got {period}")
        self.period = period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_true_range(
        self,
        high: float,
        low: float,
        prev_close: float,
    ) -> float:
        """Return the True Range for a single bar.

        TR = max(High − Low, |High − Prev_Close|, |Low − Prev_Close|)
        """
        return max(
            high - low,
            abs(high - prev_close),
            abs(low  - prev_close),
        )

    def calculate_true_range_series(self, candles: List[Candle]) -> List[float]:
        """Return a TR value for every bar except the first (no prev_close).

        The returned list has len(candles) - 1 elements.
        Index 0 of the result corresponds to candles[1].
        """
        if len(candles) < 2:
            return []
        return [
            self.calculate_true_range(
                candles[i].high,
                candles[i].low,
                candles[i - 1].close,
            )
            for i in range(1, len(candles))
        ]

    def calculate_atr(self, candles: List[Candle]) -> Optional[float]:
        """Return the current ATR value using Wilder's smoothing.

        Requires at least *period* + 1 candles (one extra for the initial
        prev_close).  Returns None when insufficient data is available.

        Algorithm
        ---------
        1. Compute TR for each bar starting at bar index 1.
        2. Seed ATR as SMA of the first *period* TRs.
        3. For each subsequent TR: ATR = (ATR × (n−1) + TR) / n
        """
        trs = self.calculate_true_range_series(candles)

        if len(trs) < self.period:
            logger.debug(
                "atr_calculator: insufficient TRs (%d < period %d) for %s — returning None",
                len(trs),
                self.period,
                candles[-1].symbol if candles else "?",
            )
            return None

        # Seed: SMA of the first `period` True Ranges
        atr = sum(trs[: self.period]) / self.period

        # Wilder's smoothing for all remaining True Ranges
        for tr in trs[self.period :]:
            atr = (atr * (self.period - 1) + tr) / self.period

        logger.debug(
            "atr_calculator: %s ATR(%d)=%.5f from %d candles",
            candles[-1].symbol if candles else "?",
            self.period,
            atr,
            len(candles),
        )
        return atr
