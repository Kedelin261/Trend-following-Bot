"""Volatility analysis — categorises market conditions by ATR percent.

Research use: determine in which volatility environment the strategy
has the most consistent edge.

LOW     : ATR/Price < low_threshold  (e.g. < 0.5 %)
MEDIUM  : low_threshold ≤ ATR/Price ≤ high_threshold  (0.5 % – 2 %)
HIGH    : ATR/Price > high_threshold  (e.g. > 2 %)

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest import performance_metrics as pm
from src.data.models import Candle
from src.risk.atr_calculator import ATRCalculator

logger = logging.getLogger(__name__)


class VolatilityCategory(str, Enum):
    LOW     = "LOW"
    MEDIUM  = "MEDIUM"
    HIGH    = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class VolatilityPerformance:
    """Strategy performance within one volatility bucket."""

    category:      VolatilityCategory
    trade_count:   int
    win_rate:      float
    expectancy:    float
    profit_factor: float
    avg_atr_pct:   float    # mean ATR% across trades in this bucket
    sufficient:    bool


class VolatilityFilter:
    """Categorises bars by ATR-as-percentage-of-price and slices trade results.

    Parameters
    ----------
    atr_period      : ATR lookback (default 14)
    low_threshold   : ATR% below which the market is considered LOW vol
    high_threshold  : ATR% above which the market is considered HIGH vol
    """

    def __init__(
        self,
        atr_period:      int   = 14,
        low_threshold:   float = 0.5,    # % of price
        high_threshold:  float = 2.0,    # % of price
    ) -> None:
        self.atr_period     = atr_period
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self._atr           = ATRCalculator(period=atr_period)

    # ------------------------------------------------------------------
    # Single-bar classification
    # ------------------------------------------------------------------

    def atr_pct(self, candles: List[Candle]) -> Optional[float]:
        """Return ATR as a percentage of the last close.  None if insufficient data."""
        atr = self._atr.calculate_atr(candles)
        if atr is None:
            return None
        close = candles[-1].close
        return (atr / close * 100.0) if close > 0 else None

    def categorize(self, candles: List[Candle]) -> VolatilityCategory:
        """Classify current volatility regime."""
        pct = self.atr_pct(candles)
        if pct is None:
            return VolatilityCategory.UNKNOWN
        if pct < self.low_threshold:
            return VolatilityCategory.LOW
        if pct <= self.high_threshold:
            return VolatilityCategory.MEDIUM
        return VolatilityCategory.HIGH

    def categorize_at_timestamp(
        self,
        candles:   List[Candle],
        timestamp,
    ) -> VolatilityCategory:
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.categorize(history) if history else VolatilityCategory.UNKNOWN

    def atr_pct_at_timestamp(
        self,
        candles:   List[Candle],
        timestamp,
    ) -> Optional[float]:
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.atr_pct(history) if history else None

    # ------------------------------------------------------------------
    # Trade-level research
    # ------------------------------------------------------------------

    def analyze_trades_by_volatility(
        self,
        trades:     List[BacktestTrade],
        candles:    List[Candle],
        min_trades: int = 10,
    ) -> Dict[VolatilityCategory, VolatilityPerformance]:
        """Slice trades by the volatility regime active at each entry.

        Answers: 'Does the strategy perform better in low or high volatility?'
        """
        buckets: Dict[VolatilityCategory, List[BacktestTrade]] = {
            cat: [] for cat in VolatilityCategory
        }
        atr_pcts: Dict[VolatilityCategory, List[float]] = {
            cat: [] for cat in VolatilityCategory
        }

        for trade in trades:
            cat  = self.categorize_at_timestamp(candles, trade.entry_time)
            pct  = self.atr_pct_at_timestamp(candles, trade.entry_time)
            buckets[cat].append(trade)
            if pct is not None:
                atr_pcts[cat].append(pct)

        result: Dict[VolatilityCategory, VolatilityPerformance] = {}
        for cat, bucket in buckets.items():
            if not bucket:
                continue
            avg_pct = sum(atr_pcts[cat]) / len(atr_pcts[cat]) if atr_pcts[cat] else 0.0
            result[cat] = VolatilityPerformance(
                category      = cat,
                trade_count   = len(bucket),
                win_rate      = pm.win_rate(bucket),
                expectancy    = pm.expectancy(bucket),
                profit_factor = pm.profit_factor(bucket),
                avg_atr_pct   = avg_pct,
                sufficient    = len(bucket) >= min_trades,
            )

        return result
