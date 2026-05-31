"""Market regime detection — classifies market state as BULL, BEAR, or SIDEWAYS.

Used to slice backtest trade results by market condition and determine
in which regimes the strategy has a positive edge.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest import performance_metrics as pm
from src.data.models import Candle
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    BULL     = "BULL"
    BEAR     = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN  = "UNKNOWN"     # insufficient data


@dataclass
class RegimePerformance:
    """Strategy performance stats for a single market regime."""

    regime:        MarketRegime
    trade_count:   int
    win_rate:      float
    expectancy:    float
    profit_factor: float
    max_drawdown:  float
    sufficient:    bool   # True when trade_count >= min_trades


class MarketRegimeDetector:
    """Classifies market state using EMA50/EMA200 relationship.

    BULL     : EMA50 > EMA200 by at least sideways_threshold
    BEAR     : EMA200 > EMA50 by at least sideways_threshold
    SIDEWAYS : separation between EMAs < sideways_threshold
    """

    def __init__(
        self,
        fast_period:         int   = 50,
        slow_period:         int   = 200,
        sideways_threshold:  float = 0.01,   # 1 % EMA separation
    ) -> None:
        self.fast_period        = fast_period
        self.slow_period        = slow_period
        self.sideways_threshold = sideways_threshold
        self._detector = TrendDetector(fast_period, slow_period)

    def classify(self, candles: List[Candle]) -> MarketRegime:
        """Return the market regime for the most recent bar in *candles*."""
        if len(candles) < self.slow_period:
            return MarketRegime.UNKNOWN

        closes = [c.close for c in candles]
        ema_fast = self._detector.calculate_ema(closes, self.fast_period)
        ema_slow = self._detector.calculate_ema(closes, self.slow_period)

        if ema_fast is None or ema_slow is None or ema_slow == 0:
            return MarketRegime.UNKNOWN

        separation = abs(ema_fast - ema_slow) / ema_slow

        if separation <= self.sideways_threshold:
            return MarketRegime.SIDEWAYS
        return MarketRegime.BULL if ema_fast > ema_slow else MarketRegime.BEAR

    def classify_series(self, candles: List[Candle]) -> List[MarketRegime]:
        """Return the regime classification at every bar (None-safe)."""
        return [self.classify(candles[: i + 1]) for i in range(len(candles))]

    def regime_at_timestamp(
        self,
        candles:   List[Candle],
        timestamp,
    ) -> MarketRegime:
        """Return the regime at the bar whose timestamp matches or precedes *timestamp*."""
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.classify(history) if history else MarketRegime.UNKNOWN

    def analyze_trades_by_regime(
        self,
        trades:     List[BacktestTrade],
        candles:    List[Candle],
        min_trades: int = 10,
    ) -> Dict[MarketRegime, RegimePerformance]:
        """Slice a closed trade list by the regime active at each trade's entry.

        Returns performance stats per regime.  Regimes with fewer than
        *min_trades* are flagged as insufficient for statistical inference.
        """
        by_regime: Dict[MarketRegime, List[BacktestTrade]] = {
            r: [] for r in MarketRegime
        }

        for trade in trades:
            regime = self.regime_at_timestamp(candles, trade.entry_time)
            by_regime[regime].append(trade)

        result: Dict[MarketRegime, RegimePerformance] = {}
        for regime, bucket in by_regime.items():
            if not bucket:
                continue
            pf = pm.profit_factor(bucket)
            result[regime] = RegimePerformance(
                regime        = regime,
                trade_count   = len(bucket),
                win_rate      = pm.win_rate(bucket),
                expectancy    = pm.expectancy(bucket),
                profit_factor = pf,
                max_drawdown  = 0.0,   # equity curve not available at this level
                sufficient    = len(bucket) >= min_trades,
            )

        logger.info(
            "regime_analysis: regimes=%s",
            {r.value: len(t) for r, t in by_regime.items() if t},
        )
        return result
