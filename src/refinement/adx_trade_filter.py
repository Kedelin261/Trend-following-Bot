"""ADX trend-strength gate for Strategy V2.

Only allows trades when ADX ≥ threshold, confirming trend strength.
ADX values come from the existing ADXCalculator (Phase 4.5 research module).

Research capability: compare multiple thresholds and return the one
with the best expectancy from a set of backtest results.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.data.models import Candle
from src.research.adx_filter import ADXCalculator

logger = logging.getLogger(__name__)

RESEARCH_THRESHOLDS = [20.0, 25.0, 30.0, 35.0]


@dataclass
class ADXResearchResult:
    """ADX threshold research outcome."""
    threshold:   float
    trade_count: int
    expectancy:  float
    profit_factor: float
    sufficient:  bool   # trade_count >= min_trades


class ADXTradeFilter:
    """Gates trades when ADX(14) is below the configured threshold.

    ADX measures trend strength regardless of direction.  Requiring
    ADX ≥ 25 filters out choppy, low-conviction market conditions.

    Parameters
    ----------
    threshold : minimum ADX value to allow a trade (default 25.0)
    period    : ATR period for ADX calculation (default 14)
    """

    def __init__(self, threshold: float = 25.0, period: int = 14) -> None:
        self.threshold = threshold
        self.period    = period
        self._calc     = ADXCalculator(period=period)

    def passes(self, candles: List[Candle]) -> bool:
        """Return True when ADX meets or exceeds the threshold."""
        adx = self._calc.calculate_adx(candles)
        if adx is None:
            logger.debug("adx_filter: BLOCKED — insufficient data")
            return False
        passes = adx >= self.threshold
        if not passes:
            logger.debug(
                "adx_filter: BLOCKED — adx=%.1f < threshold=%.1f | %s",
                adx, self.threshold,
                candles[-1].symbol if candles else "?",
            )
        return passes

    def current_adx(self, candles: List[Candle]) -> Optional[float]:
        """Return the current ADX value (None if insufficient data)."""
        return self._calc.calculate_adx(candles)

    # ------------------------------------------------------------------
    # Research capability
    # ------------------------------------------------------------------

    @staticmethod
    def research_thresholds(
        completed_results,      # Dict[float, BacktestResults]
        min_trades: int = 30,
    ) -> List[ADXResearchResult]:
        """Rank ADX threshold results by expectancy.

        Accepts a dict of {threshold: BacktestResults} from pre-run backtests.
        Never auto-adopts the 'best' result — human review required.
        """
        output = []
        for thresh, bt in sorted(completed_results.items()):
            output.append(ADXResearchResult(
                threshold     = thresh,
                trade_count   = bt.total_trades,
                expectancy    = bt.expectancy,
                profit_factor = bt.profit_factor,
                sufficient    = bt.total_trades >= min_trades,
            ))
        return sorted(output, key=lambda r: r.expectancy, reverse=True)

    @staticmethod
    def best_threshold(results: List[ADXResearchResult]) -> Optional[float]:
        """Return the threshold with highest expectancy among sufficient results."""
        sufficient = [r for r in results if r.sufficient]
        return max(sufficient, key=lambda r: r.expectancy).threshold if sufficient else None
