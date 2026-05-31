"""Timeframe backtester — runs the best density profile on any timeframe.

The strategy (EMA20/50, ADX≥27, Brk≥1%, MEDIUM+HIGH vol, Bull filter)
is held constant.  Only the candle timeframe varies.

SAFEGUARD: any timeframe that produces PF < 1.50, Expectancy ≤ 0, or
DD > 15% is flagged and excluded from combination research.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults, StrategyHealth
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import (
    BEST_DENSITY_PROFILE,
    TimeframeProfile,
)

logger = logging.getLogger(__name__)

QUALITY_MIN_PF     = 1.50
QUALITY_MIN_EXP    = 0.0
QUALITY_MAX_DD     = 15.0


@dataclass
class TimeframeBacktestResult:
    """Result of running the strategy on one asset / one timeframe."""

    tf_profile:    TimeframeProfile
    symbol:        str
    results:       BacktestResults
    health:        StrategyHealth
    meets_quality: bool
    note:          str = ""

    @property
    def timeframe(self) -> str:
        return self.tf_profile.timeframe

    @property
    def trade_count(self) -> int:
        return self.results.total_trades

    @property
    def expectancy(self) -> float:
        return self.results.expectancy

    @property
    def profit_factor(self) -> float:
        return self.results.profit_factor


class TimeframeBacktester:
    """Runs a fixed strategy profile across different timeframes.

    Parameters
    ----------
    config           : settings dict (backtest, risk sections)
    strategy_profile : the strategy to test (default = BEST_DENSITY_PROFILE)
    """

    def __init__(
        self,
        config:           dict,
        strategy_profile: StrategyProfile = None,
    ) -> None:
        self._config   = config
        self._strategy = strategy_profile or BEST_DENSITY_PROFILE

    def backtest(
        self,
        symbol:     str,
        candles:    List[Candle],
        tf_profile: TimeframeProfile,
    ) -> TimeframeBacktestResult:
        """Backtest the strategy on *candles* and return a labelled result."""
        logger.info(
            "timeframe_backtest: %s/%s (%d candles)",
            symbol, tf_profile.timeframe, len(candles),
        )
        engine  = self._strategy.build_backtest_engine(self._config)
        results = engine.run(candles)
        health  = StrategyHealth.evaluate(results)

        meets = (
            results.profit_factor >= QUALITY_MIN_PF
            and results.expectancy > QUALITY_MIN_EXP
            and results.max_drawdown <= QUALITY_MAX_DD
        )
        note = "" if meets else (
            f"⚠ QUALITY FAIL: pf={results.profit_factor:.2f} "
            f"exp={results.expectancy:.2f} dd={results.max_drawdown:.1f}%"
        )

        return TimeframeBacktestResult(
            tf_profile    = tf_profile,
            symbol        = symbol,
            results       = results,
            health        = health,
            meets_quality = meets,
            note          = note,
        )

    def backtest_all_assets(
        self,
        asset_candles: Dict[str, List[Candle]],
        tf_profile:    TimeframeProfile,
    ) -> Dict[str, TimeframeBacktestResult]:
        """Run backtest on every asset for one timeframe."""
        return {
            sym: self.backtest(sym, candles, tf_profile)
            for sym, candles in asset_candles.items()
            if candles
        }
