"""BREAKOUT_V1 — current strategy wrapped as StrategyInterface.

Serves as the performance benchmark. All other strategies are judged
relative to this implementation.
"""

from typing import List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.models import Signal
from src.refinement.strategy_v2 import V2_PROFILE
from src.refinement.volatility_trade_filter import VolatilityFilterMode


class BreakoutStrategy(StrategyInterface):
    """Current V2 strategy: EMA20/50, ADX≥27, 1% breakout, MEDIUM+HIGH vol, BULL filter."""

    _name = "BREAKOUT_V1"
    _desc = "Current strategy — EMA20/50 trend + ADX≥27 + 1% breakout + regime filter"

    def __init__(self) -> None:
        from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE
        from src.refinement.market_regime_filter import MarketRegimeFilter
        from src.refinement.adx_trade_filter import ADXTradeFilter
        from src.refinement.volatility_trade_filter import VolatilityTradeFilter
        from src.signals.signal_engine import SignalEngine
        from src.signals.trend_detector import TrendDetector
        from src.signals.breakout_detector import BreakoutDetector

        base = SignalEngine(
            trend_detector=TrendDetector(
                fast_period=BEST_DENSITY_PROFILE.ema_fast,
                slow_period=BEST_DENSITY_PROFILE.ema_slow,
            ),
            breakout_detector=BreakoutDetector(
                threshold=BEST_DENSITY_PROFILE.breakout_threshold
            ),
            min_candles=BEST_DENSITY_PROFILE.ema_slow + 10,
        )
        from src.refinement.strategy_v2 import CompositeSignalEngine
        self._engine = CompositeSignalEngine(
            base_engine=base,
            regime_filter=MarketRegimeFilter() if BEST_DENSITY_PROFILE.require_bull_regime else None,
            adx_filter=ADXTradeFilter(BEST_DENSITY_PROFILE.adx_threshold)
                       if BEST_DENSITY_PROFILE.adx_threshold > 0 else None,
            vol_filter=VolatilityTradeFilter(mode=BEST_DENSITY_PROFILE.volatility_mode)
                       if BEST_DENSITY_PROFILE.volatility_mode != VolatilityFilterMode.NONE else None,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    @property
    def min_candles(self) -> int:
        return 210   # needs EMA200 for regime filter

    def generate_signal(self, candles: List[Candle]) -> Signal:
        return self._engine.generate_signal(candles)
