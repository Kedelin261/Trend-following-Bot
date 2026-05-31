"""Common interface that every Edge Lab strategy must implement.

Compatible with the existing BacktestEngine signal_engine parameter.
All strategies implement generate_signal(candles) → Signal.

CONSTRAINT: Only entry logic may differ between strategies.
Risk engine, position sizing, commission, slippage, and asset universe
are held constant for a fair comparison.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.data.models import Candle
from src.signals.models import Signal, SignalType, TrendDirection


class StrategyInterface(ABC):
    """Base class for all Edge Lab strategy implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier for reporting."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable strategy description."""

    @property
    def min_candles(self) -> int:
        """Minimum candle history required before first signal attempt."""
        return 60

    @abstractmethod
    def generate_signal(self, candles: List[Candle]) -> Signal:
        """Analyse *candles* and return a Signal.

        Must return a Signal with signal_type=NONE when no edge is present.
        Must never raise — log errors internally.
        """

    def _none_signal(self, candles: List[Candle], reason: str = "") -> Signal:
        """Return a NONE signal (no edge detected)."""
        from datetime import datetime, timezone
        latest = candles[-1] if candles else None
        return Signal(
            symbol            = latest.symbol    if latest else "UNKNOWN",
            timeframe         = latest.timeframe if latest else "D1",
            timestamp         = latest.timestamp if latest else datetime.now(tz=timezone.utc),
            signal_type       = SignalType.NONE,
            trend_direction   = TrendDirection.NEUTRAL,
            strength_score    = 0.0,
            breakout_detected = False,
            volume_confirmed  = None,
            support_level     = None,
            resistance_level  = None,
            close_price       = latest.close if latest else 0.0,
            reasoning         = [reason] if reason else [],
        )

    def _long_signal(
        self,
        candles:       List[Candle],
        strength:      float,
        breakout:      bool   = True,
        volume_conf:   Optional[bool] = None,
        support:       Optional[float] = None,
        resistance:    Optional[float] = None,
        reasoning:     Optional[list]  = None,
        ema50:         Optional[float] = None,
        ema200:        Optional[float] = None,
    ) -> Signal:
        """Return a LONG signal with the given parameters."""
        latest = candles[-1]
        return Signal(
            symbol            = latest.symbol,
            timeframe         = latest.timeframe,
            timestamp         = latest.timestamp,
            signal_type       = SignalType.LONG,
            trend_direction   = TrendDirection.BULLISH,
            strength_score    = max(0.0, min(100.0, strength)),
            breakout_detected = breakout,
            volume_confirmed  = volume_conf,
            support_level     = support,
            resistance_level  = resistance,
            close_price       = latest.close,
            reasoning         = reasoning or [self.name],
            ema50             = ema50,
            ema200            = ema200,
        )

    @staticmethod
    def _volume_ok(candles: List[Candle], lookback: int = 20) -> Optional[bool]:
        """Return volume confirmation (None if insufficient data)."""
        if len(candles) < lookback + 1:
            return None
        vols = [c.volume for c in candles[-(lookback + 1):-1]]
        meaningful = [v for v in vols if v > 1.0]
        if not meaningful:
            return None
        avg = sum(meaningful) / len(meaningful)
        return candles[-1].volume > avg if candles[-1].volume > 1.0 else None

    @staticmethod
    def _ema(closes: List[float], period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        mult = 2.0 / (period + 1)
        ema  = sum(closes[:period]) / period
        for p in closes[period:]:
            ema = p * mult + ema * (1 - mult)
        return ema
