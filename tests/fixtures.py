"""Shared deterministic candle data for signal engine tests.

All fixtures produce reproducible results (no random seeds, explicit prices).
No live connections or mocks required.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.data.models import Candle


def _candle(
    close: float,
    high: Optional[float] = None,
    low: Optional[float] = None,
    open_: Optional[float] = None,
    volume: float = 100_000_000.0,
    i: int = 0,
    symbol: str = "SPY",
    timeframe: str = "D1",
) -> Candle:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    h = high  if high  is not None else close * 1.005
    l = low   if low   is not None else close * 0.995
    o = open_ if open_ is not None else close * 0.999
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=base + timedelta(days=i),
        open=round(o, 5),
        high=round(h, 5),
        low=round(l, 5),
        close=round(close, 5),
        volume=volume,
        provider="TEST",
    )


def uptrend_candles(
    n: int = 250,
    start: float = 400.0,
    daily_gain: float = 0.0015,
    volume: float = 100_000_000.0,
    symbol: str = "SPY",
) -> List[Candle]:
    """Monotonically rising prices — produces BULLISH classification after warmup."""
    candles = []
    price = start
    for i in range(n):
        candles.append(_candle(price, volume=volume, i=i, symbol=symbol))
        price *= (1.0 + daily_gain)
    return candles


def downtrend_candles(
    n: int = 250,
    start: float = 400.0,
    daily_loss: float = 0.0015,
    volume: float = 100_000_000.0,
    symbol: str = "SPY",
) -> List[Candle]:
    """Monotonically falling prices — produces BEARISH classification after warmup."""
    candles = []
    price = start
    for i in range(n):
        candles.append(_candle(price, volume=volume, i=i, symbol=symbol))
        price *= (1.0 - daily_loss)
    return candles


def flat_candles(
    n: int = 250,
    price: float = 400.0,
    volume: float = 100_000_000.0,
    symbol: str = "SPY",
) -> List[Candle]:
    """Perfectly flat prices — produces NEUTRAL classification."""
    return [_candle(price, volume=volume, i=i, symbol=symbol) for i in range(n)]


def zigzag_candles(
    n: int = 60,
    base: float = 400.0,
    amplitude: float = 10.0,
    volume: float = 100_000_000.0,
    symbol: str = "SPY",
) -> List[Candle]:
    """Regular zigzag pattern with explicit swing highs and lows.

    Even indices → peaks (close + amplitude)
    Odd indices  → troughs (close - amplitude)
    High/low are set to the peak/trough value so swing detection works.
    """
    candles = []
    for i in range(n):
        if i % 2 == 0:
            close = base + amplitude
            high  = close + 1.0
            low   = base  - 1.0
        else:
            close = base - amplitude
            high  = base  + 1.0
            low   = close - 1.0
        candles.append(
            _candle(close, high=high, low=low, volume=volume, i=i, symbol=symbol)
        )
    return candles


def zero_volume_candles(
    n: int = 250,
    start: float = 1.0900,
    daily_gain: float = 0.0005,
    symbol: str = "EURUSD",
) -> List[Candle]:
    """Uptrending candles with zero volume (Forex-style)."""
    candles = []
    price = start
    for i in range(n):
        candles.append(_candle(price, volume=0.0, i=i, symbol=symbol, timeframe="H1"))
        price *= (1.0 + daily_gain)
    return candles


def spike_volume_candles(
    n: int = 25,
    base_price: float = 400.0,
    base_volume: float = 80_000_000.0,
    spike_multiple: float = 3.0,
    symbol: str = "SPY",
) -> List[Candle]:
    """Candles where the last bar has volume spike (spike_multiple × average)."""
    candles = []
    for i in range(n - 1):
        candles.append(_candle(base_price, volume=base_volume, i=i, symbol=symbol))
    # Final candle with volume spike
    candles.append(
        _candle(base_price * 1.003, volume=base_volume * spike_multiple,
                i=n - 1, symbol=symbol)
    )
    return candles
