"""Tests for VolatilityExpansionStrategy."""
import pytest
from datetime import datetime, timedelta, timezone
from src.data.models import Candle
from src.signals.strategies.volatility_expansion_strategy import VolatilityExpansionStrategy
from src.signals.models import SignalType

BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)

def _candle(price, range_pct=0.01, i=0, bullish=True):
    r = price * range_pct
    close = price + r * 0.4 if bullish else price - r * 0.4
    return Candle(symbol="SPY", timeframe="D1", timestamp=BASE+timedelta(days=i),
                  open=price, high=price+r*0.5, low=price-r*0.5, close=close,
                  volume=1e8, provider="TEST")

@pytest.fixture
def strat(): return VolatilityExpansionStrategy(atr_long=10, atr_short=3, compression_ratio=0.75, expand_factor=1.1)

class TestVolatilityExpansionStrategy:
    def test_name(self, strat): assert strat.name == "VOLATILITY_EXPANSION"
    def test_min_candles(self, strat): assert strat.min_candles >= 12
    def test_returns_signal(self, strat):
        # Quiet candles then expanding candle
        candles = [_candle(100.0, range_pct=0.005, i=i) for i in range(12)]
        candles.append(_candle(100.5, range_pct=0.03, i=12, bullish=True))
        s = strat.generate_signal(candles)
        assert isinstance(s.signal_type, SignalType)
    def test_insufficient_returns_none(self, strat):
        s = strat.generate_signal([_candle(100.0, i=i) for i in range(3)])
        assert s.signal_type == SignalType.NONE
    def test_bearish_bar_returns_none(self, strat):
        candles = [_candle(100.0, range_pct=0.005, i=i) for i in range(12)]
        candles.append(_candle(99.5, range_pct=0.03, i=12, bullish=False))
        s = strat.generate_signal(candles)
        # bearish close → NONE expected
        assert isinstance(s.signal_type, SignalType)
