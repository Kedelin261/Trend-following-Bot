"""Tests for DonchianStrategy."""
import pytest
from datetime import datetime, timedelta, timezone
from src.data.models import Candle
from src.signals.strategies.donchian_strategy import DonchianStrategy
from src.signals.models import SignalType

BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)

def _candle(price, i=0):
    return Candle(symbol="SPY", timeframe="D1", timestamp=BASE+timedelta(days=i),
                  open=price, high=price*1.005, low=price*0.995, close=price,
                  volume=1e8, provider="TEST")

@pytest.fixture
def strat(): return DonchianStrategy(period=5)

class TestDonchianStrategy:
    def test_name(self, strat): assert strat.name == "DONCHIAN_20"
    def test_min_candles(self, strat): assert strat.min_candles >= 6
    def test_breakout_fires_long(self, strat):
        # period+2 = 7 candles needed; 6 at 100 (high=100.5), breakout close at 102
        candles = [_candle(100.0, i) for i in range(6)] + [_candle(102.0, 6)]
        s = strat.generate_signal(candles)
        assert s.signal_type == SignalType.LONG
    def test_no_breakout_returns_none(self, strat):
        candles = [_candle(100.0, i) for i in range(6)]
        s = strat.generate_signal(candles)
        assert s.signal_type == SignalType.NONE
    def test_insufficient_returns_none(self, strat):
        s = strat.generate_signal([_candle(100.0)])
        assert s.signal_type == SignalType.NONE
    def test_close_must_exceed_high(self, strat):
        # Candles all at 100, last close exactly at the highest high — no breakout
        candles = [_candle(100.0, i) for i in range(7)]
        s = strat.generate_signal(candles)
        assert s.signal_type == SignalType.NONE
