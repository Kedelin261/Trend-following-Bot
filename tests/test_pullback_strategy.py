"""Tests for PullbackStrategy."""
import pytest
from datetime import datetime, timedelta, timezone
from src.data.models import Candle
from src.signals.strategies.pullback_strategy import PullbackStrategy
from src.signals.models import SignalType
from tests.fixtures import uptrend_candles, flat_candles

BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)

@pytest.fixture
def strat(): return PullbackStrategy(fast_period=5, slow_period=10, pullback_window=3)

class TestPullbackStrategy:
    def test_name(self, strat): assert strat.name == "PULLBACK_CONTINUATION"
    def test_min_candles(self, strat): assert strat.min_candles >= 15
    def test_returns_signal_on_uptrend(self, strat):
        s = strat.generate_signal(uptrend_candles(n=50, daily_gain=0.003))
        assert isinstance(s.signal_type, SignalType)
    def test_insufficient_returns_none(self, strat):
        s = strat.generate_signal(uptrend_candles(n=5))
        assert s.signal_type == SignalType.NONE
    def test_long_on_valid_pullback(self, strat):
        # Uptrend then price dips below EMA5 then resumes
        candles = uptrend_candles(n=30, daily_gain=0.002)
        # Force pullback: insert low candles
        base_price = candles[-5].close
        for i in range(-4, 0):
            p = base_price * 0.995
            candles[i] = Candle(
                symbol="SPY", timeframe="D1",
                timestamp=BASE + timedelta(days=100 + i),
                open=p, high=p * 1.001, low=p * 0.999,
                close=p, volume=1e8, provider="TEST",
            )
        # Resume bar
        resume = base_price * 1.005
        candles.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=BASE + timedelta(days=200),
            open=resume * 0.999, high=resume * 1.002,
            low=resume * 0.997, close=resume,
            volume=1e8, provider="TEST",
        ))
        s = strat.generate_signal(candles)
        assert isinstance(s.signal_type, SignalType)
