"""Tests for BreakoutStrategy."""
import pytest
from src.signals.strategies.breakout_strategy import BreakoutStrategy
from src.signals.models import SignalType
from tests.fixtures import uptrend_candles, flat_candles

@pytest.fixture
def strat(): return BreakoutStrategy()

class TestBreakoutStrategy:
    def test_name(self, strat): assert strat.name == "BREAKOUT_V1"
    def test_min_candles(self, strat): assert strat.min_candles >= 50
    def test_returns_signal(self, strat):
        s = strat.generate_signal(uptrend_candles(n=220))
        assert s is not None
        assert isinstance(s.signal_type, SignalType)
    def test_insufficient_returns_none(self, strat):
        s = strat.generate_signal(uptrend_candles(n=10))
        assert s.signal_type == SignalType.NONE
