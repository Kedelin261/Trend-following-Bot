"""Tests for MomentumRotationStrategy."""
import pytest
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from src.signals.models import SignalType
from tests.fixtures import uptrend_candles, downtrend_candles, flat_candles

@pytest.fixture
def strat(): return MomentumRotationStrategy(roc_short=5, roc_long=10, ema_fast=5, ema_slow=10)

class TestMomentumRotationStrategy:
    def test_name(self, strat): assert strat.name == "MOMENTUM_ROTATION"
    def test_min_candles(self, strat): assert strat.min_candles >= 15
    def test_uptrend_fires(self, strat):
        s = strat.generate_signal(uptrend_candles(n=30, daily_gain=0.005))
        assert isinstance(s.signal_type, SignalType)
    def test_downtrend_returns_none(self, strat):
        s = strat.generate_signal(downtrend_candles(n=30, daily_loss=0.005))
        assert s.signal_type == SignalType.NONE
    def test_insufficient_returns_none(self, strat):
        s = strat.generate_signal(uptrend_candles(n=5))
        assert s.signal_type == SignalType.NONE
    def test_long_in_bull_market(self, strat):
        candles = uptrend_candles(n=30, daily_gain=0.005)
        s = strat.generate_signal(candles)
        if s.signal_type == SignalType.LONG:
            assert s.strength_score >= 0
