"""Tests for StopLossEngine — ATR-based stop loss calculation."""

import pytest

from src.risk.stop_loss_engine import StopLossEngine
from src.signals.models import SignalType


@pytest.fixture
def engine() -> StopLossEngine:
    return StopLossEngine(multiplier=2.0)


class TestStopLossLong:

    def test_long_stop_below_entry(self, engine):
        stop = engine.calculate_stop_loss(SignalType.LONG, entry_price=100.0, atr=5.0)
        assert stop < 100.0

    def test_long_stop_equals_entry_minus_atr_times_mult(self, engine):
        # stop = 100 - (5 × 2) = 90
        stop = engine.calculate_stop_loss(SignalType.LONG, 100.0, 5.0)
        assert stop == pytest.approx(90.0)

    def test_long_spec_example(self, engine):
        # Spec: entry=755.64, ATR=6.25, stop=755.64-(6.25×2)=743.14
        stop = engine.calculate_stop_loss(SignalType.LONG, 755.64, 6.25)
        assert stop == pytest.approx(743.14, rel=1e-6)

    def test_long_forex_prices(self, engine):
        stop = engine.calculate_stop_loss(SignalType.LONG, 1.0900, 0.0050)
        assert stop == pytest.approx(1.0900 - 0.0100, rel=1e-6)


class TestStopLossShort:

    def test_short_stop_above_entry(self, engine):
        stop = engine.calculate_stop_loss(SignalType.SHORT, entry_price=100.0, atr=5.0)
        assert stop > 100.0

    def test_short_stop_equals_entry_plus_atr_times_mult(self, engine):
        # stop = 100 + (5 × 2) = 110
        stop = engine.calculate_stop_loss(SignalType.SHORT, 100.0, 5.0)
        assert stop == pytest.approx(110.0)

    def test_short_high_priced_asset(self, engine):
        stop = engine.calculate_stop_loss(SignalType.SHORT, 40000.0, 1200.0)
        assert stop == pytest.approx(40000.0 + 2400.0)


class TestStopLossCustomMultiplier:

    def test_multiplier_15_applied(self):
        e = StopLossEngine(multiplier=1.5)
        stop = e.calculate_stop_loss(SignalType.LONG, 100.0, 4.0)
        assert stop == pytest.approx(100.0 - 6.0)

    def test_multiplier_3_applied(self):
        e = StopLossEngine(multiplier=3.0)
        stop = e.calculate_stop_loss(SignalType.LONG, 200.0, 5.0)
        assert stop == pytest.approx(185.0)

    def test_multiplier_stored(self):
        e = StopLossEngine(multiplier=4.0)
        assert e.multiplier == 4.0


class TestStopLossInit:

    def test_zero_multiplier_raises(self):
        with pytest.raises(ValueError):
            StopLossEngine(multiplier=0.0)

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError):
            StopLossEngine(multiplier=-1.0)

    def test_default_multiplier_is_2(self):
        assert StopLossEngine().multiplier == 2.0
