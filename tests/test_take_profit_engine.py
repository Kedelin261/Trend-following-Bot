"""Tests for TakeProfitEngine — ATR-based take profit calculation."""

import pytest

from src.risk.take_profit_engine import TakeProfitEngine
from src.signals.models import SignalType


@pytest.fixture
def engine() -> TakeProfitEngine:
    return TakeProfitEngine(multiplier=3.0)


class TestTakeProfitLong:

    def test_long_target_above_entry(self, engine):
        target = engine.calculate_take_profit(SignalType.LONG, entry_price=100.0, atr=5.0)
        assert target > 100.0

    def test_long_target_equals_entry_plus_atr_times_mult(self, engine):
        # target = 100 + (5 × 3) = 115
        target = engine.calculate_take_profit(SignalType.LONG, 100.0, 5.0)
        assert target == pytest.approx(115.0)

    def test_long_spec_example(self, engine):
        # Spec: entry=755.64, ATR=6.25, target=755.64+(6.25×3)=774.39
        target = engine.calculate_take_profit(SignalType.LONG, 755.64, 6.25)
        assert target == pytest.approx(774.39, rel=1e-6)

    def test_long_forex_prices(self, engine):
        target = engine.calculate_take_profit(SignalType.LONG, 1.0900, 0.0050)
        assert target == pytest.approx(1.0900 + 0.0150, rel=1e-6)


class TestTakeProfitShort:

    def test_short_target_below_entry(self, engine):
        target = engine.calculate_take_profit(SignalType.SHORT, entry_price=100.0, atr=5.0)
        assert target < 100.0

    def test_short_target_equals_entry_minus_atr_times_mult(self, engine):
        # target = 100 - (5 × 3) = 85
        target = engine.calculate_take_profit(SignalType.SHORT, 100.0, 5.0)
        assert target == pytest.approx(85.0)

    def test_short_high_priced_asset(self, engine):
        target = engine.calculate_take_profit(SignalType.SHORT, 40000.0, 1200.0)
        assert target == pytest.approx(40000.0 - 3600.0)


class TestRiskRewardImplied:

    def test_rr_equals_target_mult_over_stop_mult(self):
        """Default 3.0 ÷ 2.0 = 1.5 R:R when same ATR used for both."""
        stop_mult   = 2.0
        target_mult = 3.0
        atr = 10.0
        entry = 100.0

        from src.risk.stop_loss_engine import StopLossEngine
        stop   = StopLossEngine(stop_mult).calculate_stop_loss(SignalType.LONG, entry, atr)
        target = TakeProfitEngine(target_mult).calculate_take_profit(SignalType.LONG, entry, atr)

        risk   = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk
        assert rr == pytest.approx(target_mult / stop_mult)


class TestTakeProfitInit:

    def test_zero_multiplier_raises(self):
        with pytest.raises(ValueError):
            TakeProfitEngine(multiplier=0.0)

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError):
            TakeProfitEngine(multiplier=-2.0)

    def test_default_multiplier_is_3(self):
        assert TakeProfitEngine().multiplier == 3.0

    def test_custom_multiplier_stored(self):
        e = TakeProfitEngine(multiplier=4.0)
        assert e.multiplier == 4.0
