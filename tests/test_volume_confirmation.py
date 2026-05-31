"""Tests for VolumeConfirmation — average volume and confirmation logic."""

import pytest
from datetime import datetime, timedelta, timezone

from src.data.models import Candle
from src.signals.volume_confirmation import VolumeConfirmation
from tests.fixtures import spike_volume_candles, uptrend_candles, zero_volume_candles


@pytest.fixture
def vc() -> VolumeConfirmation:
    return VolumeConfirmation(lookback=20)


def _candle(volume: float, i: int = 0, price: float = 400.0) -> Candle:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return Candle(
        symbol="SPY", timeframe="D1",
        timestamp=base + timedelta(days=i),
        open=price, high=price * 1.002, low=price * 0.998,
        close=price, volume=volume, provider="TEST",
    )


# ---------------------------------------------------------------------------
# average_volume
# ---------------------------------------------------------------------------

class TestAverageVolume:

    def test_returns_none_when_insufficient_candles(self, vc):
        candles = [_candle(1e8, i=i) for i in range(20)]
        assert vc.average_volume(candles) is None

    def test_returns_correct_average(self, vc):
        candles = [_candle(1e8, i=i) for i in range(22)]   # 21 reference + 1 current
        avg = vc.average_volume(candles)
        assert avg == pytest.approx(1e8, rel=1e-6)

    def test_excludes_current_candle_from_average(self, vc):
        base_vol = 1e8
        candles = [_candle(base_vol, i=i) for i in range(21)]
        # Replace last candle with spike volume
        candles[-1] = _candle(9e8, i=20)
        avg = vc.average_volume(candles)
        # Average should reflect base volume, not spike
        assert avg == pytest.approx(base_vol, rel=1e-6)

    def test_zero_volume_candles_return_none(self, vc):
        candles = zero_volume_candles(n=25)
        assert vc.average_volume(candles) is None

    def test_partial_zero_volumes_excluded_from_average(self, vc):
        candles = [_candle(0.0, i=i) for i in range(10)]
        candles += [_candle(1e8, i=i + 10) for i in range(12)]
        avg = vc.average_volume(candles)
        # Should average only non-zero volumes
        assert avg is not None
        assert avg == pytest.approx(1e8, rel=0.1)


# ---------------------------------------------------------------------------
# is_volume_confirmed
# ---------------------------------------------------------------------------

class TestIsVolumeConfirmed:

    def test_high_volume_returns_true(self, vc):
        candles = spike_volume_candles(n=25, base_volume=1e8, spike_multiple=3.0)
        result = vc.is_volume_confirmed(candles)
        assert result is True

    def test_low_volume_returns_false(self):
        vc2 = VolumeConfirmation(lookback=5)
        base = 1e8
        # 5 reference bars at 1e8, then current at 0.5e8 (below average)
        candles = [_candle(base, i=i) for i in range(6)]
        candles[-1] = _candle(base * 0.5, i=5)
        result = vc2.is_volume_confirmed(candles)
        assert result is False

    def test_zero_volume_returns_none(self, vc):
        candles = zero_volume_candles(n=30)
        assert vc.is_volume_confirmed(candles) is None

    def test_empty_candles_return_none(self, vc):
        assert vc.is_volume_confirmed([]) is None

    def test_insufficient_candles_return_none(self, vc):
        candles = [_candle(1e8, i=i) for i in range(5)]
        assert vc.is_volume_confirmed(candles) is None

    def test_normal_uptrend_volume_returns_bool(self, vc):
        candles = uptrend_candles(n=25, volume=1e8)
        result = vc.is_volume_confirmed(candles)
        assert isinstance(result, bool)

    def test_volume_exactly_equal_to_average_is_not_confirmed(self):
        vc2 = VolumeConfirmation(lookback=5)
        candles = [_candle(1e8, i=i) for i in range(7)]
        result = vc2.is_volume_confirmed(candles)
        # current == avg → not > avg → False
        assert result is False

    def test_volume_just_above_average_is_confirmed(self):
        vc2 = VolumeConfirmation(lookback=5)
        candles = [_candle(1e8, i=i) for i in range(6)]
        candles[-1] = _candle(1e8 + 1, i=5)
        result = vc2.is_volume_confirmed(candles)
        assert result is True


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestVolumeConfirmationInit:

    def test_lookback_below_2_raises(self):
        with pytest.raises(ValueError):
            VolumeConfirmation(lookback=1)

    def test_lookback_2_accepted(self):
        vc = VolumeConfirmation(lookback=2)
        assert vc.lookback == 2

    def test_custom_lookback_stored(self):
        vc = VolumeConfirmation(lookback=50)
        assert vc.lookback == 50
