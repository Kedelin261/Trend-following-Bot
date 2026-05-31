"""Tests for MarketRegimeFilter — 3-EMA BULL/BEAR/SIDEWAYS classification."""

import pytest

from src.refinement.market_regime_filter import MarketRegimeFilter, RegimeLabel
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def filt() -> MarketRegimeFilter:
    return MarketRegimeFilter(fast_period=5, mid_period=10, slow_period=20)


class TestMarketRegimeFilterInit:

    def test_default_periods(self):
        f = MarketRegimeFilter()
        assert f.fast_period == 20
        assert f.mid_period  == 50
        assert f.slow_period == 200

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            MarketRegimeFilter(fast_period=50, mid_period=20, slow_period=200)

    def test_equal_periods_raise(self):
        with pytest.raises(ValueError):
            MarketRegimeFilter(fast_period=10, mid_period=10, slow_period=20)


class TestClassify:

    def test_insufficient_candles_returns_unknown(self, filt):
        candles = uptrend_candles(n=15)  # less than slow_period=20
        assert filt.classify(candles) == RegimeLabel.UNKNOWN

    def test_strong_uptrend_is_bull(self, filt):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        result = filt.classify(candles)
        assert result == RegimeLabel.BULL

    def test_strong_downtrend_is_bear(self, filt):
        candles = downtrend_candles(n=50, daily_loss=0.005)
        result = filt.classify(candles)
        assert result == RegimeLabel.BEAR

    def test_flat_is_sideways_or_unknown(self, filt):
        candles = flat_candles(n=50)
        result = filt.classify(candles)
        assert result in (RegimeLabel.SIDEWAYS, RegimeLabel.UNKNOWN)


class TestIsBullRegime:

    def test_bull_returns_true(self, filt):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        assert filt.is_bull_regime(candles) is True

    def test_bear_returns_false(self, filt):
        candles = downtrend_candles(n=50, daily_loss=0.005)
        assert filt.is_bull_regime(candles) is False

    def test_flat_returns_false(self, filt):
        candles = flat_candles(n=50)
        assert filt.is_bull_regime(candles) is False

    def test_insufficient_data_returns_false(self, filt):
        candles = uptrend_candles(n=10)
        assert filt.is_bull_regime(candles) is False


class TestEMAValues:

    def test_returns_three_values(self, filt):
        candles = uptrend_candles(n=50)
        values = filt.ema_values(candles)
        assert len(values) == 3

    def test_bull_trend_ema_fast_highest(self, filt):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        fast, mid, slow = filt.ema_values(candles)
        if all(v is not None for v in (fast, mid, slow)):
            assert fast > slow  # fast EMA above slow EMA in uptrend
