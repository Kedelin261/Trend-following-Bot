"""Tests for TrendDetector — EMA calculations and trend classification."""

import pytest
from datetime import datetime, timedelta, timezone

from src.data.models import Candle
from src.signals.models import TrendDirection
from src.signals.trend_detector import TrendDetector
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def det() -> TrendDetector:
    return TrendDetector(fast_period=50, slow_period=200)


# ---------------------------------------------------------------------------
# calculate_ema
# ---------------------------------------------------------------------------

class TestCalculateEMA:

    def test_returns_none_when_fewer_prices_than_period(self, det):
        assert det.calculate_ema([100.0] * 49, 50) is None

    def test_returns_none_for_empty_list(self, det):
        assert det.calculate_ema([], 50) is None

    def test_exact_period_length_returns_sma(self, det):
        prices = [100.0] * 10
        result = det.calculate_ema(prices, 10)
        assert result == pytest.approx(100.0)

    def test_ema_of_flat_series_equals_price(self, det):
        prices = [200.0] * 100
        result = det.calculate_ema(prices, 50)
        assert result == pytest.approx(200.0, rel=1e-6)

    def test_ema_rises_on_increasing_prices(self, det):
        # EMA on an uptrending series should be below the last price
        prices = list(range(100, 200))   # 100 values, rising
        ema = det.calculate_ema(prices, 50)
        assert ema is not None
        assert ema < prices[-1]          # EMA lags price
        assert ema > prices[0]           # but above earliest price

    def test_ema_lags_on_declining_prices(self, det):
        prices = list(range(200, 100, -1))  # 100 declining values
        ema = det.calculate_ema(prices, 50)
        assert ema is not None
        assert ema > prices[-1]   # EMA is above current (lagging)

    def test_ema_50_vs_200_on_long_uptrend(self, det):
        prices = [400.0 * (1.001 ** i) for i in range(210)]
        ema50  = det.calculate_ema(prices, 50)
        ema200 = det.calculate_ema(prices, 200)
        assert ema50 is not None and ema200 is not None
        assert ema50 > ema200, "EMA50 should be above EMA200 in an uptrend"

    def test_single_price_beyond_period_applies_one_smoothing(self):
        det2 = TrendDetector(fast_period=3, slow_period=5)
        # Seed: SMA([10, 10, 10]) = 10; then one bar at 20
        result = det2.calculate_ema([10.0, 10.0, 10.0, 20.0], 3)
        # multiplier = 2/(3+1) = 0.5; EMA = 20×0.5 + 10×0.5 = 15
        assert result == pytest.approx(15.0)

    def test_known_ema_calculation(self):
        det2 = TrendDetector(fast_period=2, slow_period=3)
        # EMA(2) seed = SMA([4, 6]) = 5; then price 8
        # mult = 2/3; EMA = 8×(2/3) + 5×(1/3) = 16/3 + 5/3 = 7
        result = det2.calculate_ema([4.0, 6.0, 8.0], 2)
        assert result == pytest.approx(7.0, rel=1e-6)


# ---------------------------------------------------------------------------
# calculate_ema_series
# ---------------------------------------------------------------------------

class TestCalculateEMASeries:

    def test_length_matches_input(self, det):
        prices = [100.0] * 60
        series = det.calculate_ema_series(prices, 50)
        assert len(series) == 60

    def test_first_warmup_positions_are_none(self, det):
        prices = [100.0] * 60
        series = det.calculate_ema_series(prices, 50)
        assert all(v is None for v in series[:49])

    def test_position_49_equals_sma(self, det):
        prices = [100.0] * 60
        series = det.calculate_ema_series(prices, 50)
        assert series[49] == pytest.approx(100.0)

    def test_all_values_after_warmup_are_floats(self, det):
        prices = [float(x) for x in range(1, 61)]
        series = det.calculate_ema_series(prices, 10)
        for v in series[10:]:
            assert isinstance(v, float)

    def test_empty_input_returns_empty(self, det):
        assert det.calculate_ema_series([], 50) == []


# ---------------------------------------------------------------------------
# get_trend_direction
# ---------------------------------------------------------------------------

class TestGetTrendDirection:

    def test_bullish_on_uptrend(self, det):
        candles = uptrend_candles(n=250)
        assert det.get_trend_direction(candles) == TrendDirection.BULLISH

    def test_bearish_on_downtrend(self, det):
        candles = downtrend_candles(n=250)
        assert det.get_trend_direction(candles) == TrendDirection.BEARISH

    def test_neutral_on_flat_prices(self, det):
        candles = flat_candles(n=250)
        # Flat series → all EMAs equal → EMA50 == EMA200 → NEUTRAL
        assert det.get_trend_direction(candles) == TrendDirection.NEUTRAL

    def test_neutral_when_insufficient_candles(self, det):
        candles = uptrend_candles(n=199)
        assert det.get_trend_direction(candles) == TrendDirection.NEUTRAL

    def test_neutral_on_exactly_slow_period_minus_one(self, det):
        candles = uptrend_candles(n=199)
        assert det.get_trend_direction(candles) == TrendDirection.NEUTRAL

    def test_can_classify_with_exactly_slow_period_candles(self, det):
        candles = uptrend_candles(n=200)
        direction = det.get_trend_direction(candles)
        # With exactly 200 bars of uptrend, EMA50 may already exceed EMA200
        assert direction in (TrendDirection.BULLISH, TrendDirection.NEUTRAL)

    def test_bullish_requires_both_conditions(self, det):
        # Price above EMA50 but EMA50 below EMA200 → NEUTRAL
        # Simulate this by using a short uptrend after a long downtrend
        down = downtrend_candles(n=200, start=500.0)
        up   = [
            Candle(
                symbol="SPY", timeframe="D1",
                timestamp=down[-1].timestamp + timedelta(days=i + 1),
                open=down[-1].close * 1.001,
                high=down[-1].close * 1.005,
                low=down[-1].close * 0.999,
                close=down[-1].close * (1.005 ** (i + 1)),
                volume=1e8,
                provider="TEST",
            )
            for i in range(5)
        ]
        combined = down + up
        direction = det.get_trend_direction(combined)
        # After 200 downtrend bars, 5 recovery bars cannot flip EMA50 above EMA200
        # or push price above EMA50 — BULLISH requires BOTH conditions simultaneously
        assert direction != TrendDirection.BULLISH


# ---------------------------------------------------------------------------
# get_trend_strength
# ---------------------------------------------------------------------------

class TestGetTrendStrength:

    def test_returns_zero_for_insufficient_data(self, det):
        candles = uptrend_candles(n=50)
        assert det.get_trend_strength(candles) == 0.0

    def test_strength_in_valid_range(self, det):
        candles = uptrend_candles(n=250)
        strength = det.get_trend_strength(candles)
        assert 0.0 <= strength <= 100.0

    def test_strong_uptrend_scores_higher_than_weak(self, det):
        strong = uptrend_candles(n=250, daily_gain=0.003)
        weak   = uptrend_candles(n=250, daily_gain=0.0003)
        assert det.get_trend_strength(strong) > det.get_trend_strength(weak)

    def test_neutral_returns_25_baseline(self, det):
        candles = flat_candles(n=250)
        assert det.get_trend_strength(candles) == pytest.approx(25.0)

    def test_downtrend_strength_in_range(self, det):
        candles = downtrend_candles(n=250)
        strength = det.get_trend_strength(candles)
        assert 0.0 <= strength <= 100.0


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestTrendDetectorInit:

    def test_raises_if_fast_gte_slow(self):
        with pytest.raises(ValueError):
            TrendDetector(fast_period=200, slow_period=50)

    def test_raises_if_equal_periods(self):
        with pytest.raises(ValueError):
            TrendDetector(fast_period=50, slow_period=50)

    def test_custom_periods_accepted(self):
        det = TrendDetector(fast_period=10, slow_period=20)
        assert det.fast_period == 10
        assert det.slow_period == 20
