"""Tests for BreakoutDetector — bullish and bearish breakout logic."""

import pytest

from src.signals.breakout_detector import BreakoutDetector


@pytest.fixture
def det() -> BreakoutDetector:
    return BreakoutDetector(threshold=0.0025)


# ---------------------------------------------------------------------------
# Bullish breakout
# ---------------------------------------------------------------------------

class TestDetectBullishBreakout:

    def test_confirmed_breakout_above_resistance(self, det):
        # Resistance 100, threshold 0.25% → need close > 100.25
        detected, level = det.detect_bullish_breakout(101.0, [100.0])
        assert detected is True
        assert level == pytest.approx(100.0)

    def test_exactly_at_threshold_is_not_breakout(self, det):
        # close = 100.25 → not strictly greater
        detected, level = det.detect_bullish_breakout(100.25, [100.0])
        assert detected is False

    def test_just_above_threshold_is_breakout(self, det):
        detected, level = det.detect_bullish_breakout(100.251, [100.0])
        assert detected is True

    def test_below_resistance_no_breakout(self, det):
        detected, _ = det.detect_bullish_breakout(99.0, [100.0])
        assert detected is False

    def test_empty_resistance_returns_false(self, det):
        detected, level = det.detect_bullish_breakout(105.0, [])
        assert detected is False
        assert level is None

    def test_returns_highest_breached_resistance(self, det):
        # Multiple resistances broken: 95, 98; current close = 105
        detected, level = det.detect_bullish_breakout(105.0, [95.0, 98.0, 110.0])
        assert detected is True
        assert level == pytest.approx(98.0)   # highest broken level

    def test_only_most_significant_level_below_close(self, det):
        detected, level = det.detect_bullish_breakout(200.0, [50.0, 100.0, 150.0])
        assert detected is True
        assert level == pytest.approx(150.0)

    def test_zero_close_returns_false(self, det):
        detected, level = det.detect_bullish_breakout(0.0, [100.0])
        assert detected is False
        assert level is None

    def test_zero_threshold_breakout_on_any_close_above(self):
        det0 = BreakoutDetector(threshold=0.0)
        detected, level = det0.detect_bullish_breakout(100.01, [100.0])
        assert detected is True

    def test_forex_prices_eurusd(self, det):
        # EURUSD: resistance at 1.0900, close at 1.0930
        detected, level = det.detect_bullish_breakout(1.0930, [1.0900])
        assert detected is True

    def test_forex_prices_no_breakout(self, det):
        detected, _ = det.detect_bullish_breakout(1.0920, [1.0900])
        assert detected is False   # 1.0900 × 1.0025 = 1.09273 > 1.0920


# ---------------------------------------------------------------------------
# Bearish breakout
# ---------------------------------------------------------------------------

class TestDetectBearishBreakout:

    def test_confirmed_breakdown_below_support(self, det):
        # Support 100, threshold 0.25% → need close < 99.75
        detected, level = det.detect_bearish_breakout(99.0, [100.0])
        assert detected is True
        assert level == pytest.approx(100.0)

    def test_exactly_at_threshold_is_not_breakout(self, det):
        detected, _ = det.detect_bearish_breakout(99.75, [100.0])
        assert detected is False

    def test_just_below_threshold_is_breakout(self, det):
        detected, _ = det.detect_bearish_breakout(99.749, [100.0])
        assert detected is True

    def test_above_support_no_breakdown(self, det):
        detected, _ = det.detect_bearish_breakout(101.0, [100.0])
        assert detected is False

    def test_empty_support_returns_false(self, det):
        detected, level = det.detect_bearish_breakout(95.0, [])
        assert detected is False
        assert level is None

    def test_returns_lowest_broken_support(self, det):
        # Multiple supports broken: 105, 102; current = 98
        detected, level = det.detect_bearish_breakout(98.0, [105.0, 102.0, 90.0])
        assert detected is True
        assert level == pytest.approx(102.0)   # lowest broken support

    def test_zero_close_returns_false(self, det):
        detected, level = det.detect_bearish_breakout(0.0, [100.0])
        assert detected is False
        assert level is None

    def test_high_price_asset_btcusd(self, det):
        # BTC: support at 40000, close at 39800
        detected, level = det.detect_bearish_breakout(39800.0, [40000.0])
        assert detected is True


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestBreakoutDetectorInit:

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            BreakoutDetector(threshold=-0.01)

    def test_zero_threshold_accepted(self):
        det = BreakoutDetector(threshold=0.0)
        assert det.threshold == 0.0

    def test_custom_threshold_stored(self):
        det = BreakoutDetector(threshold=0.01)
        assert det.threshold == 0.01
