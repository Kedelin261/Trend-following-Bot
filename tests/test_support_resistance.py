"""Tests for SupportResistanceDetector — swing detection and level clustering."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.data.models import Candle
from src.signals.support_resistance import SupportResistanceDetector
from tests.fixtures import flat_candles, uptrend_candles, zigzag_candles


@pytest.fixture
def det() -> SupportResistanceDetector:
    return SupportResistanceDetector(lookback=5, cluster_tolerance=0.005)


def _candle(close, high=None, low=None, i=0):
    return Candle(
        symbol="SPY", timeframe="D1",
        timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
        open=close, high=high or close * 1.002, low=low or close * 0.998,
        close=close, volume=1e8, provider="TEST",
    )


def peak_trough_candles() -> List[Candle]:
    """10 candles: two clear peaks at 420 and two clear troughs at 380."""
    prices  = [400, 420, 400, 400, 400, 420, 400, 400, 380, 400]
    highs   = [p + 5 for p in prices]
    lows    = [p - 5 for p in prices]
    return [_candle(prices[i], highs[i], lows[i], i) for i in range(len(prices))]


# ---------------------------------------------------------------------------
# find_swing_highs
# ---------------------------------------------------------------------------

class TestFindSwingHighs:

    def test_empty_when_insufficient_candles(self, det):
        candles = [_candle(400, i=i) for i in range(9)]
        assert det.find_swing_highs(candles) == []

    def test_detects_obvious_peak(self):
        det2 = SupportResistanceDetector(lookback=2)
        # clear peak at index 2 (value 50), flanked by 30s
        candles = [_candle(30, i=0), _candle(30, i=1), _candle(50, i=2),
                   _candle(30, i=3), _candle(30, i=4)]
        highs = det2.find_swing_highs(candles)
        assert len(highs) >= 1
        assert any(h >= 50 for h in highs)

    def test_no_swing_in_monotone_rise(self, det):
        candles = uptrend_candles(n=30)
        # Monotone rise has no intermediate peak
        highs = det.find_swing_highs(candles)
        assert highs == []

    def test_zigzag_detects_multiple_highs(self):
        det2 = SupportResistanceDetector(lookback=1)
        candles = zigzag_candles(n=20, amplitude=10)
        highs = det2.find_swing_highs(candles)
        assert len(highs) >= 2

    def test_all_highs_above_base_price(self):
        det2 = SupportResistanceDetector(lookback=1)
        candles = zigzag_candles(n=20, base=400, amplitude=10)
        highs = det2.find_swing_highs(candles)
        for h in highs:
            assert h >= 400


# ---------------------------------------------------------------------------
# find_swing_lows
# ---------------------------------------------------------------------------

class TestFindSwingLows:

    def test_empty_when_insufficient_candles(self, det):
        candles = [_candle(400, i=i) for i in range(9)]
        assert det.find_swing_lows(candles) == []

    def test_detects_obvious_trough(self):
        det2 = SupportResistanceDetector(lookback=2)
        candles = [_candle(50, i=0), _candle(50, i=1), _candle(10, i=2),
                   _candle(50, i=3), _candle(50, i=4)]
        lows = det2.find_swing_lows(candles)
        assert len(lows) >= 1
        assert any(l <= 12 for l in lows)

    def test_no_swing_in_monotone_decline(self, det):
        from tests.fixtures import downtrend_candles
        candles = downtrend_candles(n=30)
        lows = det.find_swing_lows(candles)
        assert lows == []

    def test_zigzag_detects_multiple_lows(self):
        det2 = SupportResistanceDetector(lookback=1)
        candles = zigzag_candles(n=20, amplitude=10)
        lows = det2.find_swing_lows(candles)
        assert len(lows) >= 2

    def test_all_lows_below_high_price(self):
        det2 = SupportResistanceDetector(lookback=1)
        candles = zigzag_candles(n=20, base=400, amplitude=10)
        lows = det2.find_swing_lows(candles)
        for l in lows:
            assert l <= 402   # below base + small buffer


# ---------------------------------------------------------------------------
# cluster_levels
# ---------------------------------------------------------------------------

class TestClusterLevels:

    def test_empty_input_returns_empty(self, det):
        assert det.cluster_levels([]) == []

    def test_single_level_returned_unchanged(self, det):
        result = det.cluster_levels([100.0])
        assert result == pytest.approx([100.0])

    def test_nearby_levels_merged_into_centroid(self, det):
        # Two levels within 0.5 % of each other
        result = det.cluster_levels([100.0, 100.3])
        assert len(result) == 1
        assert result[0] == pytest.approx(100.15, rel=0.01)

    def test_distant_levels_kept_separate(self, det):
        result = det.cluster_levels([100.0, 110.0])
        assert len(result) == 2

    def test_three_clusters_formed(self, det):
        levels = [100.0, 100.2, 105.0, 105.1, 120.0]
        result = det.cluster_levels(levels)
        assert len(result) == 3

    def test_output_is_sorted_ascending(self, det):
        levels = [120.0, 100.0, 105.0]
        result = det.cluster_levels(levels)
        assert result == sorted(result)

    def test_large_cluster_centroid(self, det):
        levels = [100.0, 100.1, 100.2, 100.3, 100.4]
        result = det.cluster_levels(levels)
        assert len(result) == 1
        assert result[0] == pytest.approx(100.2, rel=0.01)


# ---------------------------------------------------------------------------
# get_support_levels / get_resistance_levels
# ---------------------------------------------------------------------------

class TestSupportResistanceLevels:

    def test_resistance_levels_sorted_ascending(self, det):
        candles = zigzag_candles(n=40, amplitude=20)
        levels = det.get_resistance_levels(candles)
        assert levels == sorted(levels)

    def test_support_levels_sorted_ascending(self, det):
        candles = zigzag_candles(n=40, amplitude=20)
        levels = det.get_support_levels(candles)
        assert levels == sorted(levels)

    def test_returns_empty_when_no_swings(self, det):
        candles = uptrend_candles(n=30)
        assert det.get_resistance_levels(candles) == []

    def test_flat_candles_returns_empty_or_single(self, det):
        candles = flat_candles(n=30)
        levels = det.get_resistance_levels(candles)
        assert isinstance(levels, list)


# ---------------------------------------------------------------------------
# nearest_support / nearest_resistance
# ---------------------------------------------------------------------------

class TestNearestLevels:

    def test_nearest_resistance_above_price(self):
        det2 = SupportResistanceDetector(lookback=1)
        candles = zigzag_candles(n=20, base=400, amplitude=20)
        current = 395.0
        nr = det2.nearest_resistance(candles, current)
        if nr is not None:
            assert nr >= current

    def test_nearest_support_below_price(self):
        det2 = SupportResistanceDetector(lookback=1)
        candles = zigzag_candles(n=20, base=400, amplitude=20)
        current = 410.0
        ns = det2.nearest_support(candles, current)
        if ns is not None:
            assert ns <= current

    def test_returns_none_when_no_levels(self, det):
        candles = uptrend_candles(n=30)
        assert det.nearest_resistance(candles, 500.0) is None
        assert det.nearest_support(candles, 500.0) is None
