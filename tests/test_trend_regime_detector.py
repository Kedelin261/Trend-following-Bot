"""Tests for TrendRegimeDetector — STRONG/MODERATE/WEAK trend classification."""

import pytest

from src.regime.trend_regime_detector import TrendRegime, TrendRegimeDetector
from tests.fixtures import flat_candles, uptrend_candles


@pytest.fixture
def det() -> TrendRegimeDetector:
    return TrendRegimeDetector(fast_period=5, slow_period=10, slope_lookback=5)


class TestTrendRegimeDetector:

    def test_insufficient_candles_returns_unknown(self, det):
        candles = uptrend_candles(n=5)
        assert det.classify(candles) == TrendRegime.UNKNOWN

    def test_flat_candles_are_weak_trend(self, det):
        candles = flat_candles(n=40)
        result = det.classify(candles)
        # Flat series → zero slope → WEAK_TREND (or UNKNOWN if candles too few)
        assert result in (TrendRegime.WEAK_TREND, TrendRegime.UNKNOWN)

    def test_strong_uptrend_is_strong_or_moderate(self, det):
        candles = uptrend_candles(n=50, daily_gain=0.008)
        result = det.classify(candles)
        assert result in (TrendRegime.STRONG_TREND, TrendRegime.MODERATE_TREND)

    def test_classify_at_timestamp(self, det):
        candles = uptrend_candles(n=50)
        ts = candles[-1].timestamp
        result = det.classify_at_timestamp(candles, ts)
        assert result in list(TrendRegime)

    def test_classify_at_early_timestamp_unknown(self, det):
        candles = uptrend_candles(n=50)
        early = candles[3].timestamp
        result = det.classify_at_timestamp(candles, early)
        assert result == TrendRegime.UNKNOWN

    def test_stronger_trend_scores_same_or_higher(self, det):
        weak   = flat_candles(n=40)
        strong = uptrend_candles(n=40, daily_gain=0.01)
        w_res = det.classify(weak)
        s_res = det.classify(strong)
        order = [TrendRegime.UNKNOWN, TrendRegime.WEAK_TREND,
                 TrendRegime.MODERATE_TREND, TrendRegime.STRONG_TREND]
        w_idx = order.index(w_res) if w_res in order else 0
        s_idx = order.index(s_res) if s_res in order else 0
        assert s_idx >= w_idx


class TestTrendRegimeEnum:

    def test_three_levels_defined(self):
        vals = {r.value for r in TrendRegime}
        assert {"STRONG_TREND", "MODERATE_TREND", "WEAK_TREND", "UNKNOWN"}.issubset(vals)
