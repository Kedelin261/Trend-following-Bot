"""Tests for TrendQualityAnalyzer."""

import pytest

from src.research.trend_quality import TrendQualityAnalyzer, TrendQualityCategory
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def analyzer() -> TrendQualityAnalyzer:
    return TrendQualityAnalyzer(fast_period=5, slow_period=10, slope_lookback=3)


class TestTrendQualityAnalyzer:

    def test_returns_unknown_for_insufficient_candles(self, analyzer):
        candles = uptrend_candles(n=5)
        tq = analyzer.analyze(candles)
        assert tq.category == TrendQualityCategory.UNKNOWN

    def test_strong_uptrend_gives_high_score(self, analyzer):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        tq = analyzer.analyze(candles)
        assert tq.quality_score > 30.0

    def test_score_in_valid_range(self, analyzer):
        for fixture in [uptrend_candles(n=50), downtrend_candles(n=50), flat_candles(n=50)]:
            tq = analyzer.analyze(fixture)
            assert 0.0 <= tq.quality_score <= 100.0

    def test_ema_fields_populated_when_sufficient_data(self, analyzer):
        candles = uptrend_candles(n=50)
        tq = analyzer.analyze(candles)
        if tq.category != TrendQualityCategory.UNKNOWN:
            assert tq.ema_fast is not None
            assert tq.ema_slow is not None

    def test_uptrend_price_above_ema_fast(self, analyzer):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        tq = analyzer.analyze(candles)
        if tq.category != TrendQualityCategory.UNKNOWN:
            assert tq.price_above_ema_fast is True

    def test_downtrend_price_below_ema_fast(self, analyzer):
        candles = downtrend_candles(n=50, daily_loss=0.005)
        tq = analyzer.analyze(candles)
        if tq.category != TrendQualityCategory.UNKNOWN:
            assert tq.price_above_ema_fast is False

    def test_strong_trend_scores_higher_than_flat(self, analyzer):
        trending = uptrend_candles(n=50, daily_gain=0.005)
        flat     = flat_candles(n=50)
        tq_t = analyzer.analyze(trending)
        tq_f = analyzer.analyze(flat)
        assert tq_t.quality_score >= tq_f.quality_score

    def test_category_strong_for_high_score(self, analyzer):
        candles = uptrend_candles(n=100, daily_gain=0.005)
        tq = analyzer.analyze(candles)
        if tq.quality_score >= 60:
            assert tq.category == TrendQualityCategory.STRONG

    def test_ema_separation_nonnegative(self, analyzer):
        candles = uptrend_candles(n=50)
        tq = analyzer.analyze(candles)
        assert tq.ema_separation_pct >= 0.0
