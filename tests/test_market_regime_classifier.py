"""Tests for MarketRegimeClassifier — five-state EMA classification."""

import pytest

from src.regime.market_regime_classifier import MarketRegime, MarketRegimeClassifier
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def clf() -> MarketRegimeClassifier:
    return MarketRegimeClassifier(ema_fast=5, ema_mid=10, ema_slow=20, strong_threshold=0.03)


class TestMarketRegimeClassifier:

    def test_insufficient_candles_returns_unknown(self, clf):
        candles = uptrend_candles(n=10)
        assert clf.classify(candles) == MarketRegime.UNKNOWN

    def test_strong_uptrend_is_strong_bull_or_bull(self, clf):
        candles = uptrend_candles(n=50, daily_gain=0.006)
        result = clf.classify(candles)
        assert result in (MarketRegime.STRONG_BULL, MarketRegime.BULL)

    def test_strong_downtrend_is_strong_bear_or_bear(self, clf):
        candles = downtrend_candles(n=50, daily_loss=0.006)
        result = clf.classify(candles)
        assert result in (MarketRegime.STRONG_BEAR, MarketRegime.BEAR)

    def test_flat_candles_are_sideways_or_unknown(self, clf):
        candles = flat_candles(n=50)
        result = clf.classify(candles)
        assert result in (MarketRegime.SIDEWAYS, MarketRegime.UNKNOWN)

    def test_regime_series_length_matches_candles(self, clf):
        candles = uptrend_candles(n=30)
        series = clf.classify_series(candles)
        assert len(series) == 30

    def test_early_bars_in_series_are_unknown(self, clf):
        candles = uptrend_candles(n=30)
        series = clf.classify_series(candles)
        assert series[0] == MarketRegime.UNKNOWN

    def test_classify_at_timestamp_early_is_unknown(self, clf):
        candles = uptrend_candles(n=30)
        early_ts = candles[5].timestamp
        result = clf.classify_at_timestamp(candles, early_ts)
        assert result == MarketRegime.UNKNOWN

    def test_strong_threshold_determines_bull_vs_strong_bull(self):
        # With very high threshold, never STRONG — always BULL
        clf_strict = MarketRegimeClassifier(
            ema_fast=5, ema_mid=10, ema_slow=20, strong_threshold=0.99
        )
        candles = uptrend_candles(n=50, daily_gain=0.005)
        result = clf_strict.classify(candles)
        if result not in (MarketRegime.UNKNOWN, MarketRegime.SIDEWAYS):
            assert result == MarketRegime.BULL  # cannot be STRONG_BULL

    def test_all_regime_values_are_strings(self):
        for r in MarketRegime:
            assert isinstance(r.value, str)


class TestMarketRegimeEnum:

    def test_five_states_defined(self):
        states = {r.value for r in MarketRegime}
        assert {"STRONG_BULL", "BULL", "SIDEWAYS", "BEAR", "STRONG_BEAR", "UNKNOWN"}.issubset(states)
