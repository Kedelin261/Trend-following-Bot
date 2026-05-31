"""Tests for SignalScorer — composite scoring and categorization."""

import pytest

from src.signals.models import SignalStrength, SignalType, TrendDirection
from src.signals.signal_scorer import SignalScorer


@pytest.fixture
def scorer() -> SignalScorer:
    return SignalScorer()


# ---------------------------------------------------------------------------
# score_signal — basic cases
# ---------------------------------------------------------------------------

class TestScoreSignal:

    def test_none_signal_returns_zero(self, scorer):
        score = scorer.score_signal(
            trend_direction=TrendDirection.BULLISH,
            trend_strength=80,
            breakout_detected=True,
            volume_confirmed=True,
            signal_type=SignalType.NONE,
        )
        assert score == 0.0

    def test_perfect_long_scores_near_100(self, scorer):
        score = scorer.score_signal(
            trend_direction=TrendDirection.BULLISH,
            trend_strength=100,
            breakout_detected=True,
            volume_confirmed=True,
            signal_type=SignalType.LONG,
        )
        assert score >= 95.0

    def test_perfect_short_scores_near_100(self, scorer):
        score = scorer.score_signal(
            trend_direction=TrendDirection.BEARISH,
            trend_strength=100,
            breakout_detected=True,
            volume_confirmed=True,
            signal_type=SignalType.SHORT,
        )
        assert score >= 95.0

    def test_no_breakout_lowers_score_significantly(self, scorer):
        with_breakout = scorer.score_signal(
            TrendDirection.BULLISH, 80, True, True, SignalType.LONG
        )
        without_breakout = scorer.score_signal(
            TrendDirection.BULLISH, 80, False, True, SignalType.LONG
        )
        assert with_breakout > without_breakout + 30

    def test_volume_not_confirmed_lowers_score(self, scorer):
        confirmed = scorer.score_signal(
            TrendDirection.BULLISH, 80, True, True, SignalType.LONG
        )
        not_confirmed = scorer.score_signal(
            TrendDirection.BULLISH, 80, True, False, SignalType.LONG
        )
        assert confirmed > not_confirmed

    def test_unknown_volume_between_confirmed_and_rejected(self, scorer):
        confirmed = scorer.score_signal(
            TrendDirection.BULLISH, 80, True, True, SignalType.LONG
        )
        unknown = scorer.score_signal(
            TrendDirection.BULLISH, 80, True, None, SignalType.LONG
        )
        not_conf = scorer.score_signal(
            TrendDirection.BULLISH, 80, True, False, SignalType.LONG
        )
        assert confirmed > unknown > not_conf

    def test_score_bounded_0_to_100(self, scorer):
        for strength in [0, 50, 100, 200]:
            score = scorer.score_signal(
                TrendDirection.BULLISH, strength, True, True, SignalType.LONG
            )
            assert 0.0 <= score <= 100.0

    def test_neutral_trend_gives_zero_trend_component(self, scorer):
        # NEUTRAL trend → trend_score = 0 → only breakout can contribute
        score = scorer.score_signal(
            TrendDirection.NEUTRAL, 80, True, True, SignalType.LONG
        )
        # trend contributes 0, breakout 100×0.4=40, volume 100×0.2=20 → 60
        assert score == pytest.approx(60.0, abs=2)

    def test_weak_trend_floor_at_40(self, scorer):
        # Directional trend always gets at least 40 trend score
        score = scorer.score_signal(
            TrendDirection.BULLISH, 0.0, False, False, SignalType.LONG
        )
        # trend=40×0.4=16, breakout=0, volume=0 → 16
        assert score == pytest.approx(16.0, abs=1)


# ---------------------------------------------------------------------------
# score_signal — weight verification
# ---------------------------------------------------------------------------

class TestScoreWeights:

    def test_trend_weight_40_percent(self, scorer):
        # Only trend contributes: neutral→0 but BULLISH with strength 100
        # trend_score=max(40,100)=100; ×0.4 = 40
        score = scorer.score_signal(
            TrendDirection.BULLISH, 100, False, False, SignalType.LONG
        )
        assert score == pytest.approx(40.0, abs=1)

    def test_breakout_weight_40_percent(self, scorer):
        # Only breakout contributes (neutral trend = 0 contribution)
        score = scorer.score_signal(
            TrendDirection.NEUTRAL, 0, True, False, SignalType.LONG
        )
        assert score == pytest.approx(40.0, abs=1)

    def test_volume_weight_20_percent(self, scorer):
        # Only volume contributes
        score = scorer.score_signal(
            TrendDirection.NEUTRAL, 0, False, True, SignalType.LONG
        )
        assert score == pytest.approx(20.0, abs=1)

    def test_unknown_volume_contributes_10_percent(self, scorer):
        # Unknown volume = 50 × 0.20 = 10
        score = scorer.score_signal(
            TrendDirection.NEUTRAL, 0, False, None, SignalType.LONG
        )
        assert score == pytest.approx(10.0, abs=1)


# ---------------------------------------------------------------------------
# categorize_score
# ---------------------------------------------------------------------------

class TestCategorizeScore:

    def test_score_70_is_strong(self, scorer):
        assert scorer.categorize_score(70.0) == SignalStrength.STRONG

    def test_score_90_is_strong(self, scorer):
        assert scorer.categorize_score(90.0) == SignalStrength.STRONG

    def test_score_100_is_strong(self, scorer):
        assert scorer.categorize_score(100.0) == SignalStrength.STRONG

    def test_score_69_is_moderate(self, scorer):
        assert scorer.categorize_score(69.9) == SignalStrength.MODERATE

    def test_score_40_is_moderate(self, scorer):
        assert scorer.categorize_score(40.0) == SignalStrength.MODERATE

    def test_score_39_is_weak(self, scorer):
        assert scorer.categorize_score(39.9) == SignalStrength.WEAK

    def test_score_0_is_weak(self, scorer):
        assert scorer.categorize_score(0.0) == SignalStrength.WEAK

    def test_score_50_is_moderate(self, scorer):
        assert scorer.categorize_score(50.0) == SignalStrength.MODERATE
