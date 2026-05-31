"""Tests for TimeframeProfile and predefined profiles."""

import pytest

from src.timeframe.timeframe_profile import (
    ALL_COMBOS,
    ALL_SINGLE_PROFILES,
    BEST_DENSITY_PROFILE,
    COMBO_D1_H4,
    D1_PROFILE,
    H1_PROFILE,
    H4_PROFILE,
    MultiTimeframeProfile,
    TimeframeProfile,
    W1_PROFILE,
)
from src.refinement.volatility_trade_filter import VolatilityFilterMode


class TestTimeframeProfile:

    def test_d1_profile_attributes(self):
        assert D1_PROFILE.name == "D1"
        assert D1_PROFILE.timeframe == "D1"
        assert D1_PROFILE.bars_per_year == 252
        assert D1_PROFILE.min_warmup == 210

    def test_h4_profile_attributes(self):
        assert H4_PROFILE.timeframe == "H4"
        assert H4_PROFILE.is_intraday is True

    def test_h1_profile_is_intraday(self):
        assert H1_PROFILE.is_intraday is True

    def test_w1_profile_is_not_intraday(self):
        assert W1_PROFILE.is_intraday is False

    def test_d1_is_daily_or_higher(self):
        assert D1_PROFILE.is_daily_or_higher is True

    def test_h4_is_not_daily_or_higher(self):
        assert H4_PROFILE.is_daily_or_higher is False

    def test_years_from_bars_d1(self):
        # 252 bars / 252 per year = 1.0 year
        years = D1_PROFILE.years_from_bars(252)
        assert years == pytest.approx(1.0, rel=0.01)

    def test_years_from_bars_h4(self):
        # 409 bars / 409 per year ≈ 1 year
        years = H4_PROFILE.years_from_bars(409)
        assert years == pytest.approx(1.0, rel=0.01)

    def test_all_single_profiles_length(self):
        assert len(ALL_SINGLE_PROFILES) == 4

    def test_all_combos_length(self):
        assert len(ALL_COMBOS) == 3


class TestMultiTimeframeProfile:

    def test_label_format(self):
        assert COMBO_D1_H4.label == "D1 + H4"

    def test_timeframes_property(self):
        assert COMBO_D1_H4.timeframes == ["D1", "H4"]

    def test_combo_has_two_profiles(self):
        assert len(COMBO_D1_H4.profiles) == 2

    def test_three_tf_combo_label(self):
        from src.timeframe.timeframe_profile import COMBO_D1_H4_H1
        assert "D1" in COMBO_D1_H4_H1.label
        assert "H4" in COMBO_D1_H4_H1.label
        assert "H1" in COMBO_D1_H4_H1.label


class TestBestDensityProfile:

    def test_adx_threshold_is_27(self):
        assert BEST_DENSITY_PROFILE.adx_threshold == 27.0

    def test_breakout_is_1_pct(self):
        assert BEST_DENSITY_PROFILE.breakout_threshold == pytest.approx(0.0100)

    def test_volatility_mode_is_medium_and_high(self):
        assert BEST_DENSITY_PROFILE.volatility_mode == VolatilityFilterMode.MEDIUM_AND_HIGH

    def test_requires_bull_regime(self):
        assert BEST_DENSITY_PROFILE.require_bull_regime is True

    def test_ema_fast_20_slow_50(self):
        assert BEST_DENSITY_PROFILE.ema_fast == 20
        assert BEST_DENSITY_PROFILE.ema_slow == 50
