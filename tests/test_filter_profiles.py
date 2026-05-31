"""Tests for FilterProfile — immutable filter configuration."""

import pytest

from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime
from src.regime_filter.filter_profiles import (
    AVOID_BEAR,
    AVOID_CRISIS_AND_CRASH,
    AVOID_EXTREME_VOL,
    AVOID_STRONG_BULL,
    AVOID_STRONG_BULL_AND_EXTREME_VOL,
    BULL_ONLY,
    COMPREHENSIVE_FILTER,
    NO_FILTER,
    RESEARCH_PROFILES,
    FilterProfile,
)


class TestFilterProfileImmutability:

    def test_frozen_dataclass_raises_on_modification(self):
        with pytest.raises((TypeError, AttributeError)):
            NO_FILTER.name = "changed"  # type: ignore[misc]

    def test_frozensets_are_hashable(self):
        # FilterProfile itself should be hashable
        _ = hash(NO_FILTER)
        _ = hash(AVOID_STRONG_BULL)


class TestNoFilter:

    def test_no_blocked_regimes(self):
        assert len(NO_FILTER.blocked_market_regimes) == 0

    def test_no_allowed_regimes(self):
        assert len(NO_FILTER.allowed_market_regimes) == 0

    def test_has_no_filter(self):
        assert NO_FILTER.has_any_filter is False

    def test_name_is_no_filter(self):
        assert NO_FILTER.name == "NO_FILTER"


class TestAvoidStrongBull:

    def test_strong_bull_is_blocked(self):
        assert MarketRegime.STRONG_BULL in AVOID_STRONG_BULL.blocked_market_regimes

    def test_bull_is_not_blocked(self):
        assert MarketRegime.BULL not in AVOID_STRONG_BULL.blocked_market_regimes

    def test_has_filter(self):
        assert AVOID_STRONG_BULL.has_any_filter is True


class TestAvoidExtremeVol:

    def test_extreme_vol_is_blocked(self):
        assert VolatilityRegime.EXTREME_VOL in AVOID_EXTREME_VOL.blocked_volatility_regimes

    def test_normal_vol_not_blocked(self):
        assert VolatilityRegime.NORMAL_VOL not in AVOID_EXTREME_VOL.blocked_volatility_regimes


class TestAvoidStrongBullAndExtremeVol:

    def test_blocks_both_dimensions(self):
        p = AVOID_STRONG_BULL_AND_EXTREME_VOL
        assert MarketRegime.STRONG_BULL in p.blocked_market_regimes
        assert VolatilityRegime.EXTREME_VOL in p.blocked_volatility_regimes


class TestBullOnly:

    def test_allowed_market_regimes_contains_only_bull(self):
        assert BULL_ONLY.allowed_market_regimes == frozenset({MarketRegime.BULL})

    def test_no_blocked_regimes_in_bull_only(self):
        assert len(BULL_ONLY.blocked_market_regimes) == 0

    def test_has_filter(self):
        assert BULL_ONLY.has_any_filter is True


class TestResearchProfiles:

    def test_no_filter_is_in_research_profiles(self):
        names = {p.name for p in RESEARCH_PROFILES}
        assert "NO_FILTER" in names

    def test_all_profiles_have_names(self):
        for p in RESEARCH_PROFILES:
            assert isinstance(p.name, str) and len(p.name) > 0

    def test_all_profiles_have_descriptions(self):
        for p in RESEARCH_PROFILES:
            assert isinstance(p.description, str) and len(p.description) > 0

    def test_names_are_unique(self):
        names = [p.name for p in RESEARCH_PROFILES]
        assert len(names) == len(set(names))
