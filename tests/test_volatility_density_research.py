"""Tests for VolatilityDensityResearcher."""

import pytest

from src.density.volatility_density_research import (
    RESEARCH_MODES,
    VolatilityDensityResearcher,
    VolatilityDensityResult,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from tests.fixtures import uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


def _mini_profile() -> StrategyProfile:
    return StrategyProfile(
        name="mini", description="mini",
        ema_fast=5, ema_slow=10,
        breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.MEDIUM_ONLY,
        atr_period=3,
    )


@pytest.fixture
def researcher(config) -> VolatilityDensityResearcher:
    return VolatilityDensityResearcher(
        config, base_profile=_mini_profile(),
        modes=[VolatilityFilterMode.MEDIUM_ONLY, VolatilityFilterMode.NONE],
    )


class TestVolatilityDensityResult:

    def test_meets_quality(self):
        r = VolatilityDensityResult(VolatilityFilterMode.NONE, 50, 30.0, 1.6, 5.0, 0.62)
        assert r.meets_quality is True

    def test_fails_quality_low_pf(self):
        r = VolatilityDensityResult(VolatilityFilterMode.NONE, 50, 30.0, 1.2, 5.0, 0.62)
        assert r.meets_quality is False

    def test_density_score_zero_when_quality_fails(self):
        r = VolatilityDensityResult(VolatilityFilterMode.NONE, 50, -5.0, 0.8, 5.0, 0.62)
        assert r.density_score == 0.0

    def test_density_score_equals_trade_count_when_passes(self):
        r = VolatilityDensityResult(VolatilityFilterMode.NONE, 50, 30.0, 1.6, 5.0, 0.62)
        assert r.density_score == 50.0


class TestVolatilityDensityResearcher:

    def test_returns_one_result_per_mode(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        assert len(results) == 2

    def test_all_four_research_modes_defined(self):
        assert VolatilityFilterMode.MEDIUM_ONLY in RESEARCH_MODES
        assert VolatilityFilterMode.LOW_AND_MEDIUM in RESEARCH_MODES
        assert VolatilityFilterMode.MEDIUM_AND_HIGH in RESEARCH_MODES
        assert VolatilityFilterMode.NONE in RESEARCH_MODES

    def test_sorted_by_density_score_descending(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        for i in range(len(results) - 1):
            assert results[i].density_score >= results[i + 1].density_score

    def test_best_mode_returns_none_when_all_fail(self, researcher):
        candles = uptrend_candles(n=15)  # too few → 0 trades → quality fails
        results = researcher.research(candles)
        if all(not r.meets_quality for r in results):
            assert researcher.best_mode(results) is None

    def test_medium_and_high_mode_exists_in_enum(self):
        assert hasattr(VolatilityFilterMode, "MEDIUM_AND_HIGH")
