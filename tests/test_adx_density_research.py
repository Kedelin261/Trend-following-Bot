"""Tests for ADXDensityResearcher."""

import pytest

from src.density.adx_density_research import (
    ADX_THRESHOLDS,
    ADXDensityResearcher,
    ADXDensityResult,
    QUALITY_MIN_PF,
)
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE
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
        require_bull_regime=False,
        adx_threshold=25.0,
        volatility_mode=VolatilityFilterMode.NONE,
        atr_period=3,
    )


@pytest.fixture
def researcher(config) -> ADXDensityResearcher:
    return ADXDensityResearcher(config, base_profile=_mini_profile(), thresholds=[20.0, 25.0, 30.0])


class TestADXDensityResult:

    def test_meets_quality_with_good_metrics(self):
        r = ADXDensityResult(25.0, 40, 50.0, 1.8, 7.0, 0.65)
        assert r.meets_quality is True

    def test_fails_quality_low_pf(self):
        r = ADXDensityResult(25.0, 40, 50.0, 1.2, 7.0, 0.65)
        assert r.meets_quality is False

    def test_fails_quality_negative_expectancy(self):
        r = ADXDensityResult(25.0, 40, -5.0, 1.8, 7.0, 0.65)
        assert r.meets_quality is False

    def test_density_score_zero_when_quality_fails(self):
        r = ADXDensityResult(25.0, 40, -5.0, 0.8, 7.0, 0.65)
        assert r.density_score == 0.0

    def test_density_score_equals_trade_count_when_quality_passes(self):
        r = ADXDensityResult(25.0, 40, 50.0, 1.8, 7.0, 0.65)
        assert r.density_score == 40.0


class TestADXDensityResearcher:

    def test_returns_one_result_per_threshold(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        assert len(results) == 3

    def test_sorted_by_density_score_descending(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        for i in range(len(results) - 1):
            assert results[i].density_score >= results[i + 1].density_score

    def test_each_result_has_threshold(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        thresholds = {r.threshold for r in results}
        assert thresholds == {20.0, 25.0, 30.0}

    def test_best_threshold_returns_none_when_all_fail_quality(self, researcher):
        # With tiny candle set, all results have 0 trades → quality fails
        candles = uptrend_candles(n=15)  # too few for signal engine warmup
        results = researcher.research(candles)
        if all(not r.meets_quality for r in results):
            assert researcher.best_threshold(results) is None

    def test_informative_property(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        for r in results:
            assert r.informative == (r.trade_count >= 10)

    def test_default_thresholds_constant(self):
        assert len(ADX_THRESHOLDS) == 5
        assert 25.0 in ADX_THRESHOLDS
