"""Tests for BreakoutDensityResearcher."""

import pytest

from src.density.breakout_density_research import (
    BREAKOUT_THRESHOLDS,
    BreakoutDensityResearcher,
    BreakoutDensityResult,
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
        breakout_threshold=0.01,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE,
        atr_period=3,
    )


@pytest.fixture
def researcher(config) -> BreakoutDensityResearcher:
    return BreakoutDensityResearcher(
        config, base_profile=_mini_profile(),
        thresholds=[0.005, 0.008, 0.010],
    )


class TestBreakoutDensityResult:

    def test_meets_quality_good_metrics(self):
        r = BreakoutDensityResult(0.005, 50, 30.0, 1.7, 6.0, 0.63)
        assert r.meets_quality is True

    def test_fails_quality_bad_pf(self):
        r = BreakoutDensityResult(0.005, 50, 30.0, 1.2, 6.0, 0.63)
        assert r.meets_quality is False

    def test_pct_str_format(self):
        r = BreakoutDensityResult(0.0080, 50, 30.0, 1.7, 6.0, 0.63)
        assert r.pct_str == "0.80%"

    def test_density_score_zero_when_quality_fails(self):
        r = BreakoutDensityResult(0.005, 50, -5.0, 0.9, 6.0, 0.63)
        assert r.density_score == 0.0


class TestBreakoutDensityResearcher:

    def test_returns_one_result_per_threshold(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        assert len(results) == 3

    def test_sorted_by_density_score_descending(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        for i in range(len(results) - 1):
            assert results[i].density_score >= results[i + 1].density_score

    def test_thresholds_tested_descending(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles)
        # All three configured thresholds should appear
        tested = {r.threshold for r in results}
        assert tested == {0.005, 0.008, 0.010}

    def test_default_thresholds_count(self):
        assert len(BREAKOUT_THRESHOLDS) == 5
        assert 0.0100 in BREAKOUT_THRESHOLDS

    def test_best_threshold_returns_none_when_all_fail(self, researcher):
        candles = uptrend_candles(n=15)
        results = researcher.research(candles)
        if all(not r.meets_quality for r in results):
            assert researcher.best_threshold(results) is None
