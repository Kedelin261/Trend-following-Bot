"""Tests for BreakoutQualityResearcher — threshold configuration and research."""

import pytest

from src.research.breakout_quality import (
    BREAKOUT_THRESHOLDS,
    SAFEGUARD_MIN_TRADES,
    BreakoutQualityResearcher,
    BreakoutResearchConfig,
)
from tests.fixtures import uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.5},
    }


@pytest.fixture
def researcher(config) -> BreakoutQualityResearcher:
    return BreakoutQualityResearcher(config, min_warmup=12)


class TestBreakoutResearchConfig:

    def test_description_auto_generated(self):
        cfg = BreakoutResearchConfig(breakout_threshold=0.0025)
        assert "0.25" in cfg.description

    def test_custom_description(self):
        cfg = BreakoutResearchConfig(0.005, "Custom desc")
        assert cfg.description == "Custom desc"


class TestBreakoutQualityResearcher:

    def test_returns_one_result_per_threshold(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles, thresholds=[0.0025, 0.0050])
        assert len(results) == 2

    def test_results_are_sorted_by_expectancy(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles, thresholds=[0.0025, 0.0050, 0.01])
        for i in range(len(results) - 1):
            assert results[i].results.expectancy >= results[i + 1].results.expectancy

    def test_insufficient_results_have_note(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles, thresholds=[0.0025])
        for r in results:
            if r.results.total_trades < SAFEGUARD_MIN_TRADES:
                assert r.note != ""
                assert "INSUFFICIENT" in r.note

    def test_best_threshold_returns_none_when_all_insufficient(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles, thresholds=[0.0025])
        best = researcher.best_threshold(results)
        # With small candle count, all results may be insufficient
        # best should be None if no sufficient results exist
        if all(not r.sufficient for r in results):
            assert best is None

    def test_each_result_has_health_object(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles, thresholds=[0.0025])
        for r in results:
            assert r.health is not None
            assert isinstance(r.health.passed, bool)

    def test_sufficient_property_matches_min_trades(self, researcher):
        candles = uptrend_candles(n=50)
        results = researcher.research(candles, thresholds=[0.0025])
        for r in results:
            assert r.sufficient == (r.results.total_trades >= SAFEGUARD_MIN_TRADES)
