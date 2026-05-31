"""Tests for BreakoutQualityFilter — V2 threshold research."""

import pytest

from src.refinement.breakout_quality_filter import (
    SAFEGUARD_MIN_TRADES,
    BreakoutQualityFilter,
    BreakoutResearchResult,
)
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


@pytest.fixture
def bqf(config) -> BreakoutQualityFilter:
    return BreakoutQualityFilter(config, ema_fast=5, ema_slow=10, min_warmup=12)


class TestBreakoutQualityFilter:

    def test_returns_one_result_per_threshold(self, bqf):
        candles = uptrend_candles(n=50)
        results = bqf.research(candles, thresholds=[0.01, 0.0125])
        assert len(results) == 2

    def test_sorted_by_expectancy_descending(self, bqf):
        candles = uptrend_candles(n=50)
        results = bqf.research(candles, thresholds=[0.01, 0.0125, 0.015])
        for i in range(len(results) - 1):
            assert results[i].results.expectancy >= results[i + 1].results.expectancy

    def test_insufficient_results_have_note(self, bqf):
        candles = uptrend_candles(n=50)
        results = bqf.research(candles, thresholds=[0.01])
        for r in results:
            if r.results.total_trades < SAFEGUARD_MIN_TRADES:
                assert r.note != ""

    def test_sufficient_property(self, bqf):
        candles = uptrend_candles(n=50)
        results = bqf.research(candles, thresholds=[0.01])
        for r in results:
            assert r.sufficient == (r.results.total_trades >= SAFEGUARD_MIN_TRADES)

    def test_best_threshold_none_when_all_insufficient(self, bqf):
        candles = uptrend_candles(n=50)
        results = bqf.research(candles, thresholds=[0.05])  # very high threshold
        if all(not r.sufficient for r in results):
            assert bqf.best_threshold(results) is None

    def test_health_object_present(self, bqf):
        candles = uptrend_candles(n=50)
        results = bqf.research(candles, thresholds=[0.01])
        for r in results:
            assert r.health is not None
            assert isinstance(r.health.passed, bool)

    def test_safeguard_min_is_30(self):
        assert SAFEGUARD_MIN_TRADES == 30
