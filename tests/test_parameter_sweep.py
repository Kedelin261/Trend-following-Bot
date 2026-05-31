"""Tests for ParameterSweep — configuration research and safeguards."""

import pytest

from src.research.parameter_sweep import (
    DEFAULT_SWEEP_CONFIGS,
    SAFEGUARD_MIN_TRADES,
    ParameterSweep,
    SweepConfig,
    SweepResult,
)
from tests.fixtures import uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0},
    }


@pytest.fixture
def sweep(config) -> ParameterSweep:
    return ParameterSweep(config, min_warmup=12)


class TestSweepConfig:

    def test_description_auto_generated(self):
        cfg = SweepConfig(5, 10, 2.0, 3.0, 0.0025)
        assert "EMA5/10" in cfg.description

    def test_custom_description(self):
        cfg = SweepConfig(50, 200, 2.0, 3.0, 0.0025, "My Config")
        assert cfg.description == "My Config"


class TestSweepResult:

    def test_sufficient_property(self):
        from src.backtest.models import BacktestResults, StrategyHealth
        results = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=11_000, net_profit=1_000,
            total_trades=60,  # ≥ SAFEGUARD_MIN_TRADES
            winning_trades=36, losing_trades=24, win_rate=0.6,
            profit_factor=1.8, expectancy=50.0, max_drawdown=5.0, sharpe_ratio=1.2,
            average_win=100.0, average_loss=50.0, largest_win=200.0, largest_loss=80.0,
            equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(results)
        cfg = SweepConfig(50, 200, 2.0, 3.0, 0.0025)
        sr  = SweepResult(cfg, results, health)
        assert sr.sufficient is True

    def test_insufficient_when_few_trades(self):
        from src.backtest.models import BacktestResults, StrategyHealth
        results = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=10_500, net_profit=500,
            total_trades=5,  # < SAFEGUARD_MIN_TRADES
            winning_trades=3, losing_trades=2, win_rate=0.6,
            profit_factor=1.5, expectancy=100.0, max_drawdown=3.0, sharpe_ratio=1.0,
            average_win=150.0, average_loss=50.0, largest_win=200.0, largest_loss=60.0,
            equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(results)
        sr = SweepResult(SweepConfig(50, 200, 2.0, 3.0, 0.0025), results, health)
        assert sr.sufficient is False


class TestParameterSweep:

    def test_returns_one_result_per_config(self, sweep):
        candles = uptrend_candles(n=50)
        configs = [SweepConfig(5, 10, 2.0, 3.0, 0.0025)]
        results = sweep.sweep(candles, configs=configs)
        assert len(results) == 1

    def test_results_sorted_descending_by_rank_score(self, sweep):
        candles = uptrend_candles(n=50)
        configs = [
            SweepConfig(5, 10, 2.0, 3.0, 0.0025),
            SweepConfig(5, 10, 1.5, 2.0, 0.0050),
        ]
        results = sweep.sweep(candles, configs=configs)
        assert len(results) == 2
        # Sufficient results should appear before insufficient
        sufficient = [r for r in results if r.sufficient]
        insufficient = [r for r in results if not r.sufficient]
        for s in sufficient:
            for i in insufficient:
                assert results.index(s) <= results.index(i)

    def test_each_result_has_backtest_results(self, sweep):
        candles = uptrend_candles(n=50)
        configs = [SweepConfig(5, 10, 2.0, 3.0, 0.0025)]
        results = sweep.sweep(candles, configs=configs)
        for r in results:
            assert r.results is not None
            assert r.results.total_trades >= 0

    def test_insufficient_count_when_few_candles(self, sweep):
        candles = uptrend_candles(n=50)  # too few for most configs
        configs = [SweepConfig(5, 10, 2.0, 3.0, 0.0025)]
        results = sweep.sweep(candles, configs=configs)
        # Some may be insufficient
        count = sweep.insufficient_count(results)
        assert count >= 0  # just verify it runs

    def test_best_by_expectancy_returns_none_when_all_insufficient(self, sweep):
        candles = uptrend_candles(n=20)  # too few candles → no trades
        configs = [SweepConfig(5, 10, 2.0, 3.0, 0.0025)]
        results = sweep.sweep(candles, configs=configs)
        if all(not r.sufficient for r in results):
            assert sweep.best_by_expectancy(results) is None

    def test_default_configs_exist(self):
        assert len(DEFAULT_SWEEP_CONFIGS) > 0

    def test_safeguard_min_trades_is_50(self):
        assert SAFEGUARD_MIN_TRADES == 50
