"""Tests for MultiAssetRunner — multi-asset research."""

import pytest
from typing import Dict, List

from src.data.models import Candle
from src.research.multi_asset_runner import AssetResearchResult, MultiAssetRunner
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0, "minimum_trades_required": 5},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


@pytest.fixture
def runner(config) -> MultiAssetRunner:
    return MultiAssetRunner(config, min_warmup=12, run_benchmark=True)


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=50, symbol="SPY"),
        "QQQ": downtrend_candles(n=50, symbol="QQQ"),
        "VOO": flat_candles(n=50, symbol="VOO"),
    }


class TestMultiAssetRunner:

    def test_returns_one_result_per_asset(self, runner):
        results = runner.run(_asset_candles())
        assert len(results) == 3

    def test_symbols_match_input(self, runner):
        results = runner.run(_asset_candles())
        symbols = {r.symbol for r in results}
        assert symbols == {"SPY", "QQQ", "VOO"}

    def test_results_sorted_by_expectancy_descending(self, runner):
        results = runner.run(_asset_candles())
        for i in range(len(results) - 1):
            assert results[i].expectancy >= results[i + 1].expectancy

    def test_candle_count_recorded(self, runner):
        results = runner.run(_asset_candles())
        for r in results:
            assert r.candle_count == 50

    def test_health_object_present(self, runner):
        results = runner.run(_asset_candles())
        for r in results:
            assert r.health is not None

    def test_benchmark_present_when_enabled(self, runner):
        results = runner.run(_asset_candles())
        for r in results:
            # benchmark may be None for empty candles but should exist otherwise
            assert r.benchmark is not None

    def test_current_regime_set(self, runner):
        results = runner.run(_asset_candles())
        from src.research.market_regime import MarketRegime
        for r in results:
            assert r.current_regime in list(MarketRegime)

    def test_empty_candle_dict_skipped(self, runner):
        asset_candles = {"SPY": uptrend_candles(n=50), "QQQ": []}
        results = runner.run(asset_candles)
        symbols = {r.symbol for r in results}
        assert "SPY" in symbols
        assert "QQQ" not in symbols

    def test_empty_input_returns_empty(self, runner):
        results = runner.run({})
        assert results == []

    def test_recommended_assets_method(self, runner):
        results = runner.run(_asset_candles())
        recommended = runner.recommended_assets(results)
        assert isinstance(recommended, list)

    def test_best_asset_is_highest_expectancy(self, runner):
        results = runner.run(_asset_candles())
        best = runner.best_asset(results)
        sufficient = [r for r in results if r.sufficient]
        if best and sufficient:
            assert best.expectancy == max(r.expectancy for r in sufficient)

    def test_worst_asset_is_lowest_expectancy(self, runner):
        results = runner.run(_asset_candles())
        worst = runner.worst_asset(results)
        sufficient = [r for r in results if r.sufficient]
        if worst and sufficient:
            assert worst.expectancy == min(r.expectancy for r in sufficient)
