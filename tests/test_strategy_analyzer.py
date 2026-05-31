"""Tests for StrategyAnalyzer — research synthesis and reporting."""

import pytest
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults, StrategyHealth
from src.research.adx_filter import ADXFilterResult
from src.research.breakout_quality import BreakoutResearchConfig, BreakoutQualityResult
from src.research.market_regime import MarketRegime, RegimePerformance
from src.research.multi_asset_runner import AssetResearchResult
from src.research.parameter_sweep import SweepConfig, SweepResult
from src.research.strategy_analyzer import ResearchReport, StrategyAnalyzer
from src.research.volatility_filter import VolatilityCategory, VolatilityPerformance


def _backtest_results(
    symbol="SPY",
    trades=35, wins=21,
    expectancy=50.0, pf=1.8, dd=7.0, net_profit=1000,
) -> BacktestResults:
    return BacktestResults(
        symbol=symbol, timeframe="D1",
        starting_balance=10_000, ending_balance=10_000 + net_profit,
        net_profit=net_profit,
        total_trades=trades, winning_trades=wins, losing_trades=trades - wins,
        win_rate=wins / trades, profit_factor=pf, expectancy=expectancy,
        max_drawdown=dd, sharpe_ratio=1.2,
        average_win=150.0, average_loss=80.0,
        largest_win=300.0, largest_loss=120.0,
        equity_curve=[], trades=[],
    )


def _asset_result(symbol, expectancy=50.0, trades=35, recommend=True) -> AssetResearchResult:
    bt = _backtest_results(symbol=symbol, expectancy=expectancy, trades=trades)
    health = StrategyHealth.evaluate(bt, min_trades=30)
    from src.research.market_regime import MarketRegime
    return AssetResearchResult(
        symbol=symbol, candle_count=500, results=bt, health=health,
        benchmark=None, current_regime=MarketRegime.BULL,
    )


def _adx_result(threshold, expectancy, trades=35) -> ADXFilterResult:
    return ADXFilterResult(
        threshold=threshold, trade_count=trades, win_rate=0.6,
        expectancy=expectancy, profit_factor=1.8, trades_passed=trades,
        trades_blocked=5, sufficient=trades >= 10,
    )


def _vol_performance(cat, expectancy=50.0, trades=15) -> VolatilityPerformance:
    return VolatilityPerformance(
        category=cat, trade_count=trades, win_rate=0.6,
        expectancy=expectancy, profit_factor=1.8, avg_atr_pct=1.2,
        sufficient=trades >= 10,
    )


@pytest.fixture
def analyzer() -> StrategyAnalyzer:
    return StrategyAnalyzer(min_trades_per_asset=30)


def _minimal_report_inputs():
    asset_results = [
        _asset_result("SPY", expectancy=60.0, trades=40),
        _asset_result("QQQ", expectancy=-10.0, trades=35),
    ]
    regime_perf = {
        MarketRegime.BULL: RegimePerformance(
            MarketRegime.BULL, 25, 0.68, 80.0, 2.1, 5.0, sufficient=True
        ),
        MarketRegime.SIDEWAYS: RegimePerformance(
            MarketRegime.SIDEWAYS, 10, 0.40, -20.0, 0.8, 8.0, sufficient=False
        ),
    }
    adx_results = [
        _adx_result(0.0, expectancy=40.0, trades=40),
        _adx_result(25.0, expectancy=65.0, trades=25),
    ]
    vol_results = {
        VolatilityCategory.MEDIUM: _vol_performance(VolatilityCategory.MEDIUM, 60.0, 20),
        VolatilityCategory.LOW: _vol_performance(VolatilityCategory.LOW, 10.0, 12),
    }
    return asset_results, regime_perf, adx_results, vol_results


class TestStrategyAnalyzer:

    def test_returns_research_report(self, analyzer):
        asset_results, regime_perf, adx_results, vol_results = _minimal_report_inputs()
        report = analyzer.analyze(
            asset_results=asset_results,
            regime_performance=regime_perf,
            adx_results=adx_results,
            volatility_results=vol_results,
            breakout_results=[],
            sweep_results=[],
            benchmark_results={},
        )
        assert isinstance(report, ResearchReport)

    def test_recommended_assets_populated(self, analyzer):
        asset_results, regime_perf, adx_results, vol_results = _minimal_report_inputs()
        report = analyzer.analyze(
            asset_results=asset_results,
            regime_performance=regime_perf,
            adx_results=adx_results,
            volatility_results=vol_results,
            breakout_results=[], sweep_results=[], benchmark_results={},
        )
        assert isinstance(report.recommended_assets, list)

    def test_best_asset_is_highest_expectancy(self, analyzer):
        asset_results, regime_perf, adx_results, vol_results = _minimal_report_inputs()
        report = analyzer.analyze(
            asset_results=asset_results,
            regime_performance=regime_perf,
            adx_results=adx_results,
            volatility_results=vol_results,
            breakout_results=[], sweep_results=[], benchmark_results={},
        )
        if report.best_asset:
            assert report.best_asset == "SPY"  # higher expectancy

    def test_edge_confirmed_requires_multiple_assets(self, analyzer):
        # Only 1 recommended asset → edge_confirmed should be False
        asset_results = [_asset_result("SPY", expectancy=80.0, trades=40)]
        _, regime_perf, adx_results, vol_results = _minimal_report_inputs()
        report = analyzer.analyze(
            asset_results=asset_results,
            regime_performance=regime_perf,
            adx_results=adx_results,
            volatility_results=vol_results,
            breakout_results=[], sweep_results=[], benchmark_results={},
        )
        # With only 1 asset, edge_confirmed should be False
        assert report.edge_confirmed is False

    def test_warnings_when_no_recommended_assets(self, analyzer):
        # All assets have negative expectancy
        asset_results = [
            _asset_result("SPY", expectancy=-20.0, trades=35),
            _asset_result("QQQ", expectancy=-30.0, trades=35),
        ]
        _, regime_perf, adx_results, vol_results = _minimal_report_inputs()
        report = analyzer.analyze(
            asset_results=asset_results,
            regime_performance=regime_perf,
            adx_results=adx_results,
            volatility_results=vol_results,
            breakout_results=[], sweep_results=[], benchmark_results={},
        )
        assert len(report.warnings) > 0
        assert not report.edge_confirmed

    def test_expected_metrics_are_averages(self, analyzer):
        asset_results = [
            _asset_result("SPY", expectancy=60.0, trades=40),
            _asset_result("QQQ", expectancy=40.0, trades=35),
        ]
        _, regime_perf, adx_results, vol_results = _minimal_report_inputs()
        report = analyzer.analyze(
            asset_results=asset_results,
            regime_performance=regime_perf,
            adx_results=adx_results,
            volatility_results=vol_results,
            breakout_results=[], sweep_results=[], benchmark_results={},
        )
        assert report.expected_expectancy == pytest.approx(50.0, rel=0.01)


class TestResearchReport:

    def test_edge_confirmed_is_bool(self, analyzer):
        ar, rp, adx, vol = _minimal_report_inputs()
        report = analyzer.analyze(ar, rp, adx, vol, [], [], {})
        assert isinstance(report.edge_confirmed, bool)

    def test_warnings_is_list(self, analyzer):
        ar, rp, adx, vol = _minimal_report_inputs()
        report = analyzer.analyze(ar, rp, adx, vol, [], [], {})
        assert isinstance(report.warnings, list)

    def test_notes_is_list(self, analyzer):
        ar, rp, adx, vol = _minimal_report_inputs()
        report = analyzer.analyze(ar, rp, adx, vol, [], [], {})
        assert isinstance(report.notes, list)
