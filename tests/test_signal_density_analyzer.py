"""Tests for SignalDensityAnalyzer — trade frequency metrics."""

import pytest
from datetime import datetime, timedelta, timezone

from src.backtest.models import BacktestResults
from src.density.signal_density_analyzer import SignalDensityAnalyzer, SignalDensityMetrics


def _results(
    symbol="SPY", trades=25, wins=15,
    exp=30.0, pf=1.7, dd=5.0,
) -> BacktestResults:
    return BacktestResults(
        symbol=symbol, timeframe="D1",
        starting_balance=10_000, ending_balance=10_750, net_profit=750,
        total_trades=trades, winning_trades=wins, losing_trades=trades - wins,
        win_rate=wins / trades, profit_factor=pf, expectancy=exp,
        max_drawdown=dd, sharpe_ratio=1.2,
        average_win=100.0, average_loss=60.0,
        largest_win=200.0, largest_loss=80.0,
        equity_curve=[], trades=[],
    )


@pytest.fixture
def analyzer() -> SignalDensityAnalyzer:
    return SignalDensityAnalyzer(min_annual_trades=10.0, bars_per_year=252)


class TestSignalDensityAnalyzer:

    def test_returns_metrics(self, analyzer):
        result = analyzer.analyze(_results(trades=25), candle_count=500)
        assert isinstance(result, SignalDensityMetrics)

    def test_symbol_propagated(self, analyzer):
        result = analyzer.analyze(_results(symbol="QQQ"), candle_count=500)
        assert result.symbol == "QQQ"

    def test_total_trades_propagated(self, analyzer):
        result = analyzer.analyze(_results(trades=25), candle_count=500)
        assert result.total_trades == 25

    def test_trading_years_calculation(self, analyzer):
        # 504 candles / 252 = 2.0 years
        result = analyzer.analyze(_results(), candle_count=504)
        assert result.trading_years == pytest.approx(2.0, rel=0.01)

    def test_trades_per_year_calculation(self, analyzer):
        # 25 trades / 2 years = 12.5 per year
        result = analyzer.analyze(_results(trades=25), candle_count=504)
        assert result.trades_per_year == pytest.approx(12.5, rel=0.01)

    def test_trades_per_month(self, analyzer):
        result = analyzer.analyze(_results(trades=24), candle_count=504)
        # 24 / 2 / 12 = 1.0 per month
        assert result.trades_per_month == pytest.approx(1.0, rel=0.01)

    def test_meets_density_goal_true(self, analyzer):
        # 25 trades / 2 years = 12.5/yr >= min_annual_trades=10
        result = analyzer.analyze(_results(trades=25), candle_count=504)
        assert result.meets_density_goal is True

    def test_meets_density_goal_false(self, analyzer):
        # 5 trades / 2 years = 2.5/yr < 10
        result = analyzer.analyze(_results(trades=5), candle_count=504)
        assert result.meets_density_goal is False

    def test_analyze_multi_returns_list(self, analyzer):
        results = {
            "SPY": _results("SPY", 25),
            "QQQ": _results("QQQ", 20),
        }
        counts = {"SPY": 500, "QQQ": 500}
        metrics = analyzer.analyze_multi(results, counts)
        assert len(metrics) == 2
