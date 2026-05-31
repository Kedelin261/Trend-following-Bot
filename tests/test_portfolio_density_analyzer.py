"""Tests for PortfolioDensityAnalyzer — portfolio-level opportunity analysis."""

import pytest

from src.backtest.models import BacktestResults
from src.density.portfolio_density_analyzer import (
    PORTFOLIO_MIN_ASSETS,
    PORTFOLIO_MIN_TRADES,
    PortfolioDensityAnalyzer,
    PortfolioDensityReport,
)


def _results(
    symbol="SPY", trades=40, wins=24,
    exp=30.0, pf=1.8, dd=6.0,
) -> BacktestResults:
    return BacktestResults(
        symbol=symbol, timeframe="D1",
        starting_balance=10_000, ending_balance=11_200, net_profit=1_200,
        total_trades=trades, winning_trades=wins, losing_trades=trades - wins,
        win_rate=wins / trades, profit_factor=pf, expectancy=exp,
        max_drawdown=dd, sharpe_ratio=1.2,
        average_win=100.0, average_loss=60.0,
        largest_win=200.0, largest_loss=80.0,
        equity_curve=[], trades=[],
    )


@pytest.fixture
def analyzer() -> PortfolioDensityAnalyzer:
    return PortfolioDensityAnalyzer(
        min_trades=100, min_pf=1.5, min_exp=0.0,
        max_dd=15.0, min_assets=2, bars_per_year=252,
    )


def _good_portfolio():
    return {
        "SPY": _results("SPY", trades=40, exp=30.0, pf=1.8, dd=5.0),
        "VOO": _results("VOO", trades=38, exp=25.0, pf=1.7, dd=4.5),
        "DIA": _results("DIA", trades=35, exp=22.0, pf=1.6, dd=6.0),
    }


def _candle_counts():
    return {"SPY": 504, "VOO": 504, "DIA": 504}


class TestPortfolioDensityAnalyzer:

    def test_returns_report(self, analyzer):
        report = analyzer.analyze(_good_portfolio(), _candle_counts())
        assert isinstance(report, PortfolioDensityReport)

    def test_total_trades_sum(self, analyzer):
        report = analyzer.analyze(_good_portfolio(), _candle_counts())
        assert report.total_trades == 40 + 38 + 35

    def test_meets_100_trades(self, analyzer):
        report = analyzer.analyze(_good_portfolio(), _candle_counts())
        assert report.meets_100_trades is True

    def test_not_meets_200_trades(self, analyzer):
        small = {"SPY": _results(trades=30), "VOO": _results(trades=25)}
        report = analyzer.analyze(small, {"SPY": 504, "VOO": 504})
        assert report.meets_200_trades is False

    def test_promoted_with_sufficient_portfolio(self, analyzer):
        report = analyzer.analyze(_good_portfolio(), _candle_counts())
        # 113 trades ≥ 100, good metrics → should be promoted
        assert report.promoted is True

    def test_not_promoted_insufficient_trades(self, analyzer):
        small = {"SPY": _results(trades=20), "VOO": _results(trades=15)}
        report = analyzer.analyze(small, {"SPY": 504, "VOO": 504})
        assert report.promoted is False

    def test_not_promoted_bad_quality(self, analyzer):
        bad = {
            "SPY": _results(trades=60, pf=0.8, exp=-5.0),
            "VOO": _results(trades=50, pf=0.9, exp=-3.0),
        }
        report = analyzer.analyze(bad, {"SPY": 504, "VOO": 504})
        assert report.promoted is False

    def test_empty_returns_report(self, analyzer):
        report = analyzer.analyze({}, {})
        assert isinstance(report, PortfolioDensityReport)
        assert report.promoted is False

    def test_trades_per_year_calculated(self, analyzer):
        report = analyzer.analyze(_good_portfolio(), _candle_counts())
        # ~113 trades / 2 years ≈ 56.5 per year
        assert report.trades_per_year > 0

    def test_promotion_constants(self):
        assert PORTFOLIO_MIN_TRADES == 100
        assert PORTFOLIO_MIN_ASSETS == 2

    def test_promotion_reason_populated(self, analyzer):
        report = analyzer.analyze(_good_portfolio(), _candle_counts())
        assert isinstance(report.promotion_reason, str)
        assert len(report.promotion_reason) > 0
