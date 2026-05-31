"""Tests for OpportunityAnalyzer — trade frequency metrics."""

import pytest

from src.backtest.models import BacktestResults
from src.timeframe.opportunity_analyzer import (
    OpportunityAnalyzer,
    OpportunityMetrics,
    PortfolioOpportunityMetrics,
)
from src.timeframe.timeframe_profile import D1_PROFILE, H4_PROFILE


def _results(symbol="SPY", trades=25, exp=30.0, pf=1.8, dd=5.0) -> BacktestResults:
    return BacktestResults(
        symbol=symbol, timeframe="D1",
        starting_balance=10_000, ending_balance=10_750, net_profit=750,
        total_trades=trades, winning_trades=int(trades * 0.6),
        losing_trades=int(trades * 0.4), win_rate=0.6,
        profit_factor=pf, expectancy=exp, max_drawdown=dd, sharpe_ratio=1.2,
        average_win=100.0, average_loss=60.0,
        largest_win=200.0, largest_loss=80.0,
        equity_curve=[], trades=[],
    )


@pytest.fixture
def analyzer() -> OpportunityAnalyzer:
    return OpportunityAnalyzer(min_annual_trades=10.0)


class TestOpportunityAnalyzer:

    def test_returns_metrics(self, analyzer):
        result = analyzer.analyze(_results(), D1_PROFILE, candle_count=504)
        assert isinstance(result, OpportunityMetrics)

    def test_symbol_propagated(self, analyzer):
        result = analyzer.analyze(_results(symbol="VOO"), D1_PROFILE, candle_count=252)
        assert result.symbol == "VOO"

    def test_trading_years_d1(self, analyzer):
        # 504 bars / 252 per year = 2.0 years
        result = analyzer.analyze(_results(), D1_PROFILE, candle_count=504)
        assert result.trading_years == pytest.approx(2.0, rel=0.01)

    def test_trades_per_year_d1(self, analyzer):
        # 25 trades / 2 years = 12.5 per year
        result = analyzer.analyze(_results(trades=25), D1_PROFILE, candle_count=504)
        assert result.trades_per_year == pytest.approx(12.5, rel=0.01)

    def test_trades_per_month(self, analyzer):
        result = analyzer.analyze(_results(trades=24), D1_PROFILE, candle_count=504)
        # 24 / 2 / 12 = 1.0 per month
        assert result.trades_per_month == pytest.approx(1.0, rel=0.01)

    def test_meets_density_goal(self, analyzer):
        result = analyzer.analyze(_results(trades=25), D1_PROFILE, candle_count=504)
        assert result.meets_density_goal is True

    def test_h4_bars_per_year_higher(self, analyzer):
        # H4 has 409 bars/year vs D1's 252
        # Same 1000 bars → fewer years for H4 than D1
        d1_result  = analyzer.analyze(_results(trades=20), D1_PROFILE, 1000)
        h4_result  = analyzer.analyze(_results(trades=20), H4_PROFILE, 1000)
        # H4 represents fewer years → higher trades/year for same trade count
        assert h4_result.trades_per_year >= d1_result.trades_per_year

    def test_analyze_portfolio_returns_portfolio_metrics(self, analyzer):
        results_map = {
            "SPY": _results("SPY", 25),
            "VOO": _results("VOO", 20),
        }
        counts = {"SPY": 504, "VOO": 504}
        portfolio = analyzer.analyze_portfolio(results_map, D1_PROFILE, counts)
        assert isinstance(portfolio, PortfolioOpportunityMetrics)
        assert portfolio.total_trades == 45

    def test_meets_100_target(self, analyzer):
        results_map = {"SPY": _results(trades=60), "VOO": _results(trades=50)}
        counts = {"SPY": 504, "VOO": 504}
        portfolio = analyzer.analyze_portfolio(results_map, D1_PROFILE, counts)
        assert portfolio.meets_100_target is True

    def test_not_meets_200_target(self, analyzer):
        results_map = {"SPY": _results(trades=25), "VOO": _results(trades=20)}
        counts = {"SPY": 504, "VOO": 504}
        portfolio = analyzer.analyze_portfolio(results_map, D1_PROFILE, counts)
        assert portfolio.meets_200_target is False
