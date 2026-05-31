"""Tests for TimeframeComparator — D1 vs H4/H1/W1."""

import pytest
from datetime import datetime, timezone

from src.backtest.models import BacktestResults, StrategyHealth
from src.timeframe.timeframe_backtester import TimeframeBacktestResult
from src.timeframe.timeframe_comparator import TimeframeComparator, TimeframeComparisonResult
from src.timeframe.timeframe_profile import D1_PROFILE, H4_PROFILE, H1_PROFILE


def _result(
    symbol="SPY", timeframe="D1",
    trades=25, wins=15,
    exp=30.0, pf=1.8, dd=5.0,
    tf_profile=None,
    meets_quality=True,
) -> TimeframeBacktestResult:
    bt = BacktestResults(
        symbol=symbol, timeframe=timeframe,
        starting_balance=10_000, ending_balance=10_750, net_profit=750,
        total_trades=trades, winning_trades=wins, losing_trades=trades - wins,
        win_rate=wins / trades, profit_factor=pf, expectancy=exp,
        max_drawdown=dd, sharpe_ratio=1.2,
        average_win=100.0, average_loss=60.0,
        largest_win=200.0, largest_loss=80.0,
        equity_curve=[], trades=[],
    )
    health = StrategyHealth.evaluate(bt)
    profile = tf_profile or D1_PROFILE
    return TimeframeBacktestResult(profile, symbol, bt, health, meets_quality)


@pytest.fixture
def comp() -> TimeframeComparator:
    return TimeframeComparator()


class TestTimeframeComparator:

    def test_returns_comparison_result(self, comp):
        baseline = _result(timeframe="D1", trades=25, tf_profile=D1_PROFILE)
        comparison = _result(timeframe="H4", trades=50, tf_profile=H4_PROFILE)
        result = comp.compare(baseline, comparison)
        assert isinstance(result, TimeframeComparisonResult)

    def test_trade_count_change(self, comp):
        baseline = _result(trades=25, tf_profile=D1_PROFILE)
        comparison = _result(trades=50, tf_profile=H4_PROFILE)
        result = comp.compare(baseline, comparison)
        assert result.trade_count_change == 25

    def test_pf_change_calculated(self, comp):
        baseline = _result(pf=1.8, tf_profile=D1_PROFILE)
        comparison = _result(pf=1.6, tf_profile=H4_PROFILE)
        result = comp.compare(baseline, comparison)
        assert result.pf_change == pytest.approx(-0.2, abs=0.01)

    def test_exp_change_calculated(self, comp):
        baseline = _result(exp=30.0, tf_profile=D1_PROFILE)
        comparison = _result(exp=20.0, tf_profile=H4_PROFILE)
        result = comp.compare(baseline, comparison)
        assert result.exp_change == pytest.approx(-10.0)

    def test_dd_change_positive_when_comparison_lower(self, comp):
        baseline = _result(dd=8.0, tf_profile=D1_PROFILE)
        comparison = _result(dd=5.0, tf_profile=H4_PROFILE)
        result = comp.compare(baseline, comparison)
        assert result.dd_change == pytest.approx(3.0)

    def test_recommendation_use_when_viable(self, comp):
        baseline = _result(trades=25, tf_profile=D1_PROFILE)
        comparison = _result(trades=50, tf_profile=H4_PROFILE, meets_quality=True)
        result = comp.compare(baseline, comparison)
        assert result.recommendation == "USE"

    def test_recommendation_reject_when_quality_fails(self, comp):
        baseline = _result(trades=25, tf_profile=D1_PROFILE)
        comparison = _result(trades=50, tf_profile=H4_PROFILE, meets_quality=False)
        result = comp.compare(baseline, comparison)
        assert result.recommendation == "REJECT"

    def test_is_viable_true_when_adds_trades_and_quality(self, comp):
        baseline = _result(trades=25, tf_profile=D1_PROFILE)
        comparison = _result(trades=60, tf_profile=H4_PROFILE, meets_quality=True)
        result = comp.compare(baseline, comparison)
        assert result.is_viable is True

    def test_compare_all_to_baseline(self, comp):
        baseline = _result(tf_profile=D1_PROFILE)
        others = [
            _result(timeframe="H4", tf_profile=H4_PROFILE),
            _result(timeframe="H1", tf_profile=H1_PROFILE),
        ]
        results = comp.compare_all_to_baseline(baseline, others)
        assert len(results) == 2

    def test_viable_timeframes_method(self, comp):
        baseline = _result(trades=25, tf_profile=D1_PROFILE)
        h4 = _result(trades=50, tf_profile=H4_PROFILE, meets_quality=True)
        h1 = _result(trades=30, tf_profile=H1_PROFILE, meets_quality=False)
        results = comp.compare_all_to_baseline(baseline, [h4, h1])
        viable = comp.viable_timeframes(results)
        assert "H4" in viable
        assert "H1" not in viable
