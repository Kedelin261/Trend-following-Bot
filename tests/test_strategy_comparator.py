"""Tests for StrategyComparator — V1 vs V2 comparison logic."""

import math
import pytest
from datetime import datetime, timezone

from src.backtest.models import BacktestResults
from src.refinement.strategy_comparator import (
    PROMOTION_MIN_PF,
    PROMOTION_MIN_TRADES,
    ComparisonResult,
    StrategyComparator,
)
from src.refinement.strategy_v2 import V1_PROFILE, V2_PROFILE


def _results(
    symbol="SPY",
    trades=40, wins=24,
    expectancy=50.0, pf=1.8, dd=7.0,
    start_bal=10_000, net=2_000,
) -> BacktestResults:
    return BacktestResults(
        symbol=symbol, timeframe="D1",
        starting_balance=start_bal, ending_balance=start_bal + net,
        net_profit=net, total_trades=trades, winning_trades=wins,
        losing_trades=trades - wins, win_rate=wins / trades,
        profit_factor=pf, expectancy=expectancy, max_drawdown=dd,
        sharpe_ratio=1.2, average_win=150.0, average_loss=80.0,
        largest_win=300.0, largest_loss=120.0, equity_curve=[], trades=[],
    )


@pytest.fixture
def comparator() -> StrategyComparator:
    return StrategyComparator()


class TestStrategyComparator:

    def test_returns_comparison_result(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(pf=1.2, expectancy=10.0),
                               _results(pf=1.8, expectancy=50.0))
        assert isinstance(r, ComparisonResult)

    def test_pf_change_calculated(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(pf=1.2), _results(pf=1.8))
        assert r.pf_change == pytest.approx(0.6, rel=1e-4)

    def test_expectancy_change_calculated(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(expectancy=20.0), _results(expectancy=60.0))
        assert r.expectancy_change == pytest.approx(40.0)

    def test_drawdown_change_positive_when_v2_lower(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(dd=10.0), _results(dd=6.0))
        assert r.drawdown_change == pytest.approx(4.0)

    def test_trade_count_change(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(trades=40), _results(trades=25))
        assert r.trade_count_change == -15

    def test_promote_true_when_all_criteria_met(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(pf=1.2, expectancy=10.0, dd=10.0),
                               _results(pf=1.8, expectancy=50.0, dd=7.0, trades=35))
        assert r.promote_v2 is True
        assert r.rejection_reason is None

    def test_promote_false_when_insufficient_trades(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(), _results(trades=5))  # only 5 V2 trades
        assert r.promote_v2 is False
        assert "insufficient" in r.rejection_reason.lower()

    def test_promote_false_when_pf_too_low(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(), _results(pf=1.2, trades=35))
        assert r.promote_v2 is False
        assert "profit factor" in r.rejection_reason.lower()

    def test_promote_false_when_negative_expectancy(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(), _results(expectancy=-10.0, trades=35))
        assert r.promote_v2 is False

    def test_promote_false_when_drawdown_too_high(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(), _results(dd=20.0, pf=2.0, expectancy=50.0, trades=35))
        assert r.promote_v2 is False
        assert "drawdown" in r.rejection_reason.lower()

    def test_compare_multi_true_when_all_promoted(self, comparator):
        good_r = _results(pf=1.8, expectancy=50.0, dd=7.0, trades=35)
        c1 = comparator.compare("SPY", V1_PROFILE, V2_PROFILE, _results(), good_r)
        c2 = comparator.compare("DIA", V1_PROFILE, V2_PROFILE, _results(), good_r)
        assert comparator.compare_multi([c1, c2]) is True

    def test_compare_multi_false_when_one_fails(self, comparator):
        good_r = _results(pf=1.8, expectancy=50.0, dd=7.0, trades=35)
        bad_r  = _results(pf=0.8, expectancy=-10.0, trades=35)
        c1 = comparator.compare("SPY", V1_PROFILE, V2_PROFILE, _results(), good_r)
        c2 = comparator.compare("QQQ", V1_PROFILE, V2_PROFILE, _results(), bad_r)
        assert comparator.compare_multi([c1, c2]) is False


class TestComparisonResultProperties:

    def test_v2_sufficient_property(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(), _results(trades=5))
        assert r.v2_sufficient is False

    def test_v2_improved_pf_true(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(pf=1.2), _results(pf=1.8))
        assert r.v2_improved_pf is True

    def test_v2_improved_expectancy_true(self, comparator):
        r = comparator.compare("SPY", V1_PROFILE, V2_PROFILE,
                               _results(expectancy=10.0), _results(expectancy=50.0))
        assert r.v2_improved_expectancy is True
