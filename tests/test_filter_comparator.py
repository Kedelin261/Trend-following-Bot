"""Tests for FilterComparator — baseline vs filtered comparison."""

import pytest

from src.regime_filter.filter_backtester import FilterBacktestResult
from src.regime_filter.filter_comparator import FilterComparator, FilterComparisonResult
from src.regime_filter.filter_profiles import AVOID_STRONG_BULL, NO_FILTER


def _result(profile, trades=50, pf=1.5, exp=25.0, dd=7.0) -> FilterBacktestResult:
    return FilterBacktestResult(
        filter_profile=profile,
        total_trades_before=50, total_trades_after=trades,
        trades_removed=50 - trades, win_rate=0.6,
        profit_factor=pf, expectancy=exp, max_drawdown=dd, net_pnl=exp * trades,
    )


@pytest.fixture
def comp() -> FilterComparator:
    return FilterComparator()


class TestFilterComparator:

    def test_returns_comparison_result(self, comp):
        baseline = _result(NO_FILTER)
        filtered = _result(AVOID_STRONG_BULL, pf=1.7, exp=35.0)
        r = comp.compare(baseline, filtered)
        assert isinstance(r, FilterComparisonResult)

    def test_pf_improvement_positive_when_filtered_better(self, comp):
        baseline = _result(NO_FILTER, pf=1.3)
        filtered = _result(AVOID_STRONG_BULL, pf=1.7)
        r = comp.compare(baseline, filtered)
        assert r.pf_improvement > 0

    def test_exp_improvement_calculated(self, comp):
        baseline = _result(NO_FILTER, exp=15.0)
        filtered = _result(AVOID_STRONG_BULL, exp=30.0)
        r = comp.compare(baseline, filtered)
        assert r.exp_improvement == pytest.approx(15.0)

    def test_dd_reduction_positive_when_filtered_lower(self, comp):
        baseline = _result(NO_FILTER, dd=10.0)
        filtered = _result(AVOID_STRONG_BULL, dd=6.0)
        r = comp.compare(baseline, filtered)
        assert r.dd_reduction == pytest.approx(4.0)

    def test_trade_reduction_calculated(self, comp):
        baseline = _result(NO_FILTER, trades=50)
        filtered = _result(AVOID_STRONG_BULL, trades=45)
        r = comp.compare(baseline, filtered)
        assert r.trade_reduction == 5

    def test_is_improvement_true_when_both_better(self, comp):
        baseline = _result(NO_FILTER, pf=1.2, exp=5.0)
        filtered = _result(AVOID_STRONG_BULL, pf=1.6, exp=25.0)
        r = comp.compare(baseline, filtered)
        assert r.is_improvement is True

    def test_compare_all_excludes_baseline(self, comp):
        baseline = _result(NO_FILTER)
        others   = [_result(AVOID_STRONG_BULL, pf=1.7, exp=30.0)]
        results  = comp.compare_all(baseline, others)
        assert len(results) == 1

    def test_compare_all_sorted_by_expectancy_improvement(self, comp):
        baseline = _result(NO_FILTER, exp=10.0)
        others   = [
            _result(AVOID_STRONG_BULL, exp=20.0),
        ]
        results = comp.compare_all(baseline, others)
        assert results[0].exp_improvement == pytest.approx(10.0)

    def test_pf_direction_up(self, comp):
        baseline = _result(NO_FILTER, pf=1.3)
        filtered = _result(AVOID_STRONG_BULL, pf=1.7)
        r = comp.compare(baseline, filtered)
        assert r.pf_direction == "↑"

    def test_best_improvement_returns_none_when_none_improve(self, comp):
        baseline = _result(NO_FILTER, pf=1.8, exp=30.0)
        # Filtered is worse
        others = [_result(AVOID_STRONG_BULL, pf=0.8, exp=-10.0, trades=5)]
        results = comp.compare_all(baseline, others)
        assert comp.best_improvement(results) is None
