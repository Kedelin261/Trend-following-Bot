"""Tests for PromotionReadinessAnalyzer — six-criteria gate."""

import pytest

from src.regime_filter.filter_backtester import FilterBacktestResult
from src.regime_filter.filter_profiles import AVOID_STRONG_BULL, NO_FILTER
from src.regime_filter.promotion_readiness_analyzer import (
    PROMO_MAX_DD,
    PROMO_MIN_EXP,
    PROMO_MIN_PF,
    PROMO_MIN_TRADES,
    PromotionReadiness,
    PromotionReadinessAnalyzer,
)


def _result(trades=110, pf=1.55, exp=25.0, dd=6.0, profile=None) -> FilterBacktestResult:
    profile = profile or AVOID_STRONG_BULL
    return FilterBacktestResult(
        filter_profile=profile,
        total_trades_before=130, total_trades_after=trades,
        trades_removed=130 - trades, win_rate=0.62,
        profit_factor=pf, expectancy=exp, max_drawdown=dd, net_pnl=exp * trades,
    )


@pytest.fixture
def analyzer() -> PromotionReadinessAnalyzer:
    return PromotionReadinessAnalyzer()


class TestPromotionReadinessAnalyzer:

    def test_all_criteria_pass(self, analyzer):
        r = analyzer.analyze(_result(), "ROBUST")
        assert r.all_passed is True

    def test_status_ready_when_all_pass(self, analyzer):
        r = analyzer.analyze(_result(), "ROBUST")
        assert r.status == "PROMOTION READY"

    def test_status_not_ready_when_fails(self, analyzer):
        r = analyzer.analyze(_result(trades=50), "ROBUST")
        assert r.status == "NOT READY"

    def test_fail_on_insufficient_trades(self, analyzer):
        r = analyzer.analyze(_result(trades=50), "ROBUST")
        assert r.trades_ok is False
        assert r.all_passed is False

    def test_fail_on_low_pf(self, analyzer):
        r = analyzer.analyze(_result(pf=1.2), "ROBUST")
        assert r.pf_ok is False

    def test_fail_on_negative_expectancy(self, analyzer):
        r = analyzer.analyze(_result(exp=-5.0), "ROBUST")
        assert r.expectancy_ok is False

    def test_fail_on_high_drawdown(self, analyzer):
        r = analyzer.analyze(_result(dd=16.0), "ROBUST")
        assert r.drawdown_ok is False

    def test_fail_on_unstable_robustness(self, analyzer):
        r = analyzer.analyze(_result(), "UNSTABLE")
        assert r.robustness_ok is False
        assert r.all_passed is False

    def test_marginal_robustness_passes(self, analyzer):
        r = analyzer.analyze(_result(), "MARGINAL")
        assert r.robustness_ok is True

    def test_at_exact_thresholds_passes(self, analyzer):
        r = analyzer.analyze(
            _result(trades=100, pf=1.50, exp=0.01, dd=14.9), "MARGINAL"
        )
        assert r.all_passed is True

    def test_pass_reasons_populated(self, analyzer):
        r = analyzer.analyze(_result(), "ROBUST")
        assert len(r.pass_reasons) == 5

    def test_failure_reasons_populated_when_fail(self, analyzer):
        r = analyzer.analyze(_result(trades=50, pf=0.8, exp=-5.0, dd=20.0), "UNSTABLE")
        assert len(r.failure_reasons) == 5

    def test_find_best_ready_profile_returns_highest_exp(self, analyzer):
        r1 = analyzer.analyze(_result(exp=20.0), "ROBUST")
        r2 = analyzer.analyze(_result(exp=35.0), "ROBUST")
        r3 = analyzer.analyze(_result(trades=5), "ROBUST")  # fails
        best = analyzer.find_best_ready_profile([r1, r2, r3])
        assert best is not None
        assert best.actual_expectancy == pytest.approx(35.0)

    def test_promotion_constants(self):
        assert PROMO_MIN_TRADES == 100
        assert PROMO_MIN_PF == 1.50
        assert PROMO_MIN_EXP == 0.0
        assert PROMO_MAX_DD == 15.0
