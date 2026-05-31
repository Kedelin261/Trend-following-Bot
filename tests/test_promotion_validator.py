"""Tests for PromotionValidator — five-criteria gate."""

import pytest

from src.promotion.promotion_validator import (
    PROMOTION_MAX_DD,
    PROMOTION_MIN_ASSETS,
    PROMOTION_MIN_EXP,
    PROMOTION_MIN_PF,
    PROMOTION_MIN_TRADES,
    PromotionValidationResult,
    PromotionValidator,
)


@pytest.fixture
def validator() -> PromotionValidator:
    return PromotionValidator()


def _all_passing_kwargs():
    return dict(
        total_trades=105, profit_factor=1.65,
        expectancy=28.0, max_drawdown=6.5, assets_passing=3,
    )


class TestPromotionValidator:

    def test_all_criteria_pass(self, validator):
        result = validator.validate(**_all_passing_kwargs())
        assert result.all_passed is True

    def test_status_label_promote(self, validator):
        result = validator.validate(**_all_passing_kwargs())
        assert result.status_label == "PROMOTE_TO_PHASE_5"

    def test_fail_trades(self, validator):
        kw = {**_all_passing_kwargs(), "total_trades": 99}
        result = validator.validate(**kw)
        assert result.all_passed is False
        assert result.trades_ok is False

    def test_fail_pf(self, validator):
        kw = {**_all_passing_kwargs(), "profit_factor": 1.2}
        result = validator.validate(**kw)
        assert result.all_passed is False
        assert result.pf_ok is False

    def test_fail_expectancy(self, validator):
        kw = {**_all_passing_kwargs(), "expectancy": -5.0}
        result = validator.validate(**kw)
        assert result.all_passed is False
        assert result.expectancy_ok is False

    def test_fail_drawdown(self, validator):
        kw = {**_all_passing_kwargs(), "max_drawdown": 16.0}
        result = validator.validate(**kw)
        assert result.all_passed is False
        assert result.drawdown_ok is False

    def test_fail_assets(self, validator):
        kw = {**_all_passing_kwargs(), "assets_passing": 1}
        result = validator.validate(**kw)
        assert result.all_passed is False
        assert result.assets_ok is False

    def test_exactly_at_thresholds_passes(self, validator):
        result = validator.validate(
            total_trades=100, profit_factor=1.50,
            expectancy=0.01, max_drawdown=14.9, assets_passing=2,
        )
        assert result.all_passed is True

    def test_failure_reasons_populated_when_fail(self, validator):
        result = validator.validate(
            total_trades=50, profit_factor=1.2,
            expectancy=-5.0, max_drawdown=20.0, assets_passing=0,
        )
        assert len(result.failure_reasons) == 5
        assert len(result.pass_reasons) == 0

    def test_pass_reasons_populated_when_pass(self, validator):
        result = validator.validate(**_all_passing_kwargs())
        assert len(result.pass_reasons) == 5
        assert len(result.failure_reasons) == 0

    def test_promotion_constants(self):
        assert PROMOTION_MIN_TRADES == 100
        assert PROMOTION_MIN_PF == 1.50
        assert PROMOTION_MIN_EXP == 0.0
        assert PROMOTION_MAX_DD == 15.0
        assert PROMOTION_MIN_ASSETS == 2
