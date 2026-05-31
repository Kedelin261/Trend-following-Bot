"""Tests for FinalRecommendationEngine — PROMOTE vs RETURN_TO_RESEARCH logic."""

import pytest

from src.promotion.final_recommendation import (
    FinalRecommendationEngine,
    FinalRecommendationResult,
    PromotionStatus,
)
from src.promotion.promotion_validator import PromotionValidationResult
from src.promotion.robustness_checker import RobustnessResult, WindowResult


def _validation(all_passed=True) -> PromotionValidationResult:
    return PromotionValidationResult(
        trades_ok=True, pf_ok=True, expectancy_ok=True, drawdown_ok=True,
        assets_ok=True, all_passed=all_passed,
        actual_trades=105, actual_pf=1.65, actual_exp=28.0,
        actual_dd=6.5, actual_assets=3,
        failure_reasons=[] if all_passed else ["Trade count 80 < 100 minimum"],
        pass_reasons=["Trade count 105 ≥ 100"] if all_passed else [],
    )


def _robustness(rating="ROBUST", windows_passing=3) -> RobustnessResult:
    return RobustnessResult([], windows_passing, rating, [f"{rating} robustness"])


@pytest.fixture
def rec_engine() -> FinalRecommendationEngine:
    return FinalRecommendationEngine()


class TestFinalRecommendationEngine:

    def test_promote_when_all_pass_and_robust(self, rec_engine):
        result = rec_engine.generate(
            _validation(True), _robustness("ROBUST"), 5000
        )
        assert result.status == PromotionStatus.PROMOTE
        assert result.promoted is True

    def test_promote_when_all_pass_and_marginal(self, rec_engine):
        result = rec_engine.generate(
            _validation(True), _robustness("MARGINAL", 2), 5000
        )
        assert result.status == PromotionStatus.PROMOTE
        assert len(result.caveats) > 0  # should warn about marginal

    def test_return_when_validation_fails(self, rec_engine):
        result = rec_engine.generate(
            _validation(False), _robustness("ROBUST"), 5000
        )
        assert result.status == PromotionStatus.RETURN
        assert result.promoted is False

    def test_return_when_unstable_even_if_validation_passes(self, rec_engine):
        result = rec_engine.generate(
            _validation(True), _robustness("UNSTABLE", 1), 5000
        )
        assert result.status == PromotionStatus.RETURN

    def test_headline_populated(self, rec_engine):
        result = rec_engine.generate(
            _validation(True), _robustness("ROBUST"), 5000
        )
        assert len(result.headline) > 0

    def test_primary_reasons_populated(self, rec_engine):
        result = rec_engine.generate(
            _validation(True), _robustness("ROBUST"), 5000
        )
        assert len(result.primary_reasons) > 0

    def test_next_steps_populated_when_return(self, rec_engine):
        result = rec_engine.generate(
            _validation(False), _robustness("ROBUST"), 5000
        )
        assert len(result.next_steps) > 0

    def test_return_reason_mentions_specific_failure(self, rec_engine):
        val = _validation(False)  # has failure_reasons
        result = rec_engine.generate(val, _robustness("ROBUST"), 5000)
        # At least one failure reason should appear in primary_reasons
        assert any("100" in r or "minimum" in r.lower() for r in result.primary_reasons)
