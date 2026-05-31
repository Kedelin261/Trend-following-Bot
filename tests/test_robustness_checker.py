"""Tests for RobustnessChecker — stability across market periods."""

import pytest

from src.promotion.robustness_checker import (
    RobustnessChecker,
    RobustnessResult,
    WindowResult,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from tests.fixtures import flat_candles, uptrend_candles


def _mini_profile() -> StrategyProfile:
    return StrategyProfile(
        name="mini", description="mini",
        ema_fast=5, ema_slow=10,
        breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE, atr_period=3,
    )


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


@pytest.fixture
def checker(config) -> RobustnessChecker:
    return RobustnessChecker(config, profile=_mini_profile())


class TestRobustnessResult:

    def test_robust_is_acceptable(self):
        r = RobustnessResult([], 3, "ROBUST", [])
        assert r.is_acceptable is True

    def test_marginal_is_acceptable(self):
        r = RobustnessResult([], 2, "MARGINAL", [])
        assert r.is_acceptable is True

    def test_unstable_is_not_acceptable(self):
        r = RobustnessResult([], 1, "UNSTABLE", [])
        assert r.is_acceptable is False


class TestRobustnessChecker:

    def test_returns_robustness_result(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        assert isinstance(result, RobustnessResult)

    def test_rating_is_valid(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        assert result.rating in ("ROBUST", "MARGINAL", "UNSTABLE")

    def test_three_windows_tested(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        assert len(result.window_results) == 3

    def test_window_names_are_correct(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        names = {w.window_name for w in result.window_results}
        assert names == {"early", "middle", "recent"}

    def test_windows_passing_count_matches_rating(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        actual_passing = sum(1 for w in result.window_results if w.passes)
        assert result.windows_passing == actual_passing

    def test_rating_maps_correctly(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        if result.windows_passing == 3:
            assert result.rating == "ROBUST"
        elif result.windows_passing == 2:
            assert result.rating == "MARGINAL"
        else:
            assert result.rating == "UNSTABLE"

    def test_notes_populated(self, checker):
        candles = {"SPY": uptrend_candles(n=100, symbol="SPY")}
        result = checker.check(candles)
        assert len(result.notes) > 0
