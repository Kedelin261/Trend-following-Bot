"""Tests for RobustnessImprovementAnalyzer."""

import pytest
from typing import Dict, List

from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.regime_filter.filter_profiles import AVOID_STRONG_BULL, NO_FILTER
from src.regime_filter.robustness_improvement_analyzer import (
    RobustnessImprovementAnalyzer,
    RobustnessImprovement,
)
from tests.fixtures import downtrend_candles, uptrend_candles


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
def analyzer(config) -> RobustnessImprovementAnalyzer:
    return RobustnessImprovementAnalyzer(config, _mini_profile())


def _asset_candles() -> Dict[str, List[Candle]]:
    return {"SPY": uptrend_candles(n=100, symbol="SPY")}


class TestRobustnessImprovementAnalyzer:

    def test_returns_robustness_improvement(self, analyzer):
        result = analyzer.analyze(_asset_candles(), NO_FILTER)
        assert isinstance(result, RobustnessImprovement)

    def test_three_windows_tested(self, analyzer):
        result = analyzer.analyze(_asset_candles(), NO_FILTER)
        assert len(result.window_results) == 3

    def test_window_names_correct(self, analyzer):
        result = analyzer.analyze(_asset_candles(), NO_FILTER)
        names = {w.window_name for w in result.window_results}
        assert names == {"early", "middle", "recent"}

    def test_rating_is_valid_value(self, analyzer):
        result = analyzer.analyze(_asset_candles(), NO_FILTER)
        assert result.rating in ("ROBUST", "MARGINAL", "UNSTABLE")

    def test_windows_passing_matches_rating(self, analyzer):
        result = analyzer.analyze(_asset_candles(), NO_FILTER)
        actual = sum(1 for w in result.window_results if w.passes)
        assert result.windows_passing == actual

    def test_improvement_over_base_is_bool(self, analyzer):
        result = analyzer.analyze(_asset_candles(), AVOID_STRONG_BULL)
        assert isinstance(result.improvement_over_base, bool)

    def test_profile_preserved(self, analyzer):
        result = analyzer.analyze(_asset_candles(), AVOID_STRONG_BULL)
        assert result.profile.name == "AVOID_STRONG_BULL"
