"""Tests for FilterBacktester — full pipeline backtest with filter applied."""

import pytest
from typing import Dict, List

from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.regime_filter.filter_backtester import FilterBacktestResult, FilterBacktester
from src.regime_filter.filter_profiles import AVOID_STRONG_BULL, NO_FILTER
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
def backtester(config) -> FilterBacktester:
    return FilterBacktester(config, _mini_profile())


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=60, symbol="SPY"),
        "DIA": downtrend_candles(n=60, symbol="DIA"),
    }


class TestFilterBacktestResult:

    def test_retention_pct_calculation(self):
        r = FilterBacktestResult(
            filter_profile=NO_FILTER,
            total_trades_before=100, total_trades_after=80,
            trades_removed=20, win_rate=0.6,
            profit_factor=1.7, expectancy=25.0,
            max_drawdown=5.0, net_pnl=500.0,
        )
        assert r.retention_pct == pytest.approx(80.0)

    def test_pf_str_for_infinity(self):
        import math
        r = FilterBacktestResult(
            filter_profile=NO_FILTER,
            total_trades_before=10, total_trades_after=10,
            trades_removed=0, win_rate=1.0,
            profit_factor=math.inf, expectancy=50.0,
            max_drawdown=3.0, net_pnl=500.0,
        )
        assert r.pf_str == "∞"

    def test_meets_quality_true(self):
        r = FilterBacktestResult(
            filter_profile=NO_FILTER,
            total_trades_before=50, total_trades_after=50,
            trades_removed=0, win_rate=0.65,
            profit_factor=1.7, expectancy=30.0,
            max_drawdown=6.0, net_pnl=1500.0,
        )
        assert r.meets_quality is True

    def test_meets_quality_false_bad_pf(self):
        r = FilterBacktestResult(
            filter_profile=NO_FILTER,
            total_trades_before=50, total_trades_after=50,
            trades_removed=0, win_rate=0.55,
            profit_factor=1.2, expectancy=10.0,
            max_drawdown=6.0, net_pnl=500.0,
        )
        assert r.meets_quality is False


class TestFilterBacktester:

    def test_backtest_with_filter_returns_result(self, backtester):
        result = backtester.backtest_with_filter(_asset_candles(), NO_FILTER)
        assert isinstance(result, FilterBacktestResult)

    def test_trades_before_lte_total_backtest(self, backtester):
        result = backtester.backtest_with_filter(_asset_candles(), NO_FILTER)
        assert result.total_trades_after <= result.total_trades_before

    def test_no_filter_retains_all_trades(self, backtester):
        r = backtester.backtest_with_filter(_asset_candles(), NO_FILTER)
        assert r.total_trades_after == r.total_trades_before

    def test_filter_removes_trades(self, backtester):
        r_no  = backtester.backtest_with_filter(_asset_candles(), NO_FILTER)
        r_flt = backtester.backtest_with_filter(_asset_candles(), AVOID_STRONG_BULL)
        assert r_flt.total_trades_after <= r_no.total_trades_after

    def test_backtest_all_profiles_returns_one_per_profile(self, backtester):
        profiles = [NO_FILTER, AVOID_STRONG_BULL]
        results = backtester.backtest_all_profiles(_asset_candles(), profiles)
        assert len(results) == 2

    def test_filter_profile_preserved_in_result(self, backtester):
        r = backtester.backtest_with_filter(_asset_candles(), AVOID_STRONG_BULL)
        assert r.filter_profile.name == "AVOID_STRONG_BULL"
