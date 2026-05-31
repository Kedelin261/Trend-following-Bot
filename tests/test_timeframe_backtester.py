"""Tests for TimeframeBacktester."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.timeframe.timeframe_backtester import (
    QUALITY_MIN_PF,
    TimeframeBacktestResult,
    TimeframeBacktester,
)
from src.timeframe.timeframe_profile import D1_PROFILE, H4_PROFILE
from tests.fixtures import downtrend_candles, uptrend_candles


def _mini_profile() -> StrategyProfile:
    return StrategyProfile(
        name="mini", description="mini",
        ema_fast=5, ema_slow=10,
        breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE,
        atr_period=3,
    )


def _h4_candles(n=50, symbol="SPY") -> List[Candle]:
    """Candles with H4 timeframe tag."""
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    price = 400.0
    result = []
    for i in range(n):
        price *= 1.002
        result.append(Candle(
            symbol=symbol, timeframe="H4",
            timestamp=base + timedelta(hours=4 * i),
            open=price, high=price * 1.003, low=price * 0.997,
            close=price, volume=1e8, provider="TEST",
        ))
    return result


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
def backtester(config) -> TimeframeBacktester:
    return TimeframeBacktester(config, strategy_profile=_mini_profile())


class TestTimeframeBacktestResult:

    def test_timeframe_property(self):
        from src.backtest.models import BacktestResults, StrategyHealth
        from src.timeframe.timeframe_profile import D1_PROFILE
        bt = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=10_500, net_profit=500,
            total_trades=10, winning_trades=6, losing_trades=4,
            win_rate=0.6, profit_factor=1.8, expectancy=50.0, max_drawdown=5.0,
            sharpe_ratio=1.2, average_win=100.0, average_loss=60.0,
            largest_win=200.0, largest_loss=80.0, equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(bt)
        r = TimeframeBacktestResult(D1_PROFILE, "SPY", bt, health, True)
        assert r.timeframe == "D1"
        assert r.trade_count == 10


class TestTimeframeBacktester:

    def test_backtest_returns_result(self, backtester):
        candles = uptrend_candles(n=50, symbol="SPY")
        result = backtester.backtest("SPY", candles, D1_PROFILE)
        assert isinstance(result, TimeframeBacktestResult)

    def test_symbol_propagated(self, backtester):
        candles = uptrend_candles(n=50, symbol="VOO")
        result = backtester.backtest("VOO", candles, D1_PROFILE)
        assert result.symbol == "VOO"

    def test_timeframe_tag_propagated(self, backtester):
        candles = _h4_candles(n=50, symbol="SPY")
        result = backtester.backtest("SPY", candles, H4_PROFILE)
        assert result.timeframe == "H4"

    def test_meets_quality_is_bool(self, backtester):
        candles = uptrend_candles(n=50)
        result = backtester.backtest("SPY", candles, D1_PROFILE)
        assert isinstance(result.meets_quality, bool)

    def test_health_object_present(self, backtester):
        candles = uptrend_candles(n=50)
        result = backtester.backtest("SPY", candles, D1_PROFILE)
        assert result.health is not None

    def test_backtest_all_assets(self, backtester):
        asset_candles = {
            "SPY": uptrend_candles(n=50, symbol="SPY"),
            "DIA": downtrend_candles(n=50, symbol="DIA"),
        }
        results = backtester.backtest_all_assets(asset_candles, D1_PROFILE)
        assert len(results) == 2
        assert "SPY" in results
        assert "DIA" in results

    def test_empty_candles_skipped(self, backtester):
        results = backtester.backtest_all_assets({"SPY": [], "DIA": downtrend_candles(n=50)}, D1_PROFILE)
        assert "SPY" not in results
        assert "DIA" in results
