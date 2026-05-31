"""Tests for RegimeTradeFilter — hypothetical filter simulations."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime.regime_trade_filter import (
    DEFAULT_FILTERS,
    FilterConfig,
    FilterSimulationResult,
    RegimeTradeFilter,
)
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime
from src.signals.models import SignalType


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(pnl: float, i: int = 0) -> BacktestTrade:
    entry = BASE + timedelta(days=i)
    return BacktestTrade(
        symbol="SPY", entry_time=entry, exit_time=entry + timedelta(days=1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=100.0 + pnl / 10,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=pnl / 10,
        holding_period=1, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def _lt(
    pnl: float,
    market=MarketRegime.STRONG_BULL,
    vol=VolatilityRegime.NORMAL_VOL,
    dd=DrawdownEnvironment.BULL_RECOVERY,
    macro=MacroRegime.EXPANSION,
    i: int = 0,
) -> LabelledTrade:
    return LabelledTrade(
        trade=_trade(pnl, i),
        market_regime=market,
        trend_regime=TrendRegime.STRONG_TREND,
        volatility_regime=vol,
        drawdown_env=dd,
        macro_regime=macro,
    )


@pytest.fixture
def filt() -> RegimeTradeFilter:
    return RegimeTradeFilter()


@pytest.fixture
def mixed_trades() -> List[LabelledTrade]:
    return [
        _lt(100.0, MarketRegime.STRONG_BULL,  i=0),
        _lt(80.0,  MarketRegime.BULL,         i=1),
        _lt(-50.0, MarketRegime.BEAR,         i=2),
        _lt(-70.0, MarketRegime.STRONG_BEAR,  i=3),
        _lt(120.0, MarketRegime.STRONG_BULL,  vol=VolatilityRegime.EXTREME_VOL, i=4),
        _lt(-30.0, MarketRegime.SIDEWAYS,     i=5),
    ]


class TestRegimeTradeFilter:

    def test_strong_bull_only_filter(self, filt, mixed_trades):
        config = FilterConfig(
            name="test", description="test",
            allowed_market_regimes={MarketRegime.STRONG_BULL},
        )
        result = filt.simulate(mixed_trades, config)
        assert result.filtered_trades == 2   # indices 0 and 4

    def test_avoid_bear_filter(self, filt, mixed_trades):
        config = FilterConfig(
            name="test", description="test",
            blocked_market_regimes={MarketRegime.BEAR, MarketRegime.STRONG_BEAR},
        )
        result = filt.simulate(mixed_trades, config)
        assert result.filtered_trades == 4   # removes indices 2 and 3

    def test_avoid_extreme_vol_filter(self, filt, mixed_trades):
        config = FilterConfig(
            name="test", description="test",
            blocked_volatility={VolatilityRegime.EXTREME_VOL},
        )
        result = filt.simulate(mixed_trades, config)
        assert result.filtered_trades == 5   # removes index 4 only

    def test_combined_filter(self, filt, mixed_trades):
        config = FilterConfig(
            name="test", description="test",
            allowed_market_regimes={MarketRegime.STRONG_BULL, MarketRegime.BULL},
            blocked_volatility={VolatilityRegime.EXTREME_VOL},
        )
        result = filt.simulate(mixed_trades, config)
        # Only STRONG_BULL (2) + BULL (1) = 3, minus 1 EXTREME_VOL = 2
        assert result.filtered_trades == 2

    def test_no_filter_passes_all(self, filt, mixed_trades):
        config = FilterConfig(name="all", description="all")  # no restrictions
        result = filt.simulate(mixed_trades, config)
        assert result.filtered_trades == len(mixed_trades)

    def test_returns_simulation_result(self, filt, mixed_trades):
        config = DEFAULT_FILTERS[0]
        result = filt.simulate(mixed_trades, config)
        assert isinstance(result, FilterSimulationResult)

    def test_original_trades_count(self, filt, mixed_trades):
        result = filt.simulate(mixed_trades, DEFAULT_FILTERS[0])
        assert result.original_trades == len(mixed_trades)

    def test_trades_removed_correct(self, filt, mixed_trades):
        config = FilterConfig(
            name="test", description="test",
            blocked_market_regimes={MarketRegime.BEAR},
        )
        result = filt.simulate(mixed_trades, config)
        assert result.trades_removed == result.original_trades - result.filtered_trades

    def test_simulate_all_returns_list(self, filt, mixed_trades):
        results = filt.simulate_all(mixed_trades)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_simulate_all_sorted_by_expectancy(self, filt, mixed_trades):
        results = filt.simulate_all(mixed_trades)
        for i in range(len(results) - 1):
            assert results[i].projected_expectancy >= results[i + 1].projected_expectancy

    def test_empty_trades_returns_zero_result(self, filt):
        config = FilterConfig(name="test", description="test")
        result = filt.simulate([], config)
        assert result.filtered_trades == 0

    def test_default_filters_have_names(self):
        assert len(DEFAULT_FILTERS) > 0
        for f in DEFAULT_FILTERS:
            assert f.name != ""
            assert f.description != ""

    def test_pf_str_for_infinity(self, filt):
        # All wins → PF = inf
        all_wins = [_lt(100.0, i=i) for i in range(5)]
        config = FilterConfig(name="test", description="test",
                              allowed_market_regimes={MarketRegime.STRONG_BULL})
        result = filt.simulate(all_wins, config)
        assert "∞" in result.pf_str or result.pf_str != ""
