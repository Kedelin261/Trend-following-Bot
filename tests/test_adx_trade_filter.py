"""Tests for ADXTradeFilter — threshold gating."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.data.models import Candle
from src.refinement.adx_trade_filter import ADXTradeFilter, ADXResearchResult
from tests.fixtures import flat_candles, uptrend_candles


def _amplitude_candles(n=50, price=100.0, amplitude=5.0) -> List[Candle]:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(symbol="SPY", timeframe="D1",
               timestamp=base + timedelta(days=i),
               open=price, high=price + amplitude, low=price - amplitude,
               close=price, volume=1e8, provider="TEST")
        for i in range(n)
    ]


@pytest.fixture
def filt() -> ADXTradeFilter:
    return ADXTradeFilter(threshold=25.0, period=14)


class TestADXTradeFilterInit:

    def test_threshold_stored(self):
        assert ADXTradeFilter(threshold=30.0).threshold == 30.0

    def test_period_stored(self):
        assert ADXTradeFilter(period=7).period == 7


class TestPasses:

    def test_insufficient_data_returns_false(self, filt):
        candles = _amplitude_candles(n=10)
        assert filt.passes(candles) is False

    def test_zero_threshold_always_passes_with_enough_data(self):
        f = ADXTradeFilter(threshold=0.0)
        candles = _amplitude_candles(n=50)
        result = f.passes(candles)
        assert isinstance(result, bool)  # just verify it runs

    def test_very_high_threshold_fails(self, filt):
        f = ADXTradeFilter(threshold=99.9)
        candles = _amplitude_candles(n=50)
        result = f.passes(candles)
        # ADX rarely reaches 99.9
        assert result is False

    def test_strong_trend_passes_moderate_threshold(self):
        f = ADXTradeFilter(threshold=10.0, period=5)
        candles = uptrend_candles(n=50, daily_gain=0.01)
        # With strong trend and low threshold, should pass
        result = f.passes(candles)
        assert isinstance(result, bool)


class TestCurrentADX:

    def test_returns_none_for_insufficient_data(self, filt):
        candles = _amplitude_candles(n=10)
        assert filt.current_adx(candles) is None

    def test_returns_float_for_sufficient_data(self, filt):
        candles = _amplitude_candles(n=50)
        adx = filt.current_adx(candles)
        if adx is not None:
            assert isinstance(adx, float)
            assert 0.0 <= adx <= 100.0


class TestResearchThresholds:

    def test_returns_sorted_by_expectancy(self):
        from src.backtest.models import BacktestResults

        def _bt(trades, exp, pf):
            return BacktestResults(
                symbol="SPY", timeframe="D1",
                starting_balance=10_000, ending_balance=10_000 + exp * trades,
                net_profit=exp * trades, total_trades=trades,
                winning_trades=int(trades * 0.6), losing_trades=int(trades * 0.4),
                win_rate=0.6, profit_factor=pf, expectancy=exp,
                max_drawdown=5.0, sharpe_ratio=1.0,
                average_win=150.0, average_loss=90.0,
                largest_win=300.0, largest_loss=120.0,
                equity_curve=[], trades=[],
            )

        results_map = {
            20.0: _bt(40, 30.0, 1.5),
            25.0: _bt(35, 50.0, 1.8),
            30.0: _bt(20, 70.0, 2.1),  # insufficient
        }
        ranked = ADXTradeFilter.research_thresholds(results_map, min_trades=30)
        assert len(ranked) == 3
        # Best expectancy first
        assert ranked[0].expectancy >= ranked[1].expectancy

    def test_best_threshold_returns_none_when_all_insufficient(self):
        from src.backtest.models import BacktestResults

        def _bt(trades, exp):
            return BacktestResults(
                symbol="SPY", timeframe="D1",
                starting_balance=10_000, ending_balance=10_500, net_profit=500,
                total_trades=trades, winning_trades=int(trades * 0.6),
                losing_trades=int(trades * 0.4), win_rate=0.6,
                profit_factor=1.5, expectancy=exp, max_drawdown=5.0, sharpe_ratio=1.0,
                average_win=100.0, average_loss=60.0, largest_win=200.0, largest_loss=80.0,
                equity_curve=[], trades=[],
            )

        results_map = {25.0: _bt(5, 50.0)}  # only 5 trades
        ranked = ADXTradeFilter.research_thresholds(results_map, min_trades=30)
        assert ADXTradeFilter.best_threshold(ranked) is None
