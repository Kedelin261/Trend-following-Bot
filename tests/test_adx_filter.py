"""Tests for ADXCalculator and ADXFilter."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.data.models import Candle
from src.research.adx_filter import ADXCalculator, ADXFilter
from src.signals.models import SignalType
from tests.fixtures import flat_candles, uptrend_candles, downtrend_candles


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _flat_candles_amp(n=50, price=100.0, amplitude=5.0) -> List[Candle]:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(symbol="SPY", timeframe="D1",
               timestamp=base + timedelta(days=i),
               open=price, high=price + amplitude, low=price - amplitude,
               close=price, volume=1e8, provider="TEST")
        for i in range(n)
    ]


def _trade(ts: datetime, pnl: float = 100.0) -> BacktestTrade:
    return BacktestTrade(
        symbol="SPY", entry_time=ts, exit_time=ts + timedelta(days=1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=110.0,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=10.0,
        holding_period=1, win_loss="WIN",
        reason_closed=ClosingReason.TARGET,
    )


@pytest.fixture
def calc() -> ADXCalculator:
    return ADXCalculator(period=14)


@pytest.fixture
def adx_filter() -> ADXFilter:
    return ADXFilter(period=14)


class TestADXCalculatorInit:

    def test_period_stored(self):
        assert ADXCalculator(period=14).period == 14

    def test_zero_period_raises(self):
        with pytest.raises(ValueError):
            ADXCalculator(period=0)


class TestCalculateADX:

    def test_returns_none_for_insufficient_data(self, calc):
        candles = _flat_candles_amp(n=10)
        assert calc.calculate_adx(candles) is None

    def test_returns_none_for_empty_list(self, calc):
        assert calc.calculate_adx([]) is None

    def test_returns_float_for_sufficient_data(self, calc):
        candles = _flat_candles_amp(n=35)
        result = calc.calculate_adx(candles)
        assert result is not None
        assert isinstance(result, float)

    def test_adx_in_valid_range(self, calc):
        candles = _flat_candles_amp(n=50)
        result = calc.calculate_adx(candles)
        if result is not None:
            assert 0.0 <= result <= 100.0

    def test_strong_uptrend_has_higher_adx_than_flat(self, calc):
        flat    = _flat_candles_amp(n=50, amplitude=0.001)
        trending = uptrend_candles(n=50, daily_gain=0.005)
        flat_adx    = calc.calculate_adx(flat)
        trend_adx   = calc.calculate_adx(trending)
        if flat_adx is not None and trend_adx is not None:
            assert trend_adx >= flat_adx

    def test_minimum_candles_threshold(self, calc):
        # period=14 → need 2*14+1 = 29 candles
        candles_28 = _flat_candles_amp(n=28)
        candles_29 = _flat_candles_amp(n=29)
        assert calc.calculate_adx(candles_28) is None
        result_29  = calc.calculate_adx(candles_29)
        assert result_29 is not None


class TestADXSeries:

    def test_series_length_matches_candles(self, calc):
        candles = _flat_candles_amp(n=40)
        series = calc.calculate_adx_series(candles)
        assert len(series) == 40

    def test_early_bars_are_none(self, calc):
        candles = _flat_candles_amp(n=40)
        series = calc.calculate_adx_series(candles)
        assert series[0] is None
        assert series[27] is None  # period=14, need 29 bars

    def test_later_bars_are_floats(self, calc):
        candles = _flat_candles_amp(n=40)
        series = calc.calculate_adx_series(candles)
        for v in series[28:]:
            assert isinstance(v, float)


class TestADXFilter:

    def test_zero_threshold_includes_all_trades(self, adx_filter):
        candles = _flat_candles_amp(n=50)
        trades  = [_trade(candles[i].timestamp) for i in range(30, 49)]
        result  = adx_filter.analyze_threshold(trades, candles, threshold=0.0)
        assert result.trades_passed == len(trades)
        assert result.trades_blocked == 0

    def test_very_high_threshold_blocks_all_trades(self, adx_filter):
        candles = _flat_candles_amp(n=50)
        trades  = [_trade(candles[i].timestamp) for i in range(30, 49)]
        result  = adx_filter.analyze_threshold(trades, candles, threshold=99.9)
        assert result.trades_passed == 0

    def test_compare_thresholds_returns_list(self, adx_filter):
        candles = _flat_candles_amp(n=50)
        trades  = [_trade(candles[i].timestamp) for i in range(30, 49)]
        results = adx_filter.compare_thresholds(trades, candles, [0.0, 20.0, 25.0])
        assert len(results) == 3

    def test_sufficient_flag_requires_min_trades(self, adx_filter):
        candles = _flat_candles_amp(n=50)
        trades  = [_trade(candles[30].timestamp)]  # only 1 trade
        result  = adx_filter.analyze_threshold(trades, candles, 0.0, min_trades=10)
        assert result.sufficient is False

    def test_best_threshold_returns_none_when_none_sufficient(self, adx_filter):
        # Create filter results with no sufficient entries
        candles = _flat_candles_amp(n=50)
        trades  = [_trade(candles[30].timestamp)]  # only 1 trade
        results = adx_filter.compare_thresholds(trades, candles, min_trades=100)
        assert adx_filter.best_threshold(results) is None

    def test_empty_trades_returns_empty_passed(self, adx_filter):
        candles = _flat_candles_amp(n=50)
        result  = adx_filter.analyze_threshold([], candles, 25.0)
        assert result.trades_passed == 0
        assert result.trades_blocked == 0
