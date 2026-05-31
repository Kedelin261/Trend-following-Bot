"""Tests for VolatilityFilter — ATR% categorization and trade slicing."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.data.models import Candle
from src.research.volatility_filter import VolatilityCategory, VolatilityFilter
from src.signals.models import SignalType
from tests.fixtures import uptrend_candles


BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _candle(price=100.0, amplitude=5.0, i=0) -> Candle:
    return Candle(symbol="SPY", timeframe="D1",
                  timestamp=BASE + timedelta(days=i),
                  open=price, high=price + amplitude, low=price - amplitude,
                  close=price, volume=1e8, provider="TEST")


def _volatile_candles(n=25, price=100.0, amplitude=10.0) -> List[Candle]:
    return [_candle(price, amplitude, i) for i in range(n)]


def _calm_candles(n=25, price=100.0, amplitude=0.2) -> List[Candle]:
    return [_candle(price, amplitude, i) for i in range(n)]


def _trade_at(i, candles, pnl=100.0) -> BacktestTrade:
    ts = candles[i].timestamp
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
def vf() -> VolatilityFilter:
    return VolatilityFilter(atr_period=5, low_threshold=1.0, high_threshold=8.0)


class TestATRPct:

    def test_returns_none_for_insufficient_data(self, vf):
        candles = _calm_candles(n=4)  # less than period=5
        assert vf.atr_pct(candles) is None

    def test_returns_float_for_sufficient_data(self, vf):
        candles = _calm_candles(n=10)
        result = vf.atr_pct(candles)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_high_amplitude_gives_high_atr_pct(self, vf):
        calm     = _calm_candles(n=20)
        volatile = _volatile_candles(n=20)
        pct_c = vf.atr_pct(calm)
        pct_v = vf.atr_pct(volatile)
        if pct_c is not None and pct_v is not None:
            assert pct_v > pct_c


class TestCategorize:

    def test_high_amplitude_is_high_volatility(self, vf):
        candles = _volatile_candles(n=20, amplitude=15.0)
        result = vf.categorize(candles)
        assert result in (VolatilityCategory.HIGH, VolatilityCategory.MEDIUM)

    def test_tiny_amplitude_is_low_volatility(self, vf):
        candles = _calm_candles(n=20, amplitude=0.05)
        result = vf.categorize(candles)
        assert result == VolatilityCategory.LOW

    def test_insufficient_data_returns_unknown(self, vf):
        candles = _calm_candles(n=3)
        assert vf.categorize(candles) == VolatilityCategory.UNKNOWN


class TestAnalyzeTradesByVolatility:

    def test_returns_dict(self, vf):
        candles = uptrend_candles(n=30, volume=1e8)
        trades = [_trade_at(i, candles) for i in range(10, 29)]
        result = vf.analyze_trades_by_volatility(trades, candles)
        assert isinstance(result, dict)

    def test_trade_count_sums_to_total(self, vf):
        candles = uptrend_candles(n=30, volume=1e8)
        trades = [_trade_at(i, candles) for i in range(10, 29)]
        result = vf.analyze_trades_by_volatility(trades, candles)
        total = sum(p.trade_count for p in result.values())
        assert total == len(trades)

    def test_sufficient_flag_based_on_min_trades(self, vf):
        candles = uptrend_candles(n=30, volume=1e8)
        trades = [_trade_at(10, candles)]  # only 1 trade
        result = vf.analyze_trades_by_volatility(trades, candles, min_trades=5)
        for perf in result.values():
            assert perf.sufficient is False

    def test_empty_trades_returns_empty(self, vf):
        candles = uptrend_candles(n=30, volume=1e8)
        result = vf.analyze_trades_by_volatility([], candles)
        assert result == {}
