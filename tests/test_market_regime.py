"""Tests for MarketRegimeDetector — classification and trade slicing."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.data.models import Candle
from src.research.market_regime import MarketRegime, MarketRegimeDetector
from src.signals.models import SignalType
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(entry_ts: datetime, pnl: float = 100.0) -> BacktestTrade:
    return BacktestTrade(
        symbol="SPY", entry_time=entry_ts,
        exit_time=entry_ts + timedelta(days=1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=110.0,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=pnl / 10,
        holding_period=1, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET,
    )


@pytest.fixture
def detector() -> MarketRegimeDetector:
    return MarketRegimeDetector(
        fast_period=5, slow_period=10, sideways_threshold=0.005
    )


class TestClassify:

    def test_uptrend_returns_bull(self, detector):
        # Use strong daily_gain so EMA5/EMA10 separation exceeds sideways_threshold=0.5%
        candles = uptrend_candles(n=50, daily_gain=0.005)
        assert detector.classify(candles) == MarketRegime.BULL

    def test_downtrend_returns_bear(self, detector):
        candles = downtrend_candles(n=50, daily_loss=0.005)
        assert detector.classify(candles) == MarketRegime.BEAR

    def test_flat_returns_sideways_or_unknown(self, detector):
        candles = flat_candles(n=50)
        result = detector.classify(candles)
        assert result in (MarketRegime.SIDEWAYS, MarketRegime.UNKNOWN)

    def test_too_few_candles_returns_unknown(self, detector):
        candles = uptrend_candles(n=5)
        assert detector.classify(candles) == MarketRegime.UNKNOWN

    def test_exactly_slow_period_candles(self, detector):
        # 10 candles = exactly slow_period; should return something
        candles = uptrend_candles(n=10)
        result = detector.classify(candles)
        assert result in list(MarketRegime)

    def test_bull_requires_fast_above_slow(self, detector):
        # Only BULL when EMA5 > EMA10
        candles = uptrend_candles(n=50, daily_gain=0.005)
        result = detector.classify(candles)
        assert result == MarketRegime.BULL


class TestClassifySeries:

    def test_series_length_matches_candles(self, detector):
        candles = uptrend_candles(n=30)
        series = detector.classify_series(candles)
        assert len(series) == len(candles)

    def test_early_bars_are_unknown(self, detector):
        candles = uptrend_candles(n=30)
        series = detector.classify_series(candles)
        # First few bars have no regime (insufficient history)
        assert series[0] == MarketRegime.UNKNOWN

    def test_late_bars_in_uptrend_are_bull(self, detector):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        series = detector.classify_series(candles)
        assert series[-1] == MarketRegime.BULL


class TestRegimeAtTimestamp:

    def test_returns_unknown_for_early_timestamp(self, detector):
        candles = uptrend_candles(n=50)
        early_ts = candles[2].timestamp
        result = detector.regime_at_timestamp(candles, early_ts)
        assert result == MarketRegime.UNKNOWN

    def test_returns_regime_for_later_timestamp(self, detector):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        late_ts = candles[-1].timestamp
        result = detector.regime_at_timestamp(candles, late_ts)
        assert result == MarketRegime.BULL


class TestAnalyzeTradesByRegime:

    def test_returns_dict_of_regime_performance(self, detector):
        candles = uptrend_candles(n=50)
        trades = [_trade(candles[i].timestamp) for i in range(15, 49)]
        result = detector.analyze_trades_by_regime(trades, candles, min_trades=1)
        assert isinstance(result, dict)

    def test_sufficient_flag_based_on_min_trades(self, detector):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        trades = [_trade(candles[i].timestamp) for i in range(15, 49)]
        result = detector.analyze_trades_by_regime(trades, candles, min_trades=100)
        for perf in result.values():
            assert perf.sufficient is False

    def test_empty_trades_returns_empty_dict(self, detector):
        candles = uptrend_candles(n=50)
        result = detector.analyze_trades_by_regime([], candles)
        assert result == {}

    def test_performance_fields_populated(self, detector):
        candles = uptrend_candles(n=50, daily_gain=0.005)
        trades = [_trade(candles[i].timestamp) for i in range(15, 49)]
        result = detector.analyze_trades_by_regime(trades, candles, min_trades=1)
        for perf in result.values():
            assert 0.0 <= perf.win_rate <= 1.0
            assert perf.trade_count >= 0
