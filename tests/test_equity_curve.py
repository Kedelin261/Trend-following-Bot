"""Tests for EquityCurve — generation, length, and value correctness."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.equity_curve import EquityCurve, EquityPoint
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(pnl: float, i: int = 0) -> BacktestTrade:
    win = pnl > 0
    return BacktestTrade(
        symbol="SPY", entry_time=BASE + timedelta(days=i),
        exit_time=BASE + timedelta(days=i + 1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=100.0 + pnl / 10,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=pnl / 10,
        holding_period=1, win_loss="WIN" if win else "LOSS",
        reason_closed=ClosingReason.TARGET if win else ClosingReason.STOP,
    )


class TestEquityCurveConstruction:

    def test_empty_trades_has_one_point(self):
        curve = EquityCurve(starting_balance=10_000.0, trades=[])
        assert curve.length == 1

    def test_starting_point_is_starting_balance(self):
        curve = EquityCurve(10_000.0, [])
        assert curve.values[0] == pytest.approx(10_000.0)

    def test_starting_timestamp_is_none(self):
        curve = EquityCurve(10_000.0, [])
        assert curve.timestamps[0] is None

    def test_length_is_trades_plus_one(self):
        trades = [_trade(100.0), _trade(-50.0), _trade(80.0)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.length == 4  # 3 trades + starting point

    def test_values_accumulate_correctly(self):
        trades = [_trade(100.0), _trade(-50.0), _trade(200.0)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.values == pytest.approx([10_000, 10_100, 10_050, 10_250])

    def test_timestamps_match_exit_times(self):
        trades = [_trade(100.0, i=0), _trade(-50.0, i=5)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.timestamps[1] == BASE + timedelta(days=1)
        assert curve.timestamps[2] == BASE + timedelta(days=6)

    def test_trade_index_starting_point_is_minus_one(self):
        curve = EquityCurve(10_000.0, [])
        assert curve.points[0].trade_index == -1

    def test_trade_index_increments(self):
        trades = [_trade(100.0), _trade(-50.0)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.points[1].trade_index == 0
        assert curve.points[2].trade_index == 1


class TestEquityCurveProperties:

    def test_peak_value_correct(self):
        trades = [_trade(500.0), _trade(-200.0), _trade(300.0)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.peak_value == pytest.approx(10_600.0)

    def test_trough_value_correct(self):
        trades = [_trade(-200.0), _trade(-100.0), _trade(500.0)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.trough_value == pytest.approx(9_700.0)

    def test_as_tuples_returns_correct_format(self):
        trades = [_trade(100.0, i=0)]
        curve = EquityCurve(10_000.0, trades)
        tuples = curve.as_tuples()
        assert len(tuples) == 2
        ts0, val0 = tuples[0]
        assert ts0 is None
        assert val0 == pytest.approx(10_000.0)
        ts1, val1 = tuples[1]
        assert ts1 is not None
        assert val1 == pytest.approx(10_100.0)

    def test_values_property_matches_point_values(self):
        trades = [_trade(100.0), _trade(-30.0)]
        curve = EquityCurve(10_000.0, trades)
        assert curve.values == [p.portfolio_value for p in curve.points]

    def test_monotone_wins_equity_never_falls(self):
        trades = [_trade(50.0 * (i + 1), i=i) for i in range(5)]
        curve = EquityCurve(10_000.0, trades)
        vals = curve.values
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]
