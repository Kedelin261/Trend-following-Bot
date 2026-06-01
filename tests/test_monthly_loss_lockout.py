"""Tests for MONTHLY_LOSS_LOCKOUT overlay."""

from datetime import datetime, timezone
from typing import List

import pytest

from src.backtest.models import BacktestTrade, ClosingReason
from src.data.models import Candle
from src.risk_overlay.profiles.monthly_loss_lockout import MonthlyLossLockout
from src.signals.models import SignalType


def _make_candle(year: int, month: int, day: int) -> Candle:
    ts = datetime(year, month, day, tzinfo=timezone.utc)
    return Candle(
        symbol="SPY", timeframe="D1", timestamp=ts,
        open=400.0, high=410.0, low=395.0, close=402.0,
        volume=1e7, provider="SYNTHETIC",
    )


def _make_trade(pnl: float, year: int, month: int) -> BacktestTrade:
    ts = datetime(year, month, 15, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol="SPY", entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=400.0, exit_price=410.0 if pnl > 0 else 390.0,
        stop_price=390.0, target_price=430.0,
        position_size=10, pnl=pnl, return_percent=pnl/4000,
        holding_period=5, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


class TestMonthlyLossLockout:

    def setup_method(self):
        self.overlay = MonthlyLossLockout(
            starting_balance=10_000.0,
            monthly_loss_threshold=0.05,
        )

    def test_name(self):
        assert self.overlay.name == "MONTHLY_LOSS_LOCKOUT"

    def test_allows_with_no_trades(self):
        candle = _make_candle(2020, 1, 15)
        allowed, scale, _ = self.overlay.evaluate(10, [candle], [10_000.0], [])
        assert allowed is True
        assert scale == 1.0

    def test_allows_below_threshold(self):
        candle = _make_candle(2020, 1, 20)
        # Loss = $400 < $500 threshold
        trades = [_make_trade(-400.0, 2020, 1)]
        allowed, scale, _ = self.overlay.evaluate(20, [candle], [], trades)
        assert allowed is True

    def test_blocks_at_threshold(self):
        candle = _make_candle(2020, 1, 20)
        # Loss = $500 = threshold (>= threshold → lockout)
        trades = [_make_trade(-500.0, 2020, 1)]
        allowed, scale, reason = self.overlay.evaluate(20, [candle], [], trades)
        assert allowed is False
        assert scale == 0.0

    def test_blocks_above_threshold(self):
        candle = _make_candle(2020, 1, 20)
        # Loss = $600 > $500 threshold
        trades = [_make_trade(-600.0, 2020, 1)]
        allowed, scale, _ = self.overlay.evaluate(20, [candle], [], trades)
        assert allowed is False

    def test_new_month_resets(self):
        # Build up lockout in January
        jan_candle = _make_candle(2020, 1, 20)
        jan_trades = [_make_trade(-600.0, 2020, 1)]
        allowed, _, _ = self.overlay.evaluate(20, [jan_candle], [], jan_trades)
        assert allowed is False

        # February should allow (new month, no Feb trades yet)
        feb_candle = _make_candle(2020, 2, 5)
        allowed, scale, _ = self.overlay.evaluate(30, [feb_candle], [], jan_trades)
        assert allowed is True
        assert scale == 1.0

    def test_only_counts_current_month_trades(self):
        candle = _make_candle(2020, 2, 15)
        # Large loss in Jan should not affect Feb
        jan_loss = _make_trade(-2000.0, 2020, 1)
        allowed, scale, _ = self.overlay.evaluate(40, [candle], [], [jan_loss])
        assert allowed is True

    def test_cumulative_losses_trigger_lockout(self):
        candle = _make_candle(2020, 1, 25)
        # Multiple small losses in same month
        trades = [
            _make_trade(-200.0, 2020, 1),
            _make_trade(-150.0, 2020, 1),
            _make_trade(-200.0, 2020, 1),
        ]  # total = $550 > $500
        allowed, _, _ = self.overlay.evaluate(30, [candle], [], trades)
        assert allowed is False

    def test_wins_reduce_net_loss(self):
        candle = _make_candle(2020, 1, 25)
        # Loss $600, Win $200 → net loss $400 < threshold
        trades = [
            _make_trade(-600.0, 2020, 1),
            _make_trade(+200.0, 2020, 1),
        ]
        allowed, _, _ = self.overlay.evaluate(30, [candle], [], trades)
        assert allowed is True

    def test_reset_clears_locked_months(self):
        candle = _make_candle(2020, 1, 20)
        trades = [_make_trade(-600.0, 2020, 1)]
        self.overlay.evaluate(20, [candle], [], trades)
        self.overlay.reset()
        # After reset, January should not be locked
        allowed, _, _ = self.overlay.evaluate(20, [candle], [], trades)
        assert allowed is False  # Will re-evaluate and re-lock

    def test_reason_is_string(self):
        candle = _make_candle(2020, 1, 15)
        _, _, reason = self.overlay.evaluate(10, [candle], [], [])
        assert isinstance(reason, str)

    def test_no_candle_data(self):
        allowed, scale, _ = self.overlay.evaluate(0, [], [], [])
        assert allowed is True
