"""Tests for NO_OVERLAY — baseline pass-through."""

import pytest
from src.risk_overlay.profiles.no_overlay import NoOverlay


class TestNoOverlay:

    def setup_method(self):
        self.overlay = NoOverlay()

    def test_name(self):
        assert self.overlay.name == "NO_OVERLAY"

    def test_always_allows(self):
        allowed, scale, reason = self.overlay.evaluate(
            bar_index=100, candles=[], equity_values=[], closed_trades=[]
        )
        assert allowed is True
        assert scale == 1.0

    def test_full_scale(self):
        _, scale, _ = self.overlay.evaluate(0, [], [], [])
        assert scale == 1.0

    def test_reset_is_noop(self):
        self.overlay.reset()  # must not raise
        allowed, scale, _ = self.overlay.evaluate(0, [], [], [])
        assert allowed is True

    def test_reason_is_string(self):
        _, _, reason = self.overlay.evaluate(0, [], [], [])
        assert isinstance(reason, str)

    def test_allows_with_trades(self):
        from datetime import datetime, timezone
        from src.backtest.models import BacktestTrade, ClosingReason
        from src.signals.models import SignalType
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        trade = BacktestTrade(
            symbol="SPY", entry_time=ts, exit_time=ts,
            signal_type=SignalType.LONG,
            entry_price=400.0, exit_price=390.0,
            stop_price=385.0, target_price=430.0,
            position_size=10, pnl=-100.0, return_percent=-0.025,
            holding_period=3, win_loss="LOSS", reason_closed=ClosingReason.STOP,
        )
        allowed, scale, _ = self.overlay.evaluate(100, [], [9000.0, 8900.0], [trade])
        assert allowed is True
        assert scale == 1.0
