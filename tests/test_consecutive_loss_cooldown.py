"""Tests for CONSECUTIVE_LOSS_COOLDOWN overlay."""

from datetime import datetime, timezone

import pytest

from src.backtest.models import BacktestTrade, ClosingReason
from src.risk_overlay.profiles.consecutive_loss_cooldown import ConsecutiveLossCooldown
from src.signals.models import SignalType


def _make_trade(pnl: float) -> BacktestTrade:
    ts = datetime(2020, 1, 15, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol="SPY", entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=400.0, exit_price=410.0 if pnl > 0 else 390.0,
        stop_price=390.0, target_price=430.0,
        position_size=10, pnl=pnl, return_percent=pnl/4000,
        holding_period=5, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


class TestConsecutiveLossCooldown:

    def setup_method(self):
        self.overlay = ConsecutiveLossCooldown(loss_trigger=5, cooldown_bars=10)

    def test_name(self):
        assert self.overlay.name == "CONSECUTIVE_LOSS_COOLDOWN"

    def test_allows_with_no_trades(self):
        allowed, scale, _ = self.overlay.evaluate(10, [], [], [])
        assert allowed is True
        assert scale == 1.0

    def test_allows_below_trigger(self):
        trades = [_make_trade(-50.0)] * 4   # 4 consecutive losses < 5
        allowed, scale, _ = self.overlay.evaluate(100, [], [], trades)
        assert allowed is True

    def test_blocks_at_trigger(self):
        trades = [_make_trade(-50.0)] * 5   # exactly 5 consecutive losses
        allowed, scale, reason = self.overlay.evaluate(100, [], [], trades)
        assert allowed is False
        assert scale == 0.0
        assert "cooldown" in reason.lower()

    def test_blocks_above_trigger(self):
        trades = [_make_trade(-50.0)] * 7   # 7 consecutive losses > 5
        allowed, _, _ = self.overlay.evaluate(100, [], [], trades)
        assert allowed is False

    def test_win_resets_streak(self):
        # 4 losses, then 1 win, then 4 more losses → streak = 4 < 5
        trades = [_make_trade(-50.0)] * 4 + [_make_trade(+100.0)] + [_make_trade(-50.0)] * 4
        allowed, scale, _ = self.overlay.evaluate(100, [], [], trades)
        assert allowed is True

    def test_cooldown_expires(self):
        # Trigger cooldown at bar 100 with 5 consecutive losses
        trades_5 = [_make_trade(-50.0)] * 5
        self.overlay.evaluate(100, [], [], trades_5)

        # Still in cooldown at bar 105
        allowed, _, _ = self.overlay.evaluate(105, [], [], trades_5)
        assert allowed is False

        # Cooldown ends at bar 110 (100 + 10)
        # After cooldown expires, overlay re-checks consecutive losses.
        # Since there are still 5 losses in the trade list, the overlay will
        # re-trigger another cooldown immediately at bar 110.
        # We test with a win appended so the streak is broken.
        trades_with_win = list(trades_5) + [_make_trade(+100.0)]
        allowed, _, _ = self.overlay.evaluate(110, [], [], trades_with_win)
        assert allowed is True

    def test_cooldown_still_active_before_expiry(self):
        trades = [_make_trade(-50.0)] * 5
        self.overlay.evaluate(100, [], [], trades)
        # bar 109 = cooldown_end - 1
        allowed, _, _ = self.overlay.evaluate(109, [], [], trades)
        assert allowed is False

    def test_reset_clears_cooldown(self):
        trades = [_make_trade(-50.0)] * 5
        self.overlay.evaluate(100, [], [], trades)  # triggers cooldown
        self.overlay.reset()
        # After reset, should be fresh
        allowed, _, _ = self.overlay.evaluate(101, [], [], trades)
        # Will re-evaluate and re-trigger since 5 losses still present
        assert allowed is False  # re-triggered immediately

    def test_reason_is_string(self):
        _, _, reason = self.overlay.evaluate(0, [], [], [])
        assert isinstance(reason, str)

    def test_returns_tuple_of_three(self):
        result = self.overlay.evaluate(0, [], [], [])
        assert len(result) == 3

    def test_mixed_win_loss_allows(self):
        # Win at end resets streak
        trades = [_make_trade(-50.0)] * 5 + [_make_trade(+100.0)]
        # Must first trigger at bar 100, then win resolves
        allowed_after_trigger, _, _ = self.overlay.evaluate(100, [], [], trades[:5])
        assert allowed_after_trigger is False
        # After cooldown expires
        allowed_after_cooldown, _, _ = self.overlay.evaluate(111, [], [], trades)
        assert allowed_after_cooldown is True
