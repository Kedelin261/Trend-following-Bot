"""Tests for EQUITY_CURVE_PAUSE overlay."""

import pytest
from src.risk_overlay.profiles.equity_curve_pause import EquityCurvePause


class TestEquityCurvePause:

    def setup_method(self):
        self.overlay = EquityCurvePause(pause_threshold=0.10)

    def test_name(self):
        assert self.overlay.name == "EQUITY_CURVE_PAUSE"

    def test_allows_when_no_drawdown(self):
        equity = [10_000.0, 10_100.0, 10_200.0]
        allowed, scale, _ = self.overlay.evaluate(10, [], equity, [])
        assert allowed is True
        assert scale == 1.0

    def test_blocks_when_drawdown_exceeds_threshold(self):
        # Peak was 10_000, now at 8_900 = 11% drawdown
        equity = [10_000.0, 9_500.0, 8_900.0]
        # First call establishes peak at 10_000
        self.overlay.evaluate(1, [], [10_000.0], [])
        allowed, scale, reason = self.overlay.evaluate(10, [], equity, [])
        assert allowed is False
        assert scale == 0.0
        assert "pause" in reason.lower()

    def test_allows_exactly_at_threshold_below(self):
        # 9.9% drawdown — just below threshold
        equity = [10_000.0, 9_010.0]
        self.overlay.evaluate(1, [], [10_000.0], [])
        allowed, _, _ = self.overlay.evaluate(10, [], equity, [])
        assert allowed is True

    def test_blocks_at_exactly_threshold(self):
        # Exactly 10% drawdown
        equity = [10_000.0, 9_000.0]
        self.overlay.evaluate(1, [], [10_000.0], [])
        allowed, scale, _ = self.overlay.evaluate(10, [], equity, [])
        assert allowed is False

    def test_resumes_after_recovery(self):
        # Step 1: drawdown to 8_900 (blocks)
        self.overlay.evaluate(1, [], [10_000.0], [])
        allowed, _, _ = self.overlay.evaluate(2, [], [10_000.0, 8_900.0], [])
        assert allowed is False

        # Step 2: equity recovers above peak
        allowed, scale, _ = self.overlay.evaluate(3, [], [10_000.0, 8_900.0, 10_100.0], [])
        assert allowed is True
        assert scale == 1.0

    def test_reset_clears_state(self):
        # Establish a drawdown
        self.overlay.evaluate(1, [], [10_000.0], [])
        self.overlay.evaluate(2, [], [10_000.0, 8_900.0], [])
        self.overlay.reset()
        # After reset, fresh start — new equity series establishes new peak
        allowed, _, _ = self.overlay.evaluate(3, [], [8_900.0], [])
        assert allowed is True

    def test_allows_empty_equity(self):
        allowed, scale, _ = self.overlay.evaluate(0, [], [], [])
        assert allowed is True

    def test_reason_is_string(self):
        _, _, reason = self.overlay.evaluate(0, [], [10_000.0], [])
        assert isinstance(reason, str)
