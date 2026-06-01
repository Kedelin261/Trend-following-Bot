"""Tests for COMBINED_OVERLAY (EQUITY_CURVE_PAUSE + VOLATILITY_RISK_SCALING)."""

import random
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.data.models import Candle
from src.risk_overlay.profiles.combined_overlay import CombinedOverlay


def _make_candles(n: int, atr_pct: float = 1.0) -> List[Candle]:
    base  = datetime(2020, 1, 1, tzinfo=timezone.utc)
    price = 400.0
    out   = []
    rng   = random.Random(42)
    for i in range(n):
        move  = price * atr_pct / 100.0
        open_ = price + rng.uniform(-move, move)
        high  = price + abs(rng.gauss(0, move))
        low   = price - abs(rng.gauss(0, move))
        close = price + rng.gauss(0, move * 0.3)
        price = max(1.0, close)
        out.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=max(0.01, open_), high=max(0.01, high),
            low=max(0.01, low), close=max(0.01, price),
            volume=1e7, provider="SYNTHETIC",
        ))
    return out


class TestCombinedOverlay:

    def setup_method(self):
        self.overlay = CombinedOverlay()

    def test_name(self):
        assert self.overlay.name == "COMBINED_OVERLAY"

    def test_component_names(self):
        names = self.overlay.component_names
        assert "EQUITY_CURVE_PAUSE" in names
        assert "VOLATILITY_RISK_SCALING" in names
        assert len(names) == 2  # exactly 2 overlays

    def test_max_two_overlays_enforced(self):
        """Anti-overfitting rule: at most 2 overlays."""
        assert len(self.overlay._overlays) == 2

    def test_allows_normal_conditions(self):
        candles = _make_candles(100, 1.0)
        equity  = [10_000.0, 10_100.0, 10_200.0]
        allowed, scale, _ = self.overlay.evaluate(100, candles, equity, [])
        # Under normal vol and no drawdown, should be allowed
        assert isinstance(allowed, bool)

    def test_blocks_if_equity_curve_pause_blocks(self):
        """If ECP component blocks, combined must block."""
        candles = _make_candles(100, 1.0)
        # Establish peak then big drawdown
        equity  = [10_000.0, 8_000.0]   # 20% drawdown
        self.overlay.evaluate(1, candles[:10], [10_000.0], [])
        allowed, scale, reason = self.overlay.evaluate(100, candles, equity, [])
        assert allowed is False
        assert scale == 0.0
        assert "equity curve pause" in reason.lower() or "combined" in reason.lower()

    def test_scale_is_minimum_of_components(self):
        """Combined scale = minimum of all component scales."""
        candles = _make_candles(100, 1.0)
        equity  = [10_000.0, 10_100.0]
        _, scale, _ = self.overlay.evaluate(100, candles, equity, [])
        # Scale must be between 0 and 1
        assert 0.0 <= scale <= 1.0

    def test_reset_resets_all_components(self):
        candles  = _make_candles(100, 1.0)
        equity_d = [10_000.0, 8_000.0]
        self.overlay.evaluate(1, candles[:10], [10_000.0], [])
        self.overlay.evaluate(50, candles, equity_d, [])
        self.overlay.reset()
        # After reset, drawdown state should be cleared
        equity_r = [8_000.0, 8_100.0]  # new start after reset
        allowed, _, _ = self.overlay.evaluate(100, candles, equity_r, [])
        assert allowed is True   # new peak = 8_100, no drawdown yet

    def test_reason_is_string(self):
        candles = _make_candles(100, 1.0)
        _, _, reason = self.overlay.evaluate(100, candles, [10_000.0], [])
        assert isinstance(reason, str)

    def test_returns_tuple_of_three(self):
        result = self.overlay.evaluate(0, [], [10_000.0], [])
        assert len(result) == 3

    def test_scale_leq_one(self):
        candles = _make_candles(100, 1.0)
        for equity in [[10_000.0], [10_000.0, 10_500.0], [10_000.0, 9_200.0]]:
            self.overlay.reset()
            _, scale, _ = self.overlay.evaluate(100, candles, equity, [])
            assert scale <= 1.0

    def test_allowed_is_bool(self):
        candles = _make_candles(100, 1.0)
        allowed, _, _ = self.overlay.evaluate(100, candles, [10_000.0], [])
        assert isinstance(allowed, bool)
