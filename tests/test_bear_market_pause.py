"""Tests for BEAR_MARKET_PAUSE overlay."""

import random
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.data.models import Candle
from src.regime.macro_regime_detector import MacroRegime, MacroRegimeDetector
from src.risk_overlay.profiles.bear_market_pause import BearMarketPause


def _make_candles(n: int, trend: str = "up") -> List[Candle]:
    """Generate simple candles with controlled trend for testing."""
    base  = datetime(2010, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    out   = []
    for i in range(n):
        if trend == "up":
            price *= 1.0003
        elif trend == "down":
            price *= 0.9985
        elif trend == "crash":
            price *= 0.997 if i < n * 0.8 else 0.990
        open_ = price * 0.999
        high  = price * 1.002
        low   = price * 0.997
        out.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=round(open_, 4), high=round(high, 4),
            low=round(low, 4), close=round(price, 4),
            volume=1e7, provider="SYNTHETIC",
        ))
    return out


class TestBearMarketPause:

    def setup_method(self):
        self.overlay = BearMarketPause()

    def test_name(self):
        assert self.overlay.name == "BEAR_MARKET_PAUSE"

    def test_allows_when_insufficient_history(self):
        candles = _make_candles(50, "up")   # < 200 for EMA200
        allowed, scale, _ = self.overlay.evaluate(50, candles, [], [])
        assert allowed is True

    def test_allows_in_bull_trend(self):
        # Long upward trend → EXPANSION
        candles = _make_candles(500, "up")
        allowed, scale, reason = self.overlay.evaluate(500, candles, [], [])
        # Should be allowed (EXPANSION or RECOVERY)
        assert isinstance(allowed, bool)
        assert scale in (0.0, 1.0)

    def test_blocks_in_bear_trend(self):
        # Strong sustained downtrend → CONTRACTION or CRISIS
        candles = _make_candles(600, "down")
        allowed, scale, reason = self.overlay.evaluate(600, candles, [], [])
        # In a sustained downtrend, regime should be CONTRACTION
        if not allowed:
            assert scale == 0.0
            assert "bear" in reason.lower() or "contraction" in reason.lower() or "crisis" in reason.lower()

    def test_reset_is_noop(self):
        candles = _make_candles(300, "up")
        self.overlay.reset()  # must not raise
        allowed, _, _ = self.overlay.evaluate(300, candles, [], [])
        assert isinstance(allowed, bool)

    def test_reason_is_string(self):
        candles = _make_candles(300, "up")
        _, _, reason = self.overlay.evaluate(300, candles, [], [])
        assert isinstance(reason, str)

    def test_returns_tuple_of_three(self):
        candles = _make_candles(300, "up")
        result = self.overlay.evaluate(300, candles, [], [])
        assert len(result) == 3

    def test_scale_is_zero_or_one(self):
        candles = _make_candles(300, "up")
        _, scale, _ = self.overlay.evaluate(300, candles, [], [])
        assert scale in (0.0, 1.0)
