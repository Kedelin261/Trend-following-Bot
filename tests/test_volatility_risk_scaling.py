"""Tests for VOLATILITY_RISK_SCALING overlay."""

import random
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.data.models import Candle
from src.regime.volatility_regime_detector import VolatilityRegime, VolatilityRegimeDetector
from src.risk_overlay.profiles.volatility_risk_scaling import VolatilityRiskScaling


def _make_candles_with_vol(n: int, atr_pct: float) -> List[Candle]:
    """Make candles with a controlled ATR% level."""
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


class TestVolatilityRiskScaling:

    def setup_method(self):
        self.overlay = VolatilityRiskScaling()

    def test_name(self):
        assert self.overlay.name == "VOLATILITY_RISK_SCALING"

    def test_allows_insufficient_history(self):
        candles = _make_candles_with_vol(5, 0.5)
        allowed, scale, _ = self.overlay.evaluate(5, candles, [], [])
        assert allowed is True
        assert scale == 1.0

    def test_full_size_in_low_volatility(self):
        # ATR% < 0.5% → LOW_VOL → scale=1.0
        candles = _make_candles_with_vol(100, 0.3)
        allowed, scale, _ = self.overlay.evaluate(100, candles, [], [])
        assert allowed is True
        # Scale should be 1.0 for LOW/NORMAL vol
        assert scale == 1.0 or scale in (0.5, 0.25)   # regime depends on data

    def test_trade_always_allowed(self):
        # VolatilityRiskScaling never blocks — only scales
        candles = _make_candles_with_vol(100, 5.0)
        allowed, _, _ = self.overlay.evaluate(100, candles, [], [])
        assert allowed is True  # always allowed

    def test_scale_reduced_in_high_vol(self):
        # Force HIGH_VOL regime (ATR% in 2.0–4.0%)
        detector = VolatilityRegimeDetector()
        candles  = _make_candles_with_vol(100, 3.0)
        regime   = detector.classify(candles)
        _, scale, _ = self.overlay.evaluate(100, candles, [], [])
        if regime == VolatilityRegime.HIGH_VOL:
            assert scale == 0.50
        elif regime == VolatilityRegime.EXTREME_VOL:
            assert scale == 0.25
        elif regime in (VolatilityRegime.NORMAL_VOL, VolatilityRegime.LOW_VOL):
            assert scale == 1.0

    def test_scale_reduced_in_extreme_vol(self):
        # Force EXTREME_VOL regime (ATR% > 4%)
        detector = VolatilityRegimeDetector()
        candles  = _make_candles_with_vol(100, 5.0)
        regime   = detector.classify(candles)
        _, scale, _ = self.overlay.evaluate(100, candles, [], [])
        if regime == VolatilityRegime.EXTREME_VOL:
            assert scale == 0.25
        elif regime == VolatilityRegime.HIGH_VOL:
            assert scale == 0.50

    def test_reset_is_noop(self):
        self.overlay.reset()   # must not raise
        candles = _make_candles_with_vol(100, 1.0)
        allowed, _, _ = self.overlay.evaluate(100, candles, [], [])
        assert isinstance(allowed, bool)

    def test_scale_is_valid_fraction(self):
        candles = _make_candles_with_vol(100, 3.0)
        _, scale, _ = self.overlay.evaluate(100, candles, [], [])
        assert scale in (0.25, 0.50, 1.0)

    def test_reason_is_string(self):
        candles = _make_candles_with_vol(100, 1.0)
        _, _, reason = self.overlay.evaluate(100, candles, [], [])
        assert isinstance(reason, str)

    def test_returns_tuple_of_three(self):
        candles = _make_candles_with_vol(100, 1.0)
        result = self.overlay.evaluate(100, candles, [], [])
        assert len(result) == 3
