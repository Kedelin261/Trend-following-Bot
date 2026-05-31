"""Tests for VolatilityRegimeDetector — ATR% classification."""

import pytest
from datetime import datetime, timedelta, timezone

from src.data.models import Candle
from src.regime.volatility_regime_detector import VolatilityRegime, VolatilityRegimeDetector


BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _candles(n=25, price=100.0, amplitude=5.0):
    return [
        Candle(symbol="SPY", timeframe="D1",
               timestamp=BASE + timedelta(days=i),
               open=price, high=price + amplitude, low=price - amplitude,
               close=price, volume=1e8, provider="TEST")
        for i in range(n)
    ]


@pytest.fixture
def det() -> VolatilityRegimeDetector:
    return VolatilityRegimeDetector(
        period=5, low_thresh=0.5, high_thresh=4.0, extreme_thresh=8.0
    )


class TestVolatilityRegimeDetector:

    def test_returns_unknown_for_insufficient_data(self, det):
        candles = _candles(n=3)
        assert det.classify(candles) == VolatilityRegime.UNKNOWN

    def test_tiny_amplitude_is_low_vol(self, det):
        candles = _candles(n=20, amplitude=0.1)  # ATR ≈ 0.2% → LOW
        result = det.classify(candles)
        assert result == VolatilityRegime.LOW_VOL

    def test_high_amplitude_is_high_vol(self, det):
        candles = _candles(n=20, price=100.0, amplitude=5.0)  # ATR ≈ 10% → HIGH/EXTREME
        result = det.classify(candles)
        assert result in (VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL)

    def test_extreme_amplitude_is_extreme_vol(self, det):
        candles = _candles(n=20, price=100.0, amplitude=10.0)  # ATR ≈ 20% → EXTREME
        result = det.classify(candles)
        assert result == VolatilityRegime.EXTREME_VOL

    def test_atr_pct_returns_float(self, det):
        candles = _candles(n=20)
        pct = det.atr_pct(candles)
        assert pct is not None
        assert isinstance(pct, float)
        assert pct > 0

    def test_atr_pct_none_for_insufficient_data(self, det):
        candles = _candles(n=2)
        assert det.atr_pct(candles) is None

    def test_classify_at_timestamp(self, det):
        candles = _candles(n=25)
        ts = candles[-1].timestamp
        result = det.classify_at_timestamp(candles, ts)
        assert result in list(VolatilityRegime)

    def test_higher_amplitude_gives_higher_vol_category(self, det):
        low_vol  = _candles(n=20, amplitude=0.1)
        high_vol = _candles(n=20, amplitude=8.0)
        l_cat = det.classify(low_vol)
        h_cat = det.classify(high_vol)
        order = {
            VolatilityRegime.LOW_VOL: 0,
            VolatilityRegime.NORMAL_VOL: 1,
            VolatilityRegime.HIGH_VOL: 2,
            VolatilityRegime.EXTREME_VOL: 3,
            VolatilityRegime.UNKNOWN: -1,
        }
        assert order.get(h_cat, -1) >= order.get(l_cat, -1)


class TestVolatilityRegimeEnum:

    def test_four_levels_defined(self):
        vals = {r.value for r in VolatilityRegime}
        assert {"LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "EXTREME_VOL", "UNKNOWN"}.issubset(vals)
