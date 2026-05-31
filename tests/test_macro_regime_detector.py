"""Tests for MacroRegimeDetector — EXPANSION/RECOVERY/CONTRACTION/CRISIS."""

import pytest
from datetime import datetime, timedelta, timezone

from src.data.models import Candle
from src.regime.macro_regime_detector import MacroRegime, MacroRegimeDetector
from tests.fixtures import downtrend_candles, uptrend_candles


@pytest.fixture
def det() -> MacroRegimeDetector:
    return MacroRegimeDetector(
        ema_period=20,
        slope_lookback=5,
        crisis_decline=0.20,
        crisis_lookback=15,
    )


BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _candles_from_prices(prices):
    return [
        Candle(symbol="SPY", timeframe="D1",
               timestamp=BASE + timedelta(days=i),
               open=p, high=p * 1.005, low=p * 0.995,
               close=p, volume=1e8, provider="TEST")
        for i, p in enumerate(prices)
    ]


class TestMacroRegimeDetector:

    def test_insufficient_candles_returns_unknown(self, det):
        candles = uptrend_candles(n=10)
        assert det.classify(candles) == MacroRegime.UNKNOWN

    def test_sustained_uptrend_is_expansion(self, det):
        candles = uptrend_candles(n=50, daily_gain=0.003)
        result = det.classify(candles)
        assert result in (MacroRegime.EXPANSION, MacroRegime.RECOVERY)

    def test_sustained_downtrend_is_contraction(self, det):
        candles = downtrend_candles(n=50, daily_loss=0.003)
        result = det.classify(candles)
        assert result == MacroRegime.CONTRACTION

    def test_rapid_decline_is_crisis(self, det):
        # 5 flat bars then a 50% crash within the 15-bar crisis lookback window
        # Peak 100 → current 50 in 14 bars → 50% decline → CRISIS
        prices = (
            [100.0] * 5
            + [100.0, 95.0, 85.0, 75.0, 70.0, 65.0, 60.0, 55.0,
               52.0, 50.0, 49.0, 48.0, 47.0, 46.0, 45.0]
        )
        candles = _candles_from_prices(prices)
        result = det.classify(candles)
        assert result == MacroRegime.CRISIS

    def test_classify_at_timestamp(self, det):
        candles = uptrend_candles(n=50)
        ts = candles[-1].timestamp
        result = det.classify_at_timestamp(candles, ts)
        assert result in list(MacroRegime)

    def test_early_timestamp_is_unknown(self, det):
        candles = uptrend_candles(n=50)
        early_ts = candles[5].timestamp
        result = det.classify_at_timestamp(candles, early_ts)
        assert result == MacroRegime.UNKNOWN

    def test_no_crisis_for_small_decline(self, det):
        # 10% decline — below crisis threshold of 20%
        prices = [100.0] * 20 + [92.0, 91.0, 90.5, 90.0]
        candles = _candles_from_prices(prices)
        result = det.classify(candles)
        assert result != MacroRegime.CRISIS


class TestMacroRegimeEnum:

    def test_four_states_defined(self):
        vals = {r.value for r in MacroRegime}
        assert {"EXPANSION", "RECOVERY", "CONTRACTION", "CRISIS", "UNKNOWN"}.issubset(vals)
