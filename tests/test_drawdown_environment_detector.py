"""Tests for DrawdownEnvironmentDetector — market stress level classification."""

import pytest
from datetime import datetime, timedelta, timezone

from src.data.models import Candle
from src.regime.drawdown_environment_detector import (
    DrawdownEnvironment,
    DrawdownEnvironmentDetector,
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


@pytest.fixture
def det() -> DrawdownEnvironmentDetector:
    return DrawdownEnvironmentDetector(
        lookback=20,
        correction_thresh=0.05,
        bear_thresh=0.15,
        crash_thresh=0.30,
    )


class TestDrawdownEnvironmentDetector:

    def test_near_peak_is_bull_recovery(self, det):
        # Prices near all-time high within window
        prices = [100.0] * 10 + [102.0, 103.0, 104.0, 105.0]
        candles = _candles_from_prices(prices)
        result = det.classify(candles)
        assert result == DrawdownEnvironment.BULL_RECOVERY

    def test_ten_percent_drop_is_correction(self, det):
        # Peak 100, current 90 = 10% drawdown
        prices = [100.0] * 15 + [90.0]
        candles = _candles_from_prices(prices)
        result = det.classify(candles)
        assert result == DrawdownEnvironment.CORRECTION

    def test_twenty_percent_drop_is_bear(self, det):
        prices = [100.0] * 15 + [78.0]  # 22% drawdown
        candles = _candles_from_prices(prices)
        result = det.classify(candles)
        assert result == DrawdownEnvironment.BEAR_MARKET

    def test_forty_percent_drop_is_crash(self, det):
        prices = [100.0] * 15 + [55.0]  # 45% drawdown
        candles = _candles_from_prices(prices)
        result = det.classify(candles)
        assert result == DrawdownEnvironment.CRASH

    def test_empty_returns_unknown(self, det):
        assert det.classify([]) == DrawdownEnvironment.UNKNOWN

    def test_current_drawdown_pct_near_zero(self, det):
        prices = [100.0] * 15 + [99.0]  # 1% below peak
        candles = _candles_from_prices(prices)
        pct = det.current_drawdown_pct(candles)
        assert pct is not None
        assert pct < 0.05

    def test_classify_at_timestamp(self, det):
        prices = [100.0 + i * 0.5 for i in range(20)]
        candles = _candles_from_prices(prices)
        ts = candles[-1].timestamp
        result = det.classify_at_timestamp(candles, ts)
        assert result in list(DrawdownEnvironment)

    def test_lookback_window_limits_peak_search(self):
        # Peak very far back (beyond lookback) should not affect classification
        det_short = DrawdownEnvironmentDetector(lookback=5)
        prices = [200.0] + [100.0] * 20  # old peak 200, recent prices ~100
        candles = _candles_from_prices(prices)
        result = det_short.classify(candles)
        # With lookback=5, only recent 5 bars matter → peak ≈ 100 → bull recovery
        assert result == DrawdownEnvironment.BULL_RECOVERY


class TestDrawdownEnvironmentEnum:

    def test_four_states_defined(self):
        vals = {e.value for e in DrawdownEnvironment}
        assert {"BULL_RECOVERY", "CORRECTION", "BEAR_MARKET", "CRASH", "UNKNOWN"}.issubset(vals)
