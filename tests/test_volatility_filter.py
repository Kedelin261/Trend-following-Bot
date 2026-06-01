"""Tests for VolatilityFilter — Phase 5.4."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from src.amplification_validation.volatility_filter import VolatilityFilter
from src.regime.volatility_regime_detector import VolatilityRegime
from src.data.models import Candle


def _candles(n: int = 20, atr_pct: float = 1.0, symbol: str = "SPY") -> list:
    """Generate candles with controlled ATR% for deterministic regime testing."""
    import random
    rng = random.Random(42)
    base_price = 400.0
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    out = []
    # Use a spread that gives approximately the requested ATR% of price
    # ATR% = ATR/close * 100 ; ATR ≈ (high-low) * 0.7 for simple approximation
    spread = base_price * (atr_pct / 100.0) / 0.7
    for i in range(n):
        close = base_price * (1 + rng.gauss(0, 0.001))
        high  = close + spread * 0.5
        low   = close - spread * 0.5
        out.append(Candle(
            symbol=symbol, timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=close, high=high, low=low, close=close,
            volume=1e6, provider="TEST",
        ))
    return out


class TestVolatilityFilter:

    def test_allows_when_no_candles(self):
        """Empty candle list → allow (cannot classify)."""
        f = VolatilityFilter()
        assert f.allows([]) is True

    def test_blocks_high_vol_regime(self):
        """Filter must block HIGH_VOL regime."""
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.HIGH_VOL
        f = VolatilityFilter(detector=mock_det)
        candles = _candles()
        assert f.allows(candles) is False

    def test_allows_normal_vol_regime(self):
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.NORMAL_VOL
        f = VolatilityFilter(detector=mock_det)
        assert f.allows(_candles()) is True

    def test_allows_low_vol_regime(self):
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.LOW_VOL
        f = VolatilityFilter(detector=mock_det)
        assert f.allows(_candles()) is True

    def test_allows_extreme_vol_regime(self):
        """EXTREME_VOL is not in the blocked set for Phase 5.4 — only HIGH_VOL."""
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.EXTREME_VOL
        f = VolatilityFilter(detector=mock_det)
        assert f.allows(_candles()) is True

    def test_allows_unknown_regime(self):
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.UNKNOWN
        f = VolatilityFilter(detector=mock_det)
        assert f.allows(_candles()) is True

    def test_detector_called_with_candles(self):
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.NORMAL_VOL
        f = VolatilityFilter(detector=mock_det)
        candles = _candles()
        f.allows(candles)
        mock_det.classify.assert_called_once_with(candles)

    def test_default_detector_created(self):
        """No-arg constructor creates a real VolatilityRegimeDetector."""
        from src.regime.volatility_regime_detector import VolatilityRegimeDetector
        f = VolatilityFilter()
        assert isinstance(f.detector, VolatilityRegimeDetector)

    def test_blocked_regimes_contains_high_vol(self):
        assert VolatilityRegime.HIGH_VOL in VolatilityFilter.BLOCKED_REGIMES

    def test_blocked_regimes_does_not_contain_normal(self):
        assert VolatilityRegime.NORMAL_VOL not in VolatilityFilter.BLOCKED_REGIMES

    def test_classify_returns_regime(self):
        mock_det = MagicMock()
        mock_det.classify.return_value = VolatilityRegime.HIGH_VOL
        f = VolatilityFilter(detector=mock_det)
        assert f.classify(_candles()) == VolatilityRegime.HIGH_VOL

    def test_classify_empty_candles_returns_unknown(self):
        f = VolatilityFilter()
        assert f.classify([]) == VolatilityRegime.UNKNOWN

    def test_repr(self):
        f = VolatilityFilter()
        assert "HIGH_VOL" in repr(f)
        assert "VolatilityFilter" in repr(f)

    def test_remove_high_vol_profile(self):
        """REMOVE_HIGH_VOL scenario has exclude_high_vol=True."""
        from src.amplification_validation.filter_profiles import SCENARIO_REMOVE_HIGH_VOL
        assert SCENARIO_REMOVE_HIGH_VOL.exclude_high_vol is True

    def test_baseline_profile_no_vol_filter(self):
        """BASELINE has exclude_high_vol=False."""
        from src.amplification_validation.filter_profiles import SCENARIO_BASELINE
        assert SCENARIO_BASELINE.exclude_high_vol is False
