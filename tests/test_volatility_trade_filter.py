"""Tests for VolatilityTradeFilter — ATR% category gating."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.data.models import Candle
from src.refinement.volatility_trade_filter import VolatilityFilterMode, VolatilityTradeFilter
from src.research.volatility_filter import VolatilityCategory


def _candles(n=25, price=100.0, amplitude=5.0) -> List[Candle]:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(symbol="SPY", timeframe="D1",
               timestamp=base + timedelta(days=i),
               open=price, high=price + amplitude, low=price - amplitude,
               close=price, volume=1e8, provider="TEST")
        for i in range(n)
    ]


class TestVolatilityTradeFilter:

    def test_none_mode_always_passes(self):
        f = VolatilityTradeFilter(mode=VolatilityFilterMode.NONE)
        assert f.passes(_candles(n=5)) is True
        assert f.passes(_candles(n=30)) is True

    def test_medium_only_blocks_high_volatility(self):
        # amplitude=20 on price=100 → ATR% ≈ 40% (HIGH)
        f = VolatilityTradeFilter(
            mode=VolatilityFilterMode.MEDIUM_ONLY,
            atr_period=5, low_threshold=0.5, high_threshold=5.0,
        )
        candles = _candles(n=25, price=100.0, amplitude=30.0)
        result = f.passes(candles)
        # ATR% > high_threshold → blocked
        assert result is False

    def test_medium_only_blocks_low_volatility(self):
        # amplitude=0.1 on price=100 → ATR% ≈ 0.2% (LOW)
        f = VolatilityTradeFilter(
            mode=VolatilityFilterMode.MEDIUM_ONLY,
            atr_period=5, low_threshold=1.0, high_threshold=5.0,
        )
        candles = _candles(n=25, price=100.0, amplitude=0.1)
        result = f.passes(candles)
        # ATR% < low_threshold → blocked
        assert result is False

    def test_low_and_medium_allows_low_vol(self):
        # amplitude=0.1 → LOW vol; LOW+MEDIUM mode should pass
        f = VolatilityTradeFilter(
            mode=VolatilityFilterMode.LOW_AND_MEDIUM,
            atr_period=5, low_threshold=5.0, high_threshold=20.0,
        )
        candles = _candles(n=25, amplitude=0.05)
        result = f.passes(candles)
        # ATR% very low → LOW category → allowed in LOW_AND_MEDIUM mode
        assert result is True

    def test_insufficient_data_returns_false(self):
        f = VolatilityTradeFilter(
            mode=VolatilityFilterMode.MEDIUM_ONLY, atr_period=14
        )
        candles = _candles(n=5)  # less than period
        assert f.passes(candles) is False

    def test_current_category_returns_category(self):
        f = VolatilityTradeFilter(mode=VolatilityFilterMode.MEDIUM_ONLY, atr_period=5)
        candles = _candles(n=25)
        cat = f.current_category(candles)
        assert isinstance(cat, VolatilityCategory)

    def test_current_atr_pct_returns_float_or_none(self):
        f = VolatilityTradeFilter(mode=VolatilityFilterMode.NONE, atr_period=5)
        candles = _candles(n=25)
        pct = f.current_atr_pct(candles)
        if pct is not None:
            assert isinstance(pct, float)
            assert pct >= 0.0
