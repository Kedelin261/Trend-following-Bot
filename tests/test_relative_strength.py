"""Tests for RelativeStrengthStrategy (Family #3).

Verifies:
- min_candles property
- No signal on insufficient history
- Signal fires in strong uptrend (top quartile returns)
- No signal in downtrend
- No signal when EMA20 not above EMA50
- Strength score bounds
- Name and description
"""

import pytest
from tests.fixtures import _candle, uptrend_candles
from src.edge_discovery.relative_strength_research import RelativeStrengthStrategy
from src.signals.models import SignalType


@pytest.fixture
def strategy():
    return RelativeStrengthStrategy()


@pytest.fixture
def strong_uptrend():
    """Variable uptrend with recent acceleration — produces top-quartile returns."""
    return _make_accelerating_candles()


def _make_accelerating_candles():
    """Build candles with slow drift then sudden acceleration."""
    from datetime import datetime, timedelta, timezone
    from src.data.models import Candle
    base_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 400.0
    # Phase 1: slow drift (160 bars — builds distribution of low returns)
    for i in range(160):
        price *= 1.0003  # 0.03% daily
        candles.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=base_dt + timedelta(days=i),
            open=price * 0.999, high=price * 1.003, low=price * 0.997,
            close=round(price, 4), volume=1e8, provider="TEST",
        ))
    # Phase 2: sharp acceleration (40 bars — recent returns well above distribution)
    for i in range(40):
        price *= 1.006  # 0.6% daily — 20x the base rate
        candles.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=base_dt + timedelta(days=160 + i),
            open=price * 0.999, high=price * 1.008, low=price * 0.997,
            close=round(price, 4), volume=1e8, provider="TEST",
        ))
    return candles


class TestRelativeStrengthBasics:
    def test_name(self, strategy):
        assert strategy.name == "RELATIVE_STRENGTH"

    def test_description_not_empty(self, strategy):
        assert len(strategy.description) > 10

    def test_min_candles_at_least_100(self, strategy):
        assert strategy.min_candles >= 100

    def test_no_signal_empty(self, strategy):
        sig = strategy.generate_signal([])
        assert sig.signal_type == SignalType.NONE

    def test_no_signal_insufficient(self, strategy):
        candles = uptrend_candles(n=strategy.min_candles - 1)
        sig = strategy.generate_signal(candles)
        assert sig.signal_type == SignalType.NONE


class TestRelativeStrengthSignal:
    def test_signal_fires_in_strong_uptrend(self, strategy, strong_uptrend):
        """Accelerating uptrend → recent returns in top quartile → should signal."""
        found = False
        for i in range(strategy.min_candles, len(strong_uptrend)):
            sig = strategy.generate_signal(strong_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                found = True
                break
        assert found, "Expected LONG signal in accelerating uptrend"

    def test_no_signal_in_downtrend(self, strategy):
        from tests.fixtures import downtrend_candles
        candles = downtrend_candles(n=250, start=500.0, daily_loss=0.002)
        for i in range(strategy.min_candles, len(candles)):
            sig = strategy.generate_signal(candles[:i + 1])
            assert sig.signal_type == SignalType.NONE

    def test_strength_at_least_60(self, strategy, strong_uptrend):
        for i in range(strategy.min_candles, len(strong_uptrend)):
            sig = strategy.generate_signal(strong_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.strength_score >= 60.0
                break

    def test_strength_not_above_100(self, strategy, strong_uptrend):
        for i in range(strategy.min_candles, len(strong_uptrend)):
            sig = strategy.generate_signal(strong_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.strength_score <= 100.0

    def test_signal_symbol_correct(self, strategy, strong_uptrend):
        for i in range(strategy.min_candles, len(strong_uptrend)):
            sig = strategy.generate_signal(strong_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.symbol == strong_uptrend[0].symbol
                break

    def test_never_raises_on_random(self, strategy):
        import random
        rng = random.Random(123)
        candles = [_candle(close=rng.uniform(1.0, 600.0), i=i) for i in range(300)]
        sig = strategy.generate_signal(candles)
        assert sig is not None

    def test_never_raises_on_uptrend(self, strategy, strong_uptrend):
        sig = strategy.generate_signal(strong_uptrend)
        assert sig is not None

    def test_ema_structure_required(self, strategy):
        """Close must be above both EMAs for signal — test with EMA20 above EMA50."""
        # A candle series where EMA20 < EMA50 (short MA below long MA = downtrend)
        from tests.fixtures import downtrend_candles
        candles = downtrend_candles(n=200, start=400.0, daily_loss=0.001)
        found_signal = False
        for i in range(strategy.min_candles, len(candles)):
            sig = strategy.generate_signal(candles[:i + 1])
            if sig.signal_type == SignalType.LONG:
                found_signal = True
                break
        # In downtrend, EMA structure will not be aligned — should produce no signal
        assert not found_signal
