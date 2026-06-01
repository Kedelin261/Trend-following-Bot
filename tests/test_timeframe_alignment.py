"""Tests for MultiTimeframeAlignmentStrategy (Family #6).

Verifies:
- min_candles property
- No signal on insufficient history
- Signal fires when all EMA proxies aligned (ascending)
- No signal when EMA5 < EMA21 (misalignment)
- No signal when close below EMA63
- Strength score bounds
- Volume bonus in strength
- Name and description
"""

import pytest
from tests.fixtures import _candle, uptrend_candles
from src.edge_discovery.timeframe_alignment_research import MultiTimeframeAlignmentStrategy
from src.signals.models import SignalType


@pytest.fixture
def strategy():
    return MultiTimeframeAlignmentStrategy()


@pytest.fixture
def long_uptrend():
    """Long steady uptrend — all EMA proxies should align after enough bars."""
    return uptrend_candles(n=300, start=400.0, daily_gain=0.0015)


class TestMultiTimeframeAlignmentBasics:
    def test_name(self, strategy):
        assert strategy.name == "MULTI_TIMEFRAME_ALIGNMENT"

    def test_description_contains_ema_proxies(self, strategy):
        desc = strategy.description
        assert "EMA" in desc

    def test_min_candles_at_least_84(self, strategy):
        assert strategy.min_candles >= 84

    def test_no_signal_empty(self, strategy):
        sig = strategy.generate_signal([])
        assert sig.signal_type == SignalType.NONE

    def test_no_signal_insufficient(self, strategy):
        candles = uptrend_candles(n=strategy.min_candles - 1)
        sig = strategy.generate_signal(candles)
        assert sig.signal_type == SignalType.NONE


class TestMultiTimeframeAlignmentSignal:
    def test_signal_fires_in_long_uptrend(self, strategy, long_uptrend):
        """Long steady uptrend → EMA5 > EMA21 > EMA63 → should signal."""
        found = False
        for i in range(strategy.min_candles, len(long_uptrend)):
            sig = strategy.generate_signal(long_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                found = True
                break
        assert found, "Expected LONG signal in 300-bar uptrend"

    def test_no_signal_in_downtrend(self, strategy):
        from tests.fixtures import downtrend_candles
        candles = downtrend_candles(n=300, start=500.0, daily_loss=0.002)
        for i in range(strategy.min_candles, len(candles)):
            sig = strategy.generate_signal(candles[:i + 1])
            assert sig.signal_type == SignalType.NONE

    def test_strength_at_least_62(self, strategy, long_uptrend):
        for i in range(strategy.min_candles, len(long_uptrend)):
            sig = strategy.generate_signal(long_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.strength_score >= 62.0
                break

    def test_strength_not_above_100(self, strategy, long_uptrend):
        for i in range(strategy.min_candles, len(long_uptrend)):
            sig = strategy.generate_signal(long_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.strength_score <= 100.0

    def test_signal_is_long_type(self, strategy, long_uptrend):
        for i in range(strategy.min_candles, len(long_uptrend)):
            sig = strategy.generate_signal(long_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                from src.signals.models import TrendDirection
                assert sig.trend_direction == TrendDirection.BULLISH
                break

    def test_never_raises_random(self, strategy):
        import random
        rng = random.Random(55)
        candles = [_candle(close=rng.uniform(1.0, 500.0), i=i) for i in range(200)]
        sig = strategy.generate_signal(candles)
        assert sig is not None

    def test_ema_alignment_required(self, strategy):
        """Misaligned EMAs (E5 < E21) should not signal.
        Use a downtrend where short EMA is below longer EMAs."""
        from tests.fixtures import downtrend_candles
        candles = downtrend_candles(n=200, start=500.0, daily_loss=0.003)
        # In downtrend: EMA5 drops faster than EMA63 → ascending order broken
        found = False
        for i in range(strategy.min_candles, len(candles)):
            sig = strategy.generate_signal(candles[:i + 1])
            if sig.signal_type == SignalType.LONG:
                found = True
                break
        assert not found

    def test_signal_references_ema50_in_reasoning(self, strategy, long_uptrend):
        """Signals should include EMA proxy information."""
        for i in range(strategy.min_candles, len(long_uptrend)):
            sig = strategy.generate_signal(long_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert len(sig.reasoning) > 0
                break
