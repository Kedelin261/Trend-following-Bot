"""Tests for SignalEngine — end-to-end signal generation."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.data.models import Candle
from src.signals.breakout_detector import BreakoutDetector
from src.signals.models import SignalType, TrendDirection
from src.signals.signal_engine import SignalEngine
from src.signals.signal_scorer import SignalScorer
from src.signals.support_resistance import SupportResistanceDetector
from src.signals.trend_detector import TrendDetector
from src.signals.volume_confirmation import VolumeConfirmation
from tests.fixtures import (
    downtrend_candles,
    flat_candles,
    uptrend_candles,
    zero_volume_candles,
)


@pytest.fixture
def engine() -> SignalEngine:
    """Engine with min_candles=210 (default)."""
    return SignalEngine()


def _minimal_engine() -> SignalEngine:
    """Engine wired for small datasets (lookback=1, 30 min candles)."""
    return SignalEngine(
        trend_detector=TrendDetector(fast_period=5, slow_period=10),
        sr_detector=SupportResistanceDetector(lookback=1),
        breakout_detector=BreakoutDetector(threshold=0.001),
        volume_confirmation=VolumeConfirmation(lookback=3),
        min_candles=12,
    )


# ---------------------------------------------------------------------------
# Edge cases — no / insufficient data
# ---------------------------------------------------------------------------

class TestSignalEngineEdgeCases:

    def test_empty_candles_returns_signal(self, engine):
        signal = engine.generate_signal([])
        assert signal is not None
        assert signal.signal_type == SignalType.NONE

    def test_insufficient_candles_returns_none_signal(self, engine):
        candles = uptrend_candles(n=50)
        signal = engine.generate_signal(candles)
        assert signal.signal_type == SignalType.NONE
        assert "Insufficient" in signal.reasoning[0]

    def test_insufficient_signal_has_valid_close(self, engine):
        candles = uptrend_candles(n=50)
        signal = engine.generate_signal(candles)
        assert signal.close_price > 0

    def test_empty_signal_has_unknown_symbol(self, engine):
        signal = engine.generate_signal([])
        assert signal.symbol == "UNKNOWN"


# ---------------------------------------------------------------------------
# Signal fields completeness
# ---------------------------------------------------------------------------

class TestSignalFields:

    def test_all_required_fields_populated(self):
        engine = _minimal_engine()
        candles = uptrend_candles(n=50, symbol="SPY")
        signal = engine.generate_signal(candles)

        assert signal.symbol    == "SPY"
        assert signal.timeframe == "D1"
        assert isinstance(signal.timestamp, datetime)
        assert isinstance(signal.signal_type, SignalType)
        assert isinstance(signal.trend_direction, TrendDirection)
        assert 0.0 <= signal.strength_score <= 100.0
        assert isinstance(signal.breakout_detected, bool)
        assert isinstance(signal.reasoning, list)
        assert len(signal.reasoning) > 0

    def test_strength_category_derivable(self):
        engine = _minimal_engine()
        candles = uptrend_candles(n=50)
        signal = engine.generate_signal(candles)
        # Property should not raise
        _ = signal.strength_category

    def test_is_actionable_false_for_none_signal(self):
        engine = _minimal_engine()
        candles = flat_candles(n=50)
        signal = engine.generate_signal(candles)
        if signal.signal_type == SignalType.NONE:
            assert signal.is_actionable is False

    def test_is_actionable_true_for_long_short(self):
        from src.signals.models import Signal, SignalType, TrendDirection
        sig = Signal(
            symbol="SPY", timeframe="D1",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            signal_type=SignalType.LONG,
            trend_direction=TrendDirection.BULLISH,
            strength_score=85.0,
            breakout_detected=True,
            volume_confirmed=True,
            support_level=390.0,
            resistance_level=410.0,
            close_price=412.0,
        )
        assert sig.is_actionable is True


# ---------------------------------------------------------------------------
# Trend direction passes through to signal
# ---------------------------------------------------------------------------

class TestSignalTrendPassthrough:

    def test_neutral_trend_produces_none_signal(self):
        engine = _minimal_engine()
        candles = flat_candles(n=50)
        signal = engine.generate_signal(candles)
        # Flat candles → neutral or no breakout → NONE
        assert signal.signal_type == SignalType.NONE

    def test_bullish_trend_reflected_in_signal(self):
        engine = _minimal_engine()
        candles = uptrend_candles(n=50, daily_gain=0.003)
        signal = engine.generate_signal(candles)
        assert signal.trend_direction == TrendDirection.BULLISH

    def test_bearish_trend_reflected_in_signal(self):
        engine = _minimal_engine()
        candles = downtrend_candles(n=50, daily_loss=0.003)
        signal = engine.generate_signal(candles)
        assert signal.trend_direction == TrendDirection.BEARISH


# ---------------------------------------------------------------------------
# Volume handling
# ---------------------------------------------------------------------------

class TestSignalVolumeHandling:

    def test_forex_zero_volume_does_not_block_signal(self):
        """Signals can still be LONG/SHORT when volume is UNKNOWN (Forex)."""
        engine = _minimal_engine()
        candles = zero_volume_candles(n=50, daily_gain=0.003, symbol="EURUSD")
        signal = engine.generate_signal(candles)
        # volume_confirmed should be None, not False
        assert signal.volume_confirmed is None

    def test_unknown_volume_signal_not_penalised(self):
        engine = _minimal_engine()
        candles_vol  = uptrend_candles(n=50, daily_gain=0.003, volume=1e8)
        candles_zero = zero_volume_candles(n=50, daily_gain=0.003)
        sig_vol  = engine.generate_signal(candles_vol)
        sig_zero = engine.generate_signal(candles_zero)
        # Both should produce the same signal_type (zero volume doesn't block)
        assert sig_zero.signal_type == sig_vol.signal_type


# ---------------------------------------------------------------------------
# Signal score is always bounded
# ---------------------------------------------------------------------------

class TestSignalScoreBounds:

    def test_score_between_0_and_100(self):
        engine = _minimal_engine()
        for fixture in [
            uptrend_candles(n=50),
            downtrend_candles(n=50),
            flat_candles(n=50),
        ]:
            signal = engine.generate_signal(fixture)
            assert 0.0 <= signal.strength_score <= 100.0

    def test_none_signal_score_is_zero(self):
        engine = _minimal_engine()
        signal = engine.generate_signal(flat_candles(n=50))
        if signal.signal_type == SignalType.NONE:
            assert signal.strength_score == 0.0


# ---------------------------------------------------------------------------
# Signal rule: LONG only when bullish AND breakout AND vol OK
# ---------------------------------------------------------------------------

class TestSignalRules:

    def test_long_requires_bullish_trend(self):
        engine = _minimal_engine()
        # Downtrend should not produce LONG
        candles = downtrend_candles(n=50, daily_loss=0.002)
        signal = engine.generate_signal(candles)
        assert signal.signal_type != SignalType.LONG

    def test_short_requires_bearish_trend(self):
        engine = _minimal_engine()
        # Uptrend should not produce SHORT
        candles = uptrend_candles(n=50, daily_gain=0.002)
        signal = engine.generate_signal(candles)
        assert signal.signal_type != SignalType.SHORT

    def test_symbol_propagated_to_signal(self):
        engine = _minimal_engine()
        candles = uptrend_candles(n=50, symbol="QQQ")
        signal = engine.generate_signal(candles)
        assert signal.symbol == "QQQ"

    def test_timeframe_propagated_to_signal(self):
        engine = _minimal_engine()
        # uptrend_candles returns D1 timeframe
        candles = uptrend_candles(n=50)
        signal = engine.generate_signal(candles)
        assert signal.timeframe == "D1"


# ---------------------------------------------------------------------------
# EMA values on signal
# ---------------------------------------------------------------------------

class TestSignalEMAValues:

    def test_ema50_ema200_present_when_sufficient_data(self, engine):
        candles = uptrend_candles(n=250)
        signal = engine.generate_signal(candles)
        assert signal.ema50  is not None
        assert signal.ema200 is not None
        assert signal.ema50 > 0
        assert signal.ema200 > 0

    def test_ema50_above_ema200_in_bullish_signal(self, engine):
        candles = uptrend_candles(n=250)
        signal = engine.generate_signal(candles)
        if signal.trend_direction == TrendDirection.BULLISH:
            assert signal.ema50 > signal.ema200

    def test_ema_none_on_insufficient_data(self, engine):
        candles = uptrend_candles(n=50)
        signal = engine.generate_signal(candles)
        # Insufficient → engine returns early without computing EMAs
        assert signal.ema50  is None
        assert signal.ema200 is None


# ---------------------------------------------------------------------------
# Provider-agnostic: engine ignores candle.provider field
# ---------------------------------------------------------------------------

class TestSignalEngineAgnostic:

    def test_works_with_mt5_provider_tag(self):
        engine = _minimal_engine()
        candles = uptrend_candles(n=50)
        for c in candles:
            object.__setattr__(c, "provider", "MT5")
        signal = engine.generate_signal(candles)
        assert signal is not None

    def test_works_with_ibkr_provider_tag(self):
        engine = _minimal_engine()
        candles = uptrend_candles(n=50)
        for c in candles:
            object.__setattr__(c, "provider", "IBKR")
        signal = engine.generate_signal(candles)
        assert signal is not None

    def test_no_broker_imports_in_signal_modules(self):
        """Ensure no IBKR / MT5 references inside the signals package."""
        import importlib, pathlib
        signals_dir = pathlib.Path("src/signals")
        for pyfile in signals_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "MetaTrader" not in text, f"{pyfile} imports MetaTrader"
            assert "ib_insync"  not in text, f"{pyfile} imports ib_insync"
            assert "ibkr"       not in text.lower() or "ibkr" in pyfile.name.lower()
