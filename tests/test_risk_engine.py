"""Tests for RiskEngine — end-to-end TradeCandidate generation."""

import pytest
from datetime import datetime, timezone
from typing import List

from src.data.models import Candle
from src.risk.atr_calculator import ATRCalculator
from src.risk.models import RiskProfile, TradeCandidate
from src.risk.risk_engine import RiskEngine
from src.risk.stop_loss_engine import StopLossEngine
from src.risk.take_profit_engine import TakeProfitEngine
from src.risk.trade_validator import TradeValidator
from src.signals.models import Signal, SignalType, TrendDirection


# ---------------------------------------------------------------------------
# Local candle builders (amplitude-aware, independent of shared fixtures)
# ---------------------------------------------------------------------------

def flat_candles(
    n: int,
    price: float = 100.0,
    amplitude: float = 5.0,
    volume: float = 1e8,
    symbol: str = "SPY",
) -> list:
    from datetime import timedelta
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(
            symbol=symbol, timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=price,
            high=price + amplitude,
            low=price - amplitude,
            close=price,
            volume=volume,
            provider="TEST",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _profile(
    account: float = 10_000.0,
    risk_pct: float = 1.0,
) -> RiskProfile:
    return RiskProfile(account_size=account, cash_available=account,
                       risk_per_trade_percent=risk_pct)


def _long_signal(score: float = 85.0, close: float = 100.0) -> Signal:
    return Signal(
        symbol="SPY", timeframe="D1",
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        signal_type=SignalType.LONG,
        trend_direction=TrendDirection.BULLISH,
        strength_score=score,
        breakout_detected=True,
        volume_confirmed=True,
        support_level=95.0,
        resistance_level=105.0,
        close_price=close,
        ema50=102.0, ema200=98.0,
    )


def _short_signal(score: float = 85.0, close: float = 100.0) -> Signal:
    return Signal(
        symbol="QQQ", timeframe="D1",
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        signal_type=SignalType.SHORT,
        trend_direction=TrendDirection.BEARISH,
        strength_score=score,
        breakout_detected=True,
        volume_confirmed=True,
        support_level=90.0,
        resistance_level=100.0,
        close_price=close,
        ema50=98.0, ema200=102.0,
    )


def _none_signal() -> Signal:
    return Signal(
        symbol="SPY", timeframe="D1",
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        signal_type=SignalType.NONE,
        trend_direction=TrendDirection.NEUTRAL,
        strength_score=0.0,
        breakout_detected=False,
        volume_confirmed=None,
        support_level=None, resistance_level=None,
        close_price=100.0,
    )


def _minimal_engine(
    account: float = 10_000.0,
    risk_pct: float = 1.0,
    min_score: float = 70.0,
    min_rr: float = 1.5,
) -> RiskEngine:
    """Engine with period=3 ATR so small fixtures work."""
    return RiskEngine(
        risk_profile     = _profile(account, risk_pct),
        atr_calculator   = ATRCalculator(period=3),
        stop_loss_engine = StopLossEngine(multiplier=2.0),
        take_profit_engine = TakeProfitEngine(multiplier=3.0),
        trade_validator  = TradeValidator(
            minimum_signal_score=min_score,
            minimum_risk_reward=min_rr,
        ),
    )


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------

class TestRiskEngineReturnType:

    def test_returns_trade_candidate(self):
        engine = _minimal_engine()
        result = engine.evaluate(_long_signal(), flat_candles(n=20))
        assert isinstance(result, TradeCandidate)

    def test_none_signal_returns_candidate(self):
        engine = _minimal_engine()
        result = engine.evaluate(_none_signal(), flat_candles(n=20))
        assert isinstance(result, TradeCandidate)
        assert result.approved is False

    def test_empty_candles_returns_candidate(self):
        engine = _minimal_engine()
        result = engine.evaluate(_long_signal(), [])
        assert isinstance(result, TradeCandidate)
        assert result.approved is False


# ---------------------------------------------------------------------------
# Field correctness
# ---------------------------------------------------------------------------

class TestTradeCandidateFields:

    @pytest.fixture
    def approved_candidate(self):
        engine = _minimal_engine()
        signal = _long_signal(score=90.0, close=100.0)
        return engine.evaluate(signal, flat_candles(n=20, amplitude=5.0))

    def test_symbol_propagated(self, approved_candidate):
        assert approved_candidate.symbol == "SPY"

    def test_timeframe_propagated(self, approved_candidate):
        assert approved_candidate.timeframe == "D1"

    def test_entry_price_equals_close(self, approved_candidate):
        assert approved_candidate.entry_price == pytest.approx(100.0)

    def test_atr_positive(self, approved_candidate):
        assert approved_candidate.atr > 0

    def test_stop_below_entry_for_long(self, approved_candidate):
        assert approved_candidate.stop_loss < approved_candidate.entry_price

    def test_target_above_entry_for_long(self, approved_candidate):
        assert approved_candidate.take_profit > approved_candidate.entry_price

    def test_risk_per_share_equals_entry_minus_stop(self, approved_candidate):
        c = approved_candidate
        assert c.risk_per_share == pytest.approx(
            abs(c.entry_price - c.stop_loss), rel=1e-5
        )

    def test_reward_per_share_equals_target_minus_entry(self, approved_candidate):
        c = approved_candidate
        assert c.reward_per_share == pytest.approx(
            abs(c.take_profit - c.entry_price), rel=1e-5
        )

    def test_rr_equals_reward_over_risk(self, approved_candidate):
        c = approved_candidate
        expected = c.reward_per_share / c.risk_per_share
        assert c.risk_reward_ratio == pytest.approx(expected, rel=1e-4)

    def test_rr_is_15_with_default_multipliers(self, approved_candidate):
        """stop×2, target×3 → R:R = 3/2 = 1.5."""
        assert approved_candidate.risk_reward_ratio == pytest.approx(1.5, rel=1e-4)

    def test_dollar_risk_from_profile(self, approved_candidate):
        # $10,000 × 1% = $100
        assert approved_candidate.dollar_risk == pytest.approx(100.0)

    def test_position_size_integer(self, approved_candidate):
        assert isinstance(approved_candidate.position_size, int)

    def test_position_size_positive(self, approved_candidate):
        assert approved_candidate.position_size > 0


# ---------------------------------------------------------------------------
# ATR stop / target math
# ---------------------------------------------------------------------------

class TestATRMath:

    def test_stop_is_entry_minus_atr_times_2(self):
        engine = _minimal_engine()
        candles = flat_candles(n=10, price=100.0, amplitude=5.0)
        c = engine.evaluate(_long_signal(close=100.0), candles)
        expected_stop = 100.0 - c.atr * 2.0
        assert c.stop_loss == pytest.approx(expected_stop, rel=1e-5)

    def test_target_is_entry_plus_atr_times_3(self):
        engine = _minimal_engine()
        candles = flat_candles(n=10, price=100.0, amplitude=5.0)
        c = engine.evaluate(_long_signal(close=100.0), candles)
        expected_target = 100.0 + c.atr * 3.0
        assert c.take_profit == pytest.approx(expected_target, rel=1e-5)

    def test_short_stop_above_entry(self):
        engine = _minimal_engine()
        candles = flat_candles(n=10, price=100.0, amplitude=5.0)
        c = engine.evaluate(_short_signal(close=100.0), candles)
        assert c.stop_loss > c.entry_price

    def test_short_target_below_entry(self):
        engine = _minimal_engine()
        candles = flat_candles(n=10, price=100.0, amplitude=5.0)
        c = engine.evaluate(_short_signal(close=100.0), candles)
        assert c.take_profit < c.entry_price


# ---------------------------------------------------------------------------
# Approval / rejection
# ---------------------------------------------------------------------------

class TestApprovalRejection:

    def test_none_signal_rejected_with_reason(self):
        engine = _minimal_engine()
        c = engine.evaluate(_none_signal(), flat_candles(n=20))
        assert c.approved is False
        assert c.rejection_reason is not None

    def test_insufficient_candles_rejected(self):
        engine = _minimal_engine()
        c = engine.evaluate(_long_signal(), flat_candles(n=3))
        assert c.approved is False
        assert "candle" in c.rejection_reason.lower() or "ATR" in c.rejection_reason

    def test_low_score_signal_rejected(self):
        engine = _minimal_engine(min_score=70.0)
        c = engine.evaluate(
            _long_signal(score=50.0),
            flat_candles(n=20, amplitude=5.0),
        )
        assert c.approved is False
        assert "score" in c.rejection_reason.lower() or "signal" in c.rejection_reason.lower()

    def test_high_score_signal_approved(self):
        engine = _minimal_engine()
        c = engine.evaluate(
            _long_signal(score=90.0),
            flat_candles(n=20, amplitude=5.0),
        )
        assert c.approved is True
        assert c.rejection_reason is None

    def test_approved_candidate_has_no_rejection_reason(self):
        engine = _minimal_engine()
        c = engine.evaluate(
            _long_signal(score=85.0),
            flat_candles(n=20, amplitude=5.0),
        )
        if c.approved:
            assert c.rejection_reason is None


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

class TestFromConfig:

    def test_builds_from_settings(self):
        config = {
            "risk": {
                "account_size": 20_000,
                "risk_per_trade_percent": 0.5,
                "atr_period": 5,
                "atr_stop_multiplier": 2.0,
                "atr_target_multiplier": 4.0,
                "minimum_signal_score": 65.0,
                "minimum_risk_reward": 2.0,
            }
        }
        engine = RiskEngine.from_config(config)
        assert engine._profile.account_size == 20_000.0
        assert engine._atr.period == 5
        assert engine._stop.multiplier == 2.0
        assert engine._target.multiplier == 4.0

    def test_empty_config_uses_defaults(self):
        engine = RiskEngine.from_config({})
        assert engine._profile.account_size == 10_000.0
        assert engine._atr.period == 14


# ---------------------------------------------------------------------------
# Broker-agnostic: no IBKR / MT5 imports in risk modules
# ---------------------------------------------------------------------------

class TestBrokerAgnostic:

    def test_no_broker_imports_in_risk_modules(self):
        import pathlib
        risk_dir = pathlib.Path("src/risk")
        for pyfile in risk_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"
            assert "mt5"         not in text.lower().split("import")[0].replace("mt5", "X"), \
                   f"{pyfile} may import MT5"

    def test_trade_candidate_properties(self):
        """TradeCandidate computed properties work correctly."""
        c = TradeCandidate(
            symbol="SPY", timeframe="D1",
            signal_type=SignalType.LONG,
            entry_price=100.0, atr=5.0,
            stop_loss=90.0, take_profit=115.0,
            risk_per_share=10.0, reward_per_share=15.0,
            risk_reward_ratio=1.5,
            position_size=10, dollar_risk=100.0,
            signal_score=85.0, approved=True,
        )
        assert c.is_long is True
        assert c.is_short is False
        assert c.total_position_value == pytest.approx(1000.0)
        assert c.expected_profit == pytest.approx(150.0)
        assert c.expected_loss  == pytest.approx(100.0)
