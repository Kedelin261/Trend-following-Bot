"""Tests for TradeValidator — rejection rules and approval logic."""

import pytest
from datetime import datetime, timezone

from src.risk.models import RiskProfile, TradeCandidate
from src.risk.trade_validator import TradeValidator
from src.signals.models import SignalType


def _candidate(
    symbol:          str   = "SPY",
    signal_type:     SignalType = SignalType.LONG,
    entry_price:     float = 100.0,
    atr:             float = 5.0,
    stop_loss:       float = 90.0,
    take_profit:     float = 115.0,
    risk_per_share:  float = 10.0,
    reward_per_share:float = 15.0,
    risk_reward_ratio: float = 1.5,
    position_size:   int   = 10,
    dollar_risk:     float = 100.0,
    signal_score:    float = 80.0,
    approved:        bool  = False,
) -> TradeCandidate:
    return TradeCandidate(
        symbol=symbol, timeframe="D1",
        signal_type=signal_type,
        entry_price=entry_price,
        atr=atr,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=risk_reward_ratio,
        position_size=position_size,
        dollar_risk=dollar_risk,
        signal_score=signal_score,
        approved=approved,
        rejection_reason=None,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def validator() -> TradeValidator:
    return TradeValidator(minimum_signal_score=70.0, minimum_risk_reward=1.5)


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

class TestApproval:

    def test_valid_trade_is_approved(self, validator):
        approved, reason = validator.validate_trade(_candidate())
        assert approved is True
        assert reason is None

    def test_approved_returns_none_reason(self, validator):
        _, reason = validator.validate_trade(_candidate())
        assert reason is None

    def test_minimum_score_exactly_70_is_approved(self, validator):
        """Score == minimum (not strictly less than) → approved."""
        approved, _ = validator.validate_trade(_candidate(signal_score=70.0))
        assert approved is True

    def test_minimum_rr_exactly_15_is_approved(self, validator):
        """R:R == minimum (not strictly less than) → approved."""
        approved, _ = validator.validate_trade(
            _candidate(risk_reward_ratio=1.5, reward_per_share=15.0)
        )
        assert approved is True

    def test_short_signal_approved(self, validator):
        approved, _ = validator.validate_trade(
            _candidate(signal_type=SignalType.SHORT, stop_loss=110.0, take_profit=85.0)
        )
        assert approved is True


# ---------------------------------------------------------------------------
# Rule 1: ATR ≤ 0
# ---------------------------------------------------------------------------

class TestATRRule:

    def test_zero_atr_rejected(self, validator):
        approved, reason = validator.validate_trade(_candidate(atr=0.0))
        assert approved is False
        assert reason is not None
        assert "ATR" in reason

    def test_negative_atr_rejected(self, validator):
        approved, _ = validator.validate_trade(_candidate(atr=-1.0))
        assert approved is False

    def test_tiny_positive_atr_not_rejected_by_atr_rule(self, validator):
        # Should pass ATR rule (atr > 0), may fail other rules
        candidate = _candidate(
            atr=0.0001,
            stop_loss=99.9999,
            risk_per_share=0.0001,
            risk_reward_ratio=1.5,
            position_size=1,
        )
        approved, reason = validator.validate_trade(candidate)
        # ATR rule should not trigger
        assert reason is None or "ATR" not in reason


# ---------------------------------------------------------------------------
# Rule 2: Entry == Stop
# ---------------------------------------------------------------------------

class TestEntryEqualsStopRule:

    def test_entry_equals_stop_rejected(self, validator):
        approved, reason = validator.validate_trade(
            _candidate(entry_price=100.0, stop_loss=100.0, risk_per_share=0.0)
        )
        assert approved is False
        assert "stop" in reason.lower() or "entry" in reason.lower()

    def test_entry_nearly_equal_to_stop_rejected(self, validator):
        """Difference < epsilon is treated as equal."""
        approved, _ = validator.validate_trade(
            _candidate(entry_price=100.0, stop_loss=100.0 + 1e-10)
        )
        assert approved is False

    def test_meaningful_stop_distance_passes_rule(self, validator):
        approved, reason = validator.validate_trade(
            _candidate(entry_price=100.0, stop_loss=90.0)
        )
        # Should not be rejected by entry==stop rule
        assert reason is None or "equal" not in reason.lower()


# ---------------------------------------------------------------------------
# Rule 3: Signal Score < minimum
# ---------------------------------------------------------------------------

class TestSignalScoreRule:

    def test_score_below_minimum_rejected(self, validator):
        approved, reason = validator.validate_trade(_candidate(signal_score=69.9))
        assert approved is False
        assert "score" in reason.lower() or "signal" in reason.lower()

    def test_score_well_below_minimum_rejected(self, validator):
        approved, _ = validator.validate_trade(_candidate(signal_score=0.0))
        assert approved is False

    def test_score_exactly_at_minimum_approved(self, validator):
        approved, _ = validator.validate_trade(_candidate(signal_score=70.0))
        assert approved is True


# ---------------------------------------------------------------------------
# Rule 4: Risk/Reward < minimum
# ---------------------------------------------------------------------------

class TestRiskRewardRule:

    def test_rr_below_minimum_rejected(self, validator):
        approved, reason = validator.validate_trade(
            _candidate(risk_reward_ratio=1.49, reward_per_share=14.9)
        )
        assert approved is False
        assert "risk" in reason.lower() or "reward" in reason.lower()

    def test_rr_zero_rejected(self, validator):
        approved, _ = validator.validate_trade(
            _candidate(risk_reward_ratio=0.0, reward_per_share=0.0)
        )
        assert approved is False

    def test_rr_exactly_minimum_approved(self, validator):
        approved, _ = validator.validate_trade(
            _candidate(risk_reward_ratio=1.5, reward_per_share=15.0)
        )
        assert approved is True

    def test_rr_above_minimum_approved(self, validator):
        approved, _ = validator.validate_trade(
            _candidate(risk_reward_ratio=2.0, reward_per_share=20.0)
        )
        assert approved is True


# ---------------------------------------------------------------------------
# Rule 5: Position Size ≤ 0
# ---------------------------------------------------------------------------

class TestPositionSizeRule:

    def test_zero_position_size_rejected(self, validator):
        approved, reason = validator.validate_trade(_candidate(position_size=0))
        assert approved is False
        assert "position" in reason.lower() or "size" in reason.lower()

    def test_negative_position_size_rejected(self, validator):
        approved, _ = validator.validate_trade(_candidate(position_size=-1))
        assert approved is False

    def test_one_share_not_rejected(self, validator):
        approved, _ = validator.validate_trade(_candidate(position_size=1))
        assert approved is True


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

class TestConfigurableThresholds:

    def test_custom_score_threshold(self):
        v = TradeValidator(minimum_signal_score=85.0, minimum_risk_reward=1.5)
        approved, _ = v.validate_trade(_candidate(signal_score=80.0))
        assert approved is False

    def test_custom_rr_threshold(self):
        v = TradeValidator(minimum_signal_score=70.0, minimum_risk_reward=2.0)
        approved, _ = v.validate_trade(_candidate(risk_reward_ratio=1.9))
        assert approved is False

    def test_relaxed_thresholds_approve_more(self):
        strict  = TradeValidator(minimum_signal_score=90.0, minimum_risk_reward=3.0)
        relaxed = TradeValidator(minimum_signal_score=50.0, minimum_risk_reward=1.0)
        c = _candidate(signal_score=75.0, risk_reward_ratio=1.5)
        approved_strict, _  = strict.validate_trade(c)
        approved_relaxed, _ = relaxed.validate_trade(c)
        assert not approved_strict
        assert approved_relaxed
