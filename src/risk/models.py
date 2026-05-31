"""Risk Engine data models for Phase 3.

No execution code. No broker references.
Designed for injection of live account values in future phases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.signals.models import SignalType


@dataclass
class RiskProfile:
    """Describes the capital and risk parameters for a trading account.

    Accepted by RiskEngine, never mutated.  Future phases inject live
    values (e.g. from IBKR account summary) without changing any risk
    logic — only the RiskProfile changes.

    Parameters
    ----------
    account_size           : Total account equity ($)
    cash_available         : Cash deployable for new trades ($)
    risk_per_trade_percent : Maximum loss per trade as % of cash_available
    max_daily_loss_percent : Maximum daily drawdown before trading halts (%)
    max_weekly_loss_percent: Maximum weekly drawdown before trading halts (%)
    """

    account_size:             float
    cash_available:           float
    risk_per_trade_percent:   float = 1.0
    max_daily_loss_percent:   float = 3.0
    max_weekly_loss_percent:  float = 5.0

    @property
    def dollar_risk_per_trade(self) -> float:
        """Maximum dollar loss permitted on a single trade."""
        return self.cash_available * (self.risk_per_trade_percent / 100.0)

    @property
    def max_daily_loss_dollar(self) -> float:
        return self.account_size * (self.max_daily_loss_percent / 100.0)

    @property
    def max_weekly_loss_dollar(self) -> float:
        return self.account_size * (self.max_weekly_loss_percent / 100.0)

    @classmethod
    def from_config(cls, config: dict) -> "RiskProfile":
        """Build a RiskProfile from the 'risk' section of settings.yaml.

        All fields default to conservative values when absent from config.
        """
        r = config.get("risk", {})
        account_size = float(r.get("account_size", 10_000.0))
        return cls(
            account_size            = account_size,
            cash_available          = float(r.get("cash_available", account_size)),
            risk_per_trade_percent  = float(r.get("risk_per_trade_percent", 1.0)),
            max_daily_loss_percent  = float(r.get("max_daily_loss_percent", 3.0)),
            max_weekly_loss_percent = float(r.get("max_weekly_loss_percent", 5.0)),
        )


@dataclass
class TradeCandidate:
    """Fully-evaluated trade candidate produced by the RiskEngine.

    Downstream consumers (Phase 4 Backtester, Phase 6 Paper Trading,
    Phase 7 Live Trading) read this object; they never call the risk
    engine internals directly.

    Fields
    ------
    approved         : True when all validation rules pass
    rejection_reason : Human-readable reason when approved=False, else None
    """

    symbol:             str
    timeframe:          str
    signal_type:        SignalType

    entry_price:        float
    atr:                float

    stop_loss:          float
    take_profit:        float

    risk_per_share:     float
    reward_per_share:   float
    risk_reward_ratio:  float

    position_size:      int
    dollar_risk:        float

    signal_score:       float
    approved:           bool
    rejection_reason:   Optional[str]   = field(default=None)
    timestamp:          Optional[datetime] = field(default=None)

    @property
    def is_long(self) -> bool:
        return self.signal_type == SignalType.LONG

    @property
    def is_short(self) -> bool:
        return self.signal_type == SignalType.SHORT

    @property
    def total_position_value(self) -> float:
        """Notional value of the full position at entry."""
        return self.entry_price * self.position_size

    @property
    def expected_profit(self) -> float:
        """Maximum expected gain at take-profit (pre-commission)."""
        return self.reward_per_share * self.position_size

    @property
    def expected_loss(self) -> float:
        """Maximum expected loss at stop-loss (pre-commission)."""
        return self.risk_per_share * self.position_size
