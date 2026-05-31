"""Portfolio — tracks cash balance and trade history during backtesting.

No execution code. No broker mutations.
Accepts a RiskProfile for future live-capital injection without code changes.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from src.backtest.models import BacktestTrade
from src.risk.models import RiskProfile

logger = logging.getLogger(__name__)


class Portfolio:
    """Simulated cash portfolio for a single backtest run.

    Designed to be reset between walk-forward windows without reinstantiation.

    Parameters
    ----------
    starting_balance : initial cash before any trades
    """

    def __init__(self, starting_balance: float) -> None:
        self._starting: float = starting_balance
        self._balance:  float = starting_balance
        self._trades:   List[BacktestTrade] = []
        # Equity history: list of (timestamp, portfolio_value)
        # Index 0 is the starting point (before any trades)
        self._equity:   List[Tuple[Optional[datetime], float]] = [
            (None, starting_balance)
        ]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def process_trade(self, trade: BacktestTrade) -> None:
        """Apply a completed trade's PnL to the portfolio balance."""
        self._balance += trade.pnl
        self._trades.append(trade)
        self._equity.append((trade.exit_time, self._balance))

        logger.debug(
            "portfolio_update: %s pnl=%.2f  balance=%.2f",
            trade.symbol,
            trade.pnl,
            self._balance,
        )

    def reset(self) -> None:
        """Return portfolio to its starting state (call before each backtest run)."""
        self._balance = self._starting
        self._trades  = []
        self._equity  = [(None, self._starting)]

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Current cash balance."""
        return self._balance

    @property
    def starting_balance(self) -> float:
        return self._starting

    @property
    def trades(self) -> List[BacktestTrade]:
        """Immutable copy of all closed trades."""
        return list(self._trades)

    @property
    def equity_history(self) -> List[Tuple[Optional[datetime], float]]:
        """List of (timestamp, portfolio_value) including starting point."""
        return list(self._equity)

    @property
    def equity_values(self) -> List[float]:
        """Just the portfolio value at each equity point."""
        return [v for _, v in self._equity]

    @property
    def net_profit(self) -> float:
        return self._balance - self._starting

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_risk_profile(cls, profile: RiskProfile) -> "Portfolio":
        """Build a Portfolio using a RiskProfile's available cash.

        Future phases inject live IBKR account balance here without
        modifying the Portfolio class.
        """
        return cls(starting_balance=profile.cash_available)
