"""Equity curve — time-series of portfolio values across a backtest.

Generated from a list of closed BacktestTrade objects.
Designed for future dashboard charting and drawdown visualisation.

No broker code. No mutable state after construction.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from src.backtest.models import BacktestTrade

logger = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    """A single data point on the equity curve."""

    timestamp:       Optional[datetime]
    portfolio_value: float
    trade_index:     int    # index in closed-trade list; -1 = starting point


class EquityCurve:
    """Generates and exposes the equity curve for a completed backtest.

    The starting point (before any trades) is always included at index -1.
    Each subsequent point corresponds to one closed trade.
    """

    def __init__(
        self,
        starting_balance: float,
        trades:           List[BacktestTrade],
    ) -> None:
        self._points: List[EquityPoint] = self._build(starting_balance, trades)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def points(self) -> List[EquityPoint]:
        return list(self._points)

    @property
    def values(self) -> List[float]:
        """Portfolio value at each equity point."""
        return [p.portfolio_value for p in self._points]

    @property
    def timestamps(self) -> List[Optional[datetime]]:
        return [p.timestamp for p in self._points]

    def as_tuples(self) -> List[Tuple[Optional[datetime], float]]:
        """Return (timestamp, portfolio_value) pairs for downstream consumers."""
        return [(p.timestamp, p.portfolio_value) for p in self._points]

    @property
    def peak_value(self) -> float:
        return max(p.portfolio_value for p in self._points)

    @property
    def trough_value(self) -> float:
        return min(p.portfolio_value for p in self._points)

    @property
    def length(self) -> int:
        """Number of equity points including the starting point."""
        return len(self._points)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build(
        starting_balance: float,
        trades:           List[BacktestTrade],
    ) -> List[EquityPoint]:
        points = [
            EquityPoint(
                timestamp       = None,
                portfolio_value = starting_balance,
                trade_index     = -1,
            )
        ]
        balance = starting_balance

        for idx, trade in enumerate(trades):
            balance += trade.pnl
            points.append(
                EquityPoint(
                    timestamp       = trade.exit_time,
                    portfolio_value = balance,
                    trade_index     = idx,
                )
            )

        logger.debug("equity_curve: built %d points from %d trades", len(points), len(trades))
        return points
