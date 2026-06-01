"""MONTHLY_LOSS_LOCKOUT — stop new entries if monthly loss exceeds threshold.

Logic:
    1. Track the cumulative PnL of all trades closed in the current calendar month.
    2. If total monthly loss exceeds MONTHLY_LOSS_THRESHOLD (expressed as a
       fraction of starting balance), lock out new entries for the rest of
       the month.
    3. Reset at the start of each new calendar month.

Default threshold: 5% of starting balance ($10,000 × 5% = $500 max loss/month).
Fixed — not optimised.

Research only. No execution. No broker code. No live trading.
"""

from typing import List, Optional

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.risk_overlay.base_overlay import RiskOverlay

# Fixed default: 5% of starting balance per month
MONTHLY_LOSS_THRESHOLD_PCT = 0.05


class MonthlyLossLockout(RiskOverlay):
    """Lock out new entries for the rest of the month after exceeding monthly loss cap.

    Parameters
    ----------
    starting_balance        : account starting balance for computing dollar threshold
    monthly_loss_threshold  : fraction of starting balance (default 0.05 = 5%)
    """

    def __init__(
        self,
        starting_balance:       float = 10_000.0,
        monthly_loss_threshold: float = MONTHLY_LOSS_THRESHOLD_PCT,
    ) -> None:
        self._starting_balance = starting_balance
        self._threshold_pct    = monthly_loss_threshold
        self._threshold_dollar = starting_balance * monthly_loss_threshold
        self._locked_months:   set = set()   # set of (year, month) tuples

    @property
    def name(self) -> str:
        return "MONTHLY_LOSS_LOCKOUT"

    def reset(self) -> None:
        self._locked_months = set()

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Allow trade only if this month's losses have not exceeded the cap."""
        if not candles:
            return True, 1.0, "no candle data"

        current_ts    = candles[-1].timestamp
        current_month = (current_ts.year, current_ts.month)

        # If this month already locked, block
        if current_month in self._locked_months:
            return (
                False, 0.0,
                f"monthly lockout: {current_month[0]}-{current_month[1]:02d} "
                f"already locked",
            )

        # Sum PnL for all trades closed in the current month
        monthly_pnl = 0.0
        for trade in closed_trades:
            t_month = (trade.exit_time.year, trade.exit_time.month)
            if t_month == current_month:
                monthly_pnl += trade.pnl

        monthly_loss = -monthly_pnl   # positive = loss

        if monthly_loss >= self._threshold_dollar:
            # Lock out rest of this month
            self._locked_months.add(current_month)
            return (
                False, 0.0,
                f"monthly lockout: loss=${monthly_loss:.0f} >= "
                f"threshold=${self._threshold_dollar:.0f} "
                f"({self._threshold_pct:.0%})",
            )

        return (
            True, 1.0,
            f"monthly loss ${monthly_loss:.0f} < threshold ${self._threshold_dollar:.0f}",
        )
