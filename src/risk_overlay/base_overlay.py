"""Base class for all Phase 5.2 risk overlays.

An overlay intercepts the "should I open a new trade?" decision point.
It receives the current bar index, the full candle history up to this
bar, the current equity curve, and the list of already-closed trades.

It returns:
    allowed     : bool — whether a new trade may be opened
    size_scale  : float in (0.0, 1.0] — position size multiplier
                  1.0 means no size reduction; 0.0 means no trade
    reason      : str — human-readable explanation

Fairness rules:
    - Same signals, same assets, same candles, same backtester as Phase 5.1
    - Same risk engine, same slippage, same commissions
    - Same position sizing baseline (size_scale=1.0 for non-scaling overlays)
    - Only the overlay logic differs between runs
    - No parameter sweeps, no brute-force tuning, no ML

Research only. No execution. No broker code. No live trading.
"""

from abc import ABC, abstractmethod
from typing import List

from src.backtest.models import BacktestTrade
from src.data.models import Candle


class RiskOverlay(ABC):
    """Abstract base class for all risk overlays."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique overlay identifier for reporting."""

    def reset(self) -> None:
        """Reset overlay state between backtest runs (called before each run)."""

    @abstractmethod
    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Evaluate whether a new trade may be opened at this bar.

        Parameters
        ----------
        bar_index     : current bar index in the full candle series
        candles       : all candles up to and including the current bar
        equity_values : portfolio equity at each closed-trade point
                        (first element is starting balance)
        closed_trades : all trades closed so far in this backtest run

        Returns
        -------
        (allowed: bool, size_scale: float, reason: str)
        """
