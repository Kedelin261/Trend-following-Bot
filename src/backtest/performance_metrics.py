"""Performance metric calculations for backtesting results.

All functions are pure and stateless: they take closed BacktestTrade lists
and/or equity curve values and return numerical metrics.

No broker code. No mutable state. No API calls.
"""

import math
import statistics
from typing import List

from src.backtest.models import BacktestTrade


# ------------------------------------------------------------------
# Trade-level metrics
# ------------------------------------------------------------------

def win_rate(trades: List[BacktestTrade]) -> float:
    """Fraction of winning trades in [0.0, 1.0]."""
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.is_win) / len(trades)


def winning_trades(trades: List[BacktestTrade]) -> int:
    return sum(1 for t in trades if t.is_win)


def losing_trades(trades: List[BacktestTrade]) -> int:
    return sum(1 for t in trades if t.is_loss)


def profit_factor(trades: List[BacktestTrade]) -> float:
    """Ratio of gross wins to gross losses.

    Returns 0.0 with no trades, float('inf') with wins but no losses.
    """
    gross_wins   = sum(t.pnl for t in trades if t.is_win)
    gross_losses = abs(sum(t.pnl for t in trades if t.is_loss))

    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else 0.0
    return gross_wins / gross_losses


def expectancy(trades: List[BacktestTrade]) -> float:
    """Expected dollar profit per trade.

    Expectancy = (Win Rate × Avg Win) − (Loss Rate × Avg Loss)

    Positive expectancy confirms a statistical edge.
    Negative expectancy means the strategy loses money long-term.
    """
    if not trades:
        return 0.0

    wr = win_rate(trades)
    lr = 1.0 - wr

    wins_pnl   = [t.pnl       for t in trades if t.is_win]
    losses_pnl = [abs(t.pnl)  for t in trades if t.is_loss]

    avg_win  = statistics.mean(wins_pnl)   if wins_pnl   else 0.0
    avg_loss = statistics.mean(losses_pnl) if losses_pnl else 0.0

    return (wr * avg_win) - (lr * avg_loss)


def average_win(trades: List[BacktestTrade]) -> float:
    """Mean positive PnL (dollar)."""
    wins = [t.pnl for t in trades if t.is_win]
    return statistics.mean(wins) if wins else 0.0


def average_loss(trades: List[BacktestTrade]) -> float:
    """Mean absolute loss (returned as a positive dollar amount)."""
    losses = [abs(t.pnl) for t in trades if t.is_loss]
    return statistics.mean(losses) if losses else 0.0


def largest_win(trades: List[BacktestTrade]) -> float:
    wins = [t.pnl for t in trades if t.is_win]
    return max(wins) if wins else 0.0


def largest_loss(trades: List[BacktestTrade]) -> float:
    """Largest single loss as a positive dollar amount."""
    losses = [abs(t.pnl) for t in trades if t.is_loss]
    return max(losses) if losses else 0.0


# ------------------------------------------------------------------
# Equity-curve metrics
# ------------------------------------------------------------------

def max_drawdown(equity_values: List[float]) -> float:
    """Largest peak-to-trough decline expressed as a percentage of the peak.

    Returns 0.0 when fewer than 2 equity points exist.
    """
    if len(equity_values) < 2:
        return 0.0

    peak   = equity_values[0]
    max_dd = 0.0

    for value in equity_values[1:]:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    return max_dd


def sharpe_ratio(
    equity_values:    List[float],
    risk_free_rate:   float = 0.0,
    periods_per_year: int   = 252,
) -> float:
    """Annualized Sharpe ratio calculated from equity curve returns.

    Uses trade-to-trade returns (each closed trade is one return period).
    Returns 0.0 when insufficient data exists or standard deviation is zero.

    Parameters
    ----------
    equity_values    : portfolio value at each equity curve point
    risk_free_rate   : annual risk-free rate (e.g. 0.05 for 5 %)
    periods_per_year : trading periods per year (252 for equities, 365 for crypto)
    """
    if len(equity_values) < 3:
        return 0.0

    returns = [
        (equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
        for i in range(1, len(equity_values))
        if equity_values[i - 1] != 0
    ]

    if len(returns) < 2:
        return 0.0

    mean_ret = statistics.mean(returns)
    std_ret  = statistics.stdev(returns)

    if std_ret == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    return (mean_ret - daily_rf) / std_ret * math.sqrt(periods_per_year)
