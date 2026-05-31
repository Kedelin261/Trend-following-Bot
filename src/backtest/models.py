"""Backtest data models for Phase 4.

No execution code. No broker references.
Designed for future dashboard replay and parameter sweep support.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from src.signals.models import SignalType


class ClosingReason(str, Enum):
    STOP      = "STOP"
    TARGET    = "TARGET"
    TIME_EXIT = "TIME_EXIT"


@dataclass
class BacktestTrade:
    """A single completed simulated trade.

    Stores enough information for full trade replay in future dashboards.
    """

    symbol:         str
    entry_time:     datetime
    exit_time:      datetime
    signal_type:    SignalType
    entry_price:    float
    exit_price:     float
    stop_price:     float
    target_price:   float
    position_size:  int
    pnl:            float           # net (after commission and slippage)
    return_percent: float
    holding_period: int             # bars from entry to exit
    win_loss:       str             # "WIN" or "LOSS"
    reason_closed:  ClosingReason

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        return self.pnl <= 0


@dataclass
class BacktestResults:
    """Complete results of a single backtest run.

    The ``trades`` list enables full trade replay.
    ``equity_curve`` enables dashboard charting.
    ``return_percent`` property is derived so no duplication exists.
    """

    symbol:           str
    timeframe:        str
    starting_balance: float
    ending_balance:   float
    net_profit:       float
    total_trades:     int
    winning_trades:   int
    losing_trades:    int
    win_rate:         float         # 0.0–1.0
    profit_factor:    float
    expectancy:       float         # expected $ per trade
    max_drawdown:     float         # peak-to-trough as %
    sharpe_ratio:     float
    average_win:      float
    average_loss:     float         # as positive dollar amount
    largest_win:      float
    largest_loss:     float         # as positive dollar amount
    equity_curve:     List[Tuple[Optional[datetime], float]]
    trades:           List[BacktestTrade] = field(default_factory=list)

    @property
    def return_percent(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.net_profit / self.starting_balance) * 100.0


@dataclass
class StrategyHealth:
    """PASS / FAIL evaluation of backtest results against risk thresholds.

    All four conditions must pass for an overall PASS verdict.
    """

    passed:           bool
    expectancy_ok:    bool
    profit_factor_ok: bool
    drawdown_ok:      bool
    min_trades_ok:    bool
    reasons_passed:   List[str]
    reasons_failed:   List[str]

    @classmethod
    def evaluate(
        cls,
        results:            "BacktestResults",
        min_trades:         int   = 30,
        min_expectancy:     float = 0.0,
        min_profit_factor:  float = 1.5,
        max_drawdown_pct:   float = 20.0,
    ) -> "StrategyHealth":
        """Evaluate results against institutional viability thresholds."""
        passed_reasons: List[str] = []
        failed_reasons: List[str] = []

        exp_ok = results.expectancy > min_expectancy
        if exp_ok:
            passed_reasons.append(f"Positive expectancy (${results.expectancy:.2f}/trade)")
        else:
            failed_reasons.append(f"Negative expectancy (${results.expectancy:.2f}/trade)")

        pf_ok = results.profit_factor >= min_profit_factor
        if pf_ok:
            passed_reasons.append(
                f"Profit factor {results.profit_factor:.2f} ≥ {min_profit_factor}"
            )
        else:
            failed_reasons.append(
                f"Profit factor {results.profit_factor:.2f} < {min_profit_factor}"
            )

        dd_ok = results.max_drawdown <= max_drawdown_pct
        if dd_ok:
            passed_reasons.append(
                f"Max drawdown {results.max_drawdown:.1f}% ≤ {max_drawdown_pct}%"
            )
        else:
            failed_reasons.append(
                f"Max drawdown {results.max_drawdown:.1f}% > {max_drawdown_pct}%"
            )

        trades_ok = results.total_trades >= min_trades
        if trades_ok:
            passed_reasons.append(
                f"{results.total_trades} trades ≥ minimum {min_trades}"
            )
        else:
            failed_reasons.append(
                f"Only {results.total_trades} trades < minimum {min_trades} "
                "(insufficient sample size for statistical significance)"
            )

        return cls(
            passed           = exp_ok and pf_ok and dd_ok and trades_ok,
            expectancy_ok    = exp_ok,
            profit_factor_ok = pf_ok,
            drawdown_ok      = dd_ok,
            min_trades_ok    = trades_ok,
            reasons_passed   = passed_reasons,
            reasons_failed   = failed_reasons,
        )
