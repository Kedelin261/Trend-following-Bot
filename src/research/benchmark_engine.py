"""Benchmark engine — compares strategy to a buy-and-hold baseline.

Primary question: Is this strategy actually outperforming passive investing?
A strategy that underperforms buy-and-hold while adding risk and complexity
provides no value regardless of its absolute return.

No broker code. No API calls. Pure arithmetic.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from src.backtest.models import BacktestResults
from src.backtest import performance_metrics as pm
from src.data.models import Candle

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Side-by-side comparison of strategy vs buy-and-hold."""

    symbol:                  str

    # Strategy
    strategy_return_pct:     float
    strategy_sharpe:         float
    strategy_max_drawdown:   float
    strategy_total_trades:   int

    # Buy & hold
    buyhold_return_pct:      float
    buyhold_sharpe:          float
    buyhold_max_drawdown:    float

    # Derived
    alpha_pct:               float   # strategy_return - buyhold_return
    strategy_outperforms:    bool    # True when alpha > 0
    notes:                   List[str]

    @property
    def risk_adjusted_outperforms(self) -> bool:
        """True when strategy has better Sharpe AND positive alpha."""
        return self.strategy_sharpe > self.buyhold_sharpe and self.alpha_pct > 0


class BenchmarkEngine:
    """Computes buy-and-hold performance for any candle series and
    compares it to backtest results on the same data.
    """

    def compare(
        self,
        backtest:         BacktestResults,
        candles:          List[Candle],
        starting_balance: Optional[float] = None,
    ) -> BenchmarkResult:
        """Generate a benchmark comparison report.

        Parameters
        ----------
        backtest         : result from BacktestEngine.run()
        candles          : the same candle series used in the backtest
        starting_balance : override starting capital (defaults to backtest value)
        """
        if not candles:
            return self._empty_result(backtest)

        bal = starting_balance or backtest.starting_balance

        # Buy-and-hold: invest entire balance at first bar, sell at last bar
        start_price = candles[0].close
        bh_equity   = [bal * (c.close / start_price) for c in candles]

        bh_return  = (bh_equity[-1] - bh_equity[0]) / bh_equity[0] * 100.0
        bh_dd      = pm.max_drawdown(bh_equity)
        bh_sharpe  = pm.sharpe_ratio(bh_equity)

        alpha  = backtest.return_percent - bh_return
        outper = alpha > 0

        notes = self._generate_notes(backtest, bh_return, bh_sharpe, bh_dd)

        logger.info(
            "benchmark: %s strategy=%.1f%% buyhold=%.1f%% alpha=%.1f%% outperforms=%s",
            backtest.symbol,
            backtest.return_percent,
            bh_return,
            alpha,
            outper,
        )

        return BenchmarkResult(
            symbol                 = backtest.symbol,
            strategy_return_pct   = backtest.return_percent,
            strategy_sharpe       = backtest.sharpe_ratio,
            strategy_max_drawdown = backtest.max_drawdown,
            strategy_total_trades = backtest.total_trades,
            buyhold_return_pct    = bh_return,
            buyhold_sharpe        = bh_sharpe,
            buyhold_max_drawdown  = bh_dd,
            alpha_pct             = alpha,
            strategy_outperforms  = outper,
            notes                 = notes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_notes(
        bt:        BacktestResults,
        bh_return: float,
        bh_sharpe: float,
        bh_dd:     float,
    ) -> List[str]:
        notes = []
        if bt.return_percent < bh_return:
            notes.append(
                f"Strategy underperforms buy-and-hold by "
                f"{bh_return - bt.return_percent:.1f}%"
            )
        else:
            notes.append(
                f"Strategy outperforms buy-and-hold by "
                f"{bt.return_percent - bh_return:.1f}%"
            )
        if bt.max_drawdown > bh_dd:
            notes.append(
                f"Strategy has higher drawdown ({bt.max_drawdown:.1f}%) "
                f"than buy-and-hold ({bh_dd:.1f}%)"
            )
        if bt.sharpe_ratio > bh_sharpe:
            notes.append(
                f"Strategy has better risk-adjusted returns "
                f"(Sharpe {bt.sharpe_ratio:.2f} vs {bh_sharpe:.2f})"
            )
        if bt.total_trades < 30:
            notes.append(
                "Insufficient trades for statistical comparison — interpret with caution"
            )
        return notes

    @staticmethod
    def _empty_result(bt: BacktestResults) -> BenchmarkResult:
        return BenchmarkResult(
            symbol                = bt.symbol,
            strategy_return_pct   = bt.return_percent,
            strategy_sharpe       = bt.sharpe_ratio,
            strategy_max_drawdown = bt.max_drawdown,
            strategy_total_trades = bt.total_trades,
            buyhold_return_pct    = 0.0,
            buyhold_sharpe        = 0.0,
            buyhold_max_drawdown  = 0.0,
            alpha_pct             = 0.0,
            strategy_outperforms  = False,
            notes                 = ["No candle data for benchmark comparison"],
        )
