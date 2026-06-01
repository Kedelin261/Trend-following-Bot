"""Module 1 — Asset Contribution Analysis.

Answers: Which assets generate the most profits? Which destroy edge?

For each asset (SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD):
  - Trades, PF, Expectancy, Win Rate, Max Drawdown
  - Gross profit contribution % of total

Ranked best → worst by PF then Expectancy.

No execution code. No broker code. No live trading.
No entry logic modifications.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest.performance_metrics import (
    profit_factor, expectancy, win_rate, max_drawdown,
)

MIN_SAMPLE = 50   # anti-curve-fitting safeguard


@dataclass
class AssetContribution:
    """Metrics for a single asset."""
    symbol:           str
    trades:           int
    profit_factor:    float
    expectancy:       float
    win_rate:         float
    max_drawdown_pct: float
    gross_profit:     float     # sum of winning pnl
    gross_loss:       float     # abs sum of losing pnl (positive)
    net_pnl:          float
    contribution_pct: float     # net_pnl / total_net_pnl × 100
    rank:             int       = 0
    insufficient_sample: bool  = False

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


class AssetContributionAnalyzer:
    """Breaks down trade results by asset symbol.

    Parameters
    ----------
    trades : list of BacktestTrade from full multi-asset backtest
    equity_by_asset : optional per-asset equity curves for drawdown
    """

    def __init__(
        self,
        trades: List[BacktestTrade],
        equity_by_asset: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self._trades = trades
        self._equity = equity_by_asset or {}

    def analyze(self) -> List[AssetContribution]:
        """Return per-asset contributions ranked best → worst by PF."""
        by_symbol: Dict[str, List[BacktestTrade]] = {}
        for t in self._trades:
            by_symbol.setdefault(t.symbol, []).append(t)

        total_net = sum(t.pnl for t in self._trades)

        results: List[AssetContribution] = []
        for sym, sym_trades in by_symbol.items():
            wins   = [t for t in sym_trades if t.is_win]
            losses = [t for t in sym_trades if t.is_loss]
            gp     = sum(t.pnl for t in wins)
            gl     = abs(sum(t.pnl for t in losses))
            net    = sum(t.pnl for t in sym_trades)
            n      = len(sym_trades)

            pf_val  = profit_factor(sym_trades)
            exp_val = expectancy(sym_trades)
            wr_val  = win_rate(sym_trades)

            # DD from per-asset equity if available, else 0
            eq_vals  = self._equity.get(sym, [])
            dd_val   = max_drawdown(eq_vals) if len(eq_vals) >= 2 else 0.0

            contrib  = (net / total_net * 100.0) if total_net != 0 else 0.0
            insuff   = n < MIN_SAMPLE

            results.append(AssetContribution(
                symbol           = sym,
                trades           = n,
                profit_factor    = pf_val,
                expectancy       = exp_val,
                win_rate         = wr_val,
                max_drawdown_pct = dd_val,
                gross_profit     = gp,
                gross_loss       = gl,
                net_pnl          = net,
                contribution_pct = contrib,
                insufficient_sample = insuff,
            ))

        # Rank best → worst: primary=PF, secondary=Expectancy
        def _sort_key(r: AssetContribution):
            pf = r.profit_factor if not math.isinf(r.profit_factor) else 99.0
            return (not r.insufficient_sample, pf, r.expectancy)

        results.sort(key=_sort_key, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results
