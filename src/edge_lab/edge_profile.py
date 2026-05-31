"""EdgeProfile — aggregated backtest results for one strategy.

Immutable snapshot of a strategy's performance across multiple assets.
Used for ranking and promotion evaluation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults


@dataclass
class EdgeProfile:
    """Complete performance profile for one strategy candidate."""

    strategy_name:   str
    description:     str

    # Aggregate metrics (across all assets)
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    win_rate:        float
    profit_factor:   float
    expectancy:      float
    max_drawdown:    float
    sharpe_ratio:    float
    net_pnl:         float

    # Asset-level results (for inspection)
    asset_results:   Dict[str, BacktestResults] = field(default_factory=dict)

    # Stability across time windows
    robustness_rating: str = "UNKNOWN"   # ROBUST / MARGINAL / UNSTABLE
    window_pf:         List[float] = field(default_factory=list)  # [early, mid, recent]

    # Computed scores
    edge_score:       float = 0.0
    promotion_candidate: bool = False
    rejection_reason: Optional[str] = None

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"

    @property
    def assets_passing(self) -> int:
        return sum(
            1 for bt in self.asset_results.values()
            if bt.profit_factor >= 1.5 and bt.expectancy > 0 and bt.max_drawdown < 15
        )

    @classmethod
    def from_asset_results(
        cls,
        strategy_name: str,
        description:   str,
        asset_results: Dict[str, BacktestResults],
    ) -> "EdgeProfile":
        """Build an EdgeProfile by aggregating per-asset BacktestResults."""
        all_trades = [t for bt in asset_results.values() for t in bt.trades]
        total  = len(all_trades)
        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if t.is_loss]

        gross_wins   = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        exp = sum(t.pnl for t in all_trades) / total if total > 0 else 0.0
        wr  = len(wins) / total if total > 0 else 0.0

        # Worst-asset drawdown
        worst_dd = max((bt.max_drawdown for bt in asset_results.values()), default=0.0)

        # Net PnL sum
        net = sum(bt.net_profit for bt in asset_results.values())

        # Sharpe from equity values
        all_eq = []
        for bt in asset_results.values():
            all_eq.extend(v for _, v in bt.equity_curve)
        from src.backtest.performance_metrics import sharpe_ratio
        sharpe = sharpe_ratio(all_eq) if len(all_eq) >= 3 else 0.0

        return cls(
            strategy_name = strategy_name,
            description   = description,
            total_trades  = total,
            winning_trades = len(wins),
            losing_trades  = len(losses),
            win_rate       = wr,
            profit_factor  = pf,
            expectancy     = exp,
            max_drawdown   = worst_dd,
            sharpe_ratio   = sharpe,
            net_pnl        = net,
            asset_results  = asset_results,
        )
