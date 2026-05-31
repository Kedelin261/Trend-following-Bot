"""Regime performance analyzer — measures strategy outcomes per regime bucket.

Groups closed BacktestTrades by regime label and computes:
  - Trade count, win rate, PF, expectancy per bucket
  - Profit concentration: % of all profits from this regime
  - Loss concentration: % of all losses from this regime

The concentration metrics answer the key questions:
  "What % of profits come from STRONG_BULL?"
  "What % of losses come from BEAR?"

No broker code. No API calls. Pure analytics.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest import performance_metrics as pm
from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime

logger = logging.getLogger(__name__)


@dataclass
class LabelledTrade:
    """A BacktestTrade enriched with regime labels at its entry bar."""

    trade:              BacktestTrade
    market_regime:      MarketRegime
    trend_regime:       TrendRegime
    volatility_regime:  VolatilityRegime
    drawdown_env:       DrawdownEnvironment
    macro_regime:       MacroRegime


@dataclass
class RegimePerformance:
    """Strategy performance statistics for one regime bucket."""

    regime_name:          str
    trade_count:          int
    win_rate:             float
    expectancy:           float
    profit_factor:        float
    total_pnl:            float
    profit_concentration: float   # fraction of all gains from this regime
    loss_concentration:   float   # fraction of all losses from this regime
    is_net_positive:      bool

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


class RegimePerformanceAnalyzer:
    """Groups trades by regime label and computes per-regime statistics."""

    DIMENSIONS = {
        "market_regime":     "Market Regime (5-state EMA)",
        "trend_regime":      "Trend Quality",
        "volatility_regime": "Volatility",
        "drawdown_env":      "Drawdown Environment",
        "macro_regime":      "Macro Regime",
    }

    def analyze(
        self,
        labelled_trades: List[LabelledTrade],
        dimension:       str,
    ) -> Dict[str, RegimePerformance]:
        """Compute per-regime performance for one classification dimension.

        Parameters
        ----------
        labelled_trades : trades with all regime labels attached
        dimension       : attribute name on LabelledTrade to group by
        """
        if dimension not in self.DIMENSIONS:
            raise ValueError(
                f"Unknown dimension '{dimension}'. Valid: {list(self.DIMENSIONS)}"
            )

        # Group by regime — use .value to get clean string ("BEAR" not "MarketRegime.BEAR")
        buckets: Dict[str, List[BacktestTrade]] = defaultdict(list)
        for lt in labelled_trades:
            regime_attr = getattr(lt, dimension)
            label = regime_attr.value if hasattr(regime_attr, "value") else str(regime_attr)
            buckets[label].append(lt.trade)

        all_trades = [lt.trade for lt in labelled_trades]
        total_profit = sum(t.pnl for t in all_trades if t.is_win)
        total_loss   = abs(sum(t.pnl for t in all_trades if t.is_loss))

        results: Dict[str, RegimePerformance] = {}
        for regime_str, trades in sorted(buckets.items()):
            wins   = [t for t in trades if t.is_win]
            losses = [t for t in trades if t.is_loss]

            gross_wins   = sum(t.pnl for t in wins)
            gross_losses = abs(sum(t.pnl for t in losses))

            pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
            exp = sum(t.pnl for t in trades) / len(trades) if trades else 0.0
            wr  = len(wins) / len(trades) if trades else 0.0

            profit_conc = gross_wins   / total_profit if total_profit > 0 else 0.0
            loss_conc   = gross_losses / total_loss   if total_loss   > 0 else 0.0

            results[regime_str] = RegimePerformance(
                regime_name          = regime_str,
                trade_count          = len(trades),
                win_rate             = wr,
                expectancy           = exp,
                profit_factor        = pf,
                total_pnl            = sum(t.pnl for t in trades),
                profit_concentration = profit_conc,
                loss_concentration   = loss_conc,
                is_net_positive      = exp > 0,
            )

        return results

    def analyze_all(
        self, labelled_trades: List[LabelledTrade]
    ) -> Dict[str, Dict[str, RegimePerformance]]:
        """Analyze every dimension and return a nested dict."""
        return {
            dim: self.analyze(labelled_trades, dim)
            for dim in self.DIMENSIONS
        }

    def best_regime(
        self, performance: Dict[str, RegimePerformance], min_trades: int = 5
    ) -> Optional[str]:
        """Return regime with highest expectancy (with sufficient sample)."""
        candidates = [
            (k, v) for k, v in performance.items()
            if v.trade_count >= min_trades and v.is_net_positive
        ]
        return max(candidates, key=lambda x: x[1].expectancy)[0] if candidates else None

    def worst_regime(
        self, performance: Dict[str, RegimePerformance], min_trades: int = 5
    ) -> Optional[str]:
        """Return regime with lowest expectancy (with sufficient sample)."""
        candidates = [
            (k, v) for k, v in performance.items()
            if v.trade_count >= min_trades
        ]
        return min(candidates, key=lambda x: x[1].expectancy)[0] if candidates else None
