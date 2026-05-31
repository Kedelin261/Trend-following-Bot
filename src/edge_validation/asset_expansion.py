"""Asset expansion — measures per-asset contribution and portfolio aggregate.

Reuses EdgeEngine.evaluate_all with varying asset subsets.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.data.models import Candle
from src.edge_lab.edge_engine import EdgeEngine
from src.edge_lab.edge_profile import EdgeProfile
from src.edge_lab.strategy_interface import StrategyInterface


@dataclass
class AssetContribution:
    symbol:      str
    trades:      int
    pf:          float
    expectancy:  float
    drawdown:    float
    adds_value:  bool    # True when expectancy > 0 and PF >= 1.0

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.pf) else f"{self.pf:.2f}"


@dataclass
class AssetExpansionResult:
    strategy_name:       str
    asset_contributions: List[AssetContribution]
    portfolio_profile:   EdgeProfile

    @property
    def approved_assets(self) -> List[str]:
        return [a.symbol for a in self.asset_contributions if a.adds_value]

    @property
    def total_trades(self) -> int:
        return self.portfolio_profile.total_trades


class AssetExpansionResearcher:
    """Evaluates strategy performance per asset and in aggregate."""

    def __init__(self, config: dict) -> None:
        self._engine = EdgeEngine(config)

    def research(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
    ) -> AssetExpansionResult:
        # Per-asset
        contributions = []
        for sym, candles in asset_candles.items():
            [p] = self._engine.evaluate_all([strategy], {sym: candles})
            contributions.append(AssetContribution(
                symbol     = sym,
                trades     = p.total_trades,
                pf         = p.profit_factor,
                expectancy = p.expectancy,
                drawdown   = p.max_drawdown,
                adds_value = p.expectancy > 0 and
                             (math.isinf(p.profit_factor) or p.profit_factor >= 1.0),
            ))

        # Portfolio aggregate (all assets together)
        [portfolio] = self._engine.evaluate_all([strategy], asset_candles)

        return AssetExpansionResult(
            strategy_name       = strategy.name,
            asset_contributions = sorted(contributions, key=lambda a: a.expectancy, reverse=True),
            portfolio_profile   = portfolio,
        )
