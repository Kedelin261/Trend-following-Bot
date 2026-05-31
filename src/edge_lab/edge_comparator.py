"""Edge comparator — side-by-side metrics for strategy pairs."""

import math
from dataclasses import dataclass
from typing import Optional

from src.edge_lab.edge_profile import EdgeProfile


@dataclass
class EdgeComparison:
    """Side-by-side delta between two EdgeProfiles."""

    challenger:        str
    benchmark:         str
    pf_delta:          float
    exp_delta:         float
    dd_delta:          float   # positive = challenger has less DD
    trade_delta:       int
    score_delta:       float
    challenger_wins:   bool


class EdgeComparator:
    """Compares one EdgeProfile against a benchmark."""

    def compare(
        self,
        challenger: EdgeProfile,
        benchmark:  EdgeProfile,
    ) -> EdgeComparison:
        c_pf = challenger.profit_factor if not math.isinf(challenger.profit_factor) else 10.0
        b_pf = benchmark.profit_factor  if not math.isinf(benchmark.profit_factor)  else 10.0

        return EdgeComparison(
            challenger      = challenger.strategy_name,
            benchmark       = benchmark.strategy_name,
            pf_delta        = round(c_pf - b_pf, 3),
            exp_delta       = round(challenger.expectancy - benchmark.expectancy, 2),
            dd_delta        = round(benchmark.max_drawdown - challenger.max_drawdown, 1),
            trade_delta     = challenger.total_trades - benchmark.total_trades,
            score_delta     = round(challenger.edge_score - benchmark.edge_score, 1),
            challenger_wins = challenger.edge_score > benchmark.edge_score,
        )
