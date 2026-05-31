"""Scalability analyzer — determines whether trade count grows proportionally
with history while quality metrics remain stable.
"""

import math
from dataclasses import dataclass
from typing import List

from src.edge_validation.history_expansion import HistorySliceResult


@dataclass
class ScalabilityResult:
    strategy_name:      str
    scales_trade_count: bool    # more history → more trades
    pf_stable:          bool    # PF doesn't collapse with more history
    exp_stable:         bool    # expectancy stays positive
    dd_controlled:      bool    # drawdown stays under 15%
    summary:            str

    @property
    def is_scalable(self) -> bool:
        return self.scales_trade_count and self.pf_stable and self.exp_stable and self.dd_controlled


def analyze_scalability(
    strategy_name: str,
    slices:        List[HistorySliceResult],
) -> ScalabilityResult:
    if len(slices) < 2:
        return ScalabilityResult(strategy_name, False, False, False, False, "insufficient windows")

    valid   = [s for s in slices if s.actual_bars > 0]
    sorted_ = sorted(valid, key=lambda s: s.actual_bars)

    # Trade count grows with history
    trade_counts = [s.trades for s in sorted_]
    scales       = trade_counts[-1] > trade_counts[0]

    # PF stable: no single window collapses below 1.0
    pf_values    = [s.pf for s in sorted_ if not math.isinf(s.pf)]
    pf_stable    = all(v >= 1.0 for v in pf_values) if pf_values else False

    # Expectancy positive in every window with enough trades
    exp_stable   = all(
        s.expectancy > 0
        for s in sorted_
        if s.trades >= 10
    )

    # DD controlled everywhere
    dd_ok        = all(s.drawdown < 15.0 for s in sorted_)

    lines = []
    if scales:
        lines.append(f"trades grow {trade_counts[0]}→{trade_counts[-1]}")
    else:
        lines.append("trades NOT growing with history")
    if pf_stable:
        lines.append("PF stable")
    else:
        worst_pf = min(pf_values) if pf_values else 0
        lines.append(f"PF degrades (worst {worst_pf:.2f})")
    if not exp_stable:
        lines.append("expectancy turns negative in some windows")
    if not dd_ok:
        worst_dd = max(s.drawdown for s in sorted_)
        lines.append(f"DD breach (worst {worst_dd:.1f}%)")

    return ScalabilityResult(
        strategy_name      = strategy_name,
        scales_trade_count = scales,
        pf_stable          = pf_stable,
        exp_stable         = exp_stable,
        dd_controlled      = dd_ok,
        summary            = " | ".join(lines),
    )
