"""Strategy survivability — determines if strategy survives long history,
multiple assets, and multiple market cycles.

SURVIVES: quality holds across all tested conditions.
FAILS: quality collapses in at least one dimension.
"""

import math
from dataclasses import dataclass
from typing import List

from src.edge_validation.history_expansion import HistorySliceResult
from src.edge_validation.asset_expansion import AssetExpansionResult
from src.edge_validation.robustness_validator import RobustnessValidationResult

MIN_WINDOW_PF  = 1.0
MIN_WINDOW_EXP = 0.0


@dataclass
class SurvivabilityResult:
    strategy_name:     str
    survives:          bool
    history_survives:  bool
    asset_survives:    bool
    robustness_ok:     bool
    failure_reasons:   List[str]
    pass_reasons:      List[str]

    @property
    def verdict(self) -> str:
        return "SURVIVES" if self.survives else "FAILS"


def evaluate_survivability(
    strategy_name:  str,
    history_slices: List[HistorySliceResult],
    asset_result:   AssetExpansionResult,
    robustness:     RobustnessValidationResult,
) -> SurvivabilityResult:
    failures = []
    passes   = []

    # History survival: no window collapses completely
    for s in history_slices:
        if s.trades < 5:
            continue
        safe_pf = s.pf if not math.isinf(s.pf) else 99.0
        if safe_pf < MIN_WINDOW_PF:
            failures.append(f"PF collapses to {safe_pf:.2f} at {s.actual_bars} bars")
        if s.expectancy < MIN_WINDOW_EXP:
            failures.append(f"Expectancy negative (${s.expectancy:.2f}) at {s.actual_bars} bars")

    history_ok = not any("collapses" in f or "negative" in f for f in failures)
    if history_ok:
        passes.append("Quality holds across all history windows")

    # Asset survival: portfolio PF positive, at least 2 assets add value
    p = asset_result.portfolio_profile
    safe_port_pf = p.profit_factor if not math.isinf(p.profit_factor) else 99.0
    if safe_port_pf < MIN_WINDOW_PF:
        failures.append(f"Portfolio PF collapses to {safe_port_pf:.2f}")
    if p.expectancy < MIN_WINDOW_EXP:
        failures.append(f"Portfolio expectancy negative (${p.expectancy:.2f})")

    approved = asset_result.approved_assets
    if len(approved) < 2:
        failures.append(f"Only {len(approved)} asset(s) add value (need ≥ 2)")

    asset_ok = (safe_port_pf >= MIN_WINDOW_PF and p.expectancy >= MIN_WINDOW_EXP
                and len(approved) >= 2)
    if asset_ok:
        passes.append(f"Portfolio quality holds across {len(asset_result.asset_contributions)} assets")

    # Robustness
    rob_ok = robustness.rating != "UNSTABLE"
    if rob_ok:
        passes.append(f"Robustness: {robustness.rating} ({robustness.windows_passing}/3 windows)")
    else:
        failures.append(f"Robustness UNSTABLE — only {robustness.windows_passing}/3 windows pass")

    survives = history_ok and asset_ok and rob_ok

    return SurvivabilityResult(
        strategy_name    = strategy_name,
        survives         = survives,
        history_survives = history_ok,
        asset_survives   = asset_ok,
        robustness_ok    = rob_ok,
        failure_reasons  = failures,
        pass_reasons     = passes,
    )
