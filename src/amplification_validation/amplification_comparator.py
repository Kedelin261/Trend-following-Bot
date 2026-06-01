"""Amplification Comparator — Phase 5.4.

Compares all 5 scenario results against the baseline,
computes delta PF and delta trades, ranks scenarios,
identifies promotion candidates, and derives the final conclusion.

Research only. No execution. No broker code.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.amplification_validation.amplification_validator import ScenarioResult


@dataclass
class ScenarioRanking:
    """Ranked entry for one scenario."""

    rank:          int
    scenario_name: str
    trades:        int
    profit_factor: float
    expectancy:    float
    max_drawdown:  float
    robustness:    str
    delta_pf:      Optional[float]    # vs baseline; None for baseline itself
    delta_trades:  Optional[int]
    is_promoted:   bool
    is_baseline:   bool

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"

    @property
    def delta_pf_str(self) -> str:
        if self.delta_pf is None:
            return "—"
        sign = "+" if self.delta_pf >= 0 else ""
        return f"{sign}{self.delta_pf:.3f}"

    @property
    def delta_trades_str(self) -> str:
        if self.delta_trades is None:
            return "—"
        sign = "+" if self.delta_trades >= 0 else ""
        return f"{sign}{self.delta_trades}"


@dataclass
class ComparisonReport:
    """Full comparison output from AmplificationComparator."""

    baseline:             ScenarioResult
    scenarios:            List[ScenarioResult]          # all 5 including baseline
    rankings:             List[ScenarioRanking]         # sorted best→worst by PF
    promotion_candidates: List[ScenarioResult]
    best_candidate:       Optional[ScenarioResult]
    most_effective_filter: Optional[str]                # scenario name
    best_pf_improvement:  Optional[float]
    best_trade_reduction: Optional[int]
    phase53_findings_held: bool                         # True if any filter improved PF
    recommendation:       str                           # PROCEED TO PHASE 5.5 or RETURN


class AmplificationComparator:
    """Compares scenarios, ranks them, and produces a ComparisonReport.

    Parameters
    ----------
    scenarios : list of ScenarioResult (must include BASELINE as first element
                or identifiable by scenario_name == "BASELINE")
    """

    def __init__(self, scenarios: List[ScenarioResult]) -> None:
        if not scenarios:
            raise ValueError("AmplificationComparator: scenarios list cannot be empty")
        self._scenarios = scenarios

    def compare(self) -> ComparisonReport:
        """Run comparison and return ComparisonReport."""
        baseline = self._find_baseline()

        # Compute deltas for every non-baseline scenario
        for sc in self._scenarios:
            if sc.scenario_name != "BASELINE":
                sc.delta_pf     = self._delta_pf(sc, baseline)
                sc.delta_trades = sc.trades - baseline.trades
            else:
                sc.delta_pf     = None
                sc.delta_trades = None

        # Rank all scenarios by PF descending; ties broken by expectancy
        sorted_sc = sorted(
            self._scenarios,
            key=lambda s: (
                s.profit_factor if not math.isinf(s.profit_factor) else 999.0,
                s.expectancy,
            ),
            reverse=True,
        )

        rankings = [
            ScenarioRanking(
                rank          = i + 1,
                scenario_name = sc.scenario_name,
                trades        = sc.trades,
                profit_factor = sc.profit_factor,
                expectancy    = sc.expectancy,
                max_drawdown  = sc.max_drawdown,
                robustness    = sc.robustness,
                delta_pf      = sc.delta_pf,
                delta_trades  = sc.delta_trades,
                is_promoted   = sc.is_promoted,
                is_baseline   = sc.scenario_name == "BASELINE",
            )
            for i, sc in enumerate(sorted_sc)
        ]

        # Promotion candidates (all scenarios where is_promoted=True)
        promotion_candidates = [s for s in self._scenarios if s.is_promoted]

        # Best promotion candidate: highest expectancy among promoted
        best_candidate: Optional[ScenarioResult] = None
        if promotion_candidates:
            best_candidate = max(
                promotion_candidates, key=lambda s: s.expectancy
            )

        # Most effective filter = non-baseline with highest PF improvement
        non_baseline = [s for s in self._scenarios if s.scenario_name != "BASELINE"]
        most_effective: Optional[str]   = None
        best_pf_imp:    Optional[float] = None
        best_trade_red: Optional[int]   = None

        if non_baseline:
            best = max(
                non_baseline,
                key=lambda s: s.delta_pf if s.delta_pf is not None else -999.0,
            )
            most_effective = best.scenario_name
            best_pf_imp    = best.delta_pf
            best_trade_red = best.delta_trades

        # Did Phase 5.3 findings hold? True if any filter improved PF
        phase53_held = any(
            (s.delta_pf is not None and s.delta_pf > 0)
            for s in non_baseline
        )

        # Recommendation
        if best_candidate is not None:
            recommendation = "PROCEED TO PHASE 5.5"
        elif phase53_held:
            # PF improved but didn't reach promotion threshold
            recommendation = "RETURN TO EDGE RESEARCH"
        else:
            recommendation = "RETURN TO EDGE RESEARCH"

        return ComparisonReport(
            baseline              = baseline,
            scenarios             = self._scenarios,
            rankings              = rankings,
            promotion_candidates  = promotion_candidates,
            best_candidate        = best_candidate,
            most_effective_filter = most_effective,
            best_pf_improvement   = best_pf_imp,
            best_trade_reduction  = best_trade_red,
            phase53_findings_held = phase53_held,
            recommendation        = recommendation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_baseline(self) -> ScenarioResult:
        for sc in self._scenarios:
            if sc.scenario_name == "BASELINE":
                return sc
        # Fallback: first element
        return self._scenarios[0]

    @staticmethod
    def _delta_pf(sc: ScenarioResult, baseline: ScenarioResult) -> float:
        """Return PF change vs baseline.  inf → 99 for arithmetic."""
        base_pf = baseline.profit_factor
        sc_pf   = sc.profit_factor
        if math.isinf(base_pf):
            base_pf = 99.0
        if math.isinf(sc_pf):
            sc_pf = 99.0
        return sc_pf - base_pf
