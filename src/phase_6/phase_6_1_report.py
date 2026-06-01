"""Phase 6.1 Report Formatter.

Formats and prints the PHASE 6.1 VALIDATION REPORT in the exact
required format:

    PHASE 6.1 VALIDATION REPORT
    DATA SOURCE: ...
    RESULTS: ...
    GATE STATUS: PASS / FAIL
    NEXT ACTION: ...
"""

from __future__ import annotations

from typing import List, Optional

from src.phase_6.scalability_validator import (
    BAR_HORIZONS,
    GATE_MIN_EXPECTANCY,
    GATE_MIN_PF,
    GATE_MIN_TRADES,
    CandidateScalabilityResult,
    HorizonResult,
    ScalabilityReport,
)


class Phase61Report:
    """Formats and prints the Phase 6.1 Validation Report."""

    def __init__(self, report: ScalabilityReport) -> None:
        self._report = report

    def print(self) -> None:
        """Print the full validation report to stdout."""
        lines = self._format()
        for line in lines:
            print(line)

    def _format(self) -> List[str]:
        r   = self._report
        out = []

        out += [
            "",
            "=" * 70,
            "  PHASE 6.1 VALIDATION REPORT",
            "  Edge Scalability Validation",
            "=" * 70,
            "",
            f"DATA SOURCE: {r.data_source}",
            "",
            f"Gate Criteria:",
            f"  PF >= {GATE_MIN_PF}",
            f"  Expectancy > {GATE_MIN_EXPECTANCY}",
            f"  Trades >= {GATE_MIN_TRADES}",
            f"  (At least one candidate must satisfy ALL at any horizon)",
            "",
            "=" * 70,
            "  RESULTS",
            "=" * 70,
            "",
        ]

        # Candidate sections
        for cand in r.candidates:
            out += self._format_candidate(cand)

        # Benchmark section
        if r.benchmark:
            out += [
                "─" * 70,
                f"  BENCHMARK: {r.benchmark.family_name}",
                "─" * 70,
            ]
            out += self._format_horizon_table(r.benchmark)
            out += [""]

        # Gate summary
        out += [
            "=" * 70,
            "  GATE EVALUATION",
            "=" * 70,
            "",
        ]
        for cand in r.candidates:
            for bars in BAR_HORIZONS:
                h = cand.get_horizon(bars)
                if h:
                    status = "PASS" if h.passes_gate else "FAIL"
                    out.append(
                        f"  {cand.family_name:<35s} @ {bars:>5d} bars: {status}"
                    )
        out += [""]

        # Gate decision
        gate_str = "PASS" if r.gate_passes else "FAIL"
        out += [
            "=" * 70,
            f"  GATE STATUS: {gate_str}",
            "=" * 70,
            "",
            f"  {r.gate_reason}",
            "",
        ]

        # Next action
        if r.gate_passes:
            out += [
                "=" * 70,
                "  NEXT ACTION: Proceed to Phase 6.2",
                "=" * 70,
            ]
        else:
            out += [
                "=" * 70,
                "  NEXT ACTION: Return to Discovery",
                "=" * 70,
            ]

        out += [""]
        return out

    def _format_candidate(
        self, cand: CandidateScalabilityResult
    ) -> List[str]:
        out = [
            "─" * 70,
            f"  CANDIDATE: {cand.family_name}",
            "─" * 70,
        ]
        out += self._format_horizon_table(cand)
        out += [""]
        return out

    @staticmethod
    def _format_horizon_table(
        cand: CandidateScalabilityResult,
    ) -> List[str]:
        out = [
            f"  {'Horizon':>8}  {'Trades':>7}  {'PF':>8}  "
            f"{'Exp($)':>8}  {'DD%':>6}  {'WR%':>6}  {'Robust':>8}  Gate",
            "  " + "-" * 66,
        ]
        for bars in BAR_HORIZONS:
            h = cand.get_horizon(bars)
            if h is None:
                out.append(f"  {bars:>8d}  {'N/A':>7}  {'N/A':>8}  "
                           f"{'N/A':>8}  {'N/A':>6}  {'N/A':>6}  {'N/A':>8}  N/A")
                continue
            pf_str = "inf" if h.profit_factor == float("inf") else f"{h.profit_factor:.3f}"
            gate   = "PASS" if h.passes_gate else "fail"
            out.append(
                f"  {h.bar_count:>8d}  {h.trades:>7d}  {pf_str:>8}  "
                f"{h.expectancy:>8.2f}  {h.max_drawdown:>6.1f}  "
                f"{h.win_rate*100:>6.1f}  {h.robustness:>8}  {gate}"
            )
        return out
