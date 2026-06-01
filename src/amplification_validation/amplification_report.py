"""Amplification Validation Report — Phase 5.4.

Formats the ComparisonReport into the exact output layout specified
in the Phase 5.4 brief.

Research only. No execution. No broker code.
"""

import math
import sys
from typing import IO, Optional

from src.amplification_validation.amplification_comparator import (
    ComparisonReport,
    ScenarioRanking,
)
from src.amplification_validation.amplification_validator import ScenarioResult

SEP  = "=" * 66
SEP2 = "-" * 66


class AmplificationValidationReport:
    """Renders the full Phase 5.4 validation report to stdout (or any stream).

    Parameters
    ----------
    report : ComparisonReport produced by AmplificationComparator
    out    : output stream (default sys.stdout)
    """

    def __init__(self, report: ComparisonReport, out: IO = None) -> None:
        self._r   = report
        self._out = out or sys.stdout

    def print(self) -> None:
        """Render the complete report."""
        r = self._r

        # --- Header ---
        self._line(SEP)
        self._line("  EDGE AMPLIFICATION VALIDATION REPORT")
        self._line(f"  Phase 5.4 — MOMENTUM_ROTATION Strategy")
        self._line(f"  Data Source: {r.baseline.data_source}")
        if r.baseline.data_source == "SYNTHETIC":
            self._line("  WARNING: Synthetic data — re-run with IBKR for official results")
        self._line(SEP)

        # --- Scenario sections ---
        self._section_result(r.baseline, baseline_pf=r.baseline.profit_factor)

        for sc in r.scenarios:
            if sc.scenario_name != "BASELINE":
                self._section_result(sc, baseline_pf=r.baseline.profit_factor)

        # --- Final rankings ---
        self._line(SEP)
        self._line("  FINAL RANKINGS")
        self._line(SEP)
        self._line("")
        self._line(
            f"  {'#':<4} {'Scenario':<26} {'Trades':>7} {'PF':>6} "
            f"{'Exp':>9} {'DD%':>6} {'Rob':<10} {'ΔPF':>8}"
        )
        self._line(f"  {SEP2}")

        for rank in r.rankings:
            delta = rank.delta_pf_str
            promo = "  ← PROMOTION CANDIDATE" if rank.is_promoted else ""
            self._line(
                f"  {rank.rank:<4} {rank.scenario_name:<26} "
                f"{rank.trades:>7} {rank.pf_str:>6} "
                f"${rank.expectancy:>+8.2f} "
                f"{rank.max_drawdown:>5.1f}% "
                f"{rank.robustness:<10} "
                f"{delta:>8}"
                f"{promo}"
            )
        self._line("")

        # --- Promotion candidate ---
        self._line(SEP)
        self._line("  PROMOTION CANDIDATE")
        self._line(SEP)
        self._line("")

        if r.best_candidate is not None:
            c = r.best_candidate
            self._line(f"  YES")
            self._line("")
            self._line(f"  Name:        {c.scenario_name}")
            self._line(f"  PF:          {c.pf_str}")
            self._line(f"  Trades:      {c.trades}")
            self._line(f"  Expectancy:  ${c.expectancy:+.2f}/trade")
            self._line(f"  DD:          {c.max_drawdown:.1f}%")
            self._line(f"  Robustness:  {c.robustness}")
            self._line("")
            self._line("  Promotion criteria met:")
            for pc in c.pass_criteria:
                self._line(f"    [✓] {pc}")
        else:
            self._line("  NO")
            self._line("")
            self._line("  No scenario met all promotion criteria.")
            # Show near-miss: best non-baseline by PF
            non_bl = [s for s in r.scenarios if s.scenario_name != "BASELINE"]
            if non_bl:
                near = max(non_bl, key=lambda s: s.profit_factor if not math.isinf(s.profit_factor) else 99.0)
                self._line(f"  Closest: {near.scenario_name} (PF={near.pf_str})")
                if near.fail_criteria:
                    self._line("  Promotion blockers:")
                    for fc in near.fail_criteria:
                        self._line(f"    [✗] {fc}")
        self._line("")

        # --- Final conclusion ---
        self._line(SEP)
        self._line("  FINAL CONCLUSION")
        self._line(SEP)
        self._line("")
        self._line(
            f"  Did Phase 5.3 findings hold up?  "
            f"{'YES' if r.phase53_findings_held else 'NO'}"
        )
        self._line("")
        if r.most_effective_filter is not None:
            pf_imp = r.best_pf_improvement or 0.0
            trade_r = r.best_trade_reduction or 0
            self._line(f"  Most effective filter:  {r.most_effective_filter}")
            self._line(
                f"  PF Improvement:         {'+' if pf_imp >= 0 else ''}{pf_imp:.3f}"
            )
            self._line(
                f"  Trade Reduction:        {trade_r:+d} trades"
            )
        else:
            self._line("  Most effective filter:  N/A")
        self._line("")

        # --- Recommendation ---
        self._line(SEP)
        self._line("  RECOMMENDATION")
        self._line(SEP)
        self._line("")
        self._line(f"  {r.recommendation}")
        self._line("")
        self._line(SEP)

    # ------------------------------------------------------------------
    # Section helpers
    # ------------------------------------------------------------------

    def _section_result(
        self,
        sc: ScenarioResult,
        baseline_pf: float,
    ) -> None:
        self._line(SEP)
        self._line(f"  {sc.scenario_name}")
        self._line(SEP)
        self._line("")
        self._line(f"  Description: {sc.description}")
        self._line("")
        self._line(f"  Trades:      {sc.trades}")
        self._line(f"  PF:          {sc.pf_str}")
        self._line(f"  Expectancy:  ${sc.expectancy:+.2f}/trade")
        self._line(f"  DD:          {sc.max_drawdown:.1f}%")
        self._line(f"  Robustness:  {sc.robustness}")

        if sc.delta_pf is not None:
            sign = "+" if sc.delta_pf >= 0 else ""
            self._line(f"  Delta PF:    {sign}{sc.delta_pf:.3f}")
        if sc.delta_trades is not None:
            self._line(f"  Delta Trades:{sc.delta_trades:+d}")

        self._line("")
        self._line(f"  Promotion:   {sc.promotion_status}")

        if sc.pass_criteria:
            self._line("  Pass:")
            for p in sc.pass_criteria:
                self._line(f"    [✓] {p}")
        if sc.fail_criteria:
            self._line("  Fail:")
            for f in sc.fail_criteria:
                self._line(f"    [✗] {f}")
        self._line("")

    def _line(self, text: str = "") -> None:
        print(text, file=self._out)
