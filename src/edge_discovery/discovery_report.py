"""DiscoveryReport — prints the Phase 6.0 Edge Discovery Validation Report.

Exact format required by the Phase 6.0 specification.
Prints to stdout.  No return value.

DATA SOURCE RULE: synthetic data results are clearly labeled.
No promotion language appears on synthetic-data reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.edge_discovery.discovery_comparator import ComparisonReport

_LINE = "=" * 54
_DASH = "-" * 54


class DiscoveryReport:
    """Print the EDGE DISCOVERY VALIDATION REPORT.

    Usage:
        report = DiscoveryReport(comparison_report, data_source)
        report.print()
    """

    def __init__(
        self,
        comparison: "ComparisonReport",
        data_source: str,
        asset_names: list = None,
    ) -> None:
        self._c          = comparison
        self._data_source = data_source
        self._assets     = asset_names or []

    def print(self) -> None:
        """Output the full validation report to stdout."""
        c = self._c
        print()
        print("EDGE DISCOVERY VALIDATION REPORT")
        print(_LINE)
        print()
        print("DATA SOURCE")
        print()
        print(self._data_source)
        print()
        if self._data_source == "SYNTHETIC":
            print("NOTE: Synthetic data — no promotion recommendations.")
        print()
        print(_LINE)
        print()
        print("STRATEGY FAMILY RANKINGS")
        print()
        for r in c.rankings:
            pf_str = "∞" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
            status = "INSUFFICIENT SAMPLE" if r.insufficient else ""
            print(f"#{r.rank}")
            print(f"Name:       {r.family_name}")
            print(f"Edge Score: {r.edge_score:.1f}")
            print(f"PF:         {pf_str}")
            print(f"Expectancy: ${r.expectancy:.2f}")
            print(f"DD:         {r.max_drawdown:.1f}%")
            print(f"Trades:     {r.trade_count}")
            print(f"Robustness: {r.robustness}")
            if status:
                print(f"Status:     {status}")
            print(_DASH)

        # Benchmark
        if c.benchmark_ranking:
            b = c.benchmark_ranking
            pf_str = "∞" if b.profit_factor == float("inf") else f"{b.profit_factor:.2f}"
            print()
            print("BENCHMARK (MOMENTUM_ROTATION)")
            print(f"Name:       {b.family_name}")
            print(f"Edge Score: {b.edge_score:.1f}")
            print(f"PF:         {pf_str}")
            print(f"Expectancy: ${b.expectancy:.2f}")
            print(f"DD:         {b.max_drawdown:.1f}%")
            print(f"Trades:     {b.trade_count}")
            print(f"Robustness: {b.robustness}")
            print()

        print(_LINE)
        print()
        self._print_family_sections()

        print(_LINE)
        print()
        print("DISCOVERY WINNER")
        print()
        if c.discovery_winner:
            w = c.discovery_winner
            pf_str = "∞" if w.profit_factor == float("inf") else f"{w.profit_factor:.2f}"
            print(f"Strategy Family: {w.family_name}")
            print(f"Edge Score:      {w.edge_score:.1f}")
            reason = self._winner_reason(w)
            print(f"Reason:          {reason}")
        else:
            print("Strategy Family: NONE (all families insufficient sample)")
            print("Edge Score:      0.0")
            print("Reason:          No strategy family met minimum sample criteria")
        print()

        print(_LINE)
        print()
        print("PROMISING CANDIDATES")
        print()
        if c.promising_candidates:
            for i, cand in enumerate(c.promising_candidates, start=1):
                pf_str = "∞" if cand.profit_factor == float("inf") else f"{cand.profit_factor:.2f}"
                print(f"Candidate #{i}")
                print(f"  Name:       {cand.family_name}")
                print(f"  Edge Score: {cand.edge_score:.1f}")
                print(f"  PF:         {pf_str} | Exp: ${cand.expectancy:.2f} | DD: {cand.max_drawdown:.1f}%")
        else:
            print("No promising candidates identified at this data level.")
        print()

        print(_LINE)
        print()
        print("RECOMMENDATION")
        print()
        recommendation = self._recommendation(c)
        print(recommendation)
        print()
        print(_LINE)

    # ------------------------------------------------------------------ #
    # Per-family section printer                                           #
    # ------------------------------------------------------------------ #

    def _print_family_sections(self) -> None:
        """Print individual detailed sections for each research family."""
        family_order = [
            "TREND_PERSISTENCE",
            "BREAKOUT_CONTINUATION",
            "RELATIVE_STRENGTH",
            "MARKET_LEADERSHIP",
            "VOLATILITY_TRANSITION",
            "MULTI_TIMEFRAME_ALIGNMENT",
        ]

        # Build lookup by family_id value
        by_id = {r.family_id.value: r for r in self._c.rankings}

        titles = {
            "TREND_PERSISTENCE":          "TREND PERSISTENCE",
            "BREAKOUT_CONTINUATION":       "BREAKOUT CONTINUATION",
            "RELATIVE_STRENGTH":           "RELATIVE STRENGTH ROTATION",
            "MARKET_LEADERSHIP":           "MARKET LEADERSHIP",
            "VOLATILITY_TRANSITION":       "VOLATILITY COMPRESSION EXPANSION",
            "MULTI_TIMEFRAME_ALIGNMENT":   "MULTI-TIMEFRAME ALIGNMENT",
        }

        for fid in family_order:
            r = by_id.get(fid)
            title = titles.get(fid, fid)
            print(title)
            print()
            if r is None:
                print("  NOT EVALUATED")
            elif r.insufficient:
                print("  Trades:     INSUFFICIENT SAMPLE")
                print("  PF:         N/A")
                print("  Expectancy: N/A")
                print("  DD:         N/A")
                print("  Robustness: N/A")
                print("  Edge Score: 0.0")
            else:
                pf_str = "∞" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
                print(f"  Trades:     {r.trade_count}")
                print(f"  PF:         {pf_str}")
                print(f"  Expectancy: ${r.expectancy:.2f}")
                print(f"  DD:         {r.max_drawdown:.1f}%")
                print(f"  Robustness: {r.robustness}")
                print(f"  Edge Score: {r.edge_score:.1f}")
            print()
            print(_LINE)
            print()

    # ------------------------------------------------------------------ #
    # Recommendation logic                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _winner_reason(winner) -> str:
        if winner.profit_factor >= 1.50 and winner.trade_count >= 500:
            return "Highest edge score with strong PF and trade count"
        elif winner.profit_factor >= 1.20:
            return "Highest edge score — promising PF, further research warranted"
        elif winner.edge_score > 0:
            return "Highest edge score among evaluated families"
        else:
            return "Default selection — marginal performance"

    def _recommendation(self, c: "ComparisonReport") -> str:
        if self._data_source == "SYNTHETIC":
            synthetic_note = (
                "NOTE: Results based on SYNTHETIC data.\n"
                "Official research requires LIVE IBKR historical data.\n"
                "No promotion decisions may be made from synthetic results.\n\n"
            )
        else:
            synthetic_note = ""

        # Check if any family looks genuinely promising
        promising_count = len(c.promising_candidates)
        top_score = c.discovery_winner.edge_score if c.discovery_winner else 0.0

        if top_score >= 40.0 and promising_count >= 2:
            verdict = "PROCEED TO PHASE 6.1"
        elif top_score >= 20.0 or promising_count >= 1:
            verdict = "PROCEED TO PHASE 6.1"
        else:
            verdict = "RETURN TO EDGE DISCOVERY"

        return f"{synthetic_note}{verdict}"
