"""Amplification Report — Phase 5.3 output formatter.

Renders the AmplificationResearchResult into the exact validation report
format specified in the Phase 5.3 requirements.

No execution code. No broker code. No live trading.
"""

import math
import sys
from typing import TextIO

from src.edge_amplification.amplification_research_engine import (
    AmplificationResearchResult,
    AmplificationCandidate,
)

SEP   = "=" * 66
TSEP  = "-" * 66
INSUF = "INSUFFICIENT SAMPLE (< 50 trades)"


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


class AmplificationReport:
    """Renders the edge amplification validation report.

    Parameters
    ----------
    result : AmplificationResearchResult from AmplificationResearchEngine.run()
    out    : output stream (default stdout)
    """

    def __init__(
        self,
        result: AmplificationResearchResult,
        out: TextIO = None,
    ) -> None:
        self._r   = result
        self._out = out or sys.stdout

    def print(self) -> None:
        """Print the full validation report to the output stream."""
        r   = self._r
        out = self._out

        # ── Header ────────────────────────────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  EDGE AMPLIFICATION VALIDATION REPORT  —  Phase 5.3", file=out)
        print(f"  Strategy:    MOMENTUM_ROTATION", file=out)
        print(f"  Data source: {r.data_source}", file=out)
        print(f"  History:     {r.history_bars:,} D1 bars per asset", file=out)
        if r.data_source.startswith("SYNTHETIC"):
            print(f"  WARNING:     Synthetic data — re-run with IBKR for official results",
                  file=out)
        print(SEP, file=out)

        # ── Baseline ──────────────────────────────────────────────────────
        b = r.baseline
        print(f"\n{SEP}", file=out)
        print("  BASELINE STRATEGY", file=out)
        print(SEP, file=out)
        print(f"\n  Trades:      {b.trades}", file=out)
        print(f"  PF:          {_pf(b.profit_factor)}", file=out)
        print(f"  Expectancy:  ${b.expectancy:+.2f}/trade", file=out)
        print(f"  Win Rate:    {b.win_rate:.1%}", file=out)
        print(f"  DD:          {b.max_drawdown:.1f}%", file=out)
        print(f"  Gross Profit: ${b.gross_profit:,.2f}", file=out)
        print(f"  Gross Loss:   ${b.gross_loss:,.2f}", file=out)

        # ── Asset Contribution Ranking ────────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  ASSET CONTRIBUTION RANKING", file=out)
        print(SEP, file=out)
        print(f"\n  {'#':<4} {'Asset':<6}  {'Trades':>6}  {'PF':>6}  "
              f"{'Exp':>7}  {'WR':>6}  {'DD%':>5}  {'Contrib%':>9}  Note", file=out)
        print(f"  {TSEP}", file=out)
        for ac in r.asset_contributions:
            note = INSUF if ac.insufficient_sample else ""
            print(
                f"  {ac.rank:<4} {ac.symbol:<6}  {ac.trades:>6}  {_pf(ac.profit_factor):>6}  "
                f"${ac.expectancy:>+6.2f}  {ac.win_rate:>5.1%}  {ac.max_drawdown_pct:>4.1f}%  "
                f"{ac.contribution_pct:>+8.1f}%  {note}",
                file=out,
            )

        # ── Regime Contribution Ranking ───────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  REGIME CONTRIBUTION RANKING", file=out)
        print(SEP, file=out)
        print(f"\n  {'#':<4} {'Regime':<15}  {'Trades':>6}  {'PF':>6}  "
              f"{'Exp':>7}  {'Profit%':>8}  {'Loss%':>7}  Note", file=out)
        print(f"  {TSEP}", file=out)
        for rc in r.regime_contributions:
            note = INSUF if rc.insufficient_sample else ""
            print(
                f"  {rc.rank:<4} {rc.regime:<15}  {rc.trades:>6}  {_pf(rc.profit_factor):>6}  "
                f"${rc.expectancy:>+6.2f}  {rc.profit_pct:>7.1f}%  {rc.loss_pct:>6.1f}%  {note}",
                file=out,
            )

        # ── Volatility Contribution Ranking ───────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  VOLATILITY CONTRIBUTION RANKING", file=out)
        print(SEP, file=out)
        print(f"\n  {'#':<4} {'Regime':<15}  {'Trades':>6}  {'PF':>6}  "
              f"{'Exp':>7}  {'Profit%':>8}  {'Loss%':>7}  Note", file=out)
        print(f"  {TSEP}", file=out)
        for vc in r.vol_contributions:
            note = INSUF if vc.insufficient_sample else ""
            print(
                f"  {vc.rank:<4} {vc.regime:<15}  {vc.trades:>6}  {_pf(vc.profit_factor):>6}  "
                f"${vc.expectancy:>+6.2f}  {vc.profit_pct:>7.1f}%  {vc.loss_pct:>6.1f}%  {note}",
                file=out,
            )

        # ── Trade Quality Analysis ────────────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  TRADE QUALITY ANALYSIS", file=out)
        print(SEP, file=out)
        print(f"\n  {'Band':<10}  {'Trades':>6}  {'PF':>6}  "
              f"{'Exp':>7}  {'Edge':>8}  Note", file=out)
        print(f"  {TSEP}", file=out)
        for qb in r.quality_buckets:
            edge = "YES" if qb.has_edge else "NO"
            note = INSUF if qb.insufficient_sample else ""
            print(
                f"  {qb.band:<10}  {qb.trades:>6}  {_pf(qb.profit_factor):>6}  "
                f"${qb.expectancy:>+6.2f}  {edge:>8}  {note}",
                file=out,
            )

        # ── Holding Period Analysis ───────────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  HOLDING PERIOD ANALYSIS", file=out)
        print(SEP, file=out)
        print(f"\n  {'Band':<8}  {'Trades':>6}  {'PF':>6}  "
              f"{'Exp':>7}  {'Contrib%':>9}  Note", file=out)
        print(f"  {TSEP}", file=out)
        for hp in r.holding_periods:
            note = INSUF if hp.insufficient_sample else ""
            print(
                f"  {hp.band:<8}  {hp.trades:>6}  {_pf(hp.profit_factor):>6}  "
                f"${hp.expectancy:>+6.2f}  {hp.contribution_pct:>+8.1f}%  {note}",
                file=out,
            )

        # ── Profit Concentration ──────────────────────────────────────────
        pc = r.profit_concentration
        print(f"\n{SEP}", file=out)
        print("  PROFIT CONCENTRATION", file=out)
        print(SEP, file=out)
        print(f"\n  Total gross profit: ${pc.total_gross_profit:,.2f}", file=out)
        print(f"  Total winning trades: {pc.total_winning_trades}", file=out)

        print(f"\n  Top Contributors:", file=out)
        print(f"\n  Asset:", file=out)
        for e in pc.top_assets[:4]:
            print(f"    {e.label:<8}  ${e.gross_profit:>10,.2f}  ({e.pct_of_total:.1f}% of profits)",
                  file=out)
        if pc.asset_80_pct_threshold > 0:
            print(f"    80% of profits from top {pc.asset_80_pct_threshold:.0f}% of assets",
                  file=out)

        print(f"\n  Regime:", file=out)
        for e in pc.top_regimes[:4]:
            print(f"    {e.label:<15}  ${e.gross_profit:>10,.2f}  ({e.pct_of_total:.1f}%)",
                  file=out)

        print(f"\n  Volatility:", file=out)
        for e in pc.top_volatility[:4]:
            print(f"    {e.label:<15}  ${e.gross_profit:>10,.2f}  ({e.pct_of_total:.1f}%)",
                  file=out)

        print(f"\n  Quality:", file=out)
        for e in pc.top_quality[:4]:
            print(f"    {e.label:<10}  ${e.gross_profit:>10,.2f}  ({e.pct_of_total:.1f}%)",
                  file=out)

        # ── Loss Concentration ────────────────────────────────────────────
        lc = r.loss_concentration
        print(f"\n{SEP}", file=out)
        print("  LOSS CONCENTRATION", file=out)
        print(SEP, file=out)
        print(f"\n  Total gross loss:   ${lc.total_gross_loss:,.2f}", file=out)
        print(f"  Total losing trades: {lc.total_losing_trades}", file=out)

        print(f"\n  Worst Contributors:", file=out)
        print(f"\n  Asset:", file=out)
        for e in lc.worst_assets[:4]:
            print(f"    {e.label:<8}  ${e.gross_loss:>10,.2f}  ({e.pct_of_total:.1f}% of losses)",
                  file=out)

        print(f"\n  Regime:", file=out)
        for e in lc.worst_regimes[:4]:
            print(f"    {e.label:<15}  ${e.gross_loss:>10,.2f}  ({e.pct_of_total:.1f}%)",
                  file=out)

        print(f"\n  Volatility:", file=out)
        for e in lc.worst_volatility[:4]:
            print(f"    {e.label:<15}  ${e.gross_loss:>10,.2f}  ({e.pct_of_total:.1f}%)",
                  file=out)

        print(f"\n  Quality:", file=out)
        if isinstance(lc.worst_quality, list):
            for e in lc.worst_quality[:4]:
                print(f"    {e.label:<10}  ${e.gross_loss:>10,.2f}  ({e.pct_of_total:.1f}%)",
                      file=out)
        else:
            print(f"    {lc.worst_quality}", file=out)

        # ── Amplification Candidates ──────────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  AMPLIFICATION CANDIDATES", file=out)
        print(SEP, file=out)

        if not r.candidates:
            print(f"\n  No amplification candidates identified.", file=out)
            print(f"  All subgroups have positive expectancy or insufficient sample.", file=out)
        else:
            for c in r.candidates:
                print(f"\n  {'─'*62}", file=out)
                print(f"  Candidate #{c.candidate_id}", file=out)
                print(f"  {c.description}", file=out)
                print(f"\n  Dimension:         {c.dimension}", file=out)
                print(f"  Excluded segment:  {c.excluded_label} ({c.excluded_trades} trades)", file=out)
                print(f"  Remaining trades:  {c.remaining_trades}", file=out)
                if c.projected_pf > 0:
                    print(f"\n  Projected PF:      {_pf(c.projected_pf)}", file=out)
                    print(f"  PF Improvement:    +{c.pf_improvement:.2f}", file=out)
                    print(f"  Projected Exp:     ${c.projected_expectancy:+.2f}/trade", file=out)
                else:
                    print(f"\n  Expected PF Improvement: Positive (requires re-run to quantify)", file=out)
                print(f"\n  Trade Impact:      {c.trade_impact:+d} trades", file=out)
                print(f"  Min Trades Met:    {'YES' if c.meets_min_trades else 'NO'}", file=out)
                print(f"  Risk:              {c.risk_note}", file=out)

        # ── Final Recommendation ──────────────────────────────────────────
        print(f"\n{SEP}", file=out)
        print("  FINAL RECOMMENDATION", file=out)
        print(SEP, file=out)
        print(f"\n  {r.recommendation}", file=out)

        if r.recommendation == "PROCEED TO PHASE 5.4":
            viable = [c for c in r.candidates if c.meets_min_trades and c.pf_improvement > 0]
            print(f"\n  {len(viable)} viable amplification candidate(s) identified.", file=out)
            print(f"  All candidates require out-of-sample validation before implementation.", file=out)
        else:
            print(f"\n  No viable amplification candidates identified.", file=out)
            print(f"  Consider:", file=out)
            print(f"    1. Collecting more live data (IBKR validation)", file=out)
            print(f"    2. Reviewing strategy fundamentals", file=out)
            print(f"    3. Extending history window beyond {r.history_bars:,} bars", file=out)

        print(f"\n{SEP}\n", file=out)
