"""Edge report — generates the formatted validation output."""

import math
from typing import List, Optional

from src.edge_lab.edge_profile import EdgeProfile
from src.edge_lab.edge_comparator import EdgeComparator


def generate_report(
    profiles:        List[EdgeProfile],
    benchmark_name:  str = "BREAKOUT_V1",
) -> str:
    comparator = EdgeComparator()
    benchmark  = next((p for p in profiles if p.strategy_name == benchmark_name), None)

    lines = []
    lines.append("\n" + "=" * 64)
    lines.append("  ENTRY EDGE RESEARCH REPORT")
    lines.append("=" * 64)

    # Per-strategy results
    for p in profiles:
        lines.append("\n" + "-" * 64)
        lines.append(f"\n  {p.strategy_name}")
        lines.append(f"  {p.description}")
        lines.append("")
        lines.append(f"  Trades:      {p.total_trades}")
        lines.append(f"  PF:          {p.pf_str}")
        lines.append(f"  Expectancy:  ${p.expectancy:+.2f}/trade")
        lines.append(f"  Max DD:      {p.max_drawdown:.1f}%")
        lines.append(f"  Win Rate:    {p.win_rate*100:.1f}%")
        lines.append(f"  Sharpe:      {p.sharpe_ratio:.2f}")
        lines.append(f"  Robustness:  {p.robustness_rating}")
        if p.window_pf:
            wins = [f"{v:.2f}" if not math.isinf(v) else "∞" for v in p.window_pf]
            lines.append(f"  Windows PF:  early={wins[0]}  mid={wins[1]}  recent={wins[2]}")
        lines.append(f"  Assets OK:   {p.assets_passing}")
        lines.append(f"  Edge Score:  {p.edge_score:.1f}/100")
        lines.append(f"  Promotion:   {'✓ CANDIDATE' if p.promotion_candidate else '✗ ' + (p.rejection_reason or 'fails criteria')}")

        if benchmark and p.strategy_name != benchmark_name:
            comp = comparator.compare(p, benchmark)
            lines.append(f"  vs Benchmark: PF {comp.pf_delta:+.2f}  Exp {comp.exp_delta:+.2f}  "
                         f"DD {comp.dd_delta:+.1f}%  Trades {comp.trade_delta:+d}  "
                         f"Score {comp.score_delta:+.1f}")

    # Rankings
    lines.append("\n" + "=" * 64)
    lines.append("  FINAL RANKINGS")
    lines.append("=" * 64)
    for i, p in enumerate(profiles, 1):
        lines.append(
            f"  #{i} {p.strategy_name:<28} Score {p.edge_score:>5.1f}  "
            f"PF {p.pf_str:>5}  Exp ${p.expectancy:>+7.2f}  "
            f"{'✓' if p.promotion_candidate else '✗'}"
        )

    # Promotion candidates
    candidates = [p for p in profiles if p.promotion_candidate]
    lines.append("\n" + "=" * 64)
    if candidates:
        winner = candidates[0]
        lines.append("  PROMOTION CANDIDATE")
        lines.append("=" * 64)
        lines.append(f"\n  Name:        {winner.strategy_name}")
        lines.append(f"  Description: {winner.description}")
        lines.append(f"  PF:          {winner.pf_str}")
        lines.append(f"  Expectancy:  ${winner.expectancy:+.2f}/trade")
        lines.append(f"  Drawdown:    {winner.max_drawdown:.1f}%")
        lines.append(f"  Robustness:  {winner.robustness_rating}")
        lines.append(f"  Trades:      {winner.total_trades}")
        lines.append(f"  Edge Score:  {winner.edge_score:.1f}/100")
        lines.append("\n  RECOMMENDATION:  PROMOTE TO PHASE 5.1")
    else:
        lines.append("  NO PROMOTION CANDIDATE")
        lines.append("=" * 64)
        best = profiles[0] if profiles else None
        if best:
            lines.append(f"\n  Best strategy: {best.strategy_name} (score {best.edge_score:.1f})")
            lines.append(f"  Rejection:     {best.rejection_reason or 'criteria not met'}")
        lines.append("\n  RECOMMENDATION:  RETURN TO EDGE RESEARCH")
        if profiles:
            top = profiles[0]
            missing = []
            from src.edge_lab.edge_ranking import (PROMO_MIN_TRADES, PROMO_MIN_PF,
                                                    PROMO_MAX_DD, PROMO_MIN_ASSETS)
            if top.total_trades < PROMO_MIN_TRADES:
                missing.append(f"extend history (need {PROMO_MIN_TRADES} trades, got {top.total_trades})")
            safe = top.profit_factor if not math.isinf(top.profit_factor) else 99.0
            if safe < PROMO_MIN_PF:
                missing.append(f"improve PF (need {PROMO_MIN_PF}, got {safe:.2f})")
            if top.max_drawdown >= PROMO_MAX_DD:
                missing.append(f"reduce DD (need <{PROMO_MAX_DD}%, got {top.max_drawdown:.1f}%)")
            if missing:
                lines.append(f"  Focus:  {' | '.join(missing)}")

    lines.append("=" * 64)
    return "\n".join(lines)
