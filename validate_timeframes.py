#!/usr/bin/env python3
"""Phase 4.8 Timeframe Expansion Research Validation Script.

Downloads 3000 candles for SPY, VOO, DIA on D1, H4, H1, W1 timeframes
then runs the full timeframe research suite:

  1. Single-timeframe backtests (D1, H4, H1, W1)
  2. D1 vs H4 / H1 / W1 comparisons
  3. Multi-timeframe combinations: D1+H4, D1+H1, D1+H4+H1
  4. Signal overlap analysis (overlap ≥ 50% → rejected)
  5. Opportunity flow analysis (trades/year, trades/month)
  6. Promotion decision (≥100 unique trades, PF≥1.50, Exp>0, DD<15%)

No orders placed. No trades executed. Research only.

Usage:
    python validate_timeframes.py
    python validate_timeframes.py --candles 2000 --quick
    python validate_timeframes.py --timeframes D1 H4
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings_loader import load_settings
from src.data.models import Candle
from src.data.symbol_registry import SymbolRegistry
from src.providers.provider_factory import ProviderFactory
from src.timeframe.timeframe_engine import PORTFOLIO_MIN_TRADES, TimeframeEngine, TimeframeReport
from src.timeframe.timeframe_profile import (
    ALL_COMBOS,
    ALL_SINGLE_PROFILES,
    BEST_DENSITY_PROFILE,
    D1_PROFILE,
    H1_PROFILE,
    H4_PROFILE,
    W1_PROFILE,
    TimeframeProfile,
)

DEFAULT_ASSETS     = ["SPY", "VOO", "DIA"]
DEFAULT_TIMEFRAMES = ["D1", "H4", "H1", "W1"]
REQUEST_DELAY_S    = 2.0

TF_PROFILE_MAP: Dict[str, TimeframeProfile] = {
    "D1": D1_PROFILE,
    "H4": H4_PROFILE,
    "H1": H1_PROFILE,
    "W1": W1_PROFILE,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.8 timeframe research (read-only)")
    p.add_argument("--candles", type=int, default=3000,
                   help="Candles per asset/timeframe (default: 3000)")
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                   help="Assets to research (default: SPY VOO DIA)")
    p.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                   help="Timeframes to test (default: D1 H4 H1 W1)")
    p.add_argument("--quick", action="store_true",
                   help="D1 + H4 only, skip H1/W1 (faster run)")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/timeframe_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_report(report: TimeframeReport) -> None:
    sep = "=" * 60

    print(f"\n{sep}")
    print("  TIMEFRAME RESEARCH REPORT")
    print(sep)

    # ---- Single timeframe results ----------------------------------
    for tf, sym_res in report.single_tf_results.items():
        if not sym_res:
            continue
        print(f"\n  {'─'*54}")
        print(f"  {tf}")
        print(f"  {'─'*54}")
        print(f"  {'Asset':<8} {'Trades':>7} {'Win%':>6} {'Exp$/tr':>9} "
              f"{'PF':>6} {'DD%':>6} {'Quality':>8}")
        print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*8}")
        for sym, r in sym_res.items():
            q = "✓" if r.meets_quality else "✗"
            print(
                f"  {sym:<8} {r.trade_count:>7} {r.results.win_rate*100:>5.0f}% "
                f"{r.results.expectancy:>+9.2f} {_pf(r.results.profit_factor):>6} "
                f"{r.results.max_drawdown:>5.1f}% {q:>8}"
            )
        # Portfolio total for this TF
        total = sum(r.trade_count for r in sym_res.values())
        print(f"  {'Portfolio':<8} {total:>7}")

        if tf in report.opportunity_metrics:
            oppo = report.opportunity_metrics[tf]
            print(f"  Trades/yr: {oppo.aggregate_trades_per_year:.0f}  "
                  f"Monthly opps: {oppo.aggregate_monthly_opps:.1f}  "
                  f"{'≥100 ✓' if oppo.meets_100_target else '<100'}")

    # ---- Multi-timeframe combinations ------------------------------
    if report.combo_results:
        print(f"\n  {'─'*54}")
        print("  MULTI-TIMEFRAME COMBINATIONS")
        print(f"  {'─'*54}")
        print(f"  {'Combo':<14} {'Raw':>5} {'Unique':>7} {'Overlap':>8} "
              f"{'PF':>6} {'Exp$/tr':>9} {'DD%':>6} {'Viable':>7}")
        print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*9} {'-'*6} {'-'*7}")
        for r in report.combo_results:
            viable = "✓" if r.is_viable else "✗"
            print(
                f"  {r.label:<14} {r.total_raw_trades:>5} {r.unique_trades:>7} "
                f"{r.overlap.average_overlap_pct*100:>7.0f}% "
                f"{_pf(r.combined_pf):>6} {r.combined_expectancy:>+9.2f} "
                f"{r.max_drawdown:>5.1f}% {viable:>7}"
            )

    # ---- Comparisons to D1 baseline --------------------------------
    if report.comparisons:
        print(f"\n  D1 BASELINE COMPARISONS")
        for sym, comps in report.comparisons.items():
            if not comps:
                continue
            print(f"\n  {sym}:")
            for c in comps:
                delta_sign = "+" if c.trade_count_change >= 0 else ""
                print(
                    f"    D1 vs {c.comparison_tf:<4}: "
                    f"trades {c.baseline_trades}→{c.comparison_trades} "
                    f"({delta_sign}{c.trade_count_change})  "
                    f"PF {c.baseline_pf:.2f}→{c.comparison_pf:.2f}  "
                    f"Exp ${c.baseline_exp:.2f}→${c.comparison_exp:.2f}  "
                    f"[{c.recommendation}]"
                )

    # ---- Promotion -------------------------------------------------
    print(f"\n{sep}")
    verdict = "PROMOTE ✓" if report.promoted else "DO NOT PROMOTE ✗"
    print(f"  PORTFOLIO STATUS — {verdict}")
    print(sep)
    print(f"\n  {report.promotion_reason}")

    if report.notes:
        print("\n  Research Notes:")
        for n in report.notes:
            print(f"    → {n}")
    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")

    print(f"\n{sep}")
    print(f"  RECOMMENDATION")
    print(sep)
    print(f"  {report.recommendation}")

    if report.best_combo and report.promoted:
        bc = report.best_combo
        print(f"\n  Best combination: {bc.label}")
        print(f"    Unique trades:  {bc.unique_trades}")
        print(f"    PF:             {_pf(bc.combined_pf)}")
        print(f"    Expectancy:     ${bc.combined_expectancy:+.2f}/trade")
        print(f"    Max DD:         {bc.max_drawdown:.1f}%")
        print(f"    Overlap:        {bc.overlap.average_overlap_pct*100:.0f}%")

    print(f"{sep}\n")


def main() -> int:
    args = parse_args()
    setup_logging()

    tf_list = ["D1", "H4"] if args.quick else args.timeframes

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print("  PHASE 4.8 TIMEFRAME EXPANSION RESEARCH")
    print(f"{'='*60}")
    print(f"\n  Strategy: {BEST_DENSITY_PROFILE.name}")
    print(f"  EMA:       {BEST_DENSITY_PROFILE.ema_fast}/{BEST_DENSITY_PROFILE.ema_slow}")
    print(f"  ADX:       ≥ {BEST_DENSITY_PROFILE.adx_threshold:.0f}")
    print(f"  Breakout:  {BEST_DENSITY_PROFILE.breakout_threshold*100:.2f}%")
    print(f"  Vol mode:  {BEST_DENSITY_PROFILE.volatility_mode.value}")
    print(f"\n  Assets:    {', '.join(args.assets)}")
    print(f"  Timeframes:{', '.join(tf_list)}")
    print(f"  Candles:   {args.candles} per asset/timeframe")
    print(f"\n  Promotion: {PORTFOLIO_MIN_TRADES}+ unique trades, PF≥1.50, Exp>0, DD<15%, Overlap<50%")

    # ------------------------------------------------------------------ #
    # Download candles for each (asset, timeframe) combination            #
    # ------------------------------------------------------------------ #
    symbol_registry = SymbolRegistry.from_config(config)
    try:
        provider = ProviderFactory.create(config, symbol_registry)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    provider_name = config.get("provider", "?").upper()
    print(f"\nConnecting to {provider_name}...")
    if not provider.connect():
        print("ERROR: Could not connect to provider.", file=sys.stderr)
        return 1
    print("  OK\n")

    # Build {symbol: {timeframe: candles}}
    symbol_candles_by_tf: Dict[str, Dict[str, List[Candle]]] = {
        sym: {} for sym in args.assets
    }

    total_requests = len(args.assets) * len(tf_list)
    req_num = 0
    try:
        for symbol in args.assets:
            for tf in tf_list:
                req_num += 1
                print(f"  [{req_num}/{total_requests}] {symbol} {tf} ({args.candles} bars)...")
                candles = provider.get_candles(symbol, tf, args.candles)
                if candles:
                    symbol_candles_by_tf[symbol][tf] = candles
                    span = (f"{candles[0].timestamp.strftime('%Y-%m-%d')} → "
                            f"{candles[-1].timestamp.strftime('%Y-%m-%d')}")
                    print(f"           {len(candles)} candles  {span}")
                else:
                    print(f"           WARNING: no data for {symbol}/{tf}")
                if req_num < total_requests:
                    time.sleep(REQUEST_DELAY_S)
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    # Remove assets with no candles
    symbol_candles_by_tf = {
        sym: tf_map
        for sym, tf_map in symbol_candles_by_tf.items()
        if tf_map
    }

    if not symbol_candles_by_tf:
        print("ERROR: No candle data retrieved.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Run timeframe engine                                                 #
    # ------------------------------------------------------------------ #
    single_profiles = [TF_PROFILE_MAP[tf] for tf in tf_list if tf in TF_PROFILE_MAP]
    # Only include combos that use available timeframes
    available_tfs = set(tf_list)
    combo_profiles = [
        c for c in ALL_COMBOS
        if all(p.timeframe in available_tfs for p in c.profiles)
    ]

    print(f"Running timeframe engine...")
    print(f"  Single profiles: {[p.name for p in single_profiles]}")
    print(f"  Combinations:    {[c.label for c in combo_profiles]}\n")

    engine = TimeframeEngine(config)
    report = engine.run(
        symbol_candles_by_tf = symbol_candles_by_tf,
        single_profiles      = single_profiles,
        combo_profiles       = combo_profiles,
    )

    print_report(report)

    print("No trades placed. No orders sent. Research only.")
    print(f"Full log: logs/timeframe_validation.log\n")
    return 0 if report.promoted else 1


if __name__ == "__main__":
    sys.exit(main())
