#!/usr/bin/env python3
"""Phase 4.9 Promotion Validation Script — Final Edge Confirmation.

Downloads maximum available candles for SPY, VOO, DIA and optional
expansion candidates (XLV, SCHD, VTI), then runs the full promotion
validation suite:

  1. History expansion (3000 → 3500 → 4000 → max candles)
  2. Candidate asset validation (one at a time)
  3. Robustness check (early / middle / recent windows)
  4. Promotion validation (5 criteria gate)
  5. Final recommendation: PROMOTE or RETURN_TO_RESEARCH

The strategy is LOCKED.  Any parameter change raises StrategyTamperedError.

Locked strategy:
  EMA 20/50 | Bull filter | ADX ≥ 27 | Breakout ≥ 1% | MEDIUM+HIGH vol

No orders placed. No trades executed. Research only.

Usage:
    python validate_promotion.py
    python validate_promotion.py --candles 5000
    python validate_promotion.py --candidate XLV --candidate SCHD
    python validate_promotion.py --skip-lock   # for test profiles
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
from src.promotion.promotion_engine import (
    LOCKED_PARAMS,
    PromotionEngine,
    PromotionReport,
)
from src.promotion.promotion_validator import (
    PROMOTION_MAX_DD,
    PROMOTION_MIN_PF,
    PROMOTION_MIN_TRADES,
)
from src.providers.provider_factory import ProviderFactory
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

CORE_ASSETS    = ["SPY", "VOO", "DIA"]
ALL_CANDIDATES = ["XLV", "SCHD", "VTI"]
REQUEST_DELAY  = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.9 promotion validation (read-only)")
    p.add_argument("--candles", type=int, default=5000,
                   help="Candles per asset (default: 5000 ≈ max available D1)")
    p.add_argument("--assets", nargs="+", default=CORE_ASSETS,
                   help="Core portfolio assets (default: SPY VOO DIA)")
    p.add_argument("--candidate", dest="candidates", action="append", default=None,
                   help="Expansion candidate to test (repeatable)")
    p.add_argument("--skip-lock", action="store_true",
                   help="Skip strategy lock verification (test use only)")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/promotion_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_report(report: PromotionReport) -> None:
    sep = "=" * 62

    print(f"\n{sep}")
    print("  PROMOTION VALIDATION REPORT")
    print(sep)

    # Strategy summary
    print(f"\n  Strategy:         {report.strategy_name}")
    print(f"  Locked params:    EMA{LOCKED_PARAMS['ema_fast']}/{LOCKED_PARAMS['ema_slow']} "
          f"ADX≥{LOCKED_PARAMS['adx_threshold']:.0f} "
          f"Brk{LOCKED_PARAMS['breakout_threshold']*100:.2f}%")

    # Key metrics
    print(f"\n  {'─'*58}")
    print(f"  Trades:           {report.trades}")
    print(f"  Profit Factor:    {_pf(report.profit_factor)}")
    print(f"  Expectancy:       ${report.expectancy:+.2f}/trade")
    print(f"  Max Drawdown:     {report.drawdown:.1f}%")
    print(f"  Assets Passing:   {report.assets_passing}")
    print(f"  History Length:   {report.history_length:,} candles")
    print(f"  Robustness:       {report.robustness_rating}")

    # History expansion
    if report.history_expansion_results:
        print(f"\n  {'─'*58}")
        print("  HISTORY EXPANSION")
        print(f"  {'Candles':>8} {'Trades':>7} {'PF':>6} {'Exp$/tr':>9} "
              f"{'DD%':>6} {'≥100?':>6} {'Quality':>8}")
        print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*8}")
        for r in report.history_expansion_results:
            meets = "✓" if r.meets_threshold else "✗"
            qual  = "✓" if r.meets_quality   else "✗"
            max_flag = " (max)" if r.is_max_available else ""
            print(
                f"  {r.candle_count:>8,} {r.total_trades:>7} {r.pf_str:>6} "
                f"{r.aggregate_exp:>+9.2f} {r.worst_dd:>5.1f}% {meets:>6} {qual:>8}"
                f"{max_flag}"
            )

    # Asset validation
    if report.asset_validation_results:
        print(f"\n  {'─'*58}")
        print("  CANDIDATE ASSET VALIDATION")
        print(f"  {'Symbol':<6} {'Δ Trades':>9} {'CandPF':>8} {'CandExp':>9} {'OK?':>6}")
        print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*9} {'-'*6}")
        for r in report.asset_validation_results:
            ok = "✓ ADD" if r.approved else "✗ SKIP"
            print(
                f"  {r.candidate_symbol:<6} {r.trade_increase:>+9} "
                f"{r.candidate_pf:>8.2f} {r.candidate_exp:>+9.2f} {ok:>6}"
            )

    # Robustness windows
    print(f"\n  {'─'*58}")
    print(f"  ROBUSTNESS — {report.robustness_rating}")
    print(f"  {'Window':<8} {'Trades':>7} {'PF':>6} {'Exp$/tr':>9} {'DD%':>6} {'Pass?':>6}")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*6}")
    for w in report.robustness_result.window_results:
        ok = "✓" if w.passes else "✗"
        print(
            f"  {w.window_name:<8} {w.total_trades:>7} {_pf(w.pf):>6} "
            f"{w.expectancy:>+9.2f} {w.drawdown:>5.1f}% {ok:>6}"
        )

    # Validation criteria
    val = report.validation_result
    print(f"\n  {'─'*58}")
    print("  PROMOTION CRITERIA")
    for reason in val.pass_reasons:
        print(f"    ✓ {reason}")
    for reason in val.failure_reasons:
        print(f"    ✗ {reason}")

    # Final decision
    print(f"\n{sep}")
    verdict = "PROMOTE TO PHASE 5" if report.promoted else "RETURN TO RESEARCH"
    print(f"  PROMOTION STATUS — {verdict}")
    print(sep)
    print(f"\n  {report.final_recommendation.headline}")

    if report.final_recommendation.caveats:
        print("\n  Caveats:")
        for c in report.final_recommendation.caveats:
            print(f"    ⚠ {c}")

    if report.final_recommendation.next_steps:
        print("\n  Next Steps:")
        for s in report.final_recommendation.next_steps:
            print(f"    → {s}")

    if report.notes:
        print("\n  Research Notes:")
        for n in report.notes:
            print(f"    → {n}")
    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")

    print(f"{sep}\n")


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    candidates = args.candidates or []

    print(f"\n{'='*62}")
    print("  PHASE 4.9 — PROMOTION VALIDATION & FINAL EDGE CONFIRMATION")
    print(f"{'='*62}")
    print(f"\n  LOCKED STRATEGY PARAMETERS:")
    for k, v in LOCKED_PARAMS.items():
        print(f"    {k}: {v}")
    print(f"\n  Core assets:     {', '.join(args.assets)}")
    if candidates:
        print(f"  Candidates:      {', '.join(candidates)}")
    print(f"  Candles:         {args.candles}")
    print(f"\n  Promotion gate:")
    print(f"    Trades ≥ {PROMOTION_MIN_TRADES}")
    print(f"    PF ≥ {PROMOTION_MIN_PF}")
    print(f"    Expectancy > $0")
    print(f"    Max DD < {PROMOTION_MAX_DD:.0f}%")
    print(f"    ≥ 2 assets individually pass")

    # ------------------------------------------------------------------ #
    # Connect and download                                                 #
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

    all_symbols = list(args.assets) + [c for c in candidates if c not in args.assets]
    asset_candles: Dict[str, List[Candle]] = {}

    try:
        for i, symbol in enumerate(all_symbols):
            print(f"  [{i+1}/{len(all_symbols)}] {symbol} D1 ({args.candles} bars)...")
            candles = provider.get_candles(symbol, "D1", args.candles)
            if candles:
                asset_candles[symbol] = candles
                span = (f"{candles[0].timestamp.strftime('%Y-%m-%d')} → "
                        f"{candles[-1].timestamp.strftime('%Y-%m-%d')}")
                print(f"           {len(candles)} candles  {span}")
            else:
                print(f"           WARNING: no data for {symbol}")
            if i < len(all_symbols) - 1:
                time.sleep(REQUEST_DELAY)
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    if not asset_candles:
        print("ERROR: No candle data retrieved.", file=sys.stderr)
        return 1

    # Separate core from candidates
    core_candles = {s: c for s, c in asset_candles.items() if s in args.assets}
    cand_candles = {s: c for s, c in asset_candles.items() if s not in args.assets}

    # ------------------------------------------------------------------ #
    # Run promotion engine                                                 #
    # ------------------------------------------------------------------ #
    print("Running promotion validation engine...")
    engine = PromotionEngine(config, BEST_DENSITY_PROFILE)
    report = engine.run(
        asset_candles     = core_candles,
        candidate_candles = cand_candles if cand_candles else None,
        skip_lock_check   = args.skip_lock,
    )

    print_report(report)

    print("No trades placed. No orders sent. Research only.")
    print(f"Full log: logs/promotion_validation.log\n")
    return 0 if report.promoted else 1


if __name__ == "__main__":
    sys.exit(main())
