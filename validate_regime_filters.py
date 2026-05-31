#!/usr/bin/env python3
"""Phase 4.11 Regime-Aware Entry Filter Research Validation Script.

Downloads maximum candles for SPY, VOO, DIA, then tests all predefined
regime filter profiles:

  NO_FILTER                       (baseline)
  AVOID_STRONG_BULL
  AVOID_EXTREME_VOL
  AVOID_STRONG_BULL_AND_EXTREME_VOL
  BULL_ONLY
  AVOID_CRISIS_AND_CRASH
  COMPREHENSIVE_FILTER

For each profile:
  - Computes PF, expectancy, drawdown after filtering
  - Compares against baseline
  - Runs robustness check (early/middle/recent windows)
  - Evaluates promotion readiness

No orders placed. No trades executed. Research only.

Usage:
    python validate_regime_filters.py
    python validate_regime_filters.py --candles 5000
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
from src.regime_filter.filter_backtester import FilterBacktestResult, FilterBacktester
from src.regime_filter.filter_comparator import FilterComparator
from src.regime_filter.filter_profiles import (
    AVOID_CRISIS_AND_CRASH,
    AVOID_EXTREME_VOL,
    AVOID_STRONG_BULL,
    AVOID_STRONG_BULL_AND_EXTREME_VOL,
    BULL_ONLY,
    COMPREHENSIVE_FILTER,
    NO_FILTER,
    RESEARCH_PROFILES,
)
from src.regime_filter.promotion_readiness_analyzer import PromotionReadinessAnalyzer
from src.regime_filter.robustness_improvement_analyzer import RobustnessImprovementAnalyzer
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

DEFAULT_ASSETS = ["SPY", "VOO", "DIA"]
REQUEST_DELAY  = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.11 regime filter research (read-only)")
    p.add_argument("--candles", type=int, default=5000,
                   help="Candles per asset (default: 5000)")
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/regime_filter_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_results(
    baseline:     FilterBacktestResult,
    all_results:  List[FilterBacktestResult],
    comparator:   FilterComparator,
    rob_analyzer: RobustnessImprovementAnalyzer,
    prom_analyzer: PromotionReadinessAnalyzer,
    asset_candles: Dict[str, List[Candle]],
) -> None:
    sep = "=" * 64

    print(f"\n{sep}")
    print("  REGIME FILTER RESEARCH REPORT")
    print(sep)

    # Baseline
    print(f"\n  BASELINE (NO_FILTER)")
    print(f"  Trades:     {baseline.total_trades_after}")
    print(f"  PF:         {_pf(baseline.profit_factor)}")
    print(f"  Expectancy: ${baseline.expectancy:+.2f}/trade")
    print(f"  Max DD:     {baseline.max_drawdown:.1f}%")
    print(f"  Quality:    {'PASS' if baseline.meets_quality else 'FAIL'}")

    # Comparison table
    print(f"\n  {'─'*60}")
    print(f"  {'Filter':<35} {'Trades':>7} {'PF':>6} {'Exp$/tr':>9} {'DD%':>6} {'Qual':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*6}")

    comparisons = comparator.compare_all(baseline, [r for r in all_results
                                                     if r.filter_profile.name != "NO_FILTER"])

    for comp in comparisons:
        f = comp.filtered
        qual = "✓" if f.meets_quality else "✗"
        imp  = "↑" if comp.is_improvement else " "
        print(
            f"  {f.filter_profile.name[:35]:<35} {f.total_trades_after:>7} "
            f"{_pf(f.profit_factor):>6} {f.expectancy:>+9.2f} "
            f"{f.max_drawdown:>5.1f}% {qual:>4}{imp}"
        )

    # Detailed results + robustness + promotion for each filter
    any_ready = False
    best_ready = None

    for r in all_results:
        if r.filter_profile.name == "NO_FILTER":
            continue

        print(f"\n  {'─'*60}")
        print(f"  {r.filter_profile.name}")
        print(f"  {r.filter_profile.description}")
        print(f"  Trades: {r.total_trades_after} (removed {r.trades_removed}, "
              f"retained {r.retention_pct:.0f}%)")
        print(f"  PF:     {_pf(r.profit_factor)}")
        print(f"  Exp:    ${r.expectancy:+.2f}/trade")
        print(f"  DD:     {r.max_drawdown:.1f}%")

        # Robustness
        print(f"  Running robustness check...")
        rob = rob_analyzer.analyze(asset_candles, r.filter_profile)
        print(f"  Robustness: {rob.rating} ({rob.windows_passing}/3 windows pass)  "
              f"{'↑ improved' if rob.improvement_over_base else ''}")

        # Promotion readiness
        prom = prom_analyzer.analyze(r, rob.rating)
        print(f"  Promotion: {prom.status}")
        if prom.all_passed:
            any_ready = True
            best_ready = prom
            for reason in prom.pass_reasons:
                print(f"    ✓ {reason}")
        else:
            for reason in prom.failure_reasons:
                print(f"    ✗ {reason}")

    # Final recommendation
    print(f"\n{sep}")
    if any_ready and best_ready:
        print(f"  PROMOTION READINESS — PASS")
        print(sep)
        print(f"\n  Profile: {best_ready.filter_profile.name}")
        print(f"  Trades:  {best_ready.actual_trades}")
        print(f"  PF:      {best_ready.pf_str}")
        print(f"  Exp:     ${best_ready.actual_expectancy:+.2f}/trade")
        print(f"  DD:      {best_ready.actual_dd:.1f}%")
        print(f"  Robustness: {best_ready.robustness_rating}")
        print(f"\n  RECOMMENDATION:")
        print(f"  Implement '{best_ready.filter_profile.name}' regime gate.")
        print(f"  Strategy is now promotion-ready for Phase 5 (Trade Journal).")
    else:
        print(f"  PROMOTION READINESS — FAIL")
        print(sep)
        print(f"\n  No filter combination achieves promotion criteria.")
        print(f"  RECOMMENDATION: Continue research — consider Phase 4.12")
        print(f"  (deeper regime analysis or additional asset universe expansion).")

    print(f"{sep}\n")
    print("No trades placed. No orders sent. Research only.")
    print(f"Full log: logs/regime_filter_validation.log\n")


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'='*64}")
    print("  PHASE 4.11 — REGIME-AWARE ENTRY FILTER RESEARCH")
    print(f"{'='*64}")
    print(f"\n  Locked strategy: {BEST_DENSITY_PROFILE.name}")
    print(f"  Assets:          {', '.join(args.assets)}")
    print(f"  Candles:         {args.candles}")
    print(f"\n  Filters to test:")
    for p in RESEARCH_PROFILES:
        print(f"    {p.name}: {p.description}")

    # Download
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

    asset_candles: Dict[str, List[Candle]] = {}
    try:
        for i, symbol in enumerate(args.assets):
            print(f"  [{i+1}/{len(args.assets)}] {symbol} D1 ({args.candles} bars)...")
            candles = provider.get_candles(symbol, "D1", args.candles)
            if candles:
                asset_candles[symbol] = candles
                span = (f"{candles[0].timestamp.strftime('%Y-%m-%d')} → "
                        f"{candles[-1].timestamp.strftime('%Y-%m-%d')}")
                print(f"           {len(candles)} candles  {span}")
            else:
                print(f"           WARNING: no data for {symbol}")
            if i < len(args.assets) - 1:
                time.sleep(REQUEST_DELAY)
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    if not asset_candles:
        print("ERROR: No candle data.", file=sys.stderr)
        return 1

    # Run all filters in single backtest pass
    print("Running filter backtester (single backtest, multiple filters)...")
    backtester    = FilterBacktester(config, BEST_DENSITY_PROFILE)
    all_results   = backtester.backtest_all_profiles(asset_candles, RESEARCH_PROFILES)
    baseline      = next(r for r in all_results if r.filter_profile.name == "NO_FILTER")

    comparator    = FilterComparator()
    rob_analyzer  = RobustnessImprovementAnalyzer(config, BEST_DENSITY_PROFILE)
    prom_analyzer = PromotionReadinessAnalyzer()

    print_results(baseline, all_results, comparator, rob_analyzer, prom_analyzer, asset_candles)

    return 0


if __name__ == "__main__":
    sys.exit(main())
