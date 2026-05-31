#!/usr/bin/env python3
"""Phase 4.6 Strategy Refinement Validation Script.

Compares Strategy V1 (current) vs Strategy V2 (refined) on
SPY, VOO, and DIA using 1500 candles of historical data.

V2 adds:
  - BULL regime filter (EMA20 > EMA50 > EMA200)
  - ADX ≥ 25 trend-strength gate
  - MEDIUM volatility filter
  - 1.00 % breakout threshold (vs 0.25 % in V1)
  - EMA20/50 trend detector (vs EMA50/200 in V1)

No orders placed. No trades executed. Research only.

Usage:
    python validate_refinement.py
    python validate_refinement.py --candles 2000
    python validate_refinement.py --assets SPY DIA VOO
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
from src.refinement.asset_selector import AssetSelector
from src.refinement.refinement_engine import RefinementEngine
from src.refinement.strategy_comparator import PROMOTION_MIN_PF, PROMOTION_MIN_TRADES
from src.refinement.strategy_v2 import V1_PROFILE, V2_PROFILE

DEFAULT_ASSETS = ["SPY", "VOO", "DIA"]
REQUEST_DELAY_S = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.6 strategy refinement (read-only)")
    p.add_argument("--candles", type=int, default=1500,
                   help="Candles per asset (default: 1500 ≈ 6 years D1)")
    p.add_argument("--timeframe", default="D1",
                   help="Candle timeframe (default: D1)")
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                   help="Assets to compare (default: SPY VOO DIA)")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/refinement_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_comparison_table(report) -> None:
    sep = "=" * 62

    print(f"\n{sep}")
    print("  STRATEGY COMPARISON — V1 (Current) vs V2 (Refined)")
    print(sep)

    # Header
    print(f"\n  {'Metric':<25} {'V1 (Current)':>14} {'V2 (Refined)':>14} {'Change':>10}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")

    for c in report.comparisons:
        print(f"\n  Symbol: {c.symbol}")
        v1r, v2r = c.v1_results, c.v2_results
        rows = [
            ("Trades",       f"{v1r.total_trades:>14}",  f"{v2r.total_trades:>14}",
             f"{c.trade_count_change:>+9}"),
            ("Win Rate",     f"{v1r.win_rate*100:>13.1f}%", f"{v2r.win_rate*100:>13.1f}%", ""),
            ("Profit Factor", f"{_pf(v1r.profit_factor):>14}", f"{_pf(v2r.profit_factor):>14}",
             f"{c.pf_change:>+9.2f}"),
            ("Expectancy $/tr", f"${v1r.expectancy:>+12.2f}", f"${v2r.expectancy:>+12.2f}",
             f"${c.expectancy_change:>+8.2f}"),
            ("Max Drawdown", f"{v1r.max_drawdown:>13.1f}%", f"{v2r.max_drawdown:>13.1f}%",
             f"{-c.drawdown_change:>+9.1f}%"),
            ("Net Profit",   f"${v1r.net_profit:>+12.2f}", f"${v2r.net_profit:>+12.2f}",
             f"${c.net_profit_change:>+8.2f}"),
        ]
        for label, v1_val, v2_val, chg in rows:
            print(f"  {'  ' + label:<25} {v1_val:>14} {v2_val:>14} {chg:>10}")

        status = "✓ PROMOTE" if c.promote_v2 else "✗ REJECT"
        print(f"  {'  V2 Verdict':<25} {'':>14} {'':>14} {status:>10}")
        if not c.promote_v2 and c.rejection_reason:
            print(f"  Reason: {c.rejection_reason}")

    # Aggregate
    print(f"\n{sep}")
    print(f"  AGGREGATE V2 METRICS (across {len(report.comparisons)} assets)")
    print(sep)
    print(f"  Total V2 Trades:      {report.v2_total_trades}")
    print(f"  Avg Profit Factor:    {_pf(report.v2_avg_pf)}")
    print(f"  Avg Expectancy:       ${report.v2_avg_expectancy:+.2f}/trade")
    print(f"  Avg Max Drawdown:     {report.v2_avg_drawdown:.1f}%")

    print(f"\n{sep}")
    verdict = "✓ V2 PROMOTED" if report.v2_promoted else "✗ V2 NOT PROMOTED"
    print(f"  {verdict}")
    print(f"\n  {report.promotion_reason}")
    print(sep)

    if report.warnings:
        print(f"\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")

    if report.notes:
        print(f"\n  Notes:")
        for n in report.notes:
            print(f"    → {n}")

    print(f"\n{sep}")
    print(f"  RECOMMENDATION:")
    print(f"  {report.recommendation}")
    print(sep)

    # V2 settings if promoted
    if report.v2_promoted:
        print(f"\n  V2 Settings to Apply:")
        print(f"    EMA Fast/Slow:      {V2_PROFILE.ema_fast}/{V2_PROFILE.ema_slow}")
        print(f"    Breakout Threshold: {V2_PROFILE.breakout_threshold*100:.2f}%")
        print(f"    Bull Regime Filter: {'ON' if V2_PROFILE.require_bull_regime else 'OFF'}")
        print(f"    ADX Threshold:      ≥ {V2_PROFILE.adx_threshold:.0f}")
        print(f"    Volatility Filter:  {V2_PROFILE.volatility_mode.value}")
        print(sep)


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Print strategy profiles being compared
    print(f"\n{'='*62}")
    print("  PHASE 4.6 STRATEGY REFINEMENT VALIDATION")
    print(f"{'='*62}")
    print(f"\n  V1: {V1_PROFILE.description}")
    print(f"  V2: {V2_PROFILE.description}")
    print(f"\n  Assets: {', '.join(args.assets)}")
    print(f"  Candles: {args.candles} per asset ({args.timeframe})")
    print(f"\n  Success Criteria:")
    print(f"    Profit Factor ≥ {PROMOTION_MIN_PF}")
    print(f"    Expectancy > $0/trade")
    print(f"    Max Drawdown < 15%")
    print(f"    Trades ≥ {PROMOTION_MIN_TRADES}")

    # Connect and download
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
            print(f"  [{i+1}/{len(args.assets)}] Fetching {symbol} {args.timeframe} "
                  f"({args.candles} bars)...")
            candles = provider.get_candles(symbol, args.timeframe, args.candles)
            if candles:
                asset_candles[symbol] = candles
                print(f"           {len(candles)} candles  "
                      f"({candles[0].timestamp.strftime('%Y-%m-%d')} → "
                      f"{candles[-1].timestamp.strftime('%Y-%m-%d')})")
            else:
                print(f"           WARNING: no data for {symbol}")
            if i < len(args.assets) - 1:
                time.sleep(REQUEST_DELAY_S)
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    if not asset_candles:
        print("ERROR: No candle data retrieved.", file=sys.stderr)
        return 1

    # Run refinement engine
    selector = AssetSelector(recommended=list(asset_candles.keys()), excluded=[])
    engine = RefinementEngine(
        config         = config,
        v1_profile     = V1_PROFILE,
        v2_profile     = V2_PROFILE,
        asset_selector = selector,
    )

    print("Running strategy comparison (V1 vs V2)...")
    report = engine.run(asset_candles)

    print_comparison_table(report)

    print("\nNo trades placed. No orders sent. Research only.")
    print(f"Full log: logs/refinement_validation.log\n")
    return 0 if report.v2_promoted else 1


if __name__ == "__main__":
    sys.exit(main())
