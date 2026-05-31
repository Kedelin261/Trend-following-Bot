#!/usr/bin/env python3
"""Phase 4.7 Signal Density Optimization Validation Script.

Downloads 2000 candles for SPY, VOO, DIA, XLK, VTI, XLP, XLV, SCHD,
then runs the full density research suite:

  1. ADX threshold sweep (20 / 22 / 25 / 27 / 30)
  2. Volatility filter sweep (MEDIUM_ONLY / LOW+MEDIUM / MEDIUM+HIGH / NONE)
  3. Breakout threshold sweep (1.00% / 0.90% / 0.80% / 0.75% / 0.50%)
  4. Asset expansion research (XLK, VTI, XLP, XLV, SCHD)
  5. Portfolio-level opportunity analysis
  6. Promotion decision (≥100 trades, PF≥1.50, Exp>0, DD<15%)

No orders placed. No trades executed. Research only.

Usage:
    python validate_density.py
    python validate_density.py --candles 1500 --quick
    python validate_density.py --candles 2000 --primary SPY
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
from src.density.density_engine import DensityEngine, DensityProfile, DensityReport
from src.density.portfolio_density_analyzer import PORTFOLIO_MIN_TRADES
from src.providers.provider_factory import ProviderFactory
from src.refinement.strategy_v2 import V2_PROFILE

DEFAULT_ASSETS = ["SPY", "VOO", "DIA", "XLK", "VTI", "XLP", "XLV", "SCHD"]
REQUEST_DELAY_S = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.7 signal density optimisation (read-only)")
    p.add_argument("--candles", type=int, default=2000,
                   help="Candles per asset (default: 2000 ≈ 8 years D1)")
    p.add_argument("--timeframe", default="D1",
                   help="Candle timeframe (default: D1)")
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                   help="Assets to research")
    p.add_argument("--primary", default="SPY",
                   help="Primary asset for single-asset research (default: SPY)")
    p.add_argument("--quick", action="store_true",
                   help="Skip expansion research and breakout sweep (faster run)")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/density_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_research_table(
    label: str,
    headers: List[str],
    rows: List[List[str]],
) -> None:
    col_w = [max(len(h), max((len(r[i]) for r in rows), default=0))
             for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  " + "  ".join("-" * w for w in col_w)
    print(f"\n  {label}")
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def print_report(report: DensityReport) -> None:
    sep = "=" * 60

    # ---- Baseline -----------------------------------------------------
    print(f"\n{sep}")
    print("  SIGNAL DENSITY REPORT")
    print(sep)

    print(f"\n  BASELINE (V2 — {len(report.baseline_asset_results)} assets)")
    base_trades = sum(bt.total_trades for bt in report.baseline_asset_results.values())
    base_pf     = report.portfolio_report.aggregate_pf
    base_exp    = report.portfolio_report.aggregate_expectancy
    base_dd     = report.portfolio_report.worst_drawdown
    print(f"  Total trades: {base_trades}")
    print(f"  PF:           {_pf(base_pf)}")
    print(f"  Expectancy:   ${base_exp:+.2f}/trade")
    print(f"  Max DD:       {base_dd:.1f}%")

    # Per-asset baseline
    print_research_table(
        "Baseline — per asset",
        ["Asset", "Trades", "Win%", "Exp$/tr", "PF", "DD%"],
        [
            [sym,
             str(bt.total_trades),
             f"{bt.win_rate*100:.0f}%",
             f"${bt.expectancy:+.2f}",
             _pf(bt.profit_factor),
             f"{bt.max_drawdown:.1f}%"]
            for sym, bt in report.baseline_asset_results.items()
        ],
    )

    # ---- ADX research -------------------------------------------------
    if report.adx_results:
        print_research_table(
            "ADX Threshold Research",
            ["ADX≥", "Trades", "Exp$/tr", "PF", "DD%", "Quality", "Note"],
            [
                [f"{r.threshold:.0f}",
                 str(r.trade_count),
                 f"${r.expectancy:+.2f}",
                 _pf(r.profit_factor),
                 f"{r.max_drawdown:.1f}%",
                 "✓" if r.meets_quality else "✗",
                 r.note[:30] if r.note else ""]
                for r in sorted(report.adx_results, key=lambda x: x.threshold)
            ],
        )

    # ---- Volatility research -----------------------------------------
    if report.volatility_results:
        print_research_table(
            "Volatility Filter Research",
            ["Mode", "Trades", "Exp$/tr", "PF", "DD%", "Quality"],
            [
                [r.mode.value,
                 str(r.trade_count),
                 f"${r.expectancy:+.2f}",
                 _pf(r.profit_factor),
                 f"{r.max_drawdown:.1f}%",
                 "✓" if r.meets_quality else "✗"]
                for r in report.volatility_results
            ],
        )

    # ---- Breakout research -------------------------------------------
    if report.breakout_results:
        print_research_table(
            "Breakout Threshold Research",
            ["Threshold", "Trades", "Exp$/tr", "PF", "DD%", "Quality"],
            [
                [f"{r.threshold*100:.2f}%",
                 str(r.trade_count),
                 f"${r.expectancy:+.2f}",
                 _pf(r.profit_factor),
                 f"{r.max_drawdown:.1f}%",
                 "✓" if r.meets_quality else "✗"]
                for r in sorted(report.breakout_results, key=lambda x: x.threshold,
                                reverse=True)
            ],
        )

    # ---- Asset expansion -------------------------------------------
    if report.asset_expansion_results:
        print_research_table(
            "Asset Expansion Research",
            ["Symbol", "Trades", "Exp$/tr", "PF", "DD%", "Recommend?"],
            [
                [r.symbol,
                 str(r.trade_count),
                 f"${r.expectancy:+.2f}",
                 _pf(r.profit_factor),
                 f"{r.max_drawdown:.1f}%",
                 "✓ ADD" if r.recommended else "✗ SKIP"]
                for r in report.asset_expansion_results
            ],
        )

    # ---- Signal density metrics --------------------------------------
    if report.density_metrics:
        print_research_table(
            "Signal Density — Baseline",
            ["Asset", "Trades/yr", "Trades/mo", "Avg hold", "Goal?"],
            [
                [m.symbol,
                 f"{m.trades_per_year:.1f}",
                 f"{m.trades_per_month:.2f}",
                 f"{m.avg_holding_bars:.1f} bars",
                 "✓" if m.meets_density_goal else "⚠"]
                for m in report.density_metrics
            ],
        )

    # ---- Best profile ------------------------------------------------
    print(f"\n{sep}")
    print("  BEST DENSITY PROFILE")
    print(sep)
    bp = report.best_density_profile
    print(f"  Name:              {bp.name}")
    print(f"  Description:       {bp.description}")
    print(f"  ADX Threshold:     ≥ {bp.adx_threshold:.0f}")
    print(f"  Breakout:          {bp.breakout_threshold*100:.2f}%")
    print(f"  Volatility Mode:   {bp.volatility_mode.value}")
    print(f"  Assets:            {', '.join(bp.assets)}")

    # Best profile results
    if report.best_asset_results:
        print_research_table(
            "Best Profile — per asset",
            ["Asset", "Trades", "Win%", "Exp$/tr", "PF", "DD%"],
            [
                [sym,
                 str(bt.total_trades),
                 f"{bt.win_rate*100:.0f}%",
                 f"${bt.expectancy:+.2f}",
                 _pf(bt.profit_factor),
                 f"{bt.max_drawdown:.1f}%"]
                for sym, bt in report.best_asset_results.items()
            ],
        )

    # Portfolio summary
    bpr = report.best_portfolio_report
    print(f"\n  PORTFOLIO STATUS — BEST PROFILE")
    print(f"  Total trades:      {bpr.total_trades}  "
          f"({'≥100 ✓' if bpr.meets_100_trades else '<100 ✗'}  "
          f"{'≥150 ✓' if bpr.meets_150_trades else '<150 ✗'}  "
          f"{'≥200 ✓' if bpr.meets_200_trades else '<200 ✗'})")
    print(f"  Trades / year:     {bpr.trades_per_year:.1f}")
    print(f"  Monthly opps:      {bpr.monthly_opportunities:.1f}")
    print(f"  Aggregate PF:      {_pf(bpr.aggregate_pf)}")
    print(f"  Aggregate Exp:     ${bpr.aggregate_expectancy:+.2f}/trade")
    print(f"  Worst DD:          {bpr.worst_drawdown:.1f}%")
    print(f"  Assets passing:    {bpr.assets_passing}")

    # Promotion
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

    if report.promoted:
        print(f"\n  To apply — update config/settings.yaml or create a StrategyProfile:")
        print(f"    adx_threshold:      {bp.adx_threshold:.0f}")
        print(f"    breakout_threshold: {bp.breakout_threshold*100:.2f}%")
        print(f"    volatility_mode:    {bp.volatility_mode.value}")
        print(f"    assets:             {', '.join(bp.assets)}")
    print(f"{sep}\n")


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print("  PHASE 4.7 SIGNAL DENSITY OPTIMISATION")
    print(f"{'='*60}")
    print(f"\n  V2 Baseline:")
    print(f"    EMA: {V2_PROFILE.ema_fast}/{V2_PROFILE.ema_slow}")
    print(f"    Breakout: {V2_PROFILE.breakout_threshold*100:.2f}%")
    print(f"    ADX ≥ {V2_PROFILE.adx_threshold:.0f}")
    print(f"    Volatility: {V2_PROFILE.volatility_mode.value}")
    print(f"\n  Promotion Criteria:")
    print(f"    Portfolio trades ≥ {PORTFOLIO_MIN_TRADES}")
    print(f"    Profit Factor ≥ 1.50")
    print(f"    Expectancy > $0")
    print(f"    Max Drawdown < 15%")
    print(f"    ≥ 2 assets individually pass")

    # ------------------------------------------------------------------ #
    # Download candles                                                     #
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

    asset_candles: Dict[str, List[Candle]] = {}
    try:
        for i, symbol in enumerate(args.assets):
            print(f"  [{i+1}/{len(args.assets)}] {symbol} "
                  f"{args.timeframe} ({args.candles} bars)...")
            candles = provider.get_candles(symbol, args.timeframe, args.candles)
            if candles:
                asset_candles[symbol] = candles
                span = (f"{candles[0].timestamp.strftime('%Y-%m-%d')} → "
                        f"{candles[-1].timestamp.strftime('%Y-%m-%d')}")
                print(f"           {len(candles)} candles  {span}")
            else:
                print(f"           WARNING: no data returned for {symbol}")
            if i < len(args.assets) - 1:
                time.sleep(REQUEST_DELAY_S)
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    if not asset_candles:
        print("ERROR: No candle data retrieved.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Run density engine                                                   #
    # ------------------------------------------------------------------ #
    engine = DensityEngine(
        config         = config,
        base_profile   = V2_PROFILE,
        primary_symbol = args.primary if args.primary in asset_candles
                         else next(iter(asset_candles)),
    )

    print(f"Running density research ({len(asset_candles)} assets)...")
    print(f"  ADX sweep:       YES")
    print(f"  Volatility sweep: YES")
    print(f"  Breakout sweep:  {'NO (--quick)' if args.quick else 'YES'}")
    print(f"  Asset expansion: {'NO (--quick)' if args.quick else 'YES'}\n")

    report = engine.run(
        asset_candles   = asset_candles,
        run_adx         = True,
        run_vol         = True,
        run_breakout    = not args.quick,
        run_expansion   = not args.quick,
    )

    print_report(report)

    print("No trades placed. No orders sent. Research only.")
    print(f"Full log: logs/density_validation.log\n")
    return 0 if report.promoted else 1


if __name__ == "__main__":
    sys.exit(main())
