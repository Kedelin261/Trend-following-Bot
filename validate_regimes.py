#!/usr/bin/env python3
"""Phase 4.10 Regime Stability Research Validation Script.

Downloads maximum available candles for SPY, VOO, DIA then runs the
complete regime stability research suite:

  1. Strategy backtest (locked — BEST_DENSITY_PROFILE)
  2. Trade labelling across 5 regime dimensions
  3. Per-regime performance analysis
  4. Profit / loss concentration analysis
  5. Hypothetical regime filter simulations
  6. Regime stability assessment

Answers: 'When does the strategy work, and when should it NOT trade?'

No orders placed. No trades executed. Research only.

Usage:
    python validate_regimes.py
    python validate_regimes.py --candles 5000
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
from src.regime.regime_stability_engine import RegimeStabilityEngine, RegimeStabilityReport
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

DEFAULT_ASSETS = ["SPY", "VOO", "DIA"]
REQUEST_DELAY  = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.10 regime stability research (read-only)")
    p.add_argument("--candles", type=int, default=5000,
                   help="Candles per asset (default: 5000)")
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                   help="Assets to analyse")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/regime_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_report(report: RegimeStabilityReport) -> None:
    sep = "=" * 62

    print(f"\n{sep}")
    print("  REGIME STABILITY REPORT")
    print(sep)
    print(f"\n  Total trades analysed: {report.total_trades_analyzed}")

    if report.total_trades_analyzed == 0:
        print("  No trades generated — cannot produce regime analysis.")
        print("  Hint: extend candle history or add more assets.\n")
        return

    # ---- Per-regime performance tables ------------------------------
    dimensions = {
        "Market Regime":     report.market_regime_performance,
        "Trend Quality":     report.trend_regime_performance,
        "Volatility":        report.volatility_regime_performance,
        "Drawdown Env":      report.drawdown_env_performance,
        "Macro Regime":      report.macro_regime_performance,
    }

    for dim_name, perf_dict in dimensions.items():
        if not perf_dict:
            continue
        print(f"\n  {'─'*58}")
        print(f"  {dim_name.upper()}")
        print(f"  {'Regime':<18} {'Trades':>7} {'Win%':>6} {'Exp$/tr':>9} "
              f"{'PF':>6} {'Profit%':>8} {'Loss%':>7}")
        print(f"  {'-'*18} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*8} {'-'*7}")
        for regime, p in sorted(perf_dict.items(),
                                 key=lambda x: x[1].expectancy, reverse=True):
            pos = "✓" if p.is_net_positive else "✗"
            print(
                f"  {regime:<18} {p.trade_count:>7} {p.win_rate*100:>5.0f}% "
                f"{p.expectancy:>+9.2f} {p.pf_str:>6} "
                f"{p.profit_concentration*100:>7.0f}% {p.loss_concentration*100:>6.0f}% {pos}"
            )

    # ---- Best / worst summary --------------------------------------
    print(f"\n  {'─'*58}")
    print(f"  Best Market Regime:  {report.best_market_regime or 'N/A'}")
    print(f"  Worst Market Regime: {report.worst_market_regime or 'N/A'}")

    if report.best_market_regime and report.best_market_regime in report.market_regime_performance:
        bp = report.market_regime_performance[report.best_market_regime]
        print(f"\n  Best Regime ({report.best_market_regime}):")
        print(f"    Trades:     {bp.trade_count}")
        print(f"    PF:         {bp.pf_str}")
        print(f"    Expectancy: ${bp.expectancy:+.2f}/trade")
        print(f"    Profit %:   {bp.profit_concentration*100:.0f}% of all profits")

    if report.worst_market_regime and report.worst_market_regime in report.market_regime_performance:
        wp = report.market_regime_performance[report.worst_market_regime]
        print(f"\n  Worst Regime ({report.worst_market_regime}):")
        print(f"    Trades:     {wp.trade_count}")
        print(f"    PF:         {wp.pf_str}")
        print(f"    Expectancy: ${wp.expectancy:+.2f}/trade")
        print(f"    Loss %:     {wp.loss_concentration*100:.0f}% of all losses")

    # ---- Concentration summary -------------------------------------
    print(f"\n  {'─'*58}")
    print(f"  CONCENTRATION ANALYSIS")
    if report.top_profit_regime:
        print(
            f"  {report.profit_in_best_regime_pct*100:.0f}% of profits from "
            f"'{report.top_profit_regime}' regime"
        )
    if report.top_loss_regime:
        print(
            f"  {report.loss_in_worst_regime_pct*100:.0f}% of losses from "
            f"'{report.top_loss_regime}' regime"
        )

    # ---- Simulated filters -----------------------------------------
    if report.filter_simulations:
        print(f"\n  {'─'*58}")
        print("  SIMULATED REGIME FILTERS (top 5)")
        print(f"  {'Filter':<35} {'Keep':>5} {'PF':>6} {'Exp$/tr':>9} {'Improve?':>9}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*9} {'-'*9}")
        for r in report.filter_simulations[:5]:
            ok = "✓ YES" if r.improves_quality else "✗ NO"
            print(
                f"  {r.filter_description[:35]:<35} "
                f"{r.filtered_trades:>5} {r.pf_str:>6} "
                f"{r.projected_expectancy:>+9.2f} {ok:>9}"
            )

    # ---- Assessment ------------------------------------------------
    print(f"\n{sep}")
    dep = "REGIME DEPENDENT" if report.edge_is_regime_dependent else "REGIME ROBUST"
    print(f"  REGIME STABILITY ASSESSMENT — {dep}")
    print(sep)
    print(f"\n  {report.stability_assessment}")

    if report.recommendations:
        print("\n  Recommendations:")
        for r in report.recommendations:
            print(f"    → {r}")
    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")

    if report.best_filter:
        bf = report.best_filter
        print(f"\n  Best simulated filter: '{bf.filter_description}'")
        print(f"    Projected PF:         {bf.pf_str}")
        print(f"    Projected Expectancy: ${bf.projected_expectancy:+.2f}/trade")
        print(f"    Trades retained:      {bf.filtered_trades}/{bf.original_trades} "
              f"({bf.retention_pct*100:.0f}%)")

    print(f"\n{sep}\n")
    print("Phase 4.10 CONCLUSION:")
    if report.edge_is_regime_dependent:
        print("  The strategy's instability is REGIME-DEPENDENT, not a fundamental flaw.")
        print("  A good strategy is operating in the wrong environments.")
        print("  → Next step: Phase 4.11 — Design regime-aware entry filters.")
    else:
        print("  The strategy shows consistent edge across regimes.")
        print("  → Robustness concern may be sample-size related, not regime-related.")
    print()


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'='*62}")
    print("  PHASE 4.10 — REGIME STABILITY RESEARCH")
    print(f"{'='*62}")
    print(f"\n  Strategy:   {BEST_DENSITY_PROFILE.name}")
    print(f"  Assets:     {', '.join(args.assets)}")
    print(f"  Candles:    {args.candles}")
    print(f"\n  Research dimensions:")
    print("    1. Market Regime (STRONG_BULL / BULL / SIDEWAYS / BEAR / STRONG_BEAR)")
    print("    2. Trend Quality (STRONG / MODERATE / WEAK)")
    print("    3. Volatility    (LOW / NORMAL / HIGH / EXTREME)")
    print("    4. Drawdown Env  (BULL_RECOVERY / CORRECTION / BEAR / CRASH)")
    print("    5. Macro Regime  (EXPANSION / RECOVERY / CONTRACTION / CRISIS)")

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
        print("ERROR: No candle data retrieved.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Run regime stability engine                                          #
    # ------------------------------------------------------------------ #
    print("Running regime stability engine...")
    engine = RegimeStabilityEngine(config, BEST_DENSITY_PROFILE)
    report = engine.run(asset_candles)

    print_report(report)

    print("No trades placed. No orders sent. Research only.")
    print(f"Full log: logs/regime_validation.log\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
