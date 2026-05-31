#!/usr/bin/env python3
"""Phase 4.5 Strategy Research & Edge Discovery Validation Script.

Downloads historical data for the full ETF/equity universe and runs the
complete research suite: regime analysis, ADX filtering, volatility
analysis, breakout quality, parameter sweep, multi-asset testing,
and benchmark comparison.

Answers: 'Under what conditions does this strategy have a repeatable edge?'

No orders placed. No trades executed. Research only.

Usage:
    python validate_research.py
    python validate_research.py --quick          # SPY only, skip sweep
    python validate_research.py --candles 1000   # request more history
    python validate_research.py --primary QQQ    # set primary analysis asset
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
from src.research.market_regime import MarketRegime
from src.research.research_engine import ResearchEngine
from src.research.strategy_analyzer import ResearchReport
from src.research.volatility_filter import VolatilityCategory

# Full research universe
RESEARCH_ASSETS = ["SPY", "QQQ", "VOO", "VTI", "DIA", "IWM", "XLK", "XLF", "XLE"]
REQUEST_DELAY_S = 2.0   # IBKR pacing: space requests slightly


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.5 strategy research (read-only)")
    p.add_argument("--quick", action="store_true",
                   help="SPY only, skip parameter sweep (fast mode)")
    p.add_argument("--candles", type=int, default=1260,
                   help="Candles per asset — 1260 ≈ 5 years D1 (default: 1260)")
    p.add_argument("--timeframe", default="D1",
                   help="Candle timeframe (default: D1)")
    p.add_argument("--primary", default="SPY",
                   help="Primary symbol for regime/ADX/vol analysis (default: SPY)")
    p.add_argument("--no-sweep", action="store_true",
                   help="Skip parameter sweep (saves ~5 min)")
    p.add_argument("--no-breakout", action="store_true",
                   help="Skip breakout quality research")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler("logs/research_validation.log", encoding="utf-8")],
    )


def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_report(report: ResearchReport) -> None:
    sep = "=" * 58

    print(f"\n{sep}")
    print("  RESEARCH FINDINGS")
    print(sep)

    # Multi-asset results
    print(f"\n  {'Asset':<8} {'Trades':>7} {'Win%':>6} {'Exp$/tr':>9} "
          f"{'PF':>6} {'MaxDD%':>7} {'Health':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*7} {'-'*7}")
    for ar in report.asset_results:
        r = ar.results
        h = "PASS" if ar.health.passed else "FAIL"
        flag = "" if ar.sufficient else " ⚠"
        print(
            f"  {ar.symbol:<8} {r.total_trades:>7} {r.win_rate*100:>5.1f}% "
            f"{r.expectancy:>+9.2f} {_pf(r.profit_factor):>6} "
            f"{r.max_drawdown:>6.1f}% {h:>7}{flag}"
        )

    # Summary
    print(f"\n{sep}")
    print(f"  Best Asset:          {report.best_asset or 'N/A'}")
    print(f"  Worst Asset:         {report.worst_asset or 'N/A'}")
    if report.recommended_assets:
        print(f"  Recommended:         {', '.join(report.recommended_assets)}")
    if report.avoid_assets:
        print(f"  Avoid:               {', '.join(report.avoid_assets)}")

    # Regime analysis
    if report.regime_performance:
        print(f"\n  Market Regime Analysis:")
        for regime, perf in sorted(report.regime_performance.items(),
                                   key=lambda x: x[1].expectancy, reverse=True):
            suf = "" if perf.sufficient else " ⚠ (low sample)"
            print(f"    {regime.value:<12}  trades={perf.trade_count:>4}  "
                  f"exp={perf.expectancy:>+7.2f}  win={perf.win_rate*100:.0f}%{suf}")
        if report.best_regime:
            print(f"  Best Regime:         {report.best_regime.value}")
        print(f"  Regime Filter Rec:   {'YES' if report.regime_filter_recommended else 'NO'}")

    # ADX analysis
    if report.adx_results:
        print(f"\n  ADX Filter Analysis:")
        for r in report.adx_results:
            suf = "" if r.sufficient else " ⚠"
            print(f"    ADX ≥ {r.threshold:>4.0f}  trades={r.trades_passed:>4}  "
                  f"exp={r.expectancy:>+7.2f}{suf}")
        if report.best_adx_threshold is not None:
            print(f"  Best ADX Threshold:  ≥ {report.best_adx_threshold:.0f}")
        print(f"  ADX Filter Rec:      {'YES' if report.adx_filter_recommended else 'NO'}")

    # Volatility analysis
    if report.volatility_results:
        print(f"\n  Volatility Analysis:")
        for cat, perf in report.volatility_results.items():
            suf = "" if perf.sufficient else " ⚠"
            print(f"    {cat.value:<8}  trades={perf.trade_count:>4}  "
                  f"exp={perf.expectancy:>+7.2f}  atr%={perf.avg_atr_pct:.2f}{suf}")

    # Breakout quality
    if report.breakout_results:
        print(f"\n  Breakout Threshold Research:")
        for r in report.breakout_results[:4]:
            suf = "" if r.sufficient else " ⚠"
            print(f"    {r.config.description:<40}  "
                  f"trades={r.results.total_trades:>4}  "
                  f"exp={r.results.expectancy:>+7.2f}{suf}")

    # Parameter sweep
    if report.sweep_results:
        print(f"\n  Parameter Sweep (top 5 by score):")
        print(f"  {'Configuration':<45} {'Trades':>7} {'Exp$/tr':>9} {'PF':>6}")
        print(f"  {'-'*45} {'-'*7} {'-'*9} {'-'*6}")
        for sr in report.sweep_results[:5]:
            suf = "" if sr.sufficient else " ⚠"
            print(
                f"  {sr.config.description:<45} "
                f"{sr.results.total_trades:>7} "
                f"{sr.results.expectancy:>+9.2f} "
                f"{_pf(sr.results.profit_factor):>6}{suf}"
            )

    # Benchmark comparison
    if report.benchmark_results:
        print(f"\n  Benchmark (Strategy vs Buy-and-Hold):")
        print(f"  {'Asset':<8} {'Strat%':>7} {'BH%':>7} {'Alpha%':>8} {'Beats?':>7}")
        print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
        for sym, bm in report.benchmark_results.items():
            beats = "YES" if bm.strategy_outperforms else "NO"
            print(
                f"  {sym:<8} {bm.strategy_return_pct:>+6.1f}% "
                f"{bm.buyhold_return_pct:>+6.1f}% "
                f"{bm.alpha_pct:>+7.1f}% {beats:>7}"
            )

    # Overall assessment
    print(f"\n{sep}")
    edge = "✓ EDGE CONFIRMED" if report.edge_confirmed else "✗ EDGE NOT CONFIRMED"
    print(f"  {edge}")
    print(sep)
    print(f"  Expected Expectancy:  ${report.expected_expectancy:+.2f}/trade")
    print(f"  Expected P/F:         {_pf(report.expected_profit_factor)}")
    print(f"  Expected Max DD:      {report.expected_max_drawdown:.1f}%")

    if report.warnings:
        print(f"\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")

    if report.notes:
        print(f"\n  Research Notes:")
        for n in report.notes:
            print(f"    → {n}")

    print(f"\n{sep}")
    if report.edge_confirmed:
        print("  RECOMMENDATION: Proceed to paper trading (Phase 6) with:")
        for asset in report.recommended_assets:
            print(f"    • {asset}")
        if report.regime_filter_recommended:
            print("  Apply: BULL market regime filter")
        if report.adx_filter_recommended and report.best_adx_threshold:
            print(f"  Apply: ADX ≥ {report.best_adx_threshold:.0f} filter")
    else:
        print("  RECOMMENDATION: Do NOT proceed to paper trading.")
        print("  Continue research to improve strategy edge before deployment.")
    print(f"{sep}\n")


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    assets = ["SPY"] if args.quick else RESEARCH_ASSETS
    run_sweep = not (args.quick or args.no_sweep)
    run_breakout = not (args.quick or args.no_breakout)

    # ------------------------------------------------------------------ #
    # Download candles                                                     #
    # ------------------------------------------------------------------ #
    symbol_registry = SymbolRegistry.from_config(config)
    try:
        provider = ProviderFactory.create(config, symbol_registry)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    provider_name = config.get("provider", "?").upper()
    print(f"\nConnecting to {provider_name}...")
    if not provider.connect():
        print("ERROR: Could not connect to provider.", file=sys.stderr)
        return 1

    asset_candles: Dict[str, List[Candle]] = {}
    try:
        for i, symbol in enumerate(assets):
            print(f"  [{i + 1}/{len(assets)}] Fetching {symbol} "
                  f"{args.timeframe} ({args.candles} bars)...")
            candles = provider.get_candles(symbol, args.timeframe, args.candles)
            if candles:
                asset_candles[symbol] = candles
                print(f"           Received {len(candles)} candles  "
                      f"({candles[0].timestamp.strftime('%Y-%m-%d')} → "
                      f"{candles[-1].timestamp.strftime('%Y-%m-%d')})")
            else:
                print(f"           WARNING: no data for {symbol}")
            if i < len(assets) - 1:
                time.sleep(REQUEST_DELAY_S)
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    if not asset_candles:
        print("ERROR: No candle data retrieved.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Run research engine                                                  #
    # ------------------------------------------------------------------ #
    primary = args.primary if args.primary in asset_candles else next(iter(asset_candles))
    print(f"Running research engine (primary asset: {primary})...")
    print(f"  Assets: {', '.join(asset_candles)}")
    print(f"  Parameter sweep: {'YES' if run_sweep else 'NO'}")
    print(f"  Breakout research: {'YES' if run_breakout else 'NO'}\n")

    engine = ResearchEngine(config, min_warmup=210)
    report = engine.run(
        asset_candles    = asset_candles,
        primary_symbol   = primary,
        run_sweep        = run_sweep,
        run_breakout     = run_breakout,
    )

    print_report(report)

    print("No trades placed. No orders sent. Research only.")
    print(f"Full log: logs/research_validation.log\n")
    return 0 if report.edge_confirmed else 1


if __name__ == "__main__":
    sys.exit(main())
