#!/usr/bin/env python3
"""Phase 4 Backtesting Engine Validation Script.

Workflow:
  1. Connect to configured provider
  2. Download 2 years of SPY D1 candles (~500 bars + 210 warmup)
  3. Run full backtest (in-sample)
  4. Run walk-forward analysis (70/30 in/out-of-sample)
  5. Evaluate strategy health
  6. Print complete report

No orders placed. No trades executed. Read-only throughout.

Usage:
    python validate_backtest.py
    python validate_backtest.py --symbol QQQ --candles 750
    python validate_backtest.py --symbol SPY --no-walk-forward
"""

import argparse
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import StrategyHealth
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.config.settings_loader import load_settings
from src.data.symbol_registry import SymbolRegistry
from src.providers.provider_factory import ProviderFactory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 backtest validation (read-only)")
    p.add_argument("--symbol", default="SPY",
                   help="Symbol to backtest (default: SPY)")
    p.add_argument("--timeframe", default="D1",
                   help="Candle timeframe (default: D1)")
    p.add_argument("--candles", type=int, default=750,
                   help="Candles to fetch — 750 ≈ 2 years D1 + warmup (default: 750)")
    p.add_argument("--no-walk-forward", action="store_true",
                   help="Skip walk-forward analysis")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/backtest_validation.log", encoding="utf-8"),
        ],
    )


def _pf_str(pf: float) -> str:
    return "∞" if math.isinf(pf) else f"{pf:.2f}"


def print_results(results, label: str = "BACKTEST RESULTS") -> None:
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Symbol:             {results.symbol}")
    print(f"  Timeframe:          {results.timeframe}")
    print()
    print(f"  Starting Balance:   ${results.starting_balance:,.2f}")
    print(f"  Ending Balance:     ${results.ending_balance:,.2f}")
    print(f"  Net Profit:         ${results.net_profit:+,.2f}  ({results.return_percent:+.1f}%)")
    print()
    print(f"  Total Trades:       {results.total_trades}")
    print(f"  Winning Trades:     {results.winning_trades}")
    print(f"  Losing Trades:      {results.losing_trades}")
    print(f"  Win Rate:           {results.win_rate * 100:.1f}%")
    print()
    print(f"  Profit Factor:      {_pf_str(results.profit_factor)}")
    print(f"  Expectancy:         ${results.expectancy:+.2f} / trade")
    print()
    print(f"  Max Drawdown:       {results.max_drawdown:.1f}%")
    print(f"  Sharpe Ratio:       {results.sharpe_ratio:.2f}")
    print()
    print(f"  Average Win:        ${results.average_win:.2f}")
    print(f"  Average Loss:       ${results.average_loss:.2f}")
    print(f"  Largest Win:        ${results.largest_win:.2f}")
    print(f"  Largest Loss:       ${results.largest_loss:.2f}")
    print(sep)


def print_health(health: StrategyHealth) -> None:
    sep = "=" * 54
    verdict = "PASS ✓" if health.passed else "FAIL ✗"
    print(f"\n{sep}")
    print(f"  STRATEGY HEALTH CHECK — {verdict}")
    print(sep)
    if health.reasons_passed:
        print("  Passed:")
        for r in health.reasons_passed:
            print(f"    ✓ {r}")
    if health.reasons_failed:
        print("  Failed:")
        for r in health.reasons_failed:
            print(f"    ✗ {r}")
    print(sep)


def print_walk_forward(wf_result) -> None:
    sep = "=" * 54
    print(f"\n{sep}")
    print("  WALK-FORWARD ANALYSIS")
    print(sep)
    is_  = wf_result.in_sample
    oos_ = wf_result.out_of_sample
    split = wf_result.split

    print(f"  Split:              {int(split.train_count / (split.train_count + split.test_count) * 100)}% in-sample / "
          f"{int(split.test_count / (split.train_count + split.test_count) * 100)}% out-of-sample")
    print(f"  Train candles:      {split.train_count}")
    print(f"  Test candles:       {split.test_count}")
    print()
    print(f"  {'Metric':<22} {'In-Sample':>12} {'Out-of-Sample':>15}")
    print(f"  {'-'*22} {'-'*12} {'-'*15}")
    print(f"  {'Trades':<22} {is_.total_trades:>12} {oos_.total_trades:>15}")
    print(f"  {'Win Rate':<22} {is_.win_rate*100:>11.1f}% {oos_.win_rate*100:>14.1f}%")
    print(f"  {'Profit Factor':<22} {_pf_str(is_.profit_factor):>12} {_pf_str(oos_.profit_factor):>15}")
    print(f"  {'Expectancy ($/trade)':<22} {is_.expectancy:>+12.2f} {oos_.expectancy:>+15.2f}")
    print(f"  {'Max Drawdown':<22} {is_.max_drawdown:>11.1f}% {oos_.max_drawdown:>14.1f}%")
    print(f"  {'Net Profit':<22} ${is_.net_profit:>+11.2f} ${oos_.net_profit:>+14.2f}")
    print()
    print(f"  Consistency Ratio:  {wf_result.consistency_ratio:.2f}  "
          f"({'good' if wf_result.consistency_ratio > 0.5 else 'poor'} OOS alignment)")
    print(sep)


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    symbol_registry = SymbolRegistry.from_config(config)

    try:
        provider = ProviderFactory.create(config, symbol_registry)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    provider_name = config.get("provider", "?").upper()
    print(f"\nConnecting to {provider_name} provider...")
    if not provider.connect():
        print("ERROR: Could not connect to provider.", file=sys.stderr)
        return 1
    print("  OK\n")

    try:
        print(f"Fetching {args.symbol} {args.timeframe} ({args.candles} bars)...")
        candles = provider.get_candles(args.symbol, args.timeframe, args.candles)
        print(f"  Received: {len(candles)} candles")
        if candles:
            print(f"  From:     {candles[0].timestamp.strftime('%Y-%m-%d')}")
            print(f"  To:       {candles[-1].timestamp.strftime('%Y-%m-%d')}")
    finally:
        provider.disconnect()
        print("  Disconnected.\n")

    if not candles:
        print("ERROR: No candle data returned. Cannot backtest.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Full backtest                                                        #
    # ------------------------------------------------------------------ #
    engine = BacktestEngine.from_config(config, symbol=args.symbol, timeframe=args.timeframe)

    print("Running full backtest...")
    results = engine.run(candles)
    print_results(results, f"BACKTEST RESULTS — {args.symbol}")

    bt_cfg   = config.get("backtest", {})
    min_trades = int(bt_cfg.get("minimum_trades_required", 30))
    health = StrategyHealth.evaluate(results, min_trades=min_trades)
    print_health(health)

    # ------------------------------------------------------------------ #
    # Walk-forward analysis                                                #
    # ------------------------------------------------------------------ #
    if not args.no_walk_forward:
        train_pct = float(bt_cfg.get("walk_forward_train_pct", 0.70))
        min_warmup = int(bt_cfg.get("min_warmup", 210))
        wf = WalkForwardAnalyzer(train_pct=train_pct, warmup=min_warmup)

        print("\nRunning walk-forward analysis...")
        wf_result = wf.analyze(engine, candles)
        print_walk_forward(wf_result)

        oos_health = StrategyHealth.evaluate(wf_result.out_of_sample, min_trades=5)
        print_health(oos_health)

    print()
    print("No trades placed. No orders sent. Read-only simulation.")
    print(f"Full log: logs/backtest_validation.log")
    print()
    return 0 if health.passed else 1


if __name__ == "__main__":
    sys.exit(main())
