#!/usr/bin/env python3
"""Phase 2 Signal Engine Validation Script.

Connects to the configured provider, fetches historical candles for
configured symbols, runs the Signal Engine, and prints a full analysis.

No orders placed. No trades executed. Read-only throughout.

Usage:
    python validate_signals.py
    python validate_signals.py --symbol SPY --symbol QQQ --timeframe D1 --candles 300
    python validate_signals.py --symbol EURUSD --timeframe H1 --candles 300
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings_loader import load_settings
from src.data.symbol_registry import SymbolRegistry
from src.providers.ibkr_provider import _BAR_SIZE_MAP, _duration_str
from src.providers.provider_factory import ProviderFactory
from src.signals.signal_engine import SignalEngine
from src.signals.signal_scorer import SignalScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 signal validation (read-only)")
    p.add_argument(
        "--symbol", dest="symbols", action="append", default=None,
        help="Symbol to analyse (repeatable, default: settings.yaml symbols)",
    )
    p.add_argument(
        "--timeframe", default="D1",
        help="Candle timeframe (default: D1)",
    )
    p.add_argument(
        "--candles", type=int, default=300,
        help="Number of candles to fetch (default: 300)",
    )
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    # INFO to file so provider request details (contract, bar size, duration) are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/signal_validation.log", encoding="utf-8"),
        ],
    )


def print_request_info(
    provider_name: str,
    symbol: str,
    timeframe: str,
    candle_count: int,
) -> None:
    """Print what we are about to request so any failure is immediately diagnosable."""
    bar_size = _BAR_SIZE_MAP.get(timeframe, "UNKNOWN")
    duration = _duration_str(timeframe, candle_count)
    print(f"  Provider:    {provider_name}")
    print(f"  Symbol:      {symbol}")
    print(f"  Timeframe:   {timeframe}  →  IBKR bar size: '{bar_size}'")
    print(f"  Duration:    {duration}  (covers {candle_count} bars + buffer)")
    print(f"  Candles req: {candle_count}")


def print_signal(signal, scorer: SignalScorer) -> None:
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  SIGNAL ANALYSIS — {signal.symbol} / {signal.timeframe}")
    print(sep)
    print(f"  Timestamp:          {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Close Price:        {signal.close_price:.5f}")
    print()
    print(f"  Trend:              {signal.trend_direction.value}")
    if signal.ema50:
        print(f"  EMA50:              {signal.ema50:.5f}")
    if signal.ema200:
        print(f"  EMA200:             {signal.ema200:.5f}")
    print()
    if signal.resistance_level:
        print(f"  Resistance:         {signal.resistance_level:.5f}")
    if signal.support_level:
        print(f"  Support:            {signal.support_level:.5f}")
    print()
    print(f"  Breakout Detected:  {signal.breakout_detected}")
    vol_str = (
        "TRUE" if signal.volume_confirmed is True else
        "FALSE" if signal.volume_confirmed is False else
        "UNKNOWN (Forex / no exchange volume)"
    )
    print(f"  Volume Confirmed:   {vol_str}")
    print()
    print(f"  Signal Score:       {signal.strength_score:.1f} / 100  "
          f"[{signal.strength_category.value}]")
    print()
    print(f"  *** SIGNAL: {signal.signal_type.value} ***")
    print()
    if signal.reasoning:
        print("  Reasoning:")
        for r in signal.reasoning:
            print(f"    - {r}")
    print(sep)


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    symbols = args.symbols or config.get("symbols", ["SPY"])
    timeframe = args.timeframe
    candle_count = args.candles
    provider_name = config.get("provider", "?").upper()

    symbol_registry = SymbolRegistry.from_config(config)

    try:
        provider = ProviderFactory.create(config, symbol_registry)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nConnecting to {provider_name} provider...")
    if not provider.connect():
        print("ERROR: Could not connect to provider.", file=sys.stderr)
        print("  - For IBKR: ensure TWS is running on port 7497", file=sys.stderr)
        print("  - For MT5:  ensure MetaTrader 5 is open and logged in", file=sys.stderr)
        return 1

    print(f"  OK — connected\n")

    engine = SignalEngine()
    scorer = SignalScorer()
    results = []

    try:
        for symbol in symbols:
            print(f"\n{'─' * 54}")
            print(f"  Requesting candles for {symbol} / {timeframe}...")
            print_request_info(provider_name, symbol, timeframe, candle_count)
            print()

            candles = provider.get_candles(symbol, timeframe, candle_count)
            print(f"  Candles received: {len(candles)}")

            if not candles:
                print(
                    f"\n  WARNING: No data returned for {symbol}/{timeframe}\n"
                    f"  Check:\n"
                    f"    1. Symbol is visible in Market Watch / subscribed\n"
                    f"    2. Duration '{_duration_str(timeframe, candle_count)}' "
                    f"is valid for '{_BAR_SIZE_MAP.get(timeframe)}' bars\n"
                    f"    3. Market data subscription covers this instrument\n"
                    f"    4. Logs: logs/signal_validation.log for provider error\n"
                )
                continue

            signal = engine.generate_signal(candles)
            print_signal(signal, scorer)
            results.append(signal)

    finally:
        provider.disconnect()
        print("\n  Disconnected cleanly.")

    # Summary table
    if len(results) > 1:
        sep = "=" * 54
        print(f"\n{sep}")
        print("  SUMMARY")
        print(sep)
        print(f"  {'Symbol':<12} {'Trend':<10} {'Signal':<8} {'Score':>6}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*6}")
        for sig in results:
            print(
                f"  {sig.symbol:<12} {sig.trend_direction.value:<10} "
                f"{sig.signal_type.value:<8} {sig.strength_score:>6.1f}"
            )
        print(sep)

    print()
    print("No trades placed. No orders sent. Read-only mode.")
    print(f"Full request log: logs/signal_validation.log")
    print()
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
