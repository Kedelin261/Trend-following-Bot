#!/usr/bin/env python3
"""IBKR Phase 1.5 Connectivity Validation Script.

Connects to TWS, verifies the connection, fetches account summary and
historical candles for EUR.USD and SPY, then disconnects cleanly.

No orders are placed. No trades are executed. Read-only throughout.

Usage:
    python validate_ibkr.py
    python validate_ibkr.py --port 4002      # IB Gateway paper
    python validate_ibkr.py --port 7496      # TWS live account
    python validate_ibkr.py --client-id 2   # if client ID 1 is taken
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.providers.ibkr_provider import IBKRProvider, _IB_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IBKR Phase 1.5 connectivity validation (read-only)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="API port: 7497=TWS paper, 7496=TWS live, 4002=Gateway paper (default: 7497)",
    )
    parser.add_argument(
        "--client-id",
        dest="client_id",
        type=int,
        default=1,
        help="API client ID (default: 1)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler("logs/ibkr_validation.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    args = parse_args()
    setup_logging()

    sep = "=" * 58
    print(f"\n{sep}")
    print("  IBKR CONNECTIVITY VALIDATION — PHASE 1.5")
    print(sep)
    print(f"  Host:       {args.host}:{args.port}")
    print(f"  Client ID:  {args.client_id}")
    print(f"  Mode:       read-only (no orders, no trades)")
    print(sep)

    # Pre-flight: ib-insync installed?
    if not _IB_AVAILABLE:
        print(
            "\n  ERROR: ib-insync is not installed.\n"
            "  Fix:   pip install ib-insync\n"
        )
        return 1

    provider = IBKRProvider(
        config={},
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )

    results: dict = {
        "connection": False,
        "account_summary": False,
        "eurusd_candles": 0,
        "spy_candles": 0,
        "errors": [],
    }

    # ------------------------------------------------------------------ #
    # Step 1 — Connect                                                     #
    # ------------------------------------------------------------------ #
    print("\n[1/6]  Connecting to TWS...")
    if not provider.connect():
        print(
            "       FAILED — could not connect.\n"
            "\n"
            "       Checklist:\n"
            "         1. Open Trader Workstation and log in\n"
            "         2. File → Global Configuration → API → Settings\n"
            "            → enable 'Enable ActiveX and Socket Clients'\n"
            f"           → confirm Socket port is {args.port}\n"
            "         3. Untick 'Read-Only API' if account summary is empty\n"
            "         4. If client ID conflicts, rerun with --client-id 2\n"
        )
        return 1

    results["connection"] = True
    print("       OK — connected\n")

    try:
        # ------------------------------------------------------------------ #
        # Step 2 — Health check                                               #
        # ------------------------------------------------------------------ #
        print("[2/6]  Connection health check...")
        health = provider.health_check()
        for key, val in health.items():
            print(f"         {key}: {val}")
        print()

        # ------------------------------------------------------------------ #
        # Step 3 — Account summary                                            #
        # ------------------------------------------------------------------ #
        print("[3/6]  Account summary...")
        summary = provider.get_account_summary()
        if summary:
            results["account_summary"] = True
            highlight = [
                "AccountType", "TotalCashValue", "NetLiquidation",
                "AvailableFunds", "BuyingPower", "Currency",
            ]
            for tag in highlight:
                if tag in summary:
                    item = summary[tag]
                    val = item.get("value", "") if isinstance(item, dict) else item
                    cur = item.get("currency", "") if isinstance(item, dict) else ""
                    print(f"         {tag}: {val} {cur}".rstrip())
        else:
            print("         (empty — check API permissions in TWS)")
        print()

        # ------------------------------------------------------------------ #
        # Step 4 — EUR.USD H1 historical data                                 #
        # ------------------------------------------------------------------ #
        print("[4/6]  EUR.USD — H1 historical data (100 bars)...")
        try:
            eurusd = provider.get_historical_data("EUR.USD", "H1", 100)
            results["eurusd_candles"] = len(eurusd)
            if eurusd:
                c = eurusd[-1]
                print(f"         Received : {len(eurusd)} candles")
                print(
                    f"         Latest   : {c.timestamp}  "
                    f"O:{c.open:.5f}  H:{c.high:.5f}  "
                    f"L:{c.low:.5f}  C:{c.close:.5f}  Vol:{c.volume:.0f}"
                )
                oldest = eurusd[0]
                print(f"         Oldest   : {oldest.timestamp}")
            else:
                print("         No data returned.")
                print(
                    "         Hint: Forex data requires a market data subscription.\n"
                    "               Paper accounts often include delayed Forex data."
                )
        except Exception as exc:
            results["errors"].append(f"EUR.USD: {exc}")
            print(f"         ERROR: {exc}")
        print()

        # ------------------------------------------------------------------ #
        # Step 5 — SPY D1 historical data                                      #
        # ------------------------------------------------------------------ #
        print("[5/6]  SPY — D1 historical data (50 bars)...")
        try:
            spy = provider.get_historical_data("SPY", "D1", 50)
            results["spy_candles"] = len(spy)
            if spy:
                c = spy[-1]
                print(f"         Received : {len(spy)} candles")
                print(
                    f"         Latest   : {c.timestamp}  "
                    f"O:{c.open:.2f}  H:{c.high:.2f}  "
                    f"L:{c.low:.2f}  C:{c.close:.2f}  Vol:{c.volume:.0f}"
                )
                oldest = spy[0]
                print(f"         Oldest   : {oldest.timestamp}")
            else:
                print("         No data returned.")
                print(
                    "         Hint: US stock data needs a market data subscription.\n"
                    "               Most paper accounts include delayed US equity data."
                )
        except Exception as exc:
            results["errors"].append(f"SPY: {exc}")
            print(f"         ERROR: {exc}")
        print()

        # ------------------------------------------------------------------ #
        # Step 6 — Disconnect                                                  #
        # ------------------------------------------------------------------ #
        print("[6/6]  Disconnecting...")

    finally:
        provider.disconnect()
        print("       OK — disconnected cleanly\n")

    # ------------------------------------------------------------------ #
    # Final summary                                                        #
    # ------------------------------------------------------------------ #
    data_ok = results["eurusd_candles"] > 0 or results["spy_candles"] > 0
    all_ok = results["connection"] and data_ok and not results["errors"]
    status = "PASSED" if all_ok else ("PARTIAL" if results["connection"] else "FAILED")

    print(sep)
    print(f"  VALIDATION {status}")
    print(sep)
    print(f"  Provider:              IBKR (ib-insync)")
    print(f"  Connection:            {'OK' if results['connection'] else 'FAILED'}")
    print(f"  Account summary:       {'OK' if results['account_summary'] else 'EMPTY'}")
    print(f"  EUR.USD H1 candles:    {results['eurusd_candles']}")
    print(f"  SPY D1 candles:        {results['spy_candles']}")
    print(f"  Errors:                {len(results['errors'])}")
    print(f"  Orders placed:         0")
    print(f"  Trades executed:       0")
    print(f"  Log:                   logs/ibkr_validation.log")
    print(sep + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
