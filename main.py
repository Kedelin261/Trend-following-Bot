"""Phase 1: Market Data Foundation — entry point.

Connects to the configured provider, syncs historical candles to SQLite,
and prints a summary report. No trades are placed. No orders are sent.
"""

import logging
import sys
from pathlib import Path

from src.config.settings_loader import load_settings
from src.data.candle_store import CandleStore
from src.data.data_sync_service import DataSyncService
from src.data.symbol_registry import SymbolRegistry
from src.providers.provider_factory import ProviderFactory


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "logs/bot.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Hard safety gate — Phase 1 must never enable trading
    safety = config.get("safety", {})
    if safety.get("trading_enabled") or safety.get("orders_allowed"):
        logger.critical(
            "SAFETY VIOLATION: trading_enabled or orders_allowed is True. "
            "Phase 1 does not support trading. Aborting."
        )
        return 1

    symbol_registry = SymbolRegistry.from_config(config)
    candle_store = CandleStore(config["database"]["path"])

    try:
        provider = ProviderFactory.create(config, symbol_registry)
    except ValueError as exc:
        logger.error("Provider creation failed: %s", exc)
        return 1

    if not provider.connect():
        logger.error(
            "Connection failed. Ensure the MT5 terminal is open, "
            "logged into your account, and connected to the broker."
        )
        return 1

    account_info: dict = {}
    report = None

    try:
        account_info = provider.get_account_info()
        provider.get_terminal_info()

        candle_count: int = config.get("history", {}).get("default_candle_count", 500)
        timeframes = config.get("timeframes", [])

        sync_service = DataSyncService(
            provider=provider,
            candle_store=candle_store,
            symbol_registry=symbol_registry,
            timeframes=timeframes,
        )
        report = sync_service.sync_latest_candles(count=candle_count)

    finally:
        provider.disconnect()

    if report is None:
        logger.error("Sync did not complete.")
        return 1

    # Determine account type for display (MT5 trade_mode: 0=demo, 2=live)
    account_type = "Demo" if account_info.get("trade_mode", 0) == 0 else "Live"
    timeframes = config.get("timeframes", [])

    sep = "=" * 54
    print(f"\n{sep}")
    print("  PHASE 1 MARKET DATA SYNC COMPLETE")
    print(sep)
    print(f"  Provider:             {report.provider}")
    print(f"  Connected:            Yes")
    print(f"  Account Type:         {account_type}")
    print(f"  Login:                {account_info.get('login', 'N/A')}")
    print(f"  Server:               {account_info.get('server', 'N/A')}")
    print(f"  Symbols:              {', '.join(symbol_registry.get_symbols())}")
    print(f"  Timeframes:           {', '.join(timeframes)}")
    print(f"  Candles Requested:    {report.candles_requested}")
    print(f"  Candles Saved:        {report.candles_saved}")
    print(f"  Duplicates Skipped:   {report.duplicates_skipped}")
    print(f"  Errors:               {report.errors}")
    if report.symbol_errors:
        print(f"  Failed Pairs:         {', '.join(report.symbol_errors)}")
    print(f"  Database:             {config['database']['path']}")
    print(sep)
    print("  No trades placed.")
    print("  Trading disabled.")
    print(f"{sep}\n")

    return 0 if report.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
