"""Orchestrates fetching candles from a provider and saving them to the store.

DataSyncService depends only on MarketDataProvider (the interface) and
CandleStore, so it works identically regardless of which provider is active.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from src.data.candle_store import CandleStore
from src.data.symbol_registry import SymbolRegistry
from src.data.timeframe_registry import TimeframeRegistry
from src.providers.market_data_provider import MarketDataProvider

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Summary produced by a single sync_latest_candles run."""

    provider: str
    symbols_processed: int = 0
    timeframes_processed: int = 0
    candles_requested: int = 0
    candles_saved: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    symbol_errors: List[str] = field(default_factory=list)


class DataSyncService:
    """Pulls candles from a provider and persists them to the candle store."""

    def __init__(
        self,
        provider: MarketDataProvider,
        candle_store: CandleStore,
        symbol_registry: SymbolRegistry,
        timeframes: List[str],
    ) -> None:
        self._provider = provider
        self._store = candle_store
        self._symbol_registry = symbol_registry
        self._timeframes = [
            tf for tf in timeframes if TimeframeRegistry.is_valid(tf)
        ]

        invalid = [tf for tf in timeframes if not TimeframeRegistry.is_valid(tf)]
        if invalid:
            logger.warning(
                "data_sync_service: ignoring unknown timeframes: %s", invalid
            )

    def sync_latest_candles(self, count: int) -> SyncReport:
        """Fetch the latest *count* candles for every symbol/timeframe pair.

        Returns a SyncReport summarising the outcome.
        """
        provider_name = getattr(
            type(self._provider), "PROVIDER_NAME", type(self._provider).__name__
        )
        report = SyncReport(provider=provider_name)

        symbols = self._symbol_registry.get_symbols()
        report.symbols_processed = len(symbols)
        report.timeframes_processed = len(self._timeframes)

        for symbol in symbols:
            for timeframe in self._timeframes:
                try:
                    candles = self._provider.get_candles(symbol, timeframe, count)
                    report.candles_requested += count

                    if candles:
                        saved, dupes = self._store.save_candles(candles)
                        report.candles_saved += saved
                        report.duplicates_skipped += dupes
                    else:
                        logger.warning(
                            "sync_no_data: symbol=%s timeframe=%s "
                            "| Hint: Check Market Watch and market hours.",
                            symbol,
                            timeframe,
                        )
                except Exception as exc:
                    report.errors += 1
                    report.symbol_errors.append(f"{symbol}/{timeframe}")
                    logger.error(
                        "provider_error: sync failed | symbol=%s timeframe=%s "
                        "error=%s | Hint: Verify symbol visibility in Market Watch.",
                        symbol,
                        timeframe,
                        exc,
                    )

        logger.info(
            "sync_complete: provider=%s symbols=%d timeframes=%d "
            "requested=%d saved=%d dupes=%d errors=%d",
            provider_name,
            report.symbols_processed,
            report.timeframes_processed,
            report.candles_requested,
            report.candles_saved,
            report.duplicates_skipped,
            report.errors,
        )
        return report
