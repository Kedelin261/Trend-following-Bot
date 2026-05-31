"""Broker-agnostic abstract interface for all market data providers.

ARCHITECTURE RULE: Every downstream module — signal engine, risk engine,
backtester, journal, dashboard — must import ONLY this interface.
No module outside src/providers/ may import a concrete provider class.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from src.data.models import Candle, Quote


class MarketDataProvider(ABC):
    """Abstract base class for market data providers.

    Implementations: MT5Provider, IBKRProvider, PolygonProvider, etc.
    All methods must be safe to call: never raise, log errors internally,
    and return empty / None / False on failure.
    """

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source.

        Returns True on success. Logs errors internally and returns False
        on any failure — never raises.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly close the provider connection."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the connection is currently active."""

    # ------------------------------------------------------------------
    # Account / terminal metadata
    # ------------------------------------------------------------------

    @abstractmethod
    def get_account_info(self) -> Dict:
        """Return account metadata as a plain dict.

        Minimum expected keys: login, balance, currency, server,
        trade_mode (0=demo, 2=live). Returns {} on failure.
        """

    @abstractmethod
    def get_terminal_info(self) -> Dict:
        """Return terminal/platform metadata as a plain dict.

        Minimum expected keys: connected, trade_allowed, build/version.
        Returns {} on failure.
        """

    # ------------------------------------------------------------------
    # Symbol queries
    # ------------------------------------------------------------------

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict:
        """Return symbol metadata (digits, spread, pip size, etc.).

        Returns {} if symbol is unknown or unavailable.
        """

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Return True if the symbol is recognised by this provider."""

    @abstractmethod
    def list_available_symbols(self) -> List[str]:
        """Return all symbols currently available from the provider."""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Return the latest bid/ask quote for *symbol*.

        Returns None if unavailable (market closed, symbol invalid, etc.).
        """

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> List[Candle]:
        """Return the most recent *count* closed candles.

        *timeframe* must be a standard registry key: M1, M5, H1, H4, D1, etc.
        Returns [] if data is unavailable — never raises.
        """

    @abstractmethod
    def get_candles_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Candle]:
        """Return candles within an inclusive date range.

        Returns [] if data is unavailable — never raises.
        """
