"""Interactive Brokers market data provider using ib-insync.

Read-only mode. No order placement, no execution, no strategy logic.
Connects to TWS (Trader Workstation) or IB Gateway running locally.

Port reference:
  7497  TWS paper trading
  7496  TWS live trading
  4002  IB Gateway paper
  4001  IB Gateway live
"""

import logging
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from src.data.models import Candle, Quote
from src.data.symbol_registry import SymbolRegistry
from src.data.timeframe_registry import TimeframeRegistry
from src.providers.market_data_provider import MarketDataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe ib-insync import
# ---------------------------------------------------------------------------
try:
    from ib_insync import IB, Contract, Forex, Stock  # type: ignore[import]
    _IB_AVAILABLE = True
except ImportError:
    IB = None          # type: ignore[assignment,misc]
    Contract = None    # type: ignore[assignment,misc]
    Forex = None       # type: ignore[assignment,misc]
    Stock = None       # type: ignore[assignment,misc]
    _IB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Timeframe mappings (all IBKR-specific constants stay in this file)
# ---------------------------------------------------------------------------

# Standard timeframe string → IBKR barSizeSetting
_BAR_SIZE_MAP: Dict[str, str] = {
    "M1":  "1 min",
    "M5":  "5 mins",
    "M15": "15 mins",
    "M30": "30 mins",
    "H1":  "1 hour",
    "H4":  "4 hours",
    "D1":  "1 day",
    "W1":  "1 week",
}

# Minutes of market time per bar (used to estimate request duration)
_MINUTES_PER_BAR: Dict[str, int] = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  60,
    "H4":  240,
    "D1":  1440,
    "W1":  10080,
}

# Currency codes accepted by IBKR as Forex symbols
_FOREX_CODES = frozenset(
    "EUR GBP USD JPY CHF AUD NZD CAD HKD SGD NOK SEK DKK "
    "MXN ZAR CNH TRY PLN CZK HUF RUB BTC ETH".split()
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_server_version(ib) -> Optional[int]:
    """Return the TWS server version without raising if the attribute is absent.

    ib-insync does not expose serverVersion directly on the IB object in all
    versions. The canonical path is ib.client.serverVersion(); older builds
    may place it elsewhere. This function tries known paths and returns None
    if none succeed — a missing server version must never abort a connection.
    """
    for getter in (
        lambda: ib.client.serverVersion(),  # ib-insync standard path
        lambda: ib.serverVersion(),         # some older / patched builds
    ):
        try:
            result = getter()
            if result is not None:
                return result
        except (AttributeError, TypeError):
            continue
    return None

def _duration_str(timeframe: str, count: int) -> str:
    """Return a valid IBKR durationStr that covers at least *count* bars.

    Adds a 60 % buffer to account for weekends and non-trading hours.

    IBKR accepts: "N D" (1–365), "N W", "N M" (1–12 only), "N Y".
    To avoid generating invalid month counts (> 12) we skip the month
    format entirely and go straight from days to years.  This is safe
    because IBKR accepts multi-year durations for all daily+ bar sizes.
    """
    minutes = _MINUTES_PER_BAR.get(timeframe, 60) * count * 1.6
    days = minutes / (60 * 24)

    if days <= 365:
        return f"{max(1, int(days) + 1)} D"

    years = days / 365
    return f"{max(1, int(years) + 1)} Y"


def _to_utc_datetime(date_val) -> datetime:
    """Normalise any date/datetime returned by ib-insync to a UTC datetime."""
    if isinstance(date_val, datetime):
        return date_val if date_val.tzinfo else date_val.replace(tzinfo=timezone.utc)
    if isinstance(date_val, date):
        return datetime(date_val.year, date_val.month, date_val.day, tzinfo=timezone.utc)
    # Fallback: try string parsing
    s = str(date_val).strip()
    for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.warning("ibkr: could not parse date '%s', using now()", s)
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class IBKRProvider(MarketDataProvider):
    """Interactive Brokers market data provider via ib-insync.

    Fully implements MarketDataProvider so it is a drop-in replacement for
    MT5Provider in any downstream module.  Extra IBKR-specific methods
    (get_account_summary, get_positions, get_historical_data, health_check)
    are exposed for convenience scripts and diagnostics.
    """

    PROVIDER_NAME = "IBKR"

    def __init__(
        self,
        config: dict,
        symbol_registry: Optional[SymbolRegistry] = None,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: float = 10.0,
    ) -> None:
        self._symbol_registry = symbol_registry

        ibkr_cfg = config.get("ibkr", {})
        self._host: str = ibkr_cfg.get("host", host)
        self._port: int = int(ibkr_cfg.get("port", port))
        self._client_id: int = int(ibkr_cfg.get("client_id", client_id))
        self._timeout: float = float(ibkr_cfg.get("timeout", timeout))
        self._readonly: bool = bool(ibkr_cfg.get("readonly", True))

        self._ib: Optional[IB] = None  # type: ignore[type-arg]
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not _IB_AVAILABLE:
            logger.error(
                "provider_connect_failure: ib-insync not installed. "
                "provider=%s | Hint: pip install ib-insync",
                self.PROVIDER_NAME,
            )
            return False

        logger.info(
            "provider_connect_attempt: provider=%s host=%s port=%d clientId=%d",
            self.PROVIDER_NAME,
            self._host,
            self._port,
            self._client_id,
        )
        try:
            self._ib = IB()
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=self._timeout,
                readonly=self._readonly,
            )
            self._connected = True
            logger.info(
                "provider_connect_success: provider=%s server_version=%s",
                self.PROVIDER_NAME,
                _safe_server_version(self._ib),
            )
            return True

        except ConnectionRefusedError:
            logger.error(
                "provider_connect_failure: Connection refused at %s:%d. "
                "provider=%s | Hint: Open Trader Workstation, then go to "
                "File → Global Configuration → API → Settings and enable "
                "'Enable ActiveX and Socket Clients'.",
                self._host,
                self._port,
                self.PROVIDER_NAME,
            )
        except TimeoutError:
            logger.error(
                "provider_connect_failure: Timeout after %.1fs. "
                "provider=%s | Hint: Increase ibkr.timeout in settings.yaml.",
                self._timeout,
                self.PROVIDER_NAME,
            )
        except Exception as exc:
            logger.error(
                "provider_connect_failure: %s | provider=%s "
                "| Hint: Check TWS API configuration and client ID conflicts.",
                exc,
                self.PROVIDER_NAME,
            )

        self._connected = False
        return False

    def disconnect(self) -> None:
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception as exc:
                logger.warning("provider_disconnect_error: %s", exc)
            finally:
                self._connected = False
                logger.info("provider_disconnect: provider=%s", self.PROVIDER_NAME)

    def is_connected(self) -> bool:
        if not _IB_AVAILABLE or not self._ib:
            return False
        try:
            return self._ib.isConnected()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Account / terminal metadata  (abstract interface)
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict:
        """Return a flattened account summary dict compatible with the interface."""
        raw = self.get_account_summary()
        return {
            tag: (v["value"] if isinstance(v, dict) else v)
            for tag, v in raw.items()
        }

    def get_terminal_info(self) -> Dict:
        """Return health check data as terminal info."""
        return self.health_check()

    # ------------------------------------------------------------------
    # Symbol queries  (abstract interface)
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> Dict:
        if not self.is_connected():
            return {}
        contract = self._make_contract(symbol)
        if contract is None:
            return {}
        try:
            details = self._ib.reqContractDetails(contract)
            if not details:
                logger.warning(
                    "symbol_resolution_failure: '%s' returned no contract details. "
                    "provider=%s | Hint: Check symbol name and data subscriptions.",
                    symbol,
                    self.PROVIDER_NAME,
                )
                return {}
            cd = details[0]
            logger.info("symbol_resolution_success: %s | provider=%s", symbol, self.PROVIDER_NAME)
            return {
                "symbol": cd.contract.symbol,
                "secType": cd.contract.secType,
                "exchange": cd.contract.exchange,
                "currency": cd.contract.currency,
                "longName": cd.longName,
                "minTick": cd.minTick,
            }
        except Exception as exc:
            logger.error("provider_error: get_symbol_info | symbol=%s error=%s", symbol, exc)
            return {}

    def validate_symbol(self, symbol: str) -> bool:
        return bool(self.get_symbol_info(symbol))

    def list_available_symbols(self) -> List[str]:
        """Return symbols from the registry; IBKR has no global symbol list."""
        if self._symbol_registry:
            return self._symbol_registry.get_symbols()
        return []

    # ------------------------------------------------------------------
    # Market data  (abstract interface)
    # ------------------------------------------------------------------

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Fetch a snapshot tick. Requires live market data subscription."""
        if not self.is_connected():
            return None
        contract = self._make_contract(symbol)
        if contract is None:
            return None
        try:
            # Request snapshot (non-streaming)
            ticker = self._ib.reqMktData(contract, "", True, False)
            self._ib.sleep(1.0)  # allow ticker to populate
            self._ib.cancelMktData(contract)

            bid = float(ticker.bid or 0.0)
            ask = float(ticker.ask or 0.0)
            if bid <= 0 or ask <= 0:
                logger.warning(
                    "candles_received: no tick data for %s. provider=%s "
                    "| Hint: Market may be closed or no live data subscription.",
                    symbol,
                    self.PROVIDER_NAME,
                )
                return None
            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread=round(ask - bid, 8),
                timestamp=datetime.now(tz=timezone.utc),
                provider=self.PROVIDER_NAME,
            )
        except Exception as exc:
            logger.error("provider_error: get_latest_quote | symbol=%s error=%s", symbol, exc)
            return None

    def get_candles(self, symbol: str, timeframe: str, count: int) -> List[Candle]:
        """Return the most recent *count* candles via IBKR historical data."""
        return self.get_historical_data(symbol, timeframe, count)

    def get_candles_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Candle]:
        if not self.is_connected():
            return []

        bar_size = _BAR_SIZE_MAP.get(timeframe)
        if not bar_size:
            logger.error(
                "provider_error: unsupported timeframe '%s'. "
                "Valid: %s | provider=%s",
                timeframe,
                list(_BAR_SIZE_MAP.keys()),
                self.PROVIDER_NAME,
            )
            return []

        contract = self._make_contract(symbol)
        if contract is None:
            return []

        delta_days = max(1, (end_date - start_date).days + 1)
        if delta_days <= 365:
            duration = f"{delta_days} D"
        else:
            # Skip months format — IBKR only accepts 1–12 months; use years
            duration = f"{delta_days // 365 + 1} Y"

        logger.info(
            "candles_requested: provider=%s symbol=%s timeframe=%s range=%s to %s",
            self.PROVIDER_NAME,
            symbol,
            timeframe,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=self._what_to_show(contract),
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
            candles = [self._bar_to_candle(b, symbol, timeframe) for b in (bars or [])]
            candles = [c for c in candles if start_date <= c.timestamp <= end_date]
            logger.info(
                "candles_received: count=%d | provider=%s symbol=%s timeframe=%s",
                len(candles),
                self.PROVIDER_NAME,
                symbol,
                timeframe,
            )
            return candles
        except Exception as exc:
            logger.error(
                "provider_error: get_candles_range | symbol=%s timeframe=%s error=%s",
                symbol,
                timeframe,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # IBKR-specific public methods
    # ------------------------------------------------------------------

    def get_account_summary(self) -> Dict:
        """Return IBKR account summary as {tag: {value, currency}} dict."""
        if not self.is_connected():
            return {}
        try:
            values = self._ib.accountSummary()
            result: Dict[str, Dict] = {}
            for item in values:
                result[item.tag] = {"value": item.value, "currency": item.currency}
            logger.info(
                "account_info_loaded: provider=%s tags=%d", self.PROVIDER_NAME, len(result)
            )
            return result
        except Exception as exc:
            logger.error("provider_error: get_account_summary | %s", exc)
            return {}

    def get_positions(self) -> List[Dict]:
        """Return all open positions as a list of plain dicts.

        Read-only — this method never places or modifies orders.
        """
        if not self.is_connected():
            return []
        try:
            positions = self._ib.positions()
            result = []
            for pos in positions:
                result.append(
                    {
                        "account": pos.account,
                        "symbol": pos.contract.symbol,
                        "secType": pos.contract.secType,
                        "exchange": pos.contract.exchange,
                        "currency": pos.contract.currency,
                        "position": float(pos.position),
                        "avgCost": float(pos.avgCost),
                    }
                )
            logger.info(
                "positions_loaded: provider=%s count=%d", self.PROVIDER_NAME, len(result)
            )
            return result
        except Exception as exc:
            logger.error("provider_error: get_positions | %s", exc)
            return []

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
    ) -> List[Candle]:
        """Request historical bars from IBKR and return normalized Candles.

        *bars* is the requested count; actual returned count may differ if
        the market lacks data for the full range.
        """
        if not self.is_connected():
            return []

        if not TimeframeRegistry.is_valid(timeframe):
            logger.error(
                "provider_error: unsupported timeframe '%s'. "
                "Valid: %s | provider=%s",
                timeframe,
                TimeframeRegistry.get_all(),
                self.PROVIDER_NAME,
            )
            return []

        bar_size = _BAR_SIZE_MAP[timeframe]
        contract = self._make_contract(symbol)
        if contract is None:
            return []

        duration = _duration_str(timeframe, bars)
        what_to_show = self._what_to_show(contract)

        logger.info(
            "candles_requested: provider=%s symbol=%s "
            "contract_type=%s exchange=%s currency=%s "
            "timeframe=%s bar_size=%s duration=%s count=%d whatToShow=%s",
            self.PROVIDER_NAME,
            symbol,
            getattr(contract, "secType",  "UNKNOWN"),
            getattr(contract, "exchange", "UNKNOWN"),
            getattr(contract, "currency", "UNKNOWN"),
            timeframe,
            bar_size,
            duration,
            bars,
            what_to_show,
        )
        try:
            raw_bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            logger.error(
                "provider_error: reqHistoricalData failed | provider=%s "
                "symbol=%s timeframe=%s error=%s "
                "| Hint: Check market data subscriptions and pacing limits.",
                self.PROVIDER_NAME,
                symbol,
                timeframe,
                exc,
            )
            return []

        if not raw_bars:
            logger.warning(
                "candles_received: empty response | provider=%s symbol=%s "
                "timeframe=%s | Hint: Market may be closed, or no data "
                "subscription for this instrument.",
                self.PROVIDER_NAME,
                symbol,
                timeframe,
            )
            return []

        # Trim to the most recent *bars* candles
        trimmed = list(raw_bars)[-bars:] if len(raw_bars) > bars else list(raw_bars)
        candles = [self._bar_to_candle(b, symbol, timeframe) for b in trimmed]

        logger.info(
            "candles_received: count=%d | provider=%s symbol=%s timeframe=%s",
            len(candles),
            self.PROVIDER_NAME,
            symbol,
            timeframe,
        )
        return candles

    def health_check(self) -> Dict:
        """Return connection health and TWS version information."""
        if not self._ib:
            return {
                "connected": False,
                "provider": self.PROVIDER_NAME,
                "host": self._host,
                "port": self._port,
            }
        try:
            connected = self._ib.isConnected()
            return {
                "connected": connected,
                "provider": self.PROVIDER_NAME,
                "host": self._host,
                "port": self._port,
                "client_id": self._client_id,
                "readonly": self._readonly,
                "server_version": _safe_server_version(self._ib) if connected else None,
            }
        except Exception as exc:
            return {
                "connected": False,
                "provider": self.PROVIDER_NAME,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_contract(self, symbol: str):
        """Resolve a canonical symbol string to an ib-insync Contract.

        Detection rules:
        - Contains '.' or is 6-char alpha and both halves are forex codes
          → Forex contract on IDEALPRO
        - Anything else → Stock on SMART routing, USD currency
        """
        if not _IB_AVAILABLE:
            return None

        clean = symbol.replace(".", "").replace("/", "").upper()

        if len(clean) == 6 and clean.isalpha():
            base, quote = clean[:3], clean[3:]
            if base in _FOREX_CODES and quote in _FOREX_CODES:
                logger.debug(
                    "symbol_resolution_attempt: %s → Forex('%s') | provider=%s",
                    symbol,
                    clean,
                    self.PROVIDER_NAME,
                )
                return Forex(clean)

        logger.debug(
            "symbol_resolution_attempt: %s → Stock('%s', 'SMART', 'USD') | provider=%s",
            symbol,
            symbol.upper(),
            self.PROVIDER_NAME,
        )
        return Stock(symbol.upper(), "SMART", "USD")

    @staticmethod
    def _what_to_show(contract) -> str:
        """Return the appropriate IBKR whatToShow value for a contract type."""
        sec_type = getattr(contract, "secType", "") or ""
        # Forex / currency swaps use MIDPOINT; everything else uses TRADES
        return "MIDPOINT" if sec_type == "CASH" else "TRADES"

    def _bar_to_candle(self, bar, symbol: str, timeframe: str) -> Candle:
        """Convert an ib-insync BarData object to a standard Candle."""
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=_to_utc_datetime(bar.date),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            provider=self.PROVIDER_NAME,
        )
