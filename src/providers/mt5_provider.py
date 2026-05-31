"""MetaTrader 5 implementation of MarketDataProvider.

All MT5-specific code is isolated here. No other module in this project
imports MetaTrader5 directly. Provider quirks — symbol suffixes, timeframe
constants, tick structs — never leak out of this file.

Platform note: the MetaTrader5 Python package requires Windows and a
running MT5 terminal. On macOS/Linux this module loads safely but
connect() returns False with a clear log message.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.data.models import Candle, Quote
from src.data.symbol_registry import SymbolRegistry
from src.data.timeframe_registry import TimeframeRegistry
from src.providers.market_data_provider import MarketDataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe MT5 import — the rest of the module works without it installed
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5  # type: ignore[import]
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

# MT5 timeframe constant map (built once; lives only in this module)
_TF_MAP: Dict[str, int] = {}


def _build_tf_map() -> None:
    if not _MT5_AVAILABLE:
        return
    global _TF_MAP
    _TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }


_build_tf_map()

# ---------------------------------------------------------------------------
# MT5 error hints — surfaced in log messages so users know what to fix
# ---------------------------------------------------------------------------
_ERROR_HINTS: Dict[int, str] = {
    -10004: "MT5 terminal is not running. Open MetaTrader 5 first.",
    -10006: "Not logged into broker account. Connect in MT5 File → Login.",
    -10018: "Symbol not found in broker feed. Check Market Watch or add a "
            "broker-suffix alias in config/settings.yaml under symbol_aliases.",
    -10016: "Invalid parameters passed to MT5. Check timeframe and symbol.",
    -6:     "Algo trading is disabled. Enable it in MT5 Tools → Options → "
            "Expert Advisors, then click AutoTrading in the toolbar.",
}
_DEFAULT_HINT = (
    "Check that MT5 terminal is open, logged in, connected to broker, "
    "and that the symbol is visible in Market Watch."
)


class MT5Provider(MarketDataProvider):
    """MetaTrader 5 market data provider.

    Credentials are optional when the user is already logged into the
    running MT5 terminal.  Set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in
    .env to enable credential-based initialization.
    """

    PROVIDER_NAME = "MT5"

    def __init__(self, config: dict, symbol_registry: SymbolRegistry) -> None:
        self._config = config
        self._symbol_registry = symbol_registry
        self._connected = False
        self._available_symbols: List[str] = []

        mt5_cfg = config.get("mt5", {})
        self._login: Optional[int] = self._env_int(mt5_cfg.get("login_env", "MT5_LOGIN"))
        self._password: Optional[str] = self._env_str(mt5_cfg.get("password_env", "MT5_PASSWORD"))
        self._server: Optional[str] = self._env_str(mt5_cfg.get("server_env", "MT5_SERVER"))
        self._retry_count: int = int(mt5_cfg.get("retry_count", 3))
        self._retry_delay: float = float(mt5_cfg.get("retry_delay_seconds", 5))
        self._allow_no_creds: bool = bool(
            mt5_cfg.get("allow_initialize_without_credentials", True)
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_int(env_name: str) -> Optional[int]:
        val = os.environ.get(env_name, "").strip()
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            logger.warning("MT5 env var %s must be an integer, got: %s", env_name, val)
            return None

    @staticmethod
    def _env_str(env_name: str) -> Optional[str]:
        val = os.environ.get(env_name, "").strip()
        return val or None

    def _log_mt5_error(
        self,
        operation: str,
        symbol: str = "",
        timeframe: str = "",
    ) -> None:
        if not _MT5_AVAILABLE:
            return
        code, msg = mt5.last_error()
        hint = _ERROR_HINTS.get(code, _DEFAULT_HINT)
        logger.error(
            "provider_error: %s failed | provider=%s symbol=%s timeframe=%s "
            "mt5_code=%d mt5_msg=%s | Hint: %s",
            operation,
            self.PROVIDER_NAME,
            symbol,
            timeframe,
            code,
            msg,
            hint,
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the running MT5 terminal.

        Returns True on success. Retries up to retry_count times.
        """
        if not _MT5_AVAILABLE:
            logger.error(
                "provider_connect_failure: MetaTrader5 package not installed. "
                "provider=%s | Hint: Run 'pip install MetaTrader5' on Windows. "
                "On macOS/Linux, use a different provider or a Windows VPS.",
                self.PROVIDER_NAME,
            )
            return False

        logger.info(
            "provider_connect_attempt: provider=%s", self.PROVIDER_NAME
        )

        for attempt in range(1, self._retry_count + 1):
            if self._try_initialize():
                self._connected = True
                self._load_available_symbols()
                logger.info(
                    "provider_connect_success: provider=%s attempt=%d/%d",
                    self.PROVIDER_NAME,
                    attempt,
                    self._retry_count,
                )
                return True

            if attempt < self._retry_count:
                logger.warning(
                    "provider_connect_failure: attempt %d/%d failed, "
                    "retrying in %.1fs...",
                    attempt,
                    self._retry_count,
                    self._retry_delay,
                )
                time.sleep(self._retry_delay)

        logger.error(
            "provider_connect_failure: all %d attempts exhausted. "
            "provider=%s | Hint: Ensure MT5 terminal is open and logged in.",
            self._retry_count,
            self.PROVIDER_NAME,
        )
        return False

    def _try_initialize(self) -> bool:
        """Attempt a single mt5.initialize() call."""
        if self._login and self._password and self._server:
            ok = mt5.initialize(
                login=self._login,
                password=self._password,
                server=self._server,
            )
        elif self._allow_no_creds:
            ok = mt5.initialize()
        else:
            logger.error(
                "provider_connect_failure: No credentials provided and "
                "allow_initialize_without_credentials=false. "
                "Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env"
            )
            return False

        if not ok:
            self._log_mt5_error("mt5.initialize")
            return False
        return True

    def disconnect(self) -> None:
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("provider_disconnect: provider=%s", self.PROVIDER_NAME)

    def is_connected(self) -> bool:
        if not _MT5_AVAILABLE or not self._connected:
            return False
        try:
            return mt5.terminal_info() is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Account / terminal metadata
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict:
        if not self.is_connected():
            return {}
        info = mt5.account_info()
        if info is None:
            self._log_mt5_error("get_account_info")
            return {}
        result = dict(info._asdict())
        logger.info(
            "account_info_loaded: provider=%s login=%s server=%s "
            "trade_mode=%s balance=%s",
            self.PROVIDER_NAME,
            result.get("login"),
            result.get("server"),
            result.get("trade_mode"),
            result.get("balance"),
        )
        return result

    def get_terminal_info(self) -> Dict:
        if not self.is_connected():
            return {}
        info = mt5.terminal_info()
        if info is None:
            self._log_mt5_error("get_terminal_info")
            return {}
        result = dict(info._asdict())
        logger.info(
            "terminal_info_loaded: provider=%s connected=%s "
            "trade_allowed=%s build=%s",
            self.PROVIDER_NAME,
            result.get("connected"),
            result.get("trade_allowed"),
            result.get("build"),
        )
        return result

    # ------------------------------------------------------------------
    # Symbol management
    # ------------------------------------------------------------------

    def _load_available_symbols(self) -> None:
        symbols = mt5.symbols_get()
        if symbols:
            self._available_symbols = [s.name for s in symbols]
            logger.info(
                "symbol_registry_loaded: provider=%s count=%d",
                self.PROVIDER_NAME,
                len(self._available_symbols),
            )
        else:
            logger.warning(
                "symbol_registry_empty: No symbols returned. provider=%s "
                "| Hint: Enable symbols in Market Watch (right-click → Show All).",
                self.PROVIDER_NAME,
            )

    def _resolve_and_select(self, symbol: str) -> Optional[str]:
        """Resolve canonical symbol to broker name and ensure it is selected."""
        resolved = self._symbol_registry.resolve(symbol, self._available_symbols)
        if resolved is None:
            return None
        # Select in Market Watch so price data is streamed by the terminal
        if not mt5.symbol_select(resolved, True):
            logger.warning(
                "symbol_select_failed: %s | provider=%s "
                "| Hint: Symbol may be restricted for this account type.",
                resolved,
                self.PROVIDER_NAME,
            )
        return resolved

    def get_symbol_info(self, symbol: str) -> Dict:
        if not self.is_connected():
            return {}
        resolved = self._resolve_and_select(symbol)
        if not resolved:
            return {}
        info = mt5.symbol_info(resolved)
        if info is None:
            self._log_mt5_error("get_symbol_info", symbol=symbol)
            return {}
        return dict(info._asdict())

    def validate_symbol(self, symbol: str) -> bool:
        if not self.is_connected():
            return False
        return self._resolve_and_select(symbol) is not None

    def list_available_symbols(self) -> List[str]:
        return self._available_symbols.copy()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        if not self.is_connected():
            return None
        resolved = self._resolve_and_select(symbol)
        if not resolved:
            return None
        tick = mt5.symbol_info_tick(resolved)
        if tick is None:
            self._log_mt5_error("get_latest_quote", symbol=symbol)
            return None
        return Quote(
            symbol=symbol,
            bid=float(tick.bid),
            ask=float(tick.ask),
            spread=round(float(tick.ask) - float(tick.bid), 8),
            timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
            provider=self.PROVIDER_NAME,
        )

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> List[Candle]:
        if not self.is_connected():
            return []

        mt5_tf = self._resolve_timeframe(timeframe, symbol)
        if mt5_tf is None:
            return []

        resolved = self._resolve_and_select(symbol)
        if not resolved:
            return []

        logger.info(
            "candles_requested: provider=%s symbol=%s resolved=%s "
            "timeframe=%s count=%d",
            self.PROVIDER_NAME,
            symbol,
            resolved,
            timeframe,
            count,
        )
        rates = mt5.copy_rates_from_pos(resolved, mt5_tf, 0, count)
        return self._rates_to_candles(rates, symbol, resolved, timeframe)

    def get_candles_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Candle]:
        if not self.is_connected():
            return []

        mt5_tf = self._resolve_timeframe(timeframe, symbol)
        if mt5_tf is None:
            return []

        resolved = self._resolve_and_select(symbol)
        if not resolved:
            return []

        logger.info(
            "candles_requested: provider=%s symbol=%s timeframe=%s "
            "range=%s to %s",
            self.PROVIDER_NAME,
            symbol,
            timeframe,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        rates = mt5.copy_rates_range(resolved, mt5_tf, start_date, end_date)
        return self._rates_to_candles(rates, symbol, resolved, timeframe)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_timeframe(self, timeframe: str, symbol: str = "") -> Optional[int]:
        """Map a standard timeframe string to an MT5 constant."""
        if not TimeframeRegistry.is_valid(timeframe):
            logger.error(
                "provider_error: unsupported timeframe '%s'. "
                "Valid: %s | provider=%s symbol=%s",
                timeframe,
                TimeframeRegistry.get_all(),
                self.PROVIDER_NAME,
                symbol,
            )
            return None
        mt5_tf = _TF_MAP.get(timeframe)
        if mt5_tf is None:
            logger.error(
                "provider_error: no MT5 constant for timeframe '%s'. "
                "provider=%s",
                timeframe,
                self.PROVIDER_NAME,
            )
            return None
        return mt5_tf

    def _rates_to_candles(
        self,
        rates,
        symbol: str,
        resolved_symbol: str,
        timeframe: str,
    ) -> List[Candle]:
        """Convert a numpy rates array to a list of standard Candle objects."""
        if rates is None or len(rates) == 0:
            logger.warning(
                "candles_received: empty response | provider=%s symbol=%s "
                "timeframe=%s | Hint: Market may be closed or the symbol "
                "has no data for this timeframe. Try again during market hours.",
                self.PROVIDER_NAME,
                symbol,
                timeframe,
            )
            return []

        candles = [
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.fromtimestamp(r["time"], tz=timezone.utc),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["tick_volume"]),
                provider=self.PROVIDER_NAME,
                resolved_symbol=resolved_symbol,
            )
            for r in rates
        ]

        logger.info(
            "candles_received: count=%d | provider=%s symbol=%s timeframe=%s",
            len(candles),
            self.PROVIDER_NAME,
            symbol,
            timeframe,
        )
        return candles
