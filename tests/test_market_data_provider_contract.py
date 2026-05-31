"""Contract tests for MarketDataProvider.

Any valid implementation must pass all tests in this module.
Uses a minimal in-memory implementation — no real provider required.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.data.models import Candle, Quote
from src.providers.market_data_provider import MarketDataProvider


# ---------------------------------------------------------------------------
# Minimal concrete implementation for contract verification
# ---------------------------------------------------------------------------

class _StubProvider(MarketDataProvider):
    """Minimal implementation that satisfies the interface contract."""

    def __init__(self) -> None:
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> Dict:
        return {"login": 1234, "balance": 10000.0, "trade_mode": 0, "server": "Demo"}

    def get_terminal_info(self) -> Dict:
        return {"connected": True, "trade_allowed": False, "build": 3800}

    def get_symbol_info(self, symbol: str) -> Dict:
        return {"name": symbol, "digits": 5, "spread": 1}

    def validate_symbol(self, symbol: str) -> bool:
        return symbol in ("EURUSD", "GBPUSD")

    def list_available_symbols(self) -> List[str]:
        return ["EURUSD", "GBPUSD"]

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        return Quote(
            symbol=symbol,
            bid=1.0800,
            ask=1.0801,
            spread=0.0001,
            timestamp=datetime.now(tz=timezone.utc),
            provider="STUB",
        )

    def get_candles(self, symbol: str, timeframe: str, count: int) -> List[Candle]:
        return [
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                open=1.08,
                high=1.09,
                low=1.07,
                close=1.085,
                volume=1000.0,
                provider="STUB",
            )
        ]

    def get_candles_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Candle]:
        return self.get_candles(symbol, timeframe, 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider() -> _StubProvider:
    p = _StubProvider()
    p.connect()
    return p


# ---------------------------------------------------------------------------
# Contract tests — must pass for any MarketDataProvider implementation
# ---------------------------------------------------------------------------

class TestMarketDataProviderContract:

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            MarketDataProvider()  # type: ignore[abstract]

    def test_connect_returns_bool(self, provider):
        result = provider.connect()
        assert isinstance(result, bool)

    def test_is_connected_true_after_connect(self, provider):
        assert provider.is_connected() is True

    def test_is_connected_false_after_disconnect(self, provider):
        provider.disconnect()
        assert provider.is_connected() is False

    def test_reconnect_works(self, provider):
        provider.disconnect()
        ok = provider.connect()
        assert ok is True
        assert provider.is_connected() is True

    def test_get_account_info_returns_dict(self, provider):
        result = provider.get_account_info()
        assert isinstance(result, dict)

    def test_get_terminal_info_returns_dict(self, provider):
        result = provider.get_terminal_info()
        assert isinstance(result, dict)

    def test_get_symbol_info_returns_dict(self, provider):
        result = provider.get_symbol_info("EURUSD")
        assert isinstance(result, dict)

    def test_validate_symbol_returns_bool(self, provider):
        assert isinstance(provider.validate_symbol("EURUSD"), bool)
        assert isinstance(provider.validate_symbol("XXXXXX"), bool)

    def test_list_available_symbols_returns_list_of_strings(self, provider):
        result = provider.list_available_symbols()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_get_latest_quote_returns_quote_or_none(self, provider):
        result = provider.get_latest_quote("EURUSD")
        assert result is None or isinstance(result, Quote)

    def test_quote_has_required_fields(self, provider):
        quote = provider.get_latest_quote("EURUSD")
        if quote is not None:
            assert hasattr(quote, "symbol")
            assert hasattr(quote, "bid")
            assert hasattr(quote, "ask")
            assert hasattr(quote, "spread")
            assert hasattr(quote, "timestamp")
            assert hasattr(quote, "provider")

    def test_get_candles_returns_list(self, provider):
        result = provider.get_candles("EURUSD", "H1", 10)
        assert isinstance(result, list)

    def test_candle_has_required_fields(self, provider):
        candles = provider.get_candles("EURUSD", "H1", 1)
        if candles:
            c = candles[0]
            for attr in ("symbol", "timeframe", "timestamp",
                         "open", "high", "low", "close", "volume", "provider"):
                assert hasattr(c, attr), f"Candle missing field: {attr}"

    def test_candle_high_gte_low(self, provider):
        candles = provider.get_candles("EURUSD", "H1", 1)
        for c in candles:
            assert c.high >= c.low

    def test_candle_high_gte_open_and_close(self, provider):
        candles = provider.get_candles("EURUSD", "H1", 1)
        for c in candles:
            assert c.high >= c.open
            assert c.high >= c.close

    def test_candle_low_lte_open_and_close(self, provider):
        candles = provider.get_candles("EURUSD", "H1", 1)
        for c in candles:
            assert c.low <= c.open
            assert c.low <= c.close

    def test_get_candles_range_returns_list(self, provider):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        result = provider.get_candles_range("EURUSD", "H1", start, end)
        assert isinstance(result, list)
