"""MT5Provider tests.

Unit tests run on any platform (MT5 package optional — handled gracefully).
Integration tests require a running MT5 terminal and are auto-skipped otherwise.

Run unit tests only:      pytest tests/test_mt5_provider.py -m "not integration"
Run integration tests:    pytest tests/test_mt5_provider.py -m integration
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data.symbol_registry import SymbolRegistry
from src.data.timeframe_registry import TimeframeRegistry
from src.providers.mt5_provider import MT5Provider, _MT5_AVAILABLE, _TF_MAP


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mt5_config() -> dict:
    return {
        "mt5": {
            "login_env": "MT5_LOGIN",
            "password_env": "MT5_PASSWORD",
            "server_env": "MT5_SERVER",
            "retry_count": 1,
            "retry_delay_seconds": 0,
            "allow_initialize_without_credentials": True,
        }
    }


@pytest.fixture
def symbol_reg() -> SymbolRegistry:
    return SymbolRegistry(
        symbols=["EURUSD", "BTCUSD"],
        aliases={
            "EURUSD": ["EURUSD", "EURUSD.r"],
            "BTCUSD": ["BTCUSD", "BTCUSD.r"],
        },
    )


# ---------------------------------------------------------------------------
# Unit tests — run on any platform, no terminal required
# ---------------------------------------------------------------------------

class TestMT5ProviderUnit:

    def test_is_connected_false_before_connect(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        assert provider.is_connected() is False

    def test_get_candles_empty_when_not_connected(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        result = provider.get_candles("EURUSD", "H1", 100)
        assert result == []

    def test_get_candles_range_empty_when_not_connected(self, mt5_config, symbol_reg):
        from datetime import datetime, timezone
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        result = provider.get_candles_range(
            "EURUSD", "H1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert result == []

    def test_get_latest_quote_none_when_not_connected(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        assert provider.get_latest_quote("EURUSD") is None

    def test_get_account_info_empty_when_not_connected(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        assert provider.get_account_info() == {}

    def test_get_terminal_info_empty_when_not_connected(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        assert provider.get_terminal_info() == {}

    def test_connect_returns_false_when_mt5_unavailable(self, mt5_config, symbol_reg):
        if _MT5_AVAILABLE:
            pytest.skip("MT5 is installed; this test targets missing-package scenario")
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        assert provider.connect() is False

    def test_invalid_timeframe_returns_empty_list(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        provider._connected = True  # bypass connection guard
        result = provider.get_candles("EURUSD", "3H_INVALID", 100)
        assert result == []

    def test_provider_name_is_mt5(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        assert provider.PROVIDER_NAME == "MT5"


# ---------------------------------------------------------------------------
# Timeframe map tests — require MT5 package to be installed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MT5_AVAILABLE, reason="MetaTrader5 package not installed")
class TestMT5TimeframeMap:

    def test_all_standard_timeframes_have_mt5_mapping(self):
        for tf in TimeframeRegistry.get_all():
            assert tf in _TF_MAP, f"Missing MT5 mapping for timeframe: {tf}"

    def test_all_map_entries_are_valid_standard_timeframes(self):
        for tf in _TF_MAP:
            assert TimeframeRegistry.is_valid(tf), f"Unknown entry in _TF_MAP: {tf}"

    def test_map_values_are_integers(self):
        for tf, val in _TF_MAP.items():
            assert isinstance(val, int), f"MT5 constant for {tf} is not int: {val}"


# ---------------------------------------------------------------------------
# Mock-based unit tests — require MT5 package installed (for patching)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MT5_AVAILABLE, reason="MetaTrader5 package not installed")
class TestMT5ProviderWithMocks:

    def test_connect_returns_false_on_initialize_failure(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        with patch("src.providers.mt5_provider.mt5") as mock_mt5:
            mock_mt5.initialize.return_value = False
            mock_mt5.last_error.return_value = (-10004, "terminal not found")
            result = provider.connect()
        assert result is False

    def test_connect_returns_true_on_success(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        with patch("src.providers.mt5_provider.mt5") as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.symbols_get.return_value = []
            result = provider.connect()
        assert result is True

    def test_rates_to_candles_empty_on_none_rates(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        result = provider._rates_to_candles(None, "EURUSD", "EURUSD", "H1")
        assert result == []

    def test_rates_to_candles_empty_on_empty_array(self, mt5_config, symbol_reg):
        provider = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        result = provider._rates_to_candles([], "EURUSD", "EURUSD", "H1")
        assert result == []


# ---------------------------------------------------------------------------
# Integration tests — require a running MT5 terminal
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not _MT5_AVAILABLE, reason="MetaTrader5 package not installed")
class TestMT5Integration:
    """Full integration tests against a live MT5 terminal.

    These are automatically skipped when the terminal is not running.
    They never place trades or modify any account state.
    """

    @pytest.fixture(autouse=True)
    def connected_provider(self, mt5_config, symbol_reg):
        p = MT5Provider(config=mt5_config, symbol_registry=symbol_reg)
        if not p.connect():
            pytest.skip("MT5 terminal not running or not connected")
        yield p
        p.disconnect()

    def test_is_connected(self, connected_provider):
        assert connected_provider.is_connected() is True

    def test_account_info_has_login(self, connected_provider):
        info = connected_provider.get_account_info()
        assert isinstance(info, dict)
        assert "login" in info

    def test_terminal_info_has_connected_flag(self, connected_provider):
        info = connected_provider.get_terminal_info()
        assert isinstance(info, dict)
        assert "connected" in info

    def test_list_available_symbols_nonempty(self, connected_provider):
        symbols = connected_provider.list_available_symbols()
        assert len(symbols) > 0

    def test_get_candles_returns_list(self, connected_provider):
        candles = connected_provider.get_candles("EURUSD", "H1", 10)
        assert isinstance(candles, list)

    def test_candles_have_correct_symbol(self, connected_provider):
        candles = connected_provider.get_candles("EURUSD", "H1", 5)
        for c in candles:
            assert c.symbol == "EURUSD"
            assert c.provider == "MT5"
            assert c.timeframe == "H1"

    def test_disconnect_sets_not_connected(self, connected_provider):
        connected_provider.disconnect()
        assert connected_provider.is_connected() is False
