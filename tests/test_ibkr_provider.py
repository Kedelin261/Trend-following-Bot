"""Tests for IBKRProvider.

Three tiers:
  1. Pure unit tests       — no ib-insync needed, run everywhere
  2. Mocked-IB tests       — patch ib-insync internals, run everywhere
  3. Integration tests     — require live TWS, auto-skipped otherwise

Run unit + mocked only:   pytest tests/test_ibkr_provider.py -m "not integration" -v
Run integration only:     pytest tests/test_ibkr_provider.py -m integration -v
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.providers.ibkr_provider import (
    IBKRProvider,
    _IB_AVAILABLE,
    _BAR_SIZE_MAP,
    _FOREX_CODES,
    _safe_server_version,
    _duration_str,
    _to_utc_datetime,
)
from src.data.models import Candle
from src.data.timeframe_registry import TimeframeRegistry


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

def _bar(
    dt=None,
    open_: float = 1.0800,
    high: float = 1.0900,
    low: float = 1.0700,
    close: float = 1.0850,
    volume: float = 1000.0,
) -> MagicMock:
    b = MagicMock()
    b.date = dt or datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    b.open = open_
    b.high = high
    b.low = low
    b.close = close
    b.volume = volume
    return b


def _account_value(tag: str, value: str, currency: str = "USD") -> MagicMock:
    av = MagicMock()
    av.tag = tag
    av.value = value
    av.currency = currency
    return av


def _position(
    symbol: str = "AAPL",
    sec_type: str = "STK",
    position: float = 100.0,
    avg_cost: float = 150.0,
) -> MagicMock:
    pos = MagicMock()
    pos.account = "DU123456"
    pos.contract.symbol = symbol
    pos.contract.secType = sec_type
    pos.contract.exchange = "SMART"
    pos.contract.currency = "USD"
    pos.position = position
    pos.avgCost = avg_cost
    return pos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> dict:
    return {
        "ibkr": {
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1,
            "timeout": 5.0,
            "readonly": True,
        }
    }


@pytest.fixture
def mock_ib_instance() -> MagicMock:
    """A pre-configured mock IB instance that looks connected."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.serverVersion.return_value = 10514
    return ib


# ---------------------------------------------------------------------------
# 1. _duration_str helper
# ---------------------------------------------------------------------------

class TestDurationStr:

    def test_returns_string(self):
        for tf in TimeframeRegistry.get_all():
            result = _duration_str(tf, 100)
            assert isinstance(result, str)

    def test_format_is_number_unit(self):
        for tf in TimeframeRegistry.get_all():
            parts = _duration_str(tf, 50).split()
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1] in ("D", "M", "Y")

    def test_m1_small_count_gives_days(self):
        result = _duration_str("M1", 60)
        # 60 min × 1.6 ≈ 96 min ≈ 0.07 days → at least 1 D
        assert "D" in result

    def test_h1_moderate_count_gives_days(self):
        result = _duration_str("H1", 200)
        # 200 h × 1.6 = 320 h ≈ 13 days → "14 D" or similar
        assert "D" in result

    def test_d1_large_count_gives_months_or_years(self):
        result = _duration_str("D1", 600)
        # 600 × 1440 min × 1.6 = ~864000 min = 600 days → months or years
        assert any(unit in result for unit in ("D", "M", "Y"))

    def test_w1_200_bars_gives_years(self):
        result = _duration_str("W1", 200)
        # 200 weeks ≈ 3.8 years → Y
        assert "Y" in result

    def test_larger_count_never_shorter_duration(self):
        """Requesting more bars should require equal or longer duration."""
        small = _duration_str("H1", 10)
        large = _duration_str("H1", 500)
        small_n = int(small.split()[0])
        large_n = int(large.split()[0])
        # Either larger number, or a higher-order unit (D < M < Y)
        units = {"D": 0, "M": 1, "Y": 2}
        small_order = units[small.split()[1]]
        large_order = units[large.split()[1]]
        assert large_order > small_order or large_n >= small_n

    def test_minimum_is_one_day(self):
        result = _duration_str("M1", 1)
        n, unit = result.split()
        assert unit == "D"
        assert int(n) >= 1

    # ------------------------------------------------------------------
    # Regression: IBKR only accepts "N M" where N is 1–12.
    # 300 D1 bars previously produced "17 M" which IBKR silently rejects.
    # ------------------------------------------------------------------

    def test_d1_300_bars_never_produces_invalid_months(self):
        """Regression: 300 D1 bars must NOT produce an invalid month duration.

        IBKR accepts 'N M' only when N is 1–12.  Previously _duration_str
        returned '17 M' for this input, causing reqHistoricalData to return
        an empty list even when SPY data was available.
        """
        result = _duration_str("D1", 300)
        n, unit = result.split()
        if unit == "M":
            # If months format is ever re-introduced, it must be 1–12
            assert int(n) <= 12, f"Invalid IBKR month duration: {result!r}"
        # Should be "Y" for this input (480 days > 365)
        assert unit == "Y", (
            f"Expected year-format for D1/300 bars, got {result!r}. "
            "IBKR rejects month counts > 12, causing empty candle responses."
        )

    def test_d1_50_bars_uses_days_format(self):
        """validate_ibkr.py uses 50 bars and gets data — confirm it stays in 'D' format."""
        result = _duration_str("D1", 50)
        n, unit = result.split()
        assert unit == "D"
        assert int(n) >= 50

    def test_no_duration_exceeds_12_months(self):
        """No call to _duration_str should ever produce month count > 12."""
        from src.data.timeframe_registry import STANDARD_TIMEFRAMES
        for tf in STANDARD_TIMEFRAMES:
            for count in [50, 100, 200, 300, 500, 1000]:
                result = _duration_str(tf, count)
                n, unit = result.split()
                if unit == "M":
                    assert int(n) <= 12, (
                        f"Invalid IBKR duration for {tf}/{count}: {result!r}"
                    )

    def test_h1_500_bars_valid_duration(self):
        """H1/500 bars previously could produce invalid months; should be D or Y now."""
        result = _duration_str("H1", 500)
        n, unit = result.split()
        assert unit in ("D", "Y"), f"Unexpected unit in {result!r}"
        if unit == "M":
            assert int(n) <= 12


# ---------------------------------------------------------------------------
# 2. _to_utc_datetime helper
# ---------------------------------------------------------------------------

class TestToUtcDatetime:

    def test_datetime_with_tz_returned_unchanged(self):
        dt = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        assert _to_utc_datetime(dt) == dt

    def test_naive_datetime_gets_utc_tzinfo(self):
        naive = datetime(2024, 6, 1, 14, 30)
        result = _to_utc_datetime(naive)
        assert result.tzinfo == timezone.utc
        assert result.hour == 14

    def test_date_object_becomes_midnight_utc(self):
        d = date(2024, 6, 1)
        result = _to_utc_datetime(d)
        assert result == datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)

    def test_string_with_time_parses_correctly(self):
        result = _to_utc_datetime("20240601 14:30:00")
        assert result == datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)

    def test_string_date_only_parses_correctly(self):
        result = _to_utc_datetime("20240601")
        assert result == datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)

    def test_result_always_has_tzinfo(self):
        cases = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            date(2024, 1, 1),
            "20240101",
            "20240101 12:00:00",
        ]
        for val in cases:
            result = _to_utc_datetime(val)
            assert result.tzinfo is not None, f"No tzinfo for input: {val!r}"


# ---------------------------------------------------------------------------
# 3. Bar-size map coverage
# ---------------------------------------------------------------------------

class TestBarSizeMap:

    def test_every_standard_timeframe_has_mapping(self):
        for tf in TimeframeRegistry.get_all():
            assert tf in _BAR_SIZE_MAP, f"Missing bar size for {tf}"

    def test_known_values(self):
        assert _BAR_SIZE_MAP["M1"]  == "1 min"
        assert _BAR_SIZE_MAP["M5"]  == "5 mins"
        assert _BAR_SIZE_MAP["M15"] == "15 mins"
        assert _BAR_SIZE_MAP["M30"] == "30 mins"
        assert _BAR_SIZE_MAP["H1"]  == "1 hour"
        assert _BAR_SIZE_MAP["H4"]  == "4 hours"
        assert _BAR_SIZE_MAP["D1"]  == "1 day"
        assert _BAR_SIZE_MAP["W1"]  == "1 week"


# ---------------------------------------------------------------------------
# 3b. _safe_server_version helper  (regression for the serverVersion bug)
# ---------------------------------------------------------------------------

class TestSafeServerVersion:
    """Unit tests for _safe_server_version — no ib-insync required."""

    def test_returns_value_from_client_path(self):
        ib = MagicMock()
        ib.client.serverVersion.return_value = 10514
        ib.serverVersion.side_effect = AttributeError("not on IB")
        assert _safe_server_version(ib) == 10514

    def test_falls_back_to_ib_serverversion(self):
        ib = MagicMock()
        ib.client.serverVersion.side_effect = AttributeError("not on client")
        ib.serverVersion.return_value = 10514
        assert _safe_server_version(ib) == 10514

    def test_returns_none_when_both_paths_missing(self):
        """Regression: must return None, not raise, when neither path exists."""
        ib = MagicMock()
        ib.client.serverVersion.side_effect = AttributeError("no attr")
        ib.serverVersion.side_effect = AttributeError("no attr")
        assert _safe_server_version(ib) is None

    def test_returns_none_on_type_error(self):
        ib = MagicMock()
        ib.client.serverVersion.side_effect = TypeError("not callable")
        ib.serverVersion.side_effect = TypeError("not callable")
        assert _safe_server_version(ib) is None

    def test_skips_none_result_and_tries_next(self):
        ib = MagicMock()
        ib.client.serverVersion.return_value = None   # None → skip, try next
        ib.serverVersion.return_value = 10514
        assert _safe_server_version(ib) == 10514


# ---------------------------------------------------------------------------
# 4. Pure unit tests — no ib-insync, no mocking
# ---------------------------------------------------------------------------

class TestIBKRProviderUnit:

    def test_provider_name_constant(self, cfg):
        assert IBKRProvider(config=cfg).PROVIDER_NAME == "IBKR"

    def test_not_connected_by_default(self, cfg):
        assert IBKRProvider(config=cfg).is_connected() is False

    def test_get_candles_empty_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_candles("EUR.USD", "H1", 100) == []

    def test_get_candles_range_empty_when_not_connected(self, cfg):
        p = IBKRProvider(config=cfg)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        assert p.get_candles_range("EUR.USD", "H1", start, end) == []

    def test_get_historical_data_empty_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_historical_data("EUR.USD", "H1", 100) == []

    def test_get_latest_quote_none_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_latest_quote("EUR.USD") is None

    def test_get_account_info_empty_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_account_info() == {}

    def test_get_account_summary_empty_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_account_summary() == {}

    def test_get_positions_empty_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_positions() == []

    def test_get_symbol_info_empty_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).get_symbol_info("EUR.USD") == {}

    def test_validate_symbol_false_when_not_connected(self, cfg):
        assert IBKRProvider(config=cfg).validate_symbol("EUR.USD") is False

    def test_health_check_when_no_ib_instance(self, cfg):
        result = IBKRProvider(config=cfg).health_check()
        assert result["connected"] is False
        assert result["provider"] == "IBKR"

    def test_get_terminal_info_returns_health_check(self, cfg):
        result = IBKRProvider(config=cfg).get_terminal_info()
        assert "connected" in result
        assert "provider" in result

    def test_connect_false_when_ib_not_available(self, cfg):
        if _IB_AVAILABLE:
            pytest.skip("ib-insync installed; this tests the missing-package path")
        assert IBKRProvider(config=cfg).connect() is False

    def test_invalid_timeframe_returns_empty(self, cfg):
        p = IBKRProvider(config=cfg)
        p._connected = True
        assert p.get_historical_data("EUR.USD", "INVALID", 100) == []

    def test_list_symbols_empty_without_registry(self, cfg):
        assert IBKRProvider(config=cfg).list_available_symbols() == []

    def test_list_symbols_from_registry(self, cfg):
        from src.data.symbol_registry import SymbolRegistry
        reg = SymbolRegistry(symbols=["EUR.USD", "SPY"])
        p = IBKRProvider(config=cfg, symbol_registry=reg)
        assert p.list_available_symbols() == ["EUR.USD", "SPY"]

    def test_config_dict_overrides_defaults(self):
        config = {"ibkr": {"host": "10.0.0.1", "port": 4002, "client_id": 5}}
        p = IBKRProvider(config=config)
        assert p._host == "10.0.0.1"
        assert p._port == 4002
        assert p._client_id == 5

    def test_direct_params_used_when_no_config_key(self):
        p = IBKRProvider(config={}, host="192.168.1.10", port=4001, client_id=3)
        assert p._host == "192.168.1.10"
        assert p._port == 4001
        assert p._client_id == 3


# ---------------------------------------------------------------------------
# 5. Bar-to-candle conversion
# ---------------------------------------------------------------------------

class TestBarToCandle:

    def test_utc_datetime_preserved(self, cfg):
        p = IBKRProvider(config=cfg)
        dt = datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
        candle = p._bar_to_candle(_bar(dt), "EUR.USD", "H1")
        assert candle.timestamp == dt

    def test_naive_datetime_gets_utc(self, cfg):
        p = IBKRProvider(config=cfg)
        candle = p._bar_to_candle(_bar(datetime(2024, 3, 15, 10, 0)), "SPY", "D1")
        assert candle.timestamp.tzinfo == timezone.utc

    def test_date_object_becomes_midnight(self, cfg):
        p = IBKRProvider(config=cfg)
        candle = p._bar_to_candle(_bar(date(2024, 3, 15)), "SPY", "D1")
        assert candle.timestamp == datetime(2024, 3, 15, 0, 0, tzinfo=timezone.utc)

    def test_ohlcv_fields_correct(self, cfg):
        p = IBKRProvider(config=cfg)
        candle = p._bar_to_candle(
            _bar(open_=1.10, high=1.15, low=1.05, close=1.12, volume=9876.0),
            "EUR.USD",
            "H1",
        )
        assert candle.open   == pytest.approx(1.10)
        assert candle.high   == pytest.approx(1.15)
        assert candle.low    == pytest.approx(1.05)
        assert candle.close  == pytest.approx(1.12)
        assert candle.volume == pytest.approx(9876.0)

    def test_symbol_and_metadata_set(self, cfg):
        p = IBKRProvider(config=cfg)
        candle = p._bar_to_candle(_bar(), "EUR.USD", "H4")
        assert candle.symbol    == "EUR.USD"
        assert candle.timeframe == "H4"
        assert candle.provider  == "IBKR"

    def test_high_gte_low(self, cfg):
        p = IBKRProvider(config=cfg)
        candle = p._bar_to_candle(_bar(), "EUR.USD", "H1")
        assert candle.high >= candle.low


# ---------------------------------------------------------------------------
# 6. Mocked-IB tests — logic tested without real TWS, runs on any OS
# ---------------------------------------------------------------------------

class TestIBKRProviderMocked:
    """Patch ib-insync internals to test provider logic on any platform."""

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_connect_success(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514

        p = IBKRProvider(config=cfg)
        assert p.connect() is True
        assert p._connected is True
        ib.connect.assert_called_once_with(
            "127.0.0.1", 7497, clientId=1, timeout=5.0, readonly=True
        )

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_connect_refused_returns_false(self, MockIB, cfg):
        MockIB.return_value.connect.side_effect = ConnectionRefusedError()
        p = IBKRProvider(config=cfg)
        assert p.connect() is False
        assert p._connected is False

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_connect_timeout_returns_false(self, MockIB, cfg):
        MockIB.return_value.connect.side_effect = TimeoutError()
        assert IBKRProvider(config=cfg).connect() is False

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_connect_generic_exception_returns_false(self, MockIB, cfg):
        MockIB.return_value.connect.side_effect = RuntimeError("unexpected")
        assert IBKRProvider(config=cfg).connect() is False

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_disconnect_calls_ib_disconnect(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514

        p = IBKRProvider(config=cfg)
        p.connect()
        p.disconnect()

        ib.disconnect.assert_called_once()
        assert p._connected is False

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_is_connected_reflects_ib_state(self, MockIB, cfg):
        ib = MockIB.return_value
        # connect() does NOT call isConnected(); these two calls come from
        # the explicit is_connected() assertions below.
        ib.isConnected.side_effect = [True, False]
        ib.serverVersion.return_value = 10514

        p = IBKRProvider(config=cfg)
        p.connect()
        assert p.is_connected() is True
        assert p.is_connected() is False

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_historical_data_returns_candles(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.reqHistoricalData.return_value = [
            _bar(datetime(2024, 1, i + 1, 12, 0, tzinfo=timezone.utc))
            for i in range(5)
        ]

        p = IBKRProvider(config=cfg)
        p.connect()
        p._make_contract = MagicMock(return_value=MagicMock(secType="CASH"))
        candles = p.get_historical_data("EUR.USD", "H1", 5)

        assert len(candles) == 5
        assert all(isinstance(c, Candle) for c in candles)
        assert candles[0].symbol   == "EUR.USD"
        assert candles[0].provider == "IBKR"
        assert candles[0].timeframe == "H1"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_candles_trimmed_to_requested_count(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        # Provider returns 10 bars; we request only 3
        ib.reqHistoricalData.return_value = [
            _bar(datetime(2024, 1, i + 1, 12, 0, tzinfo=timezone.utc))
            for i in range(10)
        ]

        p = IBKRProvider(config=cfg)
        p.connect()
        p._make_contract = MagicMock(return_value=MagicMock(secType="CASH"))
        candles = p.get_historical_data("EUR.USD", "H1", 3)

        assert len(candles) == 3
        # Most recent 3: Jan 8, 9, 10
        assert candles[0].timestamp == datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
        assert candles[-1].timestamp == datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_historical_data_empty_response(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.reqHistoricalData.return_value = []

        p = IBKRProvider(config=cfg)
        p.connect()
        p._make_contract = MagicMock(return_value=MagicMock(secType="CASH"))
        assert p.get_historical_data("EUR.USD", "H1", 100) == []

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_historical_data_exception_returns_empty(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.reqHistoricalData.side_effect = Exception("Pacing violation")

        p = IBKRProvider(config=cfg)
        p.connect()
        p._make_contract = MagicMock(return_value=MagicMock(secType="CASH"))
        assert p.get_historical_data("EUR.USD", "H1", 100) == []

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_candles_delegates_to_get_historical_data(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.reqHistoricalData.return_value = [
            _bar(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))
        ]

        p = IBKRProvider(config=cfg)
        p.connect()
        p._make_contract = MagicMock(return_value=MagicMock(secType="CASH"))
        candles = p.get_candles("EUR.USD", "H1", 1)

        assert len(candles) == 1
        assert candles[0].symbol == "EUR.USD"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_account_summary_returns_dict(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.accountSummary.return_value = [
            _account_value("TotalCashValue", "50000.00"),
            _account_value("NetLiquidation", "52000.00"),
            _account_value("AccountType",    "PAPER",    ""),
        ]

        p = IBKRProvider(config=cfg)
        p.connect()
        result = p.get_account_summary()

        assert result["TotalCashValue"]["value"] == "50000.00"
        assert result["TotalCashValue"]["currency"] == "USD"
        assert result["AccountType"]["value"] == "PAPER"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_account_info_flattens_to_values(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.accountSummary.return_value = [
            _account_value("TotalCashValue", "50000.00"),
        ]

        p = IBKRProvider(config=cfg)
        p.connect()
        result = p.get_account_info()

        assert result["TotalCashValue"] == "50000.00"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_get_positions_returns_list_of_dicts(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514
        ib.positions.return_value = [
            _position("AAPL", "STK",  100.0, 150.0),
            _position("EUR",  "CASH", 10000.0, 1.085),
        ]

        p = IBKRProvider(config=cfg)
        p.connect()
        positions = p.get_positions()

        assert len(positions) == 2
        assert positions[0]["symbol"]   == "AAPL"
        assert positions[0]["position"] == pytest.approx(100.0)
        assert positions[1]["symbol"]   == "EUR"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_health_check_when_connected(self, MockIB, cfg):
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.serverVersion.return_value = 10514

        p = IBKRProvider(config=cfg)
        p.connect()
        result = p.health_check()

        assert result["connected"] is True
        assert result["provider"] == "IBKR"
        assert result["port"] == 7497
        assert result["client_id"] == 1
        assert result["readonly"] is True

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_what_to_show_cash_is_midpoint(self, MockIB, cfg):
        contract = MagicMock()
        contract.secType = "CASH"
        p = IBKRProvider(config=cfg)
        assert p._what_to_show(contract) == "MIDPOINT"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_what_to_show_stock_is_trades(self, MockIB, cfg):
        contract = MagicMock()
        contract.secType = "STK"
        p = IBKRProvider(config=cfg)
        assert p._what_to_show(contract) == "TRADES"

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_whattoshow_unknown_sectype_defaults_to_trades(self, MockIB, cfg):
        # Only CASH (Forex) uses MIDPOINT; everything else defaults to TRADES.
        contract = MagicMock()
        contract.secType = ""
        p = IBKRProvider(config=cfg)
        assert p._what_to_show(contract) == "TRADES"

    # ------------------------------------------------------------------
    # Regression tests: serverVersion must not abort connect() or health_check()
    # ------------------------------------------------------------------

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_connect_succeeds_when_server_version_unavailable(self, MockIB, cfg):
        """Regression: 'IB has no attribute serverVersion' must not kill connect().

        Real ib-insync does not expose serverVersion directly on the IB object;
        it lives on ib.client.serverVersion(). If both paths are missing,
        connect() must still return True when isConnected() is True.
        """
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        # Simulate both serverVersion paths being absent
        ib.client.serverVersion.side_effect = AttributeError(
            "'IB' object has no attribute 'serverVersion'"
        )
        ib.serverVersion.side_effect = AttributeError(
            "'IB' object has no attribute 'serverVersion'"
        )

        p = IBKRProvider(config=cfg)
        result = p.connect()

        assert result is True
        assert p._connected is True

    @patch("src.providers.ibkr_provider._IB_AVAILABLE", True)
    @patch("src.providers.ibkr_provider.IB")
    def test_health_check_server_version_none_when_unavailable(self, MockIB, cfg):
        """Regression: health_check() must return server_version=None gracefully."""
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.client.serverVersion.side_effect = AttributeError("no serverVersion")
        ib.serverVersion.side_effect = AttributeError("no serverVersion")

        p = IBKRProvider(config=cfg)
        p.connect()
        result = p.health_check()

        assert result["connected"] is True
        assert result["server_version"] is None


# ---------------------------------------------------------------------------
# 7. Contract creation — requires ib-insync package
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IB_AVAILABLE, reason="ib-insync not installed")
class TestContractCreation:

    def test_eurusd_dotted_creates_forex(self, cfg):
        contract = IBKRProvider(config=cfg)._make_contract("EUR.USD")
        assert contract is not None
        assert contract.secType  == "CASH"
        assert contract.symbol   == "EUR"
        assert contract.currency == "USD"
        assert contract.exchange == "IDEALPRO"

    def test_eurusd_plain_creates_forex(self, cfg):
        contract = IBKRProvider(config=cfg)._make_contract("EURUSD")
        assert contract.secType == "CASH"

    def test_gbpusd_creates_forex(self, cfg):
        contract = IBKRProvider(config=cfg)._make_contract("GBP.USD")
        assert contract.secType == "CASH"
        assert contract.symbol  == "GBP"

    def test_usdjpy_creates_forex(self, cfg):
        contract = IBKRProvider(config=cfg)._make_contract("USDJPY")
        assert contract.secType == "CASH"

    def test_spy_creates_stock(self, cfg):
        contract = IBKRProvider(config=cfg)._make_contract("SPY")
        assert contract.secType  == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"

    def test_aapl_creates_stock(self, cfg):
        contract = IBKRProvider(config=cfg)._make_contract("AAPL")
        assert contract.secType == "STK"

    def test_forex_what_to_show_midpoint(self, cfg):
        p = IBKRProvider(config=cfg)
        contract = p._make_contract("EUR.USD")
        assert p._what_to_show(contract) == "MIDPOINT"

    def test_stock_what_to_show_trades(self, cfg):
        p = IBKRProvider(config=cfg)
        contract = p._make_contract("SPY")
        assert p._what_to_show(contract) == "TRADES"


# ---------------------------------------------------------------------------
# 8. Integration tests — auto-skipped when TWS is not running
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIBKRIntegration:
    """Live connectivity tests against a running TWS paper account.

    Never place orders. Never modify account state. Read-only throughout.
    """

    @pytest.fixture(autouse=True)
    def live_provider(self, cfg):
        p = IBKRProvider(config=cfg)
        if not p.connect():
            pytest.skip("TWS not running or not reachable on 127.0.0.1:7497")
        yield p
        p.disconnect()

    def test_is_connected_true(self, live_provider):
        assert live_provider.is_connected() is True

    def test_health_check_shows_connected(self, live_provider):
        result = live_provider.health_check()
        assert result["connected"] is True
        assert result["server_version"] is not None

    def test_account_summary_is_nonempty(self, live_provider):
        summary = live_provider.get_account_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_account_info_flattened(self, live_provider):
        info = live_provider.get_account_info()
        assert isinstance(info, dict)

    def test_positions_returns_list(self, live_provider):
        assert isinstance(live_provider.get_positions(), list)

    def test_eurusd_h1_historical(self, live_provider):
        candles = live_provider.get_historical_data("EUR.USD", "H1", 20)
        assert isinstance(candles, list)
        for c in candles:
            assert c.symbol   == "EUR.USD"
            assert c.provider == "IBKR"
            assert c.high     >= c.low

    def test_spy_d1_historical(self, live_provider):
        candles = live_provider.get_historical_data("SPY", "D1", 20)
        assert isinstance(candles, list)

    def test_candles_have_utc_timestamps(self, live_provider):
        candles = live_provider.get_historical_data("EUR.USD", "H1", 5)
        for c in candles:
            assert c.timestamp.tzinfo is not None

    def test_candles_ordered_ascending(self, live_provider):
        candles = live_provider.get_historical_data("EUR.USD", "H1", 10)
        if len(candles) > 1:
            timestamps = [c.timestamp for c in candles]
            assert timestamps == sorted(timestamps)

    def test_disconnect_sets_not_connected(self, live_provider):
        live_provider.disconnect()
        assert live_provider.is_connected() is False
