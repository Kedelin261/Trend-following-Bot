"""Tests for CandleStore — SQLite persistence, deduplication, and queries."""

import pytest
from datetime import datetime, timezone, timedelta

from src.data.candle_store import CandleStore
from src.data.models import Candle


def _candle(
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    dt: datetime = None,
    close: float = 1.09,
    provider: str = "TEST",
) -> Candle:
    if dt is None:
        dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=dt,
        open=1.08,
        high=1.10,
        low=1.07,
        close=close,
        volume=1000.0,
        provider=provider,
    )


@pytest.fixture
def store(tmp_path) -> CandleStore:
    return CandleStore(str(tmp_path / "test.db"))


class TestCandleStoreSave:

    def test_save_single_candle(self, store):
        saved, dupes = store.save_candles([_candle()])
        assert saved == 1
        assert dupes == 0

    def test_save_returns_zero_for_empty_list(self, store):
        saved, dupes = store.save_candles([])
        assert saved == 0
        assert dupes == 0

    def test_duplicate_is_skipped(self, store):
        c = _candle()
        store.save_candles([c])
        saved, dupes = store.save_candles([c])
        assert saved == 0
        assert dupes == 1

    def test_duplicate_detection_uses_symbol_timeframe_timestamp_provider(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        c1 = _candle(dt=base, provider="MT5")
        c2 = _candle(dt=base, provider="POLYGON")  # different provider = not a dupe
        store.save_candles([c1])
        saved, dupes = store.save_candles([c2])
        assert saved == 1
        assert dupes == 0

    def test_different_timestamps_both_saved(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        candles = [_candle(dt=base + timedelta(hours=i)) for i in range(3)]
        saved, dupes = store.save_candles(candles)
        assert saved == 3
        assert dupes == 0

    def test_mixed_batch_partial_dupes(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        c1 = _candle(dt=base)
        store.save_candles([c1])
        c2 = _candle(dt=base + timedelta(hours=1))
        saved, dupes = store.save_candles([c1, c2])
        assert saved == 1
        assert dupes == 1

    def test_resolved_symbol_stored(self, store):
        c = Candle(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            open=1.08, high=1.10, low=1.07, close=1.09,
            volume=1000.0,
            provider="MT5",
            resolved_symbol="EURUSD.r",
        )
        store.save_candles([c])
        loaded = store.load_candles("EURUSD", "H1")
        assert loaded[0].resolved_symbol == "EURUSD.r"


class TestCandleStoreLoad:

    def test_load_returns_candles(self, store):
        store.save_candles([_candle()])
        loaded = store.load_candles("EURUSD", "H1")
        assert len(loaded) == 1
        assert loaded[0].symbol == "EURUSD"
        assert loaded[0].timeframe == "H1"

    def test_load_close_precision(self, store):
        store.save_candles([_candle(close=1.12345)])
        loaded = store.load_candles("EURUSD", "H1")
        assert loaded[0].close == pytest.approx(1.12345)

    def test_load_returns_empty_for_unknown_symbol(self, store):
        store.save_candles([_candle(symbol="EURUSD")])
        loaded = store.load_candles("GBPUSD", "H1")
        assert loaded == []

    def test_load_ordered_by_timestamp(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        candles = [_candle(dt=base + timedelta(hours=i)) for i in range(5)]
        store.save_candles(candles)
        loaded = store.load_candles("EURUSD", "H1")
        timestamps = [c.timestamp for c in loaded]
        assert timestamps == sorted(timestamps)

    def test_load_with_start_date_filter(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(dt=base + timedelta(hours=i)) for i in range(5)])
        loaded = store.load_candles(
            "EURUSD", "H1", start_date=base + timedelta(hours=2)
        )
        assert len(loaded) == 3
        assert loaded[0].timestamp == base + timedelta(hours=2)

    def test_load_with_end_date_filter(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(dt=base + timedelta(hours=i)) for i in range(5)])
        loaded = store.load_candles(
            "EURUSD", "H1", end_date=base + timedelta(hours=2)
        )
        assert len(loaded) == 3

    def test_load_with_date_range(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(dt=base + timedelta(hours=i)) for i in range(10)])
        loaded = store.load_candles(
            "EURUSD",
            "H1",
            start_date=base + timedelta(hours=2),
            end_date=base + timedelta(hours=5),
        )
        assert len(loaded) == 4  # hours 2, 3, 4, 5 inclusive


class TestCandleStoreQueries:

    def test_count_all(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(symbol="EURUSD", dt=base)])
        store.save_candles([_candle(symbol="GBPUSD", dt=base + timedelta(hours=1))])
        assert store.count_candles() == 2

    def test_count_filtered_by_symbol(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(symbol="EURUSD", dt=base)])
        store.save_candles([_candle(symbol="GBPUSD", dt=base + timedelta(hours=1))])
        assert store.count_candles(symbol="EURUSD") == 1
        assert store.count_candles(symbol="GBPUSD") == 1

    def test_count_filtered_by_timeframe(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(timeframe="H1", dt=base)])
        store.save_candles([_candle(timeframe="H4", dt=base + timedelta(hours=1))])
        assert store.count_candles(timeframe="H1") == 1
        assert store.count_candles(timeframe="H4") == 1

    def test_get_latest_timestamp(self, store):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        store.save_candles([_candle(dt=base + timedelta(hours=i)) for i in range(3)])
        latest = store.get_latest_timestamp("EURUSD", "H1", "TEST")
        assert latest is not None
        assert latest == base + timedelta(hours=2)

    def test_get_latest_timestamp_none_when_empty(self, store):
        result = store.get_latest_timestamp("EURUSD", "H1", "MT5")
        assert result is None

    def test_count_zero_on_empty_store(self, store):
        assert store.count_candles() == 0
