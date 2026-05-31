"""Shared fixtures for all test modules."""

import pytest
from datetime import datetime, timezone, timedelta

from src.data.candle_store import CandleStore
from src.data.models import Candle, Quote
from src.data.symbol_registry import SymbolRegistry


@pytest.fixture
def sample_candle() -> Candle:
    return Candle(
        symbol="EURUSD",
        timeframe="H1",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        open=1.0900,
        high=1.0950,
        low=1.0850,
        close=1.0920,
        volume=1000.0,
        provider="TEST",
    )


@pytest.fixture
def candle_sequence() -> list:
    """Five consecutive hourly candles for EURUSD."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        Candle(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=base + timedelta(hours=i),
            open=1.09,
            high=1.10,
            low=1.08,
            close=1.091 + i * 0.001,
            volume=float(1000 + i * 100),
            provider="TEST",
        )
        for i in range(5)
    ]


@pytest.fixture
def tmp_store(tmp_path) -> CandleStore:
    return CandleStore(str(tmp_path / "test.db"))


@pytest.fixture
def symbol_registry_with_aliases() -> SymbolRegistry:
    return SymbolRegistry(
        symbols=["EURUSD", "BTCUSD"],
        aliases={
            "EURUSD": ["EURUSD", "EURUSD.r", "EURUSDm"],
            "BTCUSD": ["BTCUSD", "BTCUSD.r", "BTCUSDm"],
        },
    )
