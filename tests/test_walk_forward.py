"""Tests for WalkForwardAnalyzer — data splitting logic."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardSplit
from src.data.models import Candle


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _candles(n: int, symbol: str = "SPY") -> List[Candle]:
    return [
        Candle(
            symbol=symbol, timeframe="D1",
            timestamp=BASE + timedelta(days=i),
            open=100.0, high=101.0, low=99.0, close=100.0,
            volume=1e8, provider="TEST",
        )
        for i in range(n)
    ]


@pytest.fixture
def analyzer() -> WalkForwardAnalyzer:
    return WalkForwardAnalyzer(train_pct=0.70, warmup=10)


class TestWalkForwardSplit:

    def test_train_length_70_percent(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        assert len(split.train_candles) == 70

    def test_test_includes_warmup_prefix(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        # Test should include 10 warmup bars from end of training
        assert len(split.test_candles) == 40  # 30 new + 10 warmup

    def test_test_count_is_30_percent(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        assert split.test_count == 30

    def test_train_count_correct(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        assert split.train_count == 70

    def test_warmup_count_correct(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        assert split.warmup_count == 10

    def test_custom_split_80_20(self):
        analyzer = WalkForwardAnalyzer(train_pct=0.80, warmup=5)
        candles = _candles(100)
        split = analyzer.split(candles)
        assert len(split.train_candles) == 80
        assert split.test_count == 20

    def test_warmup_exceeds_train_clamps_to_start(self):
        # warmup > training period → test starts at 0
        analyzer = WalkForwardAnalyzer(train_pct=0.70, warmup=200)
        candles = _candles(100)
        split = analyzer.split(candles)
        assert len(split.test_candles) == 100  # entire dataset (warmup clamped)

    def test_train_and_test_cover_full_dataset(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        assert split.train_count + split.test_count == len(candles)

    def test_train_candles_are_first_70_percent(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        # First candle of train matches first candle of full dataset
        assert split.train_candles[0].timestamp == candles[0].timestamp
        assert split.train_candles[-1].timestamp == candles[69].timestamp

    def test_test_candles_end_matches_dataset_end(self, analyzer):
        candles = _candles(100)
        split = analyzer.split(candles)
        assert split.test_candles[-1].timestamp == candles[-1].timestamp


class TestWalkForwardAnalyzerInit:

    def test_default_pct_is_70(self):
        a = WalkForwardAnalyzer()
        assert a.train_pct == 0.70

    def test_default_warmup_is_210(self):
        a = WalkForwardAnalyzer()
        assert a.warmup == 210

    def test_invalid_train_pct_raises(self):
        with pytest.raises(ValueError):
            WalkForwardAnalyzer(train_pct=0.0)

    def test_invalid_train_pct_too_high_raises(self):
        with pytest.raises(ValueError):
            WalkForwardAnalyzer(train_pct=1.0)

    def test_custom_pct_stored(self):
        a = WalkForwardAnalyzer(train_pct=0.60)
        assert a.train_pct == 0.60
