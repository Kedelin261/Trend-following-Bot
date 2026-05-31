"""Tests for SignalOverlapAnalyzer — date-based overlap detection."""

import pytest
from datetime import datetime, timedelta, timezone

from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.timeframe.signal_overlap_analyzer import (
    MAX_ACCEPTABLE_OVERLAP,
    OverlapResult,
    SignalOverlapAnalyzer,
)


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(day_offset=0, symbol="SPY") -> BacktestTrade:
    entry = BASE + timedelta(days=day_offset)
    return BacktestTrade(
        symbol=symbol, entry_time=entry, exit_time=entry + timedelta(days=1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=110.0,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=100.0, return_percent=10.0,
        holding_period=1, win_loss="WIN", reason_closed=ClosingReason.TARGET,
    )


@pytest.fixture
def analyzer() -> SignalOverlapAnalyzer:
    return SignalOverlapAnalyzer(max_overlap=0.50)


class TestCalculateOverlap:

    def test_no_overlap_when_different_days(self, analyzer):
        primary   = [_trade(0), _trade(2), _trade(4)]
        secondary = [_trade(1), _trade(3), _trade(5)]
        result = analyzer.calculate_overlap(primary, secondary, "D1", "H4")
        assert result.overlapping_trades == 0
        assert result.overlap_pct == pytest.approx(0.0)

    def test_full_overlap_when_same_days(self, analyzer):
        primary   = [_trade(0), _trade(1), _trade(2)]
        secondary = [_trade(0), _trade(1), _trade(2)]
        result = analyzer.calculate_overlap(primary, secondary, "D1", "H4")
        assert result.overlapping_trades == 3
        assert result.overlap_pct == pytest.approx(1.0)

    def test_partial_overlap(self, analyzer):
        primary   = [_trade(0), _trade(2)]
        secondary = [_trade(0), _trade(1), _trade(2), _trade(3)]
        result = analyzer.calculate_overlap(primary, secondary, "D1", "H4")
        assert result.overlapping_trades == 2
        assert result.overlap_pct == pytest.approx(0.5)

    def test_acceptable_when_below_threshold(self, analyzer):
        primary   = [_trade(0)]
        secondary = [_trade(0), _trade(1), _trade(2)]  # 33% overlap
        result = analyzer.calculate_overlap(primary, secondary, "D1", "H4")
        assert result.acceptable is True

    def test_not_acceptable_when_above_threshold(self, analyzer):
        primary   = [_trade(0), _trade(1), _trade(2)]
        secondary = [_trade(0), _trade(1), _trade(2), _trade(3)]  # 75% overlap
        result = analyzer.calculate_overlap(primary, secondary, "D1", "H4")
        assert result.acceptable is False

    def test_empty_secondary_returns_zero_overlap(self, analyzer):
        primary = [_trade(0)]
        result  = analyzer.calculate_overlap(primary, [], "D1", "H4")
        assert result.overlap_pct == 0.0
        assert result.acceptable is True

    def test_unique_secondary_count(self, analyzer):
        primary   = [_trade(0), _trade(1)]
        secondary = [_trade(0), _trade(2), _trade(4)]
        result = analyzer.calculate_overlap(primary, secondary, "D1", "H4")
        assert result.unique_secondary == 2  # trade on day 2 and 4 are unique


class TestCombinedOverlap:

    def test_single_tf_no_overlap(self, analyzer):
        trades_by_tf = {"D1": [_trade(0), _trade(5)]}
        result = analyzer.calculate_combined_overlap(trades_by_tf)
        assert result.average_overlap_pct == 0.0
        assert result.acceptable is True

    def test_combined_unique_trades_no_overlap(self, analyzer):
        trades_by_tf = {
            "D1": [_trade(0), _trade(2), _trade(4)],
            "H4": [_trade(1), _trade(3), _trade(5)],
        }
        result = analyzer.calculate_combined_overlap(trades_by_tf)
        # D1 has 3 trades on days 0,2,4; H4 has 3 trades on days 1,3,5 — no overlap
        assert result.average_overlap_pct == pytest.approx(0.0)
        assert result.estimated_unique_trades == 6

    def test_combined_with_full_overlap(self, analyzer):
        trades_by_tf = {
            "D1": [_trade(0), _trade(1), _trade(2)],
            "H4": [_trade(0), _trade(1), _trade(2)],
        }
        result = analyzer.calculate_combined_overlap(trades_by_tf)
        assert result.average_overlap_pct == pytest.approx(1.0)

    def test_empty_input(self, analyzer):
        result = analyzer.calculate_combined_overlap({})
        assert result.total_raw_trades == 0

    def test_max_acceptable_overlap_constant(self):
        assert MAX_ACCEPTABLE_OVERLAP == 0.50
