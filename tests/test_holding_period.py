"""Tests for HoldingPeriodAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timezone
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.edge_amplification.holding_period_analyzer import (
    HoldingPeriodAnalyzer, HOLDING_BANDS, _band_for_bars, MIN_SAMPLE,
)


def _trade(pnl, hp):
    ts = datetime(2020, 6, 1, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol="SPY", entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=101.0 if pnl > 0 else 99.0,
        stop_price=98.0, target_price=104.0,
        position_size=5, pnl=pnl, return_percent=pnl / 100,
        holding_period=hp, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def test_band_for_bars():
    assert _band_for_bars(0)  == "0-5"
    assert _band_for_bars(5)  == "0-5"
    assert _band_for_bars(6)  == "6-10"
    assert _band_for_bars(10) == "6-10"
    assert _band_for_bars(11) == "11-20"
    assert _band_for_bars(20) == "11-20"
    assert _band_for_bars(21) == "21+"
    assert _band_for_bars(100) == "21+"


def test_band_bucketing():
    trades = [
        _trade(50, 3),   # 0-5
        _trade(60, 8),   # 6-10
        _trade(70, 15),  # 11-20
        _trade(80, 30),  # 21+
    ]
    results = HoldingPeriodAnalyzer(trades).analyze()
    bands = {r.band for r in results}
    assert "0-5" in bands
    assert "6-10" in bands
    assert "11-20" in bands
    assert "21+" in bands


def test_contribution_pct_sums_to_100():
    trades = [
        _trade(100, 3), _trade(80, 8), _trade(60, 15), _trade(40, 25),
    ]
    results = HoldingPeriodAnalyzer(trades).analyze()
    total = sum(r.contribution_pct for r in results)
    assert total == pytest.approx(100.0, abs=1e-6)


def test_rank_assigned():
    trades = [
        _trade(100, 3), _trade(50, 8), _trade(-30, 15), _trade(-50, 25),
    ]
    results = HoldingPeriodAnalyzer(trades).analyze()
    ranks = sorted(r.rank for r in results)
    assert ranks == list(range(1, len(results) + 1))


def test_pf_best_in_rank_1():
    trades = (
        [_trade(200, 3), _trade(100, 3)]   # strong short-term
        + [_trade(-50, 25), _trade(-60, 25)]  # long-term losses
    )
    results = HoldingPeriodAnalyzer(trades).analyze()
    best = next(r for r in results if r.rank == 1)
    worst = max(results, key=lambda r: r.rank)
    assert best.profit_factor >= worst.profit_factor


def test_insufficient_sample():
    trades = [_trade(10, 3)] * 3
    results = HoldingPeriodAnalyzer(trades).analyze()
    short = next(r for r in results if r.band == "0-5")
    assert short.insufficient_sample is True


def test_sufficient_sample():
    trades = [_trade(10 if i % 2 == 0 else -5, 3) for i in range(MIN_SAMPLE)]
    results = HoldingPeriodAnalyzer(trades).analyze()
    short = next(r for r in results if r.band == "0-5")
    assert short.insufficient_sample is False


def test_empty_trades():
    results = HoldingPeriodAnalyzer([]).analyze()
    assert results == []


def test_net_pnl_correct():
    trades = [_trade(100, 5), _trade(-30, 5)]
    results = HoldingPeriodAnalyzer(trades).analyze()
    short = next(r for r in results if r.band == "0-5")
    assert short.net_pnl == pytest.approx(70.0)


def test_expectancy_positive_on_winners():
    trades = [_trade(50, 8), _trade(60, 8), _trade(-10, 8)]
    results = HoldingPeriodAnalyzer(trades).analyze()
    bucket = next(r for r in results if r.band == "6-10")
    assert bucket.expectancy > 0


def test_pf_inf_all_wins():
    import math
    trades = [_trade(50, 15)] * 5
    results = HoldingPeriodAnalyzer(trades).analyze()
    bucket = next(r for r in results if r.band == "11-20")
    assert math.isinf(bucket.profit_factor)
    assert bucket.pf_str == "∞"
