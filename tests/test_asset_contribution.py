"""Tests for AssetContributionAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timezone
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.edge_amplification.asset_contribution_analyzer import (
    AssetContributionAnalyzer, MIN_SAMPLE,
)


def _trade(symbol, pnl, hp=5):
    ts = datetime(2020, 1, 2, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol=symbol, entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=101.0 if pnl > 0 else 99.0,
        stop_price=98.0, target_price=104.0,
        position_size=10, pnl=pnl, return_percent=pnl / 100,
        holding_period=hp, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def test_single_asset_basic():
    trades = [_trade("SPY", 100), _trade("SPY", 50), _trade("SPY", -30)]
    results = AssetContributionAnalyzer(trades).analyze()
    assert len(results) == 1
    r = results[0]
    assert r.symbol == "SPY"
    assert r.trades == 3
    assert r.gross_profit == pytest.approx(150.0)
    assert r.gross_loss == pytest.approx(30.0)
    assert r.net_pnl == pytest.approx(120.0)
    assert r.contribution_pct == pytest.approx(100.0)


def test_multi_asset_ranking():
    trades = (
        [_trade("SPY", 200), _trade("SPY", 100)]        # strong PF
        + [_trade("IWM", -50), _trade("IWM", -60)]      # all losses
    )
    results = AssetContributionAnalyzer(trades).analyze()
    assert len(results) == 2
    assert results[0].symbol == "SPY"
    assert results[1].symbol == "IWM"


def test_rank_assigned():
    trades = [_trade("A", 50), _trade("B", 100), _trade("C", -10)]
    results = AssetContributionAnalyzer(trades).analyze()
    ranks = [r.rank for r in results]
    assert sorted(ranks) == list(range(1, len(results) + 1))


def test_insufficient_sample_flag():
    # Only 3 trades for SPY → below MIN_SAMPLE
    trades = [_trade("SPY", 10)] * 3
    results = AssetContributionAnalyzer(trades).analyze()
    assert results[0].insufficient_sample is True


def test_sufficient_sample_not_flagged():
    trades = [_trade("SPY", 10 if i % 2 == 0 else -5) for i in range(MIN_SAMPLE)]
    results = AssetContributionAnalyzer(trades).analyze()
    assert results[0].insufficient_sample is False


def test_all_wins_pf_inf():
    trades = [_trade("SPY", 50)] * 5
    results = AssetContributionAnalyzer(trades).analyze()
    import math
    assert math.isinf(results[0].profit_factor)


def test_contribution_pct_sums_to_100():
    trades = (
        [_trade("SPY", 100), _trade("SPY", -20)]
        + [_trade("QQQ", 80),  _trade("QQQ", -10)]
    )
    results = AssetContributionAnalyzer(trades).analyze()
    total = sum(r.contribution_pct for r in results)
    assert total == pytest.approx(100.0, abs=1e-6)


def test_empty_trades():
    results = AssetContributionAnalyzer([]).analyze()
    assert results == []


def test_win_rate_calculation():
    trades = [_trade("SPY", 50), _trade("SPY", 50), _trade("SPY", -20)]
    results = AssetContributionAnalyzer(trades).analyze()
    assert results[0].win_rate == pytest.approx(2 / 3)


def test_equity_by_asset_used_for_dd():
    trades = [_trade("SPY", 50)]
    equity = {"SPY": [10000, 9500, 9000, 9800]}  # ~10% DD
    results = AssetContributionAnalyzer(trades, equity).analyze()
    assert results[0].max_drawdown_pct > 0.0


def test_negative_contribution_pct():
    trades = [_trade("SPY", 200), _trade("IWM", -50)]
    results = AssetContributionAnalyzer(trades).analyze()
    losing_asset = next(r for r in results if r.symbol == "IWM")
    assert losing_asset.contribution_pct < 0


def test_six_assets_all_ranked():
    syms = ["SPY", "QQQ", "IWM", "DIA", "VTI", "XLV"]
    trades = [_trade(s, 50 if i % 3 != 0 else -20, hp=5)
              for i, s in enumerate(syms)]
    results = AssetContributionAnalyzer(trades).analyze()
    assert len(results) == len(syms)
    assert {r.rank for r in results} == set(range(1, len(syms) + 1))
