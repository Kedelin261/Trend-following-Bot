"""Tests for LossConcentrationAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.edge_amplification.loss_concentration_analyzer import (
    LossConcentrationAnalyzer, LossConcentrationReport,
)
from src.edge_amplification.asset_contribution_analyzer import AssetContributionAnalyzer


def _trade(symbol, pnl):
    ts = datetime(2020, 6, 1, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol=symbol, entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=101.0 if pnl > 0 else 99.0,
        stop_price=98.0, target_price=104.0,
        position_size=5, pnl=pnl, return_percent=pnl / 100,
        holding_period=5, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def _build_loss_analyzer(trades):
    asset_r = AssetContributionAnalyzer(trades).analyze()

    regime_mock = MagicMock()
    regime_mock.regime = "CONTRACTION"
    regime_mock.gross_profit = sum(t.pnl for t in trades if t.is_win)
    regime_mock.gross_loss = abs(sum(t.pnl for t in trades if t.is_loss))
    regime_mock.trades = len(trades)

    vol_mock = MagicMock()
    vol_mock.regime = "HIGH_VOL"
    vol_mock.gross_profit = sum(t.pnl for t in trades if t.is_win)
    vol_mock.gross_loss = abs(sum(t.pnl for t in trades if t.is_loss))
    vol_mock.trades = len(trades)

    qual_mock = MagicMock()
    qual_mock.band = "<60"
    qual_mock.gross_profit = sum(t.pnl for t in trades if t.is_win)
    qual_mock.gross_loss = abs(sum(t.pnl for t in trades if t.is_loss))
    qual_mock.trades = len(trades)

    return LossConcentrationAnalyzer(
        trades, asset_r, [regime_mock], [vol_mock], [qual_mock]
    )


def test_total_gross_loss():
    trades = [_trade("SPY", 100), _trade("SPY", -50), _trade("SPY", -30)]
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    assert report.total_gross_loss == pytest.approx(80.0)


def test_total_losing_trades():
    trades = [_trade("SPY", 100), _trade("SPY", -50), _trade("SPY", -30)]
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    assert report.total_losing_trades == 2


def test_worst_assets_sorted_by_loss():
    trades = (
        [_trade("IWM", -200), _trade("IWM", -100)]  # more loss
        + [_trade("SPY", -50)]                       # less loss
    )
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    if len(report.worst_assets) >= 2:
        assert report.worst_assets[0].gross_loss >= report.worst_assets[1].gross_loss


def test_worst_asset_identified():
    trades = (
        [_trade("IWM", -300), _trade("IWM", -200)]
        + [_trade("SPY", -50)]
    )
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    assert report.worst_asset == "IWM"


def test_worst_regime_populated():
    trades = [_trade("SPY", -50), _trade("SPY", 30)]
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    assert len(report.worst_regimes) >= 1
    assert report.worst_regime == "CONTRACTION"


def test_worst_volatility_populated():
    trades = [_trade("SPY", -50), _trade("SPY", 30)]
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    assert len(report.worst_volatility) >= 1


def test_pct_of_total_sums_to_100_for_assets():
    trades = [_trade("SPY", -100), _trade("QQQ", -200), _trade("IWM", -50)]
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    total_pct = sum(e.pct_of_total for e in report.worst_assets)
    assert total_pct == pytest.approx(100.0, abs=1e-5)


def test_empty_trades():
    analyzer = LossConcentrationAnalyzer([], [], [], [], [])
    report = analyzer.analyze()
    assert report.total_gross_loss == 0.0
    assert report.total_losing_trades == 0


def test_all_wins_zero_loss():
    trades = [_trade("SPY", 100), _trade("SPY", 50)]
    analyzer = _build_loss_analyzer(trades)
    report = analyzer.analyze()
    assert report.total_gross_loss == 0.0
    assert report.worst_assets == []
