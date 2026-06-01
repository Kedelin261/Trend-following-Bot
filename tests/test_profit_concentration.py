"""Tests for ProfitConcentrationAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.edge_amplification.profit_concentration_analyzer import (
    ProfitConcentrationAnalyzer, ProfitConcentrationReport,
)
from src.edge_amplification.asset_contribution_analyzer import AssetContributionAnalyzer
from src.edge_amplification.regime_contribution_analyzer import RegimeContributionAnalyzer
from src.edge_amplification.volatility_contribution_analyzer import VolatilityContributionAnalyzer
from src.edge_amplification.trade_quality_analyzer import TradeQualityAnalyzer


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


def _build_analyzer(trades):
    asset_r = AssetContributionAnalyzer(trades).analyze()
    # Use mocks for regime/vol/quality to keep unit test fast
    regime_mock = MagicMock()
    regime_mock.regime = "EXPANSION"
    regime_mock.gross_profit = sum(t.pnl for t in trades if t.is_win)
    regime_mock.gross_loss = abs(sum(t.pnl for t in trades if t.is_loss))
    regime_mock.trades = len(trades)

    vol_mock = MagicMock()
    vol_mock.regime = "NORMAL_VOL"
    vol_mock.gross_profit = sum(t.pnl for t in trades if t.is_win)
    vol_mock.gross_loss = abs(sum(t.pnl for t in trades if t.is_loss))
    vol_mock.trades = len(trades)

    qual_mock = MagicMock()
    qual_mock.band = "80-89"
    qual_mock.gross_profit = sum(t.pnl for t in trades if t.is_win)
    qual_mock.gross_loss = abs(sum(t.pnl for t in trades if t.is_loss))
    qual_mock.trades = len(trades)

    return ProfitConcentrationAnalyzer(
        trades, asset_r, [regime_mock], [vol_mock], [qual_mock]
    )


def test_total_gross_profit_correct():
    trades = [_trade("SPY", 100), _trade("SPY", 50), _trade("SPY", -30)]
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    assert report.total_gross_profit == pytest.approx(150.0)


def test_total_winning_trades():
    trades = [_trade("SPY", 100), _trade("SPY", 50), _trade("SPY", -30)]
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    assert report.total_winning_trades == 2


def test_top_assets_sorted_by_profit():
    trades = (
        [_trade("QQQ", 200), _trade("QQQ", 100)]   # more profit
        + [_trade("SPY", 50), _trade("SPY", 30)]    # less profit
    )
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    if len(report.top_assets) >= 2:
        assert report.top_assets[0].gross_profit >= report.top_assets[1].gross_profit


def test_best_asset_identified():
    trades = (
        [_trade("QQQ", 300), _trade("QQQ", 200)]
        + [_trade("SPY", 50)]
    )
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    assert report.best_asset == "QQQ"


def test_top_regimes_populated():
    trades = [_trade("SPY", 100), _trade("SPY", 50)]
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    assert len(report.top_regimes) >= 1
    assert report.best_regime == "EXPANSION"


def test_top_volatility_populated():
    trades = [_trade("SPY", 100), _trade("SPY", 50)]
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    assert len(report.top_volatility) >= 1


def test_80_pct_threshold_calculated():
    trades = (
        [_trade("SPY", 900)]   # one asset has 90% of profits
        + [_trade("QQQ", 100)]
    )
    analyzer = _build_analyzer(trades)
    report = analyzer.analyze()
    # Should reach 80% with just top asset (50% of assets)
    assert report.asset_80_pct_threshold == pytest.approx(50.0)


def test_empty_trades():
    analyzer = ProfitConcentrationAnalyzer([], [], [], [], [])
    report = analyzer.analyze()
    assert report.total_gross_profit == 0.0
    assert report.total_winning_trades == 0
