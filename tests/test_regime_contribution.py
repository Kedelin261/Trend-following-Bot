"""Tests for RegimeContributionAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.data.models import Candle
from src.regime.macro_regime_detector import MacroRegime, MacroRegimeDetector
from src.edge_amplification.regime_contribution_analyzer import (
    RegimeContributionAnalyzer, MIN_SAMPLE,
)


def _trade(symbol, pnl, entry_dt=None):
    ts = entry_dt or datetime(2020, 6, 1, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol=symbol, entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=101.0 if pnl > 0 else 99.0,
        stop_price=98.0, target_price=104.0,
        position_size=5, pnl=pnl, return_percent=pnl / 100,
        holding_period=5, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def _candle(sym, ts, close=400.0):
    return Candle(
        symbol=sym, timeframe="D1", timestamp=ts,
        open=close * 0.99, high=close * 1.01,
        low=close * 0.98, close=close, volume=1e6,
        provider="TEST",
    )


def test_buckets_created():
    trades = [_trade("SPY", 100)]
    candles = {"SPY": [
        _candle("SPY", datetime(2020, 6, 1, tzinfo=timezone.utc) - timedelta(days=i))
        for i in range(250, 0, -1)
    ]}
    analyzer = RegimeContributionAnalyzer(trades, candles)
    results = analyzer.analyze()
    assert len(results) >= 1


def test_unknown_regime_for_empty_candles():
    trades = [_trade("SPY", 50)]
    analyzer = RegimeContributionAnalyzer(trades, {})
    results = analyzer.analyze()
    unknown_buckets = [r for r in results if r.regime == MacroRegime.UNKNOWN.value]
    assert len(unknown_buckets) > 0
    assert unknown_buckets[0].trades == 1


def test_detector_called_per_trade():
    mock_detector = MagicMock(spec=MacroRegimeDetector)
    mock_detector.classify.return_value = MacroRegime.EXPANSION
    trades = [_trade("SPY", 100), _trade("SPY", 50)]
    candles = {"SPY": [_candle("SPY", datetime(2020, 6, 1, tzinfo=timezone.utc))]}
    analyzer = RegimeContributionAnalyzer(trades, candles, detector=mock_detector)
    results = analyzer.analyze()
    expansion_bucket = next(r for r in results if r.regime == MacroRegime.EXPANSION.value)
    assert expansion_bucket.trades == 2


def test_profit_pct_sums_100():
    mock_detector = MagicMock(spec=MacroRegimeDetector)
    mock_detector.classify.side_effect = [
        MacroRegime.EXPANSION,
        MacroRegime.CONTRACTION,
    ]
    trades = [_trade("SPY", 100), _trade("SPY", 80)]
    candles = {"SPY": [_candle("SPY", datetime(2020, 6, 1, tzinfo=timezone.utc))]}
    analyzer = RegimeContributionAnalyzer(trades, candles, detector=mock_detector)
    results = analyzer.analyze()
    total_pp = sum(r.profit_pct for r in results)
    assert total_pp == pytest.approx(100.0, abs=1.0)


def test_insufficient_sample_flag():
    mock_detector = MagicMock(spec=MacroRegimeDetector)
    mock_detector.classify.return_value = MacroRegime.EXPANSION
    trades = [_trade("SPY", 10)] * 3
    candles = {"SPY": [_candle("SPY", datetime(2020, 6, 1, tzinfo=timezone.utc))]}
    analyzer = RegimeContributionAnalyzer(trades, candles, detector=mock_detector)
    results = analyzer.analyze()
    expansion = next(r for r in results if r.regime == MacroRegime.EXPANSION.value)
    assert expansion.insufficient_sample is True


def test_rank_order():
    mock_detector = MagicMock(spec=MacroRegimeDetector)
    mock_detector.classify.side_effect = (
        [MacroRegime.EXPANSION] * 3 + [MacroRegime.CONTRACTION] * 3
    )
    trades = (
        [_trade("SPY", 200), _trade("SPY", 100), _trade("SPY", 80)]    # expansion wins
        + [_trade("SPY", -50), _trade("SPY", -60), _trade("SPY", -70)] # contraction losses
    )
    candles = {"SPY": [_candle("SPY", datetime(2020, 6, 1, tzinfo=timezone.utc))]}
    analyzer = RegimeContributionAnalyzer(trades, candles, detector=mock_detector)
    results = analyzer.analyze()
    non_empty = [r for r in results if r.trades > 0]
    assert non_empty[0].profit_factor > non_empty[-1].profit_factor


def test_pf_str_inf():
    mock_detector = MagicMock(spec=MacroRegimeDetector)
    mock_detector.classify.return_value = MacroRegime.EXPANSION
    trades = [_trade("SPY", 50)] * 5
    candles = {"SPY": [_candle("SPY", datetime(2020, 6, 1, tzinfo=timezone.utc))]}
    analyzer = RegimeContributionAnalyzer(trades, candles, detector=mock_detector)
    results = analyzer.analyze()
    expansion = next(r for r in results if r.regime == MacroRegime.EXPANSION.value)
    assert expansion.pf_str == "∞"


def test_empty_trades():
    analyzer = RegimeContributionAnalyzer([], {})
    results = analyzer.analyze()
    # All buckets empty → empty list
    assert results == []
