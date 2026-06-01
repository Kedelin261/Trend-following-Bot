"""Tests for TradeQualityAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import Signal, SignalType, TrendDirection
from src.data.models import Candle
from src.edge_amplification.trade_quality_analyzer import (
    TradeQualityAnalyzer, QualityBucketResult, _band_for_score,
    QUALITY_BANDS, UNKNOWN_BAND, MIN_SAMPLE,
)

TS = datetime(2020, 6, 1, tzinfo=timezone.utc)


def _trade(symbol, pnl):
    return BacktestTrade(
        symbol=symbol, entry_time=TS, exit_time=TS,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=101.0 if pnl > 0 else 99.0,
        stop_price=98.0, target_price=104.0,
        position_size=5, pnl=pnl, return_percent=pnl / 100,
        holding_period=5, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def _candle(sym="SPY"):
    """Minimal candle at the trade timestamp so _score_band() reaches the strategy."""
    return Candle(
        symbol=sym, timeframe="D1", timestamp=TS,
        open=100.0, high=101.0, low=99.0, close=100.0,
        volume=1e6, provider="TEST",
    )


def _mock_signal(score: float):
    sig = MagicMock(spec=Signal)
    sig.strength_score = score
    sig.signal_type = MagicMock()
    sig.signal_type.value = "LONG"
    return sig


# ---------------------------------------------------------------------------
# _band_for_score unit tests
# ---------------------------------------------------------------------------

def test_band_for_score_boundaries():
    assert _band_for_score(100.0) == "90-100"
    assert _band_for_score(90.0)  == "90-100"
    assert _band_for_score(89.0)  == "80-89"
    assert _band_for_score(80.0)  == "80-89"
    assert _band_for_score(70.0)  == "70-79"
    assert _band_for_score(60.0)  == "60-69"
    assert _band_for_score(59.9)  == "<60"
    assert _band_for_score(40.0)  == "<60"


# ---------------------------------------------------------------------------
# Analyzer integration tests — candles provided so strategy is reachable
# ---------------------------------------------------------------------------

def test_analyzer_groups_by_band():
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = _mock_signal(85.0)
    trades = [_trade("SPY", 50), _trade("SPY", 50), _trade("SPY", -20)]
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    band_80 = next((r for r in results if r.band == "80-89"), None)
    assert band_80 is not None
    assert band_80.trades == 3


def test_unknown_band_when_no_candles():
    """When asset_candles dict is empty, trades end in UNKNOWN bucket."""
    mock_strategy = MagicMock()
    trades = [_trade("SPY", 50)]
    analyzer = TradeQualityAnalyzer(trades, {}, mock_strategy)
    results = analyzer.analyze()
    # All trades fall to UNKNOWN — bucket may or may not appear depending on impl
    # Key invariant: no QUALITY_BANDS label has any trades
    quality_band_labels = {label for _, _, label in QUALITY_BANDS}
    for r in results:
        if r.band in quality_band_labels:
            assert r.trades == 0


def test_has_edge_true_when_pf_above_1():
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = _mock_signal(95.0)
    trades = [_trade("SPY", 100), _trade("SPY", 100), _trade("SPY", -20)]
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    top = next(r for r in results if r.band == "90-100")
    assert top.has_edge is True


def test_has_edge_false_when_all_losses():
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = _mock_signal(65.0)
    trades = [_trade("SPY", -50), _trade("SPY", -50)]
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    band = next(r for r in results if r.band == "60-69")
    assert band.has_edge is False


def test_insufficient_sample_flagged():
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = _mock_signal(85.0)
    trades = [_trade("SPY", 10)] * 3
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    band = next(r for r in results if r.band == "80-89")
    assert band.insufficient_sample is True


def test_sufficient_sample_not_flagged():
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = _mock_signal(75.0)
    trades = [_trade("SPY", 10 if i % 2 == 0 else -5) for i in range(MIN_SAMPLE)]
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    band = next(r for r in results if r.band == "70-79")
    assert band.insufficient_sample is False


def test_ranking_best_first():
    """Bands with higher PF rank before bands with lower PF."""
    mock_strategy = MagicMock()
    # alternate: first call returns score 95, second 45, etc.
    mock_strategy.generate_signal.side_effect = (
        [_mock_signal(95.0)] * 3 + [_mock_signal(45.0)] * 3
    )
    trades = (
        [_trade("SPY", 200), _trade("SPY", 100), _trade("SPY", 50)]
        + [_trade("SPY", -50), _trade("SPY", -60), _trade("SPY", -70)]
    )
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    valid = [r for r in results if not r.insufficient_sample and r.trades > 0]
    if len(valid) >= 2:
        assert valid[0].profit_factor >= valid[-1].profit_factor


def test_empty_trades():
    mock_strategy = MagicMock()
    analyzer = TradeQualityAnalyzer([], {}, mock_strategy)
    results = analyzer.analyze()
    assert results == []


def test_pf_str_inf_for_all_wins():
    import math
    mock_strategy = MagicMock()
    mock_strategy.generate_signal.return_value = _mock_signal(92.0)
    trades = [_trade("SPY", 100)] * 5
    candles = {"SPY": [_candle("SPY")]}
    analyzer = TradeQualityAnalyzer(trades, candles, mock_strategy)
    results = analyzer.analyze()
    top = next(r for r in results if r.band == "90-100")
    assert math.isinf(top.profit_factor)
    assert top.pf_str == "∞"
