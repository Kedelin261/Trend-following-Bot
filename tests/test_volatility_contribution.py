"""Tests for VolatilityContributionAnalyzer — Phase 5.3."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from src.backtest.models import BacktestTrade, ClosingReason
from src.signals.models import SignalType
from src.data.models import Candle
from src.regime.volatility_regime_detector import VolatilityRegime, VolatilityRegimeDetector
from src.edge_amplification.volatility_contribution_analyzer import (
    VolatilityContributionAnalyzer, MIN_SAMPLE,
)


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


def test_unknown_when_no_candles():
    trades = [_trade("SPY", 50)]
    analyzer = VolatilityContributionAnalyzer(trades, {})
    results = analyzer.analyze()
    unknown = next(r for r in results if r.regime == VolatilityRegime.UNKNOWN.value)
    assert unknown.trades == 1


def test_detector_called():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    mock_det.classify.return_value = VolatilityRegime.NORMAL_VOL
    trades = [_trade("SPY", 100), _trade("SPY", 50)]
    candles = {"SPY": []}
    analyzer = VolatilityContributionAnalyzer(trades, candles, detector=mock_det)
    results = analyzer.analyze()
    normal = next(r for r in results if r.regime == VolatilityRegime.NORMAL_VOL.value)
    assert normal.trades == 2


def test_profit_pct_and_loss_pct_positive():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    mock_det.classify.side_effect = [
        VolatilityRegime.NORMAL_VOL,
        VolatilityRegime.HIGH_VOL,
    ]
    trades = [_trade("SPY", 100), _trade("SPY", -50)]
    candles = {"SPY": []}
    analyzer = VolatilityContributionAnalyzer(trades, candles, detector=mock_det)
    results = analyzer.analyze()
    for r in results:
        assert r.profit_pct >= 0
        assert r.loss_pct >= 0


def test_ranking_best_first():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    mock_det.classify.side_effect = (
        [VolatilityRegime.NORMAL_VOL] * 3 + [VolatilityRegime.EXTREME_VOL] * 3
    )
    trades = (
        [_trade("SPY", 200), _trade("SPY", 100), _trade("SPY", 80)]
        + [_trade("SPY", -50), _trade("SPY", -60), _trade("SPY", -70)]
    )
    candles = {"SPY": []}
    analyzer = VolatilityContributionAnalyzer(trades, candles, detector=mock_det)
    results = analyzer.analyze()
    occupied = [r for r in results if r.trades > 0]
    # best PF should be rank 1
    assert occupied[0].rank == 1
    assert occupied[0].profit_factor > occupied[-1].profit_factor


def test_insufficient_sample_flagged():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    mock_det.classify.return_value = VolatilityRegime.HIGH_VOL
    trades = [_trade("SPY", 10)] * 3
    analyzer = VolatilityContributionAnalyzer(trades, {}, detector=mock_det)
    results = analyzer.analyze()
    high = next(r for r in results if r.regime == VolatilityRegime.HIGH_VOL.value)
    assert high.insufficient_sample is True


def test_sufficient_sample_not_flagged():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    mock_det.classify.return_value = VolatilityRegime.NORMAL_VOL
    trades = [_trade("SPY", 10 if i % 2 == 0 else -5) for i in range(MIN_SAMPLE)]
    analyzer = VolatilityContributionAnalyzer(trades, {}, detector=mock_det)
    results = analyzer.analyze()
    normal = next(r for r in results if r.regime == VolatilityRegime.NORMAL_VOL.value)
    assert normal.insufficient_sample is False


def test_empty_trades():
    analyzer = VolatilityContributionAnalyzer([], {})
    results = analyzer.analyze()
    assert results == []


def test_pf_inf_string():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    mock_det.classify.return_value = VolatilityRegime.LOW_VOL
    trades = [_trade("SPY", 100)] * 5
    analyzer = VolatilityContributionAnalyzer(trades, {}, detector=mock_det)
    results = analyzer.analyze()
    low = next(r for r in results if r.regime == VolatilityRegime.LOW_VOL.value)
    assert low.pf_str == "∞"


def test_four_vol_regimes_present_in_output():
    mock_det = MagicMock(spec=VolatilityRegimeDetector)
    regimes = [
        VolatilityRegime.LOW_VOL, VolatilityRegime.NORMAL_VOL,
        VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL,
    ]
    mock_det.classify.side_effect = regimes
    trades = [_trade("SPY", 50), _trade("SPY", 50), _trade("SPY", -20), _trade("SPY", -20)]
    analyzer = VolatilityContributionAnalyzer(trades, {}, detector=mock_det)
    results = analyzer.analyze()
    assert len(results) == 4
