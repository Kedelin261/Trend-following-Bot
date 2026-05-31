"""Tests for RegimePerformanceAnalyzer — per-regime metrics and concentration."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.regime_performance_analyzer import (
    LabelledTrade,
    RegimePerformance,
    RegimePerformanceAnalyzer,
)
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime
from src.signals.models import SignalType


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(pnl: float, i: int = 0) -> BacktestTrade:
    entry = BASE + timedelta(days=i)
    return BacktestTrade(
        symbol="SPY", entry_time=entry, exit_time=entry + timedelta(days=1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=100.0 + pnl / 10,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=pnl / 10,
        holding_period=1, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def _lt(pnl: float, regime: MarketRegime, i: int = 0) -> LabelledTrade:
    return LabelledTrade(
        trade=_trade(pnl, i),
        market_regime=regime,
        trend_regime=TrendRegime.STRONG_TREND,
        volatility_regime=VolatilityRegime.NORMAL_VOL,
        drawdown_env=DrawdownEnvironment.BULL_RECOVERY,
        macro_regime=MacroRegime.EXPANSION,
    )


@pytest.fixture
def analyzer() -> RegimePerformanceAnalyzer:
    return RegimePerformanceAnalyzer()


@pytest.fixture
def labelled_trades() -> List[LabelledTrade]:
    """3 STRONG_BULL wins, 2 BEAR losses."""
    return [
        _lt(100.0, MarketRegime.STRONG_BULL, 0),
        _lt(150.0, MarketRegime.STRONG_BULL, 1),
        _lt(80.0,  MarketRegime.STRONG_BULL, 2),
        _lt(-60.0, MarketRegime.BEAR,        3),
        _lt(-40.0, MarketRegime.BEAR,        4),
    ]


class TestRegimePerformanceAnalyzer:

    def test_returns_dict_keyed_by_regime_string(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert isinstance(result, dict)
        assert "STRONG_BULL" in result
        assert "BEAR" in result

    def test_strong_bull_trade_count(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert result["STRONG_BULL"].trade_count == 3

    def test_bear_trade_count(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert result["BEAR"].trade_count == 2

    def test_strong_bull_is_net_positive(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert result["STRONG_BULL"].is_net_positive is True

    def test_bear_is_net_negative(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert result["BEAR"].is_net_positive is False

    def test_profit_concentration_sums_to_1(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        total_conc = sum(p.profit_concentration for p in result.values())
        assert total_conc == pytest.approx(1.0, abs=0.01)

    def test_loss_concentration_sums_to_1(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        total_conc = sum(p.loss_concentration for p in result.values())
        assert total_conc == pytest.approx(1.0, abs=0.01)

    def test_strong_bull_profit_concentration_near_1(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert result["STRONG_BULL"].profit_concentration == pytest.approx(1.0, abs=0.01)

    def test_bear_loss_concentration_near_1(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        assert result["BEAR"].loss_concentration == pytest.approx(1.0, abs=0.01)

    def test_best_regime_returns_highest_expectancy(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        best = analyzer.best_regime(result, min_trades=1)
        assert best == "STRONG_BULL"

    def test_worst_regime_returns_lowest_expectancy(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, "market_regime")
        worst = analyzer.worst_regime(result, min_trades=1)
        assert worst == "BEAR"

    def test_analyze_all_returns_five_dimensions(self, analyzer, labelled_trades):
        all_results = analyzer.analyze_all(labelled_trades)
        assert set(all_results.keys()) == {
            "market_regime", "trend_regime", "volatility_regime",
            "drawdown_env", "macro_regime",
        }

    def test_invalid_dimension_raises(self, analyzer, labelled_trades):
        with pytest.raises(ValueError):
            analyzer.analyze(labelled_trades, "nonexistent_dimension")

    def test_empty_trades_returns_empty_dict(self, analyzer):
        result = analyzer.analyze([], "market_regime")
        assert result == {}
