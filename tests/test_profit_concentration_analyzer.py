"""Tests for ProfitConcentrationAnalyzer."""

import pytest
from datetime import datetime, timezone

from src.backtest.models import BacktestTrade, ClosingReason
from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime
from src.regime_filter.filter_profiles import AVOID_STRONG_BULL, NO_FILTER
from src.regime_filter.profit_concentration_analyzer import (
    FilterConcentrationResult,
    ProfitConcentrationAnalyzer,
)
from src.signals.models import SignalType

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(pnl=100.0) -> BacktestTrade:
    return BacktestTrade(
        symbol="SPY", entry_time=BASE, exit_time=BASE,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=110.0,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=10.0,
        holding_period=1, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


def _lt(pnl, regime) -> LabelledTrade:
    return LabelledTrade(
        trade=_trade(pnl),
        market_regime=regime,
        trend_regime=TrendRegime.STRONG_TREND,
        volatility_regime=VolatilityRegime.NORMAL_VOL,
        drawdown_env=DrawdownEnvironment.BULL_RECOVERY,
        macro_regime=MacroRegime.EXPANSION,
    )


@pytest.fixture
def analyzer() -> ProfitConcentrationAnalyzer:
    return ProfitConcentrationAnalyzer()


@pytest.fixture
def labelled_trades():
    return [
        _lt(100.0, MarketRegime.STRONG_BULL),
        _lt(80.0,  MarketRegime.STRONG_BULL),
        _lt(-50.0, MarketRegime.STRONG_BULL),
        _lt(150.0, MarketRegime.BULL),
        _lt(120.0, MarketRegime.BULL),
    ]


class TestProfitConcentrationAnalyzer:

    def test_returns_result(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, NO_FILTER)
        assert isinstance(result, FilterConcentrationResult)

    def test_regime_concentrations_populated(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, NO_FILTER)
        assert "STRONG_BULL" in result.regime_concentrations
        assert "BULL" in result.regime_concentrations

    def test_profit_contribution_sums_to_1(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, NO_FILTER)
        total = sum(
            c.profit_contribution
            for c in result.regime_concentrations.values()
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_no_filter_preserves_all_profits(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, NO_FILTER)
        assert result.profits_preserved_pct == pytest.approx(1.0, abs=0.01)

    def test_avoid_strong_bull_reduces_profits(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, AVOID_STRONG_BULL)
        # STRONG_BULL has some wins → removing it reduces profits
        assert result.profits_preserved_pct < 1.0

    def test_avoid_strong_bull_eliminates_some_losses(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, AVOID_STRONG_BULL)
        # STRONG_BULL has one loss → removing it eliminates those losses
        assert result.losses_eliminated_pct > 0.0

    def test_net_pnl_before_calculated(self, analyzer, labelled_trades):
        result = analyzer.analyze(labelled_trades, NO_FILTER)
        expected = sum(lt.trade.pnl for lt in labelled_trades)
        assert result.net_pnl_before == pytest.approx(expected)
