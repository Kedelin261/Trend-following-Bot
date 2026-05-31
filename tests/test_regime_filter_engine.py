"""Tests for RegimeFilterEngine — ALLOW/REJECT logic."""

import pytest
from datetime import datetime, timezone

from src.backtest.models import BacktestTrade, ClosingReason
from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime
from src.regime_filter.filter_profiles import (
    AVOID_BEAR,
    AVOID_EXTREME_VOL,
    AVOID_STRONG_BULL,
    BULL_ONLY,
    NO_FILTER,
    FilterProfile,
)
from src.regime_filter.regime_filter_engine import FilterDecision, RegimeFilterEngine
from src.signals.models import SignalType

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(pnl=100.0) -> BacktestTrade:
    return BacktestTrade(
        symbol="SPY", entry_time=BASE, exit_time=BASE,
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=110.0,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=10.0,
        holding_period=1, win_loss="WIN",
        reason_closed=ClosingReason.TARGET,
    )


def _lt(
    market=MarketRegime.BULL,
    vol=VolatilityRegime.NORMAL_VOL,
    macro=MacroRegime.EXPANSION,
    dd=DrawdownEnvironment.BULL_RECOVERY,
    trend=TrendRegime.MODERATE_TREND,
) -> LabelledTrade:
    return LabelledTrade(
        trade=_trade(),
        market_regime=market,
        trend_regime=trend,
        volatility_regime=vol,
        drawdown_env=dd,
        macro_regime=macro,
    )


@pytest.fixture
def engine() -> RegimeFilterEngine:
    return RegimeFilterEngine()


class TestEvaluate:

    def test_no_filter_always_allows(self, engine):
        for mr in MarketRegime:
            lt = _lt(market=mr)
            assert engine.evaluate(lt, NO_FILTER) == FilterDecision.ALLOW

    def test_avoid_strong_bull_blocks_strong_bull(self, engine):
        lt = _lt(market=MarketRegime.STRONG_BULL)
        assert engine.evaluate(lt, AVOID_STRONG_BULL) == FilterDecision.REJECT

    def test_avoid_strong_bull_allows_bull(self, engine):
        lt = _lt(market=MarketRegime.BULL)
        assert engine.evaluate(lt, AVOID_STRONG_BULL) == FilterDecision.ALLOW

    def test_avoid_extreme_vol_blocks_extreme(self, engine):
        lt = _lt(vol=VolatilityRegime.EXTREME_VOL)
        assert engine.evaluate(lt, AVOID_EXTREME_VOL) == FilterDecision.REJECT

    def test_avoid_extreme_vol_allows_normal(self, engine):
        lt = _lt(vol=VolatilityRegime.NORMAL_VOL)
        assert engine.evaluate(lt, AVOID_EXTREME_VOL) == FilterDecision.ALLOW

    def test_bull_only_whitelist_blocks_sideways(self, engine):
        lt = _lt(market=MarketRegime.SIDEWAYS)
        assert engine.evaluate(lt, BULL_ONLY) == FilterDecision.REJECT

    def test_bull_only_whitelist_blocks_strong_bull(self, engine):
        # BULL_ONLY allows only BULL (not STRONG_BULL)
        lt = _lt(market=MarketRegime.STRONG_BULL)
        assert engine.evaluate(lt, BULL_ONLY) == FilterDecision.REJECT

    def test_bull_only_allows_bull(self, engine):
        lt = _lt(market=MarketRegime.BULL)
        assert engine.evaluate(lt, BULL_ONLY) == FilterDecision.ALLOW

    def test_block_macro_crisis(self, engine):
        config = FilterProfile(
            name="test", description="test",
            blocked_macro_regimes=frozenset({MacroRegime.CRISIS}),
        )
        lt = _lt(macro=MacroRegime.CRISIS)
        assert engine.evaluate(lt, config) == FilterDecision.REJECT

    def test_block_crash_environment(self, engine):
        config = FilterProfile(
            name="test", description="test",
            blocked_drawdown_envs=frozenset({DrawdownEnvironment.CRASH}),
        )
        lt = _lt(dd=DrawdownEnvironment.CRASH)
        assert engine.evaluate(lt, config) == FilterDecision.REJECT

    def test_block_weak_trend(self, engine):
        config = FilterProfile(
            name="test", description="test",
            blocked_trend_regimes=frozenset({TrendRegime.WEAK_TREND}),
        )
        lt = _lt(trend=TrendRegime.WEAK_TREND)
        assert engine.evaluate(lt, config) == FilterDecision.REJECT

    def test_block_takes_priority_over_allow(self, engine):
        # Block STRONG_BULL AND allow STRONG_BULL — block should win
        config = FilterProfile(
            name="conflict", description="conflict",
            blocked_market_regimes=frozenset({MarketRegime.STRONG_BULL}),
            allowed_market_regimes=frozenset({MarketRegime.STRONG_BULL}),
        )
        lt = _lt(market=MarketRegime.STRONG_BULL)
        assert engine.evaluate(lt, config) == FilterDecision.REJECT


class TestApplyToTrades:

    def test_returns_only_allowed_trades(self, engine):
        trades = [
            _lt(market=MarketRegime.STRONG_BULL),
            _lt(market=MarketRegime.BULL),
            _lt(market=MarketRegime.BEAR),
        ]
        passing = engine.apply_to_trades(trades, AVOID_STRONG_BULL)
        assert len(passing) == 2
        for lt in passing:
            assert lt.market_regime != MarketRegime.STRONG_BULL

    def test_no_filter_returns_all(self, engine):
        trades = [_lt(market=r) for r in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]]
        passing = engine.apply_to_trades(trades, NO_FILTER)
        assert len(passing) == 3

    def test_empty_input_returns_empty(self, engine):
        assert engine.apply_to_trades([], AVOID_STRONG_BULL) == []


class TestCountByDecision:

    def test_count_allow_reject(self, engine):
        trades = [
            _lt(market=MarketRegime.STRONG_BULL),
            _lt(market=MarketRegime.BULL),
            _lt(market=MarketRegime.BULL),
        ]
        counts = engine.count_by_decision(trades, AVOID_STRONG_BULL)
        assert counts["ALLOW"] == 2
        assert counts["REJECT"] == 1
