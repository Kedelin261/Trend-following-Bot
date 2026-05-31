"""Tests for StrategyProfile, CompositeSignalEngine, V1/V2 profiles."""

import pytest
from datetime import datetime, timezone

from src.backtest.backtest_engine import BacktestEngine
from src.refinement.strategy_v2 import (
    V1_PROFILE,
    V2_PROFILE,
    CompositeSignalEngine,
    StrategyProfile,
    _none_signal,
)
from src.refinement.market_regime_filter import MarketRegimeFilter
from src.refinement.adx_trade_filter import ADXTradeFilter
from src.refinement.volatility_trade_filter import VolatilityFilterMode, VolatilityTradeFilter
from src.signals.signal_engine import SignalEngine
from src.signals.models import SignalType
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


class TestStrategyProfile:

    def test_v1_profile_has_expected_settings(self):
        assert V1_PROFILE.ema_fast == 50
        assert V1_PROFILE.ema_slow == 200
        assert V1_PROFILE.breakout_threshold == pytest.approx(0.0025)
        assert V1_PROFILE.require_bull_regime is False
        assert V1_PROFILE.adx_threshold == 0.0

    def test_v2_profile_has_expected_settings(self):
        assert V2_PROFILE.ema_fast == 20
        assert V2_PROFILE.ema_slow == 50
        assert V2_PROFILE.breakout_threshold == pytest.approx(0.0100)
        assert V2_PROFILE.require_bull_regime is True
        assert V2_PROFILE.adx_threshold == 25.0
        assert V2_PROFILE.volatility_mode == VolatilityFilterMode.MEDIUM_ONLY

    def test_v1_min_warmup(self):
        # V1: no regime filter → ema_slow + 10 = 210
        assert V1_PROFILE.min_warmup == 210

    def test_v2_min_warmup_accounts_for_regime_filter(self):
        # V2: regime filter needs EMA200 → 210
        assert V2_PROFILE.min_warmup == 210

    def test_build_backtest_engine_returns_engine(self, config):
        engine = V2_PROFILE.build_backtest_engine(config)
        assert isinstance(engine, BacktestEngine)

    def test_v1_build_returns_engine(self, config):
        engine = V1_PROFILE.build_backtest_engine(config)
        assert isinstance(engine, BacktestEngine)

    def test_run_backtest_returns_results(self, config):
        from src.backtest.models import BacktestResults
        custom = StrategyProfile(
            name="Test", description="Test profile",
            ema_fast=5, ema_slow=10,
            breakout_threshold=0.001,
            require_bull_regime=False, adx_threshold=0.0,
            volatility_mode=VolatilityFilterMode.NONE,
        )
        candles = uptrend_candles(n=50)
        results = custom.run_backtest(candles, config)
        assert isinstance(results, BacktestResults)

    def test_profile_with_no_filters_min_warmup(self):
        p = StrategyProfile(
            name="X", description="X", ema_fast=20, ema_slow=50,
            breakout_threshold=0.01,
            require_bull_regime=False, adx_threshold=0.0,
            volatility_mode=VolatilityFilterMode.NONE,
        )
        assert p.min_warmup == 60  # ema_slow + 10


class TestCompositeSignalEngine:

    def _base(self, fast=5, slow=10):
        from src.signals.trend_detector import TrendDetector
        from src.signals.breakout_detector import BreakoutDetector
        return SignalEngine(
            trend_detector=TrendDetector(fast, slow),
            breakout_detector=BreakoutDetector(0.001),
            min_candles=slow + 2,
        )

    def test_no_filters_delegates_to_base(self):
        engine = CompositeSignalEngine(base_engine=self._base())
        candles = uptrend_candles(n=50, daily_gain=0.005)
        signal = engine.generate_signal(candles)
        assert signal is not None

    def test_regime_filter_blocks_bear_market(self):
        regime_filter = MarketRegimeFilter(fast_period=5, mid_period=10, slow_period=20)
        engine = CompositeSignalEngine(
            base_engine=self._base(), regime_filter=regime_filter
        )
        candles = downtrend_candles(n=50, daily_loss=0.005)
        signal = engine.generate_signal(candles)
        assert signal.signal_type == SignalType.NONE
        assert "Regime" in signal.reasoning[0]

    def test_adx_filter_blocks_when_below_threshold(self):
        # Flat market → ADX near 0; threshold=50 → always blocked
        adx_filter = ADXTradeFilter(threshold=50.0, period=5)
        engine = CompositeSignalEngine(
            base_engine=self._base(), adx_filter=adx_filter
        )
        candles = flat_candles(n=50)  # flat → near-zero ADX → fails threshold=50
        signal = engine.generate_signal(candles)
        assert signal.signal_type == SignalType.NONE
        assert any("ADX" in r or "filter" in r.lower() for r in signal.reasoning)

    def test_vol_filter_can_block_signal(self):
        # Very tight MEDIUM band → block anything
        vol_filter = VolatilityTradeFilter(
            mode=VolatilityFilterMode.MEDIUM_ONLY,
            atr_period=5,
            low_threshold=50.0,  # impossible to satisfy (ATR% can't be >50% for SPY)
            high_threshold=60.0,
        )
        engine = CompositeSignalEngine(
            base_engine=self._base(), vol_filter=vol_filter
        )
        candles = uptrend_candles(n=50)
        signal = engine.generate_signal(candles)
        assert signal.signal_type == SignalType.NONE

    def test_min_candles_delegates_to_base(self):
        base = self._base(fast=5, slow=10)
        engine = CompositeSignalEngine(base_engine=base)
        assert engine.min_candles == base.min_candles

    def test_empty_candles_returns_none_signal(self):
        engine = CompositeSignalEngine(base_engine=self._base())
        signal = engine.generate_signal([])
        assert signal.signal_type == SignalType.NONE


class TestNoneSignal:

    def test_returns_none_signal(self):
        candles = uptrend_candles(n=5)
        sig = _none_signal(candles, "Test reason")
        assert sig.signal_type == SignalType.NONE

    def test_reason_in_reasoning(self):
        candles = uptrend_candles(n=5)
        sig = _none_signal(candles, "Test reason")
        assert "Test reason" in sig.reasoning

    def test_symbol_propagated(self):
        candles = uptrend_candles(n=5, symbol="QQQ")
        sig = _none_signal(candles, "reason")
        assert sig.symbol == "QQQ"
