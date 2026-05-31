"""Tests for RegimeStabilityEngine — end-to-end regime analysis."""

import pathlib
import pytest
from typing import Dict, List

from src.data.models import Candle
from src.promotion.promotion_engine import StrategyTamperedError
from src.regime.regime_stability_engine import (
    RegimeStabilityEngine,
    RegimeStabilityReport,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


def _mini_profile() -> StrategyProfile:
    return StrategyProfile(
        name="mini", description="mini",
        ema_fast=5, ema_slow=10,
        breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE, atr_period=3,
    )


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


@pytest.fixture
def engine(config) -> RegimeStabilityEngine:
    return RegimeStabilityEngine(
        config, strategy_profile=_mini_profile(), skip_lock_check=True
    )


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=60, symbol="SPY"),
        "DIA": downtrend_candles(n=60, symbol="DIA"),
    }


class TestRegimeStabilityEngine:

    def test_returns_regime_stability_report(self, engine):
        report = engine.run(_asset_candles())
        assert isinstance(report, RegimeStabilityReport)

    def test_empty_input_raises(self, engine):
        with pytest.raises(ValueError):
            engine.run({})

    def test_strategy_lock_verification(self, config):
        from dataclasses import replace
        bad = replace(BEST_DENSITY_PROFILE, adx_threshold=10.0)
        eng = RegimeStabilityEngine(config, strategy_profile=bad, skip_lock_check=False)
        with pytest.raises(StrategyTamperedError):
            eng.run({"SPY": uptrend_candles(n=60)})

    def test_skip_lock_check_allows_mini_profile(self, engine):
        report = engine.run({"SPY": uptrend_candles(n=60)})
        assert report is not None

    def test_market_regime_performance_populated(self, engine):
        report = engine.run(_asset_candles())
        assert isinstance(report.market_regime_performance, dict)

    def test_filter_simulations_populated(self, engine):
        report = engine.run(_asset_candles())
        assert isinstance(report.filter_simulations, list)

    def test_stability_assessment_is_string(self, engine):
        report = engine.run(_asset_candles())
        assert isinstance(report.stability_assessment, str)
        assert len(report.stability_assessment) > 0

    def test_edge_is_regime_dependent_is_bool(self, engine):
        report = engine.run(_asset_candles())
        assert isinstance(report.edge_is_regime_dependent, bool)

    def test_five_performance_dimensions_present(self, engine):
        report = engine.run(_asset_candles())
        assert report.market_regime_performance is not None
        assert report.trend_regime_performance is not None
        assert report.volatility_regime_performance is not None
        assert report.drawdown_env_performance is not None
        assert report.macro_regime_performance is not None

    def test_recommendations_is_list(self, engine):
        report = engine.run(_asset_candles())
        assert isinstance(report.recommendations, list)

    def test_no_broker_imports_in_regime_modules(self):
        regime_dir = pathlib.Path("src/regime")
        for pyfile in regime_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"
            assert "placeOrder"  not in text, f"{pyfile} places orders"

    def test_promoted_property_always_false(self, engine):
        # Phase 4.10 is research only — never promotes
        report = engine.run(_asset_candles())
        assert report.promoted is False
