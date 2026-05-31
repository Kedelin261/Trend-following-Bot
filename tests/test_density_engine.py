"""Tests for DensityEngine and DensityProfile."""

import pathlib
import pytest
from typing import Dict, List

from src.data.models import Candle
from src.density.density_engine import (
    DensityEngine,
    DensityProfile,
    DensityReport,
    V2_DENSITY_BASELINE,
)
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from tests.fixtures import downtrend_candles, uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0, "minimum_trades_required": 5},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


def _mini_profile() -> StrategyProfile:
    return StrategyProfile(
        name="mini", description="mini",
        ema_fast=5, ema_slow=10,
        breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE,
        atr_period=3,
    )


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=50, symbol="SPY"),
        "DIA": downtrend_candles(n=50, symbol="DIA"),
    }


@pytest.fixture
def engine(config) -> DensityEngine:
    return DensityEngine(config, base_profile=_mini_profile(), primary_symbol="SPY")


class TestDensityProfile:

    def test_to_strategy_profile(self):
        dp = DensityProfile(
            name="test", adx_threshold=22.0,
            breakout_threshold=0.008,
            volatility_mode=VolatilityFilterMode.LOW_AND_MEDIUM,
            assets=["SPY", "VOO"],
        )
        sp = dp.to_strategy_profile()
        assert sp.adx_threshold == 22.0
        assert sp.breakout_threshold == pytest.approx(0.008)
        assert sp.volatility_mode == VolatilityFilterMode.LOW_AND_MEDIUM

    def test_to_strategy_profile_with_custom_base(self):
        dp = DensityProfile(
            name="test", adx_threshold=20.0,
            breakout_threshold=0.005,
            volatility_mode=VolatilityFilterMode.NONE,
            assets=["SPY"],
        )
        sp = dp.to_strategy_profile(base=_mini_profile())
        assert sp.ema_fast == 5   # preserved from base
        assert sp.adx_threshold == 20.0

    def test_run_backtest_returns_results(self, config):
        from src.backtest.models import BacktestResults
        dp = DensityProfile(
            name="test", adx_threshold=0.0,
            breakout_threshold=0.001,
            volatility_mode=VolatilityFilterMode.NONE,
            assets=["SPY"],
        )
        candles = uptrend_candles(n=50, symbol="SPY")
        results = dp.run_backtest("SPY", candles, config)
        assert isinstance(results, BacktestResults)

    def test_v2_baseline_profile_exists(self):
        assert V2_DENSITY_BASELINE.adx_threshold == 25.0
        assert V2_DENSITY_BASELINE.breakout_threshold == pytest.approx(0.0100)
        assert "SPY" in V2_DENSITY_BASELINE.assets


class TestDensityEngine:

    def test_returns_density_report(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=False, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert isinstance(report, DensityReport)

    def test_empty_input_raises(self, engine):
        with pytest.raises(ValueError):
            engine.run({})

    def test_promoted_is_bool(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=False, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert isinstance(report.promoted, bool)

    def test_recommendation_is_string(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=False, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert len(report.recommendation) > 0

    def test_best_density_profile_is_density_profile(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=False, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert isinstance(report.best_density_profile, DensityProfile)

    def test_skip_research_when_flags_false(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=False, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert report.adx_results == []
        assert report.volatility_results == []
        assert report.breakout_results == []
        assert report.asset_expansion_results == []

    def test_adx_results_populated_when_enabled(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=True, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert len(report.adx_results) > 0

    def test_no_broker_imports_in_density_modules(self):
        density_dir = pathlib.Path("src/density")
        for pyfile in density_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"

    def test_baseline_results_populated(self, engine):
        report = engine.run(
            _asset_candles(),
            run_adx=False, run_vol=False, run_breakout=False, run_expansion=False,
        )
        assert len(report.baseline_asset_results) == 2
