"""Tests for RefinementEngine — orchestration and reporting."""

import pathlib
import pytest
from typing import Dict, List

from src.data.models import Candle
from src.refinement.asset_selector import AssetSelector
from src.refinement.refinement_engine import RefinementEngine, RefinementReport
from src.refinement.strategy_v2 import StrategyProfile, V1_PROFILE, V2_PROFILE
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0, "minimum_trades_required": 5},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


def _mini_v1() -> StrategyProfile:
    return StrategyProfile(
        name="V1-mini", description="test V1",
        ema_fast=5, ema_slow=10, breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE,
    )


def _mini_v2() -> StrategyProfile:
    return StrategyProfile(
        name="V2-mini", description="test V2",
        ema_fast=3, ema_slow=5, breakout_threshold=0.005,
        require_bull_regime=True, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE,
        atr_period=3,
    )


def _mini_engine(config, v1=None, v2=None) -> RefinementEngine:
    selector = AssetSelector(recommended=["SPY", "DIA"], excluded=[])
    return RefinementEngine(
        config=config,
        v1_profile=v1 or _mini_v1(),
        v2_profile=v2 or _mini_v2(),
        asset_selector=selector,
    )


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=50, symbol="SPY"),
        "DIA": uptrend_candles(n=50, symbol="DIA"),
    }


class TestRefinementEngineRun:

    def test_returns_refinement_report(self, config):
        engine = _mini_engine(config)
        report = engine.run(_asset_candles())
        assert isinstance(report, RefinementReport)

    def test_empty_input_raises(self, config):
        engine = _mini_engine(config)
        with pytest.raises(ValueError):
            engine.run({})

    def test_comparisons_populated(self, config):
        engine = _mini_engine(config)
        report = engine.run(_asset_candles())
        assert len(report.comparisons) > 0

    def test_excluded_assets_skipped(self, config):
        selector = AssetSelector(recommended=["SPY"], excluded=["DIA"])
        engine = RefinementEngine(config=config, asset_selector=selector,
                                  v1_profile=_mini_v1(), v2_profile=_mini_v2())
        report = engine.run({"SPY": uptrend_candles(n=50), "DIA": downtrend_candles(n=50)})
        symbols = {c.symbol for c in report.comparisons}
        assert "SPY" in symbols
        assert "DIA" not in symbols

    def test_v2_promoted_is_bool(self, config):
        engine = _mini_engine(config)
        report = engine.run(_asset_candles())
        assert isinstance(report.v2_promoted, bool)

    def test_recommendation_is_string(self, config):
        engine = _mini_engine(config)
        report = engine.run(_asset_candles())
        assert isinstance(report.recommendation, str)
        assert len(report.recommendation) > 0

    def test_warnings_is_list(self, config):
        engine = _mini_engine(config)
        report = engine.run(_asset_candles())
        assert isinstance(report.warnings, list)

    def test_v2_total_trades_nonnegative(self, config):
        engine = _mini_engine(config)
        report = engine.run(_asset_candles())
        assert report.v2_total_trades >= 0

    def test_not_promoted_without_trades(self, config):
        # Very strict V2 that generates 0 trades
        strict_v2 = StrategyProfile(
            name="strict", description="strict",
            ema_fast=3, ema_slow=5,
            breakout_threshold=0.99,  # impossible threshold
            require_bull_regime=False,
            adx_threshold=99.0,       # always blocks
            volatility_mode=VolatilityFilterMode.NONE,
            atr_period=3,
        )
        engine = RefinementEngine(
            config=config,
            v1_profile=_mini_v1(),
            v2_profile=strict_v2,
            asset_selector=AssetSelector(recommended=["SPY"], excluded=[]),
        )
        report = engine.run({"SPY": uptrend_candles(n=50)})
        assert report.v2_promoted is False


class TestBrokerAgnostic:

    def test_no_broker_imports_in_refinement_modules(self):
        ref_dir = pathlib.Path("src/refinement")
        for pyfile in ref_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"
