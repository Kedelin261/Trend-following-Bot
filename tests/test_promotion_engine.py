"""Tests for PromotionEngine — orchestration and strategy lock verification."""

import pathlib
import pytest
from typing import Dict, List

from src.data.models import Candle
from src.promotion.promotion_engine import (
    LOCKED_PARAMS,
    PromotionEngine,
    PromotionReport,
    StrategyTamperedError,
    verify_strategy_locked,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE
from tests.fixtures import downtrend_candles, uptrend_candles


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
def engine(config) -> PromotionEngine:
    return PromotionEngine(config, strategy_profile=_mini_profile())


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=60, symbol="SPY"),
        "DIA": downtrend_candles(n=60, symbol="DIA"),
    }


class TestVerifyStrategyLocked:

    def test_best_density_profile_passes_lock(self):
        # BEST_DENSITY_PROFILE should match LOCKED_PARAMS
        verify_strategy_locked(BEST_DENSITY_PROFILE)  # must not raise

    def test_modified_profile_raises_tampered_error(self):
        from dataclasses import replace
        modified = replace(BEST_DENSITY_PROFILE, adx_threshold=20.0)  # changed
        with pytest.raises(StrategyTamperedError):
            verify_strategy_locked(modified)

    def test_locked_params_match_best_density_profile(self):
        for param, expected in LOCKED_PARAMS.items():
            actual = getattr(BEST_DENSITY_PROFILE, param)
            assert actual == expected, f"{param}: expected {expected!r}, got {actual!r}"


class TestPromotionEngine:

    def test_returns_promotion_report(self, engine):
        report = engine.run(_asset_candles(), skip_lock_check=True)
        assert isinstance(report, PromotionReport)

    def test_empty_input_raises(self, engine):
        with pytest.raises(ValueError):
            engine.run({}, skip_lock_check=True)

    def test_strategy_tamper_raises_without_skip(self, config):
        from dataclasses import replace
        bad_profile = replace(BEST_DENSITY_PROFILE, adx_threshold=20.0)
        engine = PromotionEngine(config, strategy_profile=bad_profile)
        with pytest.raises(StrategyTamperedError):
            engine.run({"SPY": uptrend_candles(n=50)})

    def test_promotion_status_is_valid(self, engine):
        report = engine.run(_asset_candles(), skip_lock_check=True)
        assert report.promotion_status in (
            "PROMOTE_TO_PHASE_5", "RETURN_TO_RESEARCH"
        )

    def test_promoted_property(self, engine):
        report = engine.run(_asset_candles(), skip_lock_check=True)
        assert isinstance(report.promoted, bool)

    def test_report_fields_populated(self, engine):
        report = engine.run(_asset_candles(), skip_lock_check=True)
        assert report.strategy_name != ""
        assert report.robustness_rating in ("ROBUST", "MARGINAL", "UNSTABLE")
        assert isinstance(report.history_expansion_results, list)
        assert report.robustness_result is not None
        assert report.validation_result is not None
        assert report.final_recommendation is not None

    def test_candidate_validation_runs_when_provided(self, engine):
        candidates = {"XLV": uptrend_candles(n=60, symbol="XLV")}
        report = engine.run(_asset_candles(), candidate_candles=candidates,
                            skip_lock_check=True)
        assert len(report.asset_validation_results) == 1

    def test_no_execution_imports_in_promotion_modules(self):
        promo_dir = pathlib.Path("src/promotion")
        for pyfile in promo_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"
            assert "placeOrder"  not in text, f"{pyfile} places orders"
