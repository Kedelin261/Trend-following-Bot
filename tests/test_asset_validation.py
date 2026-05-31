"""Tests for AssetValidator — candidate asset approval logic."""

import pytest
from typing import Dict, List

from src.data.models import Candle
from src.promotion.asset_validation import (
    CANDIDATE_SYMBOLS,
    AssetValidationResult,
    AssetValidator,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
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
def validator(config) -> AssetValidator:
    return AssetValidator(config, profile=_mini_profile())


def _baseline() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=50, symbol="SPY"),
        "VOO": uptrend_candles(n=50, symbol="VOO"),
    }


class TestAssetValidationResult:

    def test_approved_when_adds_trades_and_quality_ok(self):
        r = AssetValidationResult(
            candidate_symbol="XLV", baseline_trades=60, with_candidate_trades=80,
            trade_increase=20, baseline_pf=1.7, with_candidate_pf=1.65,
            candidate_pf=1.6, candidate_exp=20.0, candidate_dd=5.0,
            quality_maintained=True, adds_trades=True, approved=True,
        )
        assert r.approved is True

    def test_not_approved_when_no_trade_increase(self):
        r = AssetValidationResult(
            candidate_symbol="QQQ", baseline_trades=60, with_candidate_trades=60,
            trade_increase=0, baseline_pf=1.7, with_candidate_pf=1.6,
            candidate_pf=1.5, candidate_exp=10.0, candidate_dd=5.0,
            quality_maintained=True, adds_trades=False, approved=False,
        )
        assert r.approved is False


class TestAssetValidator:

    def test_validate_returns_result(self, validator):
        candidate = uptrend_candles(n=50, symbol="XLV")
        result = validator.validate(_baseline(), "XLV", candidate)
        assert isinstance(result, AssetValidationResult)

    def test_candidate_symbol_in_result(self, validator):
        candidate = uptrend_candles(n=50, symbol="XLV")
        result = validator.validate(_baseline(), "XLV", candidate)
        assert result.candidate_symbol == "XLV"

    def test_trade_increase_calculated(self, validator):
        candidate = uptrend_candles(n=50, symbol="XLV")
        result = validator.validate(_baseline(), "XLV", candidate)
        assert result.trade_increase == result.with_candidate_trades - result.baseline_trades

    def test_validate_all_returns_list(self, validator):
        candidates = {
            "XLV":  uptrend_candles(n=50, symbol="XLV"),
            "SCHD": downtrend_candles(n=50, symbol="SCHD"),
        }
        results = validator.validate_all(_baseline(), candidates)
        assert len(results) == 2

    def test_validate_all_excludes_existing_assets(self, validator):
        # SPY is already in baseline → should be excluded from candidate testing
        candidates = {
            "SPY": uptrend_candles(n=50, symbol="SPY"),  # already in baseline
            "XLV": uptrend_candles(n=50, symbol="XLV"),
        }
        results = validator.validate_all(_baseline(), candidates)
        symbols = {r.candidate_symbol for r in results}
        assert "SPY" not in symbols
        assert "XLV" in symbols

    def test_approved_candidates_method(self, validator):
        candidates = {"XLV": uptrend_candles(n=50, symbol="XLV")}
        results = validator.validate_all(_baseline(), candidates)
        approved = validator.approved_candidates(results)
        assert isinstance(approved, list)

    def test_default_candidates_constant(self):
        assert "XLV" in CANDIDATE_SYMBOLS
        assert "SCHD" in CANDIDATE_SYMBOLS
        assert "VTI" in CANDIDATE_SYMBOLS
