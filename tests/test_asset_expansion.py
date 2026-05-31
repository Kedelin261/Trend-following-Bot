"""Tests for AssetExpansionResearcher."""

import pytest
from typing import Dict, List

from src.data.models import Candle
from src.density.asset_expansion import (
    EXPANSION_CANDIDATES,
    AssetExpansionResearcher,
    AssetExpansionResult,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
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
        "XLK":  uptrend_candles(n=50, symbol="XLK"),
        "VTI":  downtrend_candles(n=50, symbol="VTI"),
        "SCHD": flat_candles(n=50, symbol="SCHD"),
    }


@pytest.fixture
def researcher(config) -> AssetExpansionResearcher:
    return AssetExpansionResearcher(
        config,
        base_profile=_mini_profile(),
        candidates=["XLK", "VTI", "SCHD"],
    )


class TestAssetExpansionResult:

    def test_meets_quality_good_metrics(self):
        r = AssetExpansionResult("XLK", 500, 40, 30.0, 1.7, 6.0, 0.63, recommended=True)
        assert r.meets_quality is True

    def test_fails_quality_bad_pf(self):
        r = AssetExpansionResult("QQQ", 500, 40, 30.0, 1.2, 6.0, 0.63, recommended=False)
        assert r.meets_quality is False

    def test_fails_quality_negative_expectancy(self):
        r = AssetExpansionResult("XLE", 500, 40, -5.0, 1.8, 6.0, 0.63, recommended=False)
        assert r.meets_quality is False


class TestAssetExpansionResearcher:

    def test_returns_result_per_available_candidate(self, researcher):
        results = researcher.research(_asset_candles())
        assert len(results) == 3

    def test_sorted_by_expectancy_descending(self, researcher):
        results = researcher.research(_asset_candles())
        for i in range(len(results) - 1):
            assert results[i].expectancy >= results[i + 1].expectancy

    def test_missing_asset_skipped(self, researcher):
        results = researcher.research({"XLK": uptrend_candles(n=50, symbol="XLK")})
        assert len(results) == 1
        assert results[0].symbol == "XLK"

    def test_empty_input_returns_empty(self, researcher):
        results = researcher.research({})
        assert results == []

    def test_recommended_additions_method(self, researcher):
        results = researcher.research(_asset_candles())
        recs = researcher.recommended_additions(results)
        assert isinstance(recs, list)
        # All recommended must be in results
        result_syms = {r.symbol for r in results}
        for rec in recs:
            assert rec in result_syms

    def test_default_candidates_defined(self):
        assert "XLK" in EXPANSION_CANDIDATES
        assert "VTI" in EXPANSION_CANDIDATES
        assert "SCHD" in EXPANSION_CANDIDATES
        assert len(EXPANSION_CANDIDATES) == 5
