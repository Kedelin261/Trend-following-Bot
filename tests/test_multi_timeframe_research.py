"""Tests for MultiTimeframeResearcher — combination backtesting."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.timeframe.multi_timeframe_research import MultiTimeframeResearcher, MultiTimeframeResult
from src.timeframe.signal_overlap_analyzer import SignalOverlapAnalyzer
from src.timeframe.timeframe_backtester import TimeframeBacktester
from src.timeframe.timeframe_profile import COMBO_D1_H4, D1_PROFILE, H4_PROFILE
from tests.fixtures import downtrend_candles, uptrend_candles


BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _mini_profile() -> StrategyProfile:
    return StrategyProfile(
        name="mini", description="mini",
        ema_fast=5, ema_slow=10,
        breakout_threshold=0.001,
        require_bull_regime=False, adx_threshold=0.0,
        volatility_mode=VolatilityFilterMode.NONE,
        atr_period=3,
    )


def _h4_candles(n=50, symbol="SPY") -> List[Candle]:
    price = 400.0
    result = []
    for i in range(n):
        price *= 1.002
        result.append(Candle(
            symbol=symbol, timeframe="H4",
            timestamp=BASE + timedelta(hours=4 * i),
            open=price, high=price * 1.003, low=price * 0.997,
            close=price, volume=1e8, provider="TEST",
        ))
    return result


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
def researcher(config) -> MultiTimeframeResearcher:
    bt = TimeframeBacktester(config, _mini_profile())
    return MultiTimeframeResearcher(config, bt, SignalOverlapAnalyzer())


def _asset_candles() -> Dict[str, Dict[str, List[Candle]]]:
    return {
        "SPY": {
            "D1": uptrend_candles(n=50, symbol="SPY"),
            "H4": _h4_candles(n=50, symbol="SPY"),
        }
    }


class TestMultiTimeframeResearcher:

    def test_research_single_returns_result(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        assert isinstance(result, MultiTimeframeResult)

    def test_result_has_correct_label(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        assert result.label == "D1 + H4"

    def test_total_raw_trades_nonnegative(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        assert result.total_raw_trades >= 0

    def test_unique_trades_lte_total_raw(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        assert result.unique_trades <= result.total_raw_trades

    def test_overlap_has_acceptable_flag(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        assert isinstance(result.overlap.acceptable, bool)

    def test_research_all_returns_list(self, researcher):
        from src.timeframe.timeframe_profile import ALL_COMBOS
        results = researcher.research_all(_asset_candles(), [COMBO_D1_H4])
        assert len(results) == 1

    def test_meets_quality_is_bool(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        assert isinstance(result.meets_quality, bool)

    def test_is_viable_property(self, researcher):
        result = researcher.research_single(_asset_candles(), COMBO_D1_H4)
        # is_viable = meets_quality AND overlap acceptable
        expected = result.meets_quality and result.overlap.acceptable
        assert result.is_viable == expected

    def test_best_combo_none_when_all_fail(self, researcher):
        # Force failure by using downtrend candles → low quality
        dc = {"SPY": {
            "D1": downtrend_candles(n=50, symbol="SPY"),
            "H4": downtrend_candles(n=50, symbol="SPY"),
        }}
        results = researcher.research_all(dc, [COMBO_D1_H4])
        # If none viable, best_combo returns None
        best = researcher.best_combo(results)
        if all(not r.is_viable for r in results):
            assert best is None
