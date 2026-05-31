"""Tests for HistoryExpansionResearcher."""

import pytest

from src.promotion.history_expansion import (
    CANDLE_COUNTS,
    PROMOTION_MIN_TRADES,
    HistoryExpansionResearcher,
    HistoryExpansionResult,
)
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from tests.fixtures import uptrend_candles


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
def researcher(config) -> HistoryExpansionResearcher:
    return HistoryExpansionResearcher(
        config, profile=_mini_profile(), counts=[30, 40, 50]
    )


class TestHistoryExpansionResult:

    def test_meets_threshold_when_enough_trades(self):
        r = HistoryExpansionResult(3000, 105, 1.7, 30.0, 5.0, 3, meets_threshold=True)
        assert r.meets_threshold is True

    def test_does_not_meet_threshold_when_few_trades(self):
        r = HistoryExpansionResult(3000, 50, 1.7, 30.0, 5.0, 3, meets_threshold=False)
        assert r.meets_threshold is False

    def test_pf_str_for_infinity(self):
        import math
        r = HistoryExpansionResult(3000, 10, math.inf, 30.0, 5.0, 2, meets_threshold=False)
        assert r.pf_str == "∞"

    def test_pf_str_for_normal_value(self):
        r = HistoryExpansionResult(3000, 10, 1.68, 30.0, 5.0, 2, meets_threshold=False)
        assert "1.68" in r.pf_str


class TestHistoryExpansionResearcher:

    def test_returns_results_for_each_window(self, researcher):
        candles = {"SPY": uptrend_candles(n=60, symbol="SPY")}
        results = researcher.research(candles)
        # Should have 3 fixed windows + possibly max window
        assert len(results) >= 3

    def test_sorted_by_candle_count(self, researcher):
        candles = {"SPY": uptrend_candles(n=60, symbol="SPY")}
        results = researcher.research(candles)
        counts = [r.candle_count for r in results]
        assert counts == sorted(counts)

    def test_slices_candles_to_count(self, researcher):
        candles = {"SPY": uptrend_candles(n=60, symbol="SPY")}
        results = researcher.research(candles)
        # The 30-bar window should use at most 30 candles
        first = results[0]
        assert first.candle_count <= 30

    def test_meets_quality_is_bool(self, researcher):
        candles = {"SPY": uptrend_candles(n=60, symbol="SPY")}
        results = researcher.research(candles)
        for r in results:
            assert isinstance(r.meets_quality, bool)

    def test_best_window_returns_none_when_all_below_threshold(self, researcher):
        # With only 50 candles and few trades, won't hit 100 trade threshold
        candles = {"SPY": uptrend_candles(n=50, symbol="SPY")}
        results = researcher.research(candles)
        if all(not r.meets_threshold for r in results):
            assert researcher.best_window(results) is None

    def test_candle_counts_constant_defined(self):
        assert len(CANDLE_COUNTS) == 4
        assert PROMOTION_MIN_TRADES == 100
