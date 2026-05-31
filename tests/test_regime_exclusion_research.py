"""Tests for RegimeExclusionResearcher."""

import pytest

from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.regime_filter.regime_exclusion_research import (
    EXCLUSION_PROFILES,
    ExclusionResult,
    RegimeExclusionResearcher,
)
from src.regime_filter.filter_profiles import NO_FILTER
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
def researcher(config) -> RegimeExclusionResearcher:
    return RegimeExclusionResearcher(config, _mini_profile())


class TestRegimeExclusionResearcher:

    def test_returns_list_of_results(self, researcher):
        candles = {"SPY": uptrend_candles(n=60)}
        results = researcher.research(candles)
        assert isinstance(results, list)

    def test_excludes_no_filter_from_results(self, researcher):
        candles = {"SPY": uptrend_candles(n=60)}
        results = researcher.research(candles)
        names = [r.profile.name for r in results]
        assert "NO_FILTER" not in names

    def test_result_count_matches_profiles(self, researcher):
        from src.regime_filter.filter_profiles import AVOID_STRONG_BULL, AVOID_EXTREME_VOL
        candles = {"SPY": uptrend_candles(n=60)}
        results = researcher.research(candles, profiles=[AVOID_STRONG_BULL, AVOID_EXTREME_VOL])
        assert len(results) == 2

    def test_result_has_comparison(self, researcher):
        candles = {"SPY": uptrend_candles(n=60)}
        results = researcher.research(candles)
        for r in results:
            assert r.comparison is not None

    def test_is_beneficial_is_bool(self, researcher):
        candles = {"SPY": uptrend_candles(n=60)}
        results = researcher.research(candles)
        for r in results:
            assert isinstance(r.is_beneficial, bool)

    def test_beneficial_exclusions_method(self, researcher):
        candles = {"SPY": uptrend_candles(n=60)}
        results = researcher.research(candles)
        beneficial = researcher.beneficial_exclusions(results)
        assert isinstance(beneficial, list)

    def test_exclusion_profiles_constant(self):
        assert len(EXCLUSION_PROFILES) > 0
        assert all(p.name != "NO_FILTER" for p in EXCLUSION_PROFILES)


class TestExclusionResult:

    def test_trades_retained_pct(self):
        from src.regime_filter.filter_profiles import AVOID_STRONG_BULL
        from src.regime_filter.filter_backtester import FilterBacktestResult
        from src.regime_filter.filter_comparator import FilterComparisonResult

        def _r(profile, trades_after=40) -> FilterBacktestResult:
            return FilterBacktestResult(
                filter_profile=profile,
                total_trades_before=50, total_trades_after=trades_after,
                trades_removed=50 - trades_after, win_rate=0.6,
                profit_factor=1.7, expectancy=25.0, max_drawdown=5.0, net_pnl=1000.0,
            )

        r = ExclusionResult(
            profile=AVOID_STRONG_BULL,
            baseline=_r(NO_FILTER, trades_after=50),
            filtered=_r(AVOID_STRONG_BULL, trades_after=40),
            comparison=FilterComparisonResult(
                baseline=_r(NO_FILTER), filtered=_r(AVOID_STRONG_BULL),
                pf_improvement=0.2, pf_improvement_pct=15.0, exp_improvement=5.0,
                dd_reduction=1.0, trade_reduction=10,
                baseline_meets_quality=False, filtered_meets_quality=True,
                is_improvement=True,
            ),
        )
        assert r.trades_retained_pct == pytest.approx(80.0)
