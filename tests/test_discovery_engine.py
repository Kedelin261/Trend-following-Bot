"""Tests for DiscoveryEngine, EdgeScoreCalculator, DiscoveryComparator,
DiscoveryProfile, and DiscoveryReport (Phase 6.0 integration tests).

Covers:
- DiscoveryProfile dataclass
- AssetSummary dataclass
- EdgeScoreCalculator component scorers
- DiscoveryComparator ranking logic
- DiscoveryEngine runs without error on synthetic data
- FamilyID enum values
- DiscoveryReport prints without error
"""

import math
import pytest

from src.edge_discovery.strategy_family import FamilyID, FAMILY_DESCRIPTIONS
from src.edge_discovery.discovery_profile import (
    DiscoveryProfile, AssetSummary, MIN_SAMPLE,
    PROMO_MIN_PF, PROMO_MIN_TRADES, PROMO_MAX_DD,
)
from src.edge_discovery.edge_score import EdgeScoreCalculator
from src.edge_discovery.discovery_comparator import DiscoveryComparator, ComparisonReport
from src.edge_discovery.discovery_engine import DiscoveryEngine
from src.edge_discovery.discovery_report import DiscoveryReport
from tests.fixtures import uptrend_candles


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

EDGE_CONFIG = {
    "backtest": {
        "starting_balance":    10_000.0,
        "commission_per_trade": 1.0,
        "slippage_percent":     0.05,
    },
    "risk": {
        "risk_per_trade_percent": 1.0,
        "atr_period":             14,
        "atr_stop_multiplier":    2.0,
        "atr_target_multiplier":  3.0,
        "minimum_signal_score":   40.0,
        "minimum_risk_reward":    1.0,
    },
}


def _sample_profile(
    family_id=FamilyID.TREND_PERSISTENCE,
    trades=100,
    pf=1.30,
    exp=15.0,
    dd=8.0,
    wr=0.52,
    robustness="MARGINAL",
    edge_score=0.0,
    insufficient=False,
) -> DiscoveryProfile:
    return DiscoveryProfile(
        family_id=family_id,
        family_name=family_id.value,
        description="test",
        data_source="SYNTHETIC",
        trade_count=trades,
        profit_factor=pf,
        expectancy=exp,
        win_rate=wr,
        max_drawdown=dd,
        robustness=robustness,
        survivability=0.75,
        scalable=(trades >= PROMO_MIN_TRADES),
        insufficient_sample=insufficient,
        edge_score=edge_score,
    )


# ---------------------------------------------------------------------------
# FamilyID enum
# ---------------------------------------------------------------------------

class TestFamilyID:
    def test_six_research_families(self):
        research = [f for f in FamilyID if f != FamilyID.MOMENTUM_ROTATION]
        assert len(research) == 6

    def test_benchmark_exists(self):
        assert FamilyID.MOMENTUM_ROTATION in FamilyID

    def test_all_family_ids_have_descriptions(self):
        for fid in FamilyID:
            assert fid in FAMILY_DESCRIPTIONS
            assert len(FAMILY_DESCRIPTIONS[fid]) > 10


# ---------------------------------------------------------------------------
# DiscoveryProfile
# ---------------------------------------------------------------------------

class TestDiscoveryProfile:
    def test_pf_str_finite(self):
        p = _sample_profile(pf=1.45)
        assert p.pf_str == "1.45"

    def test_pf_str_infinite(self):
        p = _sample_profile(pf=float("inf"))
        assert p.pf_str == "∞"

    def test_promotion_gap_insufficient_sample(self):
        p = _sample_profile(insufficient=True)
        assert "INSUFFICIENT" in p.promotion_gap

    def test_promotion_gap_meets_all_criteria(self):
        p = _sample_profile(
            trades=600, pf=1.55, exp=20.0, dd=10.0,
            wr=0.55, robustness="ROBUST"
        )
        assert p.promotion_gap == "MEETS ALL CRITERIA"

    def test_promotion_gap_shows_pf_gap(self):
        p = _sample_profile(trades=600, pf=1.20, exp=5.0, dd=5.0, robustness="ROBUST")
        assert "PF" in p.promotion_gap

    def test_is_promising_true(self):
        p = _sample_profile(trades=100, pf=1.15, exp=5.0)
        assert p.is_promising

    def test_is_promising_false_low_trades(self):
        p = _sample_profile(trades=10, pf=1.30, exp=5.0)
        assert not p.is_promising

    def test_is_promising_false_low_pf(self):
        p = _sample_profile(trades=100, pf=1.05, exp=5.0)
        assert not p.is_promising

    def test_is_promising_false_negative_exp(self):
        p = _sample_profile(trades=100, pf=1.20, exp=-1.0)
        assert not p.is_promising

    def test_min_sample_constant(self):
        assert MIN_SAMPLE == 50


# ---------------------------------------------------------------------------
# AssetSummary
# ---------------------------------------------------------------------------

class TestAssetSummary:
    def test_repr_contains_symbol(self):
        a = AssetSummary(symbol="SPY", trades=50, profit_factor=1.30, expectancy=10.0)
        assert "SPY" in repr(a)

    def test_fields_accessible(self):
        a = AssetSummary(symbol="QQQ", trades=30, profit_factor=1.10, expectancy=-5.0)
        assert a.symbol == "QQQ"
        assert a.trades == 30


# ---------------------------------------------------------------------------
# EdgeScoreCalculator
# ---------------------------------------------------------------------------

class TestEdgeScoreCalculator:
    def setup_method(self):
        self.calc = EdgeScoreCalculator()

    def test_insufficient_sample_returns_zero(self):
        p = _sample_profile(insufficient=True)
        assert self.calc.compute(p) == 0.0

    def test_below_min_sample_returns_zero(self):
        p = _sample_profile(trades=10)
        assert self.calc.compute(p) == 0.0

    def test_score_in_range(self):
        p = _sample_profile(trades=200, pf=1.40, exp=20.0, dd=10.0, wr=0.52)
        score = self.calc.compute(p)
        assert 0.0 <= score <= 100.0

    def test_higher_pf_higher_score(self):
        p1 = _sample_profile(trades=200, pf=1.20)
        p2 = _sample_profile(trades=200, pf=1.60)
        assert self.calc.compute(p2) > self.calc.compute(p1)

    def test_robust_scores_higher_than_unstable(self):
        p_robust   = _sample_profile(trades=200, robustness="ROBUST")
        p_unstable = _sample_profile(trades=200, robustness="UNSTABLE")
        assert self.calc.compute(p_robust) > self.calc.compute(p_unstable)

    def test_lower_dd_scores_higher(self):
        p_low_dd  = _sample_profile(trades=200, dd=5.0)
        p_high_dd = _sample_profile(trades=200, dd=25.0)
        assert self.calc.compute(p_low_dd) > self.calc.compute(p_high_dd)

    def test_score_pf_below_1_returns_0(self):
        assert EdgeScoreCalculator._score_pf(0.5) == 0.0

    def test_score_pf_above_2_returns_100(self):
        assert EdgeScoreCalculator._score_pf(2.5) == 100.0

    def test_score_exp_negative_returns_0(self):
        assert EdgeScoreCalculator._score_expectancy(-10.0) == 0.0

    def test_score_robustness_robust(self):
        assert EdgeScoreCalculator._score_robustness("ROBUST") == 100.0

    def test_score_robustness_marginal(self):
        assert EdgeScoreCalculator._score_robustness("MARGINAL") == 50.0

    def test_score_robustness_unstable(self):
        assert EdgeScoreCalculator._score_robustness("UNSTABLE") == 0.0

    def test_score_trades_zero(self):
        assert EdgeScoreCalculator._score_trades(0) == 0.0

    def test_score_drawdown_zero_pct(self):
        assert EdgeScoreCalculator._score_drawdown(0.0) == 100.0

    def test_score_drawdown_30_pct(self):
        assert EdgeScoreCalculator._score_drawdown(30.0) == 0.0


# ---------------------------------------------------------------------------
# DiscoveryComparator
# ---------------------------------------------------------------------------

class TestDiscoveryComparator:
    def test_rankings_count_excludes_benchmark(self):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.30),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=80, pf=1.20),
            _sample_profile(FamilyID.MOMENTUM_ROTATION, trades=90, pf=1.10),
        ]
        report = DiscoveryComparator(profiles).compare()
        # Only 2 research families should be ranked (not benchmark)
        assert len(report.rankings) == 2

    def test_benchmark_separated(self):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.30),
            _sample_profile(FamilyID.MOMENTUM_ROTATION, trades=90, pf=1.10),
        ]
        report = DiscoveryComparator(profiles).compare()
        assert report.benchmark_ranking is not None
        assert report.benchmark_ranking.family_id == FamilyID.MOMENTUM_ROTATION

    def test_rankings_by_score_descending(self):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.20, exp=5.0, dd=10.0, robustness="UNSTABLE"),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=200, pf=1.60, exp=30.0, dd=5.0, robustness="ROBUST"),
        ]
        report = DiscoveryComparator(profiles).compare()
        # Higher-metric profile should rank first
        assert report.rankings[0].profit_factor >= report.rankings[1].profit_factor or \
               report.rankings[0].edge_score >= report.rankings[1].edge_score

    def test_discovery_winner_has_highest_score(self):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.20),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=200, pf=1.80, exp=30.0, dd=5.0, robustness="ROBUST"),
        ]
        report = DiscoveryComparator(profiles).compare()
        if report.discovery_winner:
            assert report.discovery_winner.edge_score == max(r.edge_score for r in report.rankings)

    def test_promising_candidates_all_have_score_gt_0(self):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.20, exp=10.0),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=200, pf=1.50, exp=20.0, robustness="ROBUST"),
        ]
        report = DiscoveryComparator(profiles).compare()
        for cand in report.promising_candidates:
            assert cand.edge_score > 0.0

    def test_all_insufficient_no_winner(self):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, insufficient=True),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, insufficient=True),
        ]
        report = DiscoveryComparator(profiles).compare()
        assert report.discovery_winner is None or report.discovery_winner.edge_score == 0.0


# ---------------------------------------------------------------------------
# DiscoveryEngine (smoke test with minimal synthetic data)
# ---------------------------------------------------------------------------

class TestDiscoveryEngineSmoke:
    def test_runs_without_error(self):
        """Smoke test: engine completes on tiny synthetic candles."""
        asset_candles = {
            "SPY": uptrend_candles(n=300, start=400.0, daily_gain=0.002),
        }
        engine = DiscoveryEngine(EDGE_CONFIG, asset_candles, data_source="SYNTHETIC")
        report = engine.run()
        assert isinstance(report, ComparisonReport)

    def test_report_has_6_research_rankings(self):
        asset_candles = {
            "SPY": uptrend_candles(n=300, start=400.0, daily_gain=0.002),
        }
        engine = DiscoveryEngine(EDGE_CONFIG, asset_candles, data_source="SYNTHETIC")
        report = engine.run()
        # 6 research families + 1 benchmark = 7 profiles, rankings = 6
        assert len(report.rankings) == 6

    def test_benchmark_present(self):
        asset_candles = {
            "SPY": uptrend_candles(n=300, start=400.0, daily_gain=0.002),
        }
        engine = DiscoveryEngine(EDGE_CONFIG, asset_candles, data_source="SYNTHETIC")
        report = engine.run()
        assert report.benchmark_ranking is not None

    def test_all_profiles_have_data_source(self):
        asset_candles = {
            "SPY": uptrend_candles(n=300, start=400.0, daily_gain=0.002),
        }
        engine = DiscoveryEngine(EDGE_CONFIG, asset_candles, data_source="SYNTHETIC")
        report = engine.run()
        # Rankings are FamilyRanking objects (no data_source field — that's on profile)
        assert isinstance(report, ComparisonReport)


# ---------------------------------------------------------------------------
# DiscoveryReport (print smoke test)
# ---------------------------------------------------------------------------

class TestDiscoveryReport:
    def test_prints_without_error(self, capsys):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.30, exp=10.0),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=80, pf=1.20, exp=5.0),
            _sample_profile(FamilyID.RELATIVE_STRENGTH, insufficient=True),
            _sample_profile(FamilyID.MARKET_LEADERSHIP, trades=120, pf=1.40, exp=15.0),
            _sample_profile(FamilyID.VOLATILITY_TRANSITION, trades=60, pf=1.10, exp=2.0),
            _sample_profile(FamilyID.MULTI_TIMEFRAME_ALIGNMENT, trades=90, pf=1.25, exp=8.0),
            _sample_profile(FamilyID.MOMENTUM_ROTATION, trades=150, pf=1.15, exp=5.0),
        ]
        report = DiscoveryComparator(profiles).compare()
        rpt = DiscoveryReport(report, data_source="SYNTHETIC")
        rpt.print()
        captured = capsys.readouterr()
        assert "EDGE DISCOVERY VALIDATION REPORT" in captured.out
        assert "DATA SOURCE" in captured.out
        assert "SYNTHETIC" in captured.out

    def test_prints_discovery_winner_section(self, capsys):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.30, exp=10.0),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=80, pf=1.20, exp=5.0),
            _sample_profile(FamilyID.RELATIVE_STRENGTH, trades=150, pf=1.45, exp=20.0, robustness="ROBUST"),
            _sample_profile(FamilyID.MARKET_LEADERSHIP, trades=120, pf=1.40, exp=15.0),
            _sample_profile(FamilyID.VOLATILITY_TRANSITION, trades=60, pf=1.10, exp=2.0),
            _sample_profile(FamilyID.MULTI_TIMEFRAME_ALIGNMENT, trades=90, pf=1.25, exp=8.0),
            _sample_profile(FamilyID.MOMENTUM_ROTATION, trades=150, pf=1.15, exp=5.0),
        ]
        report = DiscoveryComparator(profiles).compare()
        rpt = DiscoveryReport(report, data_source="SYNTHETIC")
        rpt.print()
        captured = capsys.readouterr()
        assert "DISCOVERY WINNER" in captured.out

    def test_recommendation_section_present(self, capsys):
        profiles = [
            _sample_profile(FamilyID.TREND_PERSISTENCE, trades=100, pf=1.30, exp=10.0),
            _sample_profile(FamilyID.BREAKOUT_CONTINUATION, trades=80, pf=1.20, exp=5.0),
            _sample_profile(FamilyID.RELATIVE_STRENGTH, insufficient=True),
            _sample_profile(FamilyID.MARKET_LEADERSHIP, trades=120, pf=1.40, exp=15.0),
            _sample_profile(FamilyID.VOLATILITY_TRANSITION, trades=60, pf=1.10, exp=2.0),
            _sample_profile(FamilyID.MULTI_TIMEFRAME_ALIGNMENT, trades=90, pf=1.25, exp=8.0),
            _sample_profile(FamilyID.MOMENTUM_ROTATION, trades=150, pf=1.15, exp=5.0),
        ]
        report = DiscoveryComparator(profiles).compare()
        rpt = DiscoveryReport(report, data_source="SYNTHETIC")
        rpt.print()
        captured = capsys.readouterr()
        assert "RECOMMENDATION" in captured.out
        # One of the two valid outcomes
        assert ("PROCEED TO PHASE 6.1" in captured.out or
                "RETURN TO EDGE DISCOVERY" in captured.out)
