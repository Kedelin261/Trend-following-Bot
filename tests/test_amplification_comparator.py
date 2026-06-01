"""Tests for AmplificationComparator — Phase 5.4."""

import math
import pytest
from src.amplification_validation.amplification_validator import ScenarioResult
from src.amplification_validation.amplification_comparator import (
    AmplificationComparator,
    ComparisonReport,
    ScenarioRanking,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario(
    name: str,
    trades: int = 600,
    pf: float = 1.20,
    exp: float = 8.0,
    dd: float = 12.0,
    robustness: str = "ROBUST",
    promoted: bool = False,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_name = name,
        description   = f"Test scenario {name}",
        trades        = trades,
        profit_factor = pf,
        expectancy    = exp,
        win_rate      = 0.46,
        max_drawdown  = dd,
        robustness    = robustness,
        history_bars  = 5000,
        data_source   = "SYNTHETIC",
        is_promoted   = promoted,
        pass_criteria = [],
        fail_criteria = [],
    )


def _five_scenarios(promoted_name: str = "") -> list:
    return [
        _scenario("BASELINE",            pf=1.15, exp=7.23, trades=1320),
        _scenario("REMOVE_SCHD",         pf=1.18, exp=8.40, trades=1178),
        _scenario("REMOVE_QUALITY_60_69",pf=1.19, exp=8.80, trades=1241),
        _scenario("REMOVE_HIGH_VOL",     pf=1.17, exp=7.90, trades=1270),
        _scenario("COMBINED_FILTERS",    pf=1.21, exp=9.10, trades=1100,
                  promoted=(promoted_name == "COMBINED_FILTERS")),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAmplificationComparator:

    def test_raises_on_empty_scenarios(self):
        with pytest.raises(ValueError):
            AmplificationComparator([])

    def test_returns_comparison_report(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        assert isinstance(report, ComparisonReport)

    def test_baseline_identified(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        assert report.baseline.scenario_name == "BASELINE"

    def test_rankings_has_five_entries(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        assert len(report.rankings) == 5

    def test_rankings_sorted_best_first(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        pfs = [r.profit_factor for r in report.rankings]
        assert pfs == sorted(pfs, reverse=True)

    def test_delta_pf_computed_for_non_baseline(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        for sc in report.scenarios:
            if sc.scenario_name != "BASELINE":
                assert sc.delta_pf is not None

    def test_delta_pf_none_for_baseline(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        assert report.baseline.delta_pf is None

    def test_delta_trades_computed(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        remove_schd = next(
            s for s in report.scenarios if s.scenario_name == "REMOVE_SCHD"
        )
        # REMOVE_SCHD has 1178 trades, BASELINE has 1320 → delta = -142
        assert remove_schd.delta_trades == 1178 - 1320

    def test_phase53_findings_held_when_pf_improves(self):
        """Any filter with positive delta PF → findings held."""
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        # All non-baseline scenarios have PF > baseline (1.15) → held
        assert report.phase53_findings_held is True

    def test_phase53_findings_not_held_when_all_worse(self):
        scenarios = [
            _scenario("BASELINE",            pf=1.15, trades=1320),
            _scenario("REMOVE_SCHD",         pf=1.10, trades=1178),
            _scenario("REMOVE_QUALITY_60_69",pf=1.08, trades=1241),
            _scenario("REMOVE_HIGH_VOL",     pf=1.12, trades=1270),
            _scenario("COMBINED_FILTERS",    pf=1.05, trades=1100),
        ]
        c = AmplificationComparator(scenarios)
        report = c.compare()
        assert report.phase53_findings_held is False

    def test_no_promotion_when_none_qualify(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        # None of our test scenarios have is_promoted=True
        assert report.best_candidate is None

    def test_promotion_candidate_found(self):
        scenarios = _five_scenarios()
        # Mark COMBINED as promoted
        scenarios[-1] = _scenario(
            "COMBINED_FILTERS", pf=1.55, exp=12.0, trades=550,
            dd=10.0, robustness="ROBUST", promoted=True,
        )
        c = AmplificationComparator(scenarios)
        report = c.compare()
        assert report.best_candidate is not None
        assert report.best_candidate.scenario_name == "COMBINED_FILTERS"

    def test_recommendation_proceed_when_promoted(self):
        scenarios = _five_scenarios()
        scenarios[-1] = _scenario(
            "COMBINED_FILTERS", pf=1.55, exp=12.0, trades=550,
            dd=10.0, robustness="ROBUST", promoted=True,
        )
        c = AmplificationComparator(scenarios)
        report = c.compare()
        assert "PROCEED TO PHASE 5.5" in report.recommendation

    def test_recommendation_return_when_not_promoted(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        assert "RETURN TO EDGE RESEARCH" in report.recommendation

    def test_most_effective_filter_identified(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        # COMBINED has highest PF (1.21) among non-baseline
        assert report.most_effective_filter == "COMBINED_FILTERS"

    def test_best_pf_improvement_positive(self):
        c = AmplificationComparator(_five_scenarios())
        report = c.compare()
        assert report.best_pf_improvement is not None
        assert report.best_pf_improvement > 0.0

    def test_scenario_ranking_pf_str_inf(self):
        r = ScenarioRanking(
            rank=1, scenario_name="TEST", trades=600,
            profit_factor=float("inf"), expectancy=10.0,
            max_drawdown=5.0, robustness="ROBUST",
            delta_pf=None, delta_trades=None,
            is_promoted=False, is_baseline=True,
        )
        assert r.pf_str == "∞"

    def test_ranking_delta_pf_str_positive(self):
        r = ScenarioRanking(
            rank=2, scenario_name="REMOVE_SCHD", trades=500,
            profit_factor=1.20, expectancy=8.0,
            max_drawdown=12.0, robustness="ROBUST",
            delta_pf=0.05, delta_trades=-142,
            is_promoted=False, is_baseline=False,
        )
        assert r.delta_pf_str == "+0.050"
        assert "-142" in r.delta_trades_str
