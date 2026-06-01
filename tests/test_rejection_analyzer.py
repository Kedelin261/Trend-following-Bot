"""Tests for RejectionCounter and RejectionSummary — Phase 6.0A.

Tests verify:
- RejectionCounter mutation methods
- Derived properties (most_common_rejection, signal_rate, rejection_pct)
- RejectionSummary.from_counters aggregation
- Multi-asset aggregation accuracy
- Edge cases (zero candidates, zero signals, single asset)
"""

import pytest
from src.edge_discovery.diagnostics.rejection_analyzer import (
    RejectionCounter,
    RejectionSummary,
)


# ---------------------------------------------------------------------------
# RejectionCounter unit tests
# ---------------------------------------------------------------------------

class TestRejectionCounter:
    """Tests for the per-asset rejection tracking data structure."""

    def test_initial_state(self):
        """Fresh counter has all zeros."""
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        assert c.total_bars   == 0
        assert c.candidates   == 0
        assert c.signals      == 0
        assert c.rejections   == 0
        assert c.rule_counts  == {}
        assert c.signal_bars  == []
        assert c.first_signal_bar is None
        assert c.last_signal_bar  is None

    def test_record_candidate_increments(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_candidate()
        c.record_candidate()
        assert c.candidates == 2

    def test_record_bar_evaluated(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        for _ in range(5):
            c.record_bar_evaluated()
        assert c.total_bars == 5

    def test_record_rejection_single_rule(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_rejection("R1:consecutive<20", bar_index=100)
        assert c.rejections == 1
        assert c.rule_counts["R1:consecutive<20"] == 1
        assert c.rule_first_bar["R1:consecutive<20"] == 100
        assert c.rule_last_bar["R1:consecutive<20"]  == 100

    def test_record_rejection_multiple_bars(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_rejection("R1:rule_a", bar_index=10)
        c.record_rejection("R1:rule_a", bar_index=20)
        c.record_rejection("R1:rule_a", bar_index=15)  # middle bar
        assert c.rule_counts["R1:rule_a"] == 3
        assert c.rule_first_bar["R1:rule_a"] == 10   # first seen
        assert c.rule_last_bar["R1:rule_a"]  == 15   # last seen

    def test_record_rejection_multiple_rules(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_rejection("R1:rule_a", 10)
        c.record_rejection("R2:rule_b", 11)
        c.record_rejection("R1:rule_a", 12)
        assert c.rejections == 3
        assert c.rule_counts["R1:rule_a"] == 2
        assert c.rule_counts["R2:rule_b"] == 1

    def test_record_signal_first_and_last(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_signal(100)
        assert c.signals           == 1
        assert c.signal_bars       == [100]
        assert c.first_signal_bar  == 100
        assert c.last_signal_bar   == 100

        c.record_signal(200)
        assert c.signals           == 2
        assert c.first_signal_bar  == 100   # unchanged
        assert c.last_signal_bar   == 200   # updated

    def test_most_common_rejection_none_when_empty(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        assert c.most_common_rejection is None

    def test_most_common_rejection_single_rule(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_rejection("R1:test", 10)
        mc = c.most_common_rejection
        assert mc is not None
        assert mc[0] == "R1:test"
        assert mc[1] == 1

    def test_most_common_rejection_picks_highest(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        for i in range(5):
            c.record_rejection("R1:high", i)
        for i in range(2):
            c.record_rejection("R2:low", i + 100)
        mc = c.most_common_rejection
        assert mc[0] == "R1:high"
        assert mc[1] == 5

    def test_signal_rate_zero_candidates(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        assert c.signal_rate == 0.0

    def test_signal_rate_calculation(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        for _ in range(10):
            c.record_candidate()
        for i in range(3):
            c.record_signal(i)
        assert abs(c.signal_rate - 0.30) < 1e-9

    def test_rejection_pct_zero_candidates(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        c.record_rejection("R1:x", 5)
        # rejections recorded without candidates → pct should be 0
        pct = c.rejection_pct
        assert pct["R1:x"] == 0.0

    def test_rejection_pct_calculation(self):
        c = RejectionCounter(family_name="TEST", asset_symbol="SPY")
        for _ in range(100):
            c.record_candidate()
        for i in range(25):
            c.record_rejection("R1:test", i)
        pct = c.rejection_pct
        assert abs(pct["R1:test"] - 25.0) < 1e-9


# ---------------------------------------------------------------------------
# RejectionSummary unit tests
# ---------------------------------------------------------------------------

class TestRejectionSummary:
    """Tests for multi-asset aggregated summary."""

    def _make_counter(
        self,
        symbol: str,
        candidates: int = 100,
        signals: int = 0,
        rejections: dict = None,
    ) -> RejectionCounter:
        c = RejectionCounter(family_name="TEST", asset_symbol=symbol)
        for _ in range(candidates):
            c.record_candidate()
        if rejections:
            bar = 0
            for rule, count in rejections.items():
                for _ in range(count):
                    c.record_rejection(rule, bar)
                    bar += 1
        for i in range(signals):
            c.record_signal(i + 500)
        return c

    def test_from_counters_empty_list(self):
        s = RejectionSummary.from_counters("TEST", [])
        assert s.total_bars        == 0
        assert s.total_candidates  == 0
        assert s.total_signals     == 0
        assert s.total_rejections  == 0
        assert s.any_signals is False
        assert s.most_common_rejection is None

    def test_from_counters_single_asset(self):
        c = self._make_counter("SPY", candidates=100, signals=5,
                               rejections={"R1:rule": 80, "R2:rule": 15})
        s = RejectionSummary.from_counters("TEST", [c])
        assert s.total_candidates == 100
        assert s.total_signals    == 5
        assert s.total_rejections == 95
        assert s.rule_totals["R1:rule"] == 80
        assert s.rule_totals["R2:rule"] == 15
        assert s.any_signals is True

    def test_from_counters_multi_asset_aggregates(self):
        c1 = self._make_counter("SPY", candidates=200, signals=3,
                                rejections={"R1:rule": 150, "R2:rule": 47})
        c2 = self._make_counter("QQQ", candidates=180, signals=2,
                                rejections={"R1:rule": 140, "R2:rule": 38})
        s = RejectionSummary.from_counters("TEST", [c1, c2])
        assert s.total_candidates  == 380
        assert s.total_signals     == 5
        assert s.rule_totals["R1:rule"] == 290
        assert s.rule_totals["R2:rule"] == 85

    def test_most_common_rejection_correct(self):
        c = self._make_counter("SPY", candidates=200,
                               rejections={"R1:dominant": 150, "R2:minor": 30})
        s = RejectionSummary.from_counters("TEST", [c])
        mc = s.most_common_rejection
        assert mc[0] == "R1:dominant"
        assert mc[1] == 150

    def test_rule_pct_correct(self):
        c = self._make_counter("SPY", candidates=100,
                               rejections={"R1:rule": 50})
        s = RejectionSummary.from_counters("TEST", [c])
        assert abs(s.rule_pct["R1:rule"] - 50.0) < 1e-9

    def test_first_signal_asset_identified(self):
        c1 = self._make_counter("SPY", candidates=100, signals=0)
        c2 = self._make_counter("QQQ", candidates=100)
        # manually add signal
        c2.record_signal(200)
        s = RejectionSummary.from_counters("TEST", [c1, c2])
        assert s.first_signal_asset == "QQQ"
        assert s.first_signal_bar   == 200

    def test_first_signal_earliest_across_assets(self):
        c1 = self._make_counter("SPY", candidates=100)
        c2 = self._make_counter("QQQ", candidates=100)
        c1.record_signal(300)
        c2.record_signal(150)   # earlier
        s = RejectionSummary.from_counters("TEST", [c1, c2])
        assert s.first_signal_asset == "QQQ"
        assert s.first_signal_bar   == 150

    def test_no_signals_any_signals_false(self):
        c1 = self._make_counter("SPY", candidates=100, signals=0)
        c2 = self._make_counter("QQQ", candidates=100, signals=0)
        s = RejectionSummary.from_counters("TEST", [c1, c2])
        assert s.any_signals is False
        assert s.first_signal_bar is None

    def test_set_conclusion(self):
        c = self._make_counter("SPY", candidates=100)
        s = RejectionSummary.from_counters("TEST", [c])
        s.set_conclusion("NO_OPPORTUNITIES")
        assert "NO_OPPORTUNITIES" in s.conclusion

    def test_rule_pct_zero_candidates(self):
        s = RejectionSummary.from_counters("TEST", [])
        assert s.rule_pct == {}
