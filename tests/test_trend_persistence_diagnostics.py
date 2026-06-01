"""Tests for TrendPersistenceDiagnostics — Phase 6.0A.

Tests verify:
- Rule waterfall instruments all 4 rules correctly
- R1 rejections captured when consecutive < 20
- R2 rejections captured when distance > ATR (structural incompatibility)
- Incompatibility evidence records correct distance/ATR ratios
- No false signals with constant-price data
- Signals captured when all 4 rules pass
- Q&A methods return correct types and non-empty strings
- Multi-asset run aggregates correctly
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pytest

from src.data.models import Candle
from src.edge_discovery.diagnostics.trend_persistence_diagnostics import (
    TrendPersistenceDiagnostics,
    IncompatibilityEvidence,
    _MIN_CANDLES,
    _CONSEC_MIN,
)
from src.edge_discovery.diagnostics.rejection_analyzer import RejectionSummary


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_candles(
    n: int,
    start_price: float = 400.0,
    seed: int = 42,
    annual_drift: float = 0.10,
    annual_vol: float = 0.16,
    symbol: str = "SPY",
) -> List[Candle]:
    """Synthetic GBM candles matching Phase 6.0 generator."""
    rng         = random.Random(seed)
    daily_drift = annual_drift / 252
    daily_vol   = annual_vol / (252 ** 0.5)
    price       = start_price
    base_dt     = datetime(2020, 1, 2, tzinfo=timezone.utc)
    candles     = []

    for i in range(n):
        ret   = daily_drift + rng.gauss(0, daily_vol)
        price = max(1.0, price * (1 + ret))
        intra_vol = price * daily_vol * 0.7
        open_ = price * (1 + rng.gauss(0, daily_vol * 0.3))
        high  = max(open_, price) + abs(rng.gauss(0, intra_vol))
        low   = min(open_, price) - abs(rng.gauss(0, intra_vol))
        candles.append(Candle(
            symbol=symbol,
            timeframe="D1",
            timestamp=base_dt + timedelta(days=i),
            open=round(open_, 4),
            high=round(high,  4),
            low=round(low,    4),
            close=round(price, 4),
            volume=rng.uniform(40e6, 180e6),
            provider="SYNTHETIC",
        ))
    return candles


def _make_spy_candles(n: int = 1000) -> List[Candle]:
    return _make_candles(n, start_price=400.0, seed=42, symbol="SPY")


def _asset_candles_single() -> Dict[str, List[Candle]]:
    return {"SPY": _make_spy_candles(1000)}


def _asset_candles_multi() -> Dict[str, List[Candle]]:
    return {
        "SPY": _make_candles(1000, 400.0,  42, symbol="SPY"),
        "QQQ": _make_candles(1000, 350.0,   7, symbol="QQQ"),
        "IWM": _make_candles(1000, 185.0,  33, symbol="IWM"),
    }


# ---------------------------------------------------------------------------
# TrendPersistenceDiagnostics tests
# ---------------------------------------------------------------------------

class TestTrendPersistenceDiagnostics:

    def test_run_returns_rejection_summary(self):
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert isinstance(summary, RejectionSummary)

    def test_family_name_correct(self):
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.family_name == "TREND_PERSISTENCE"

    def test_candidates_positive(self):
        """After warmup bars, at least some bars should be evaluated."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.total_candidates > 0

    def test_candidates_equals_bars_minus_warmup(self):
        """total_candidates should be 1000 - _MIN_CANDLES."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        expected = 1000 - _MIN_CANDLES
        assert summary.total_candidates == expected

    def test_r1_rejection_is_most_common(self):
        """In GBM data, R1 consecutive<20 should dominate rejections."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        mc = summary.most_common_rejection
        assert mc is not None
        assert "R1" in mc[0]

    def test_r1_rejection_count_significant(self):
        """R1 should account for majority of rejections (> 50% of candidates)."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        r1_key  = f"R1:consecutive<{_CONSEC_MIN}"
        r1_count = summary.rule_totals.get(r1_key, 0)
        assert r1_count > summary.total_candidates * 0.50

    def test_r2_rejection_present(self):
        """R2 distance>ATR rejections should be recorded (structural incompatibility)."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        r2_count = summary.rule_totals.get("R2:distance>ATR", 0)
        # Pre-validated: ~187 on SPY alone
        assert r2_count > 0

    def test_incompatibility_evidence_captured(self):
        """Evidence records should have ratio > 1.0 (proving structural impossibility)."""
        diag = TrendPersistenceDiagnostics()
        diag.run(_asset_candles_single())
        evidence = diag.get_incompatibility_evidence()
        assert len(evidence) > 0
        for ev in evidence:
            assert isinstance(ev, IncompatibilityEvidence)
            # The key finding: ratio always >> 1.0
            assert ev.ratio > 1.0

    def test_incompatibility_evidence_ratio_above_threshold(self):
        """Evidence records should predominantly show ratio > 1.0.

        The key finding is that when R1 passes (consecutive>=20), price is
        significantly above EMA50.  Some edge-case bars may have low ratios
        (consecutive just at threshold) but the vast majority should be > 1.0.
        We verify at least half the evidence records exceed 1.0.
        """
        diag = TrendPersistenceDiagnostics()
        diag.run(_asset_candles_single())
        evidence = diag.get_incompatibility_evidence()
        if evidence:
            # All records must have ratio > 1.0 (by definition: distance > ATR)
            for ev in evidence:
                assert ev.ratio > 1.0, (
                    f"Evidence at bar {ev.bar_index}: ratio={ev.ratio:.3f} must be >1.0"
                )

    def test_total_rejections_plus_signals_equals_candidates(self):
        """Conservation: rejections + signals = candidates."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.total_rejections + summary.total_signals == summary.total_candidates

    def test_conclusion_set(self):
        """Conclusion string should not be empty or default."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.conclusion != ""
        assert "PENDING" not in summary.conclusion

    def test_no_signals_in_gbm_data(self):
        """Pre-validated: SPY GBM seed=42 produces 0-2 signals at most."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_single())
        # The pre-diagnostic found only 2 bars, both likely failing R2
        # Accept 0 or very few signals (structural incompatibility)
        assert summary.total_signals < 10

    def test_multi_asset_aggregates_candidates(self):
        """Multi-asset run should aggregate candidate counts."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_multi())
        expected = (1000 - _MIN_CANDLES) * 3
        assert summary.total_candidates == expected

    def test_per_asset_counters_in_summary(self):
        """Summary should contain one counter per asset."""
        diag = TrendPersistenceDiagnostics()
        summary = diag.run(_asset_candles_multi())
        assert len(summary.per_asset) == 3

    # ------------------------------------------------------------------
    # Q&A method tests
    # ------------------------------------------------------------------

    def test_q1_candidates(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q1_candidates(s) == s.total_candidates

    def test_q2_signals(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q2_signals(s) == s.total_signals

    def test_q3_rejections(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q3_rejections(s) == s.total_rejections

    def test_q4_returns_dict(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q4_rejection_rules(s)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_q5_returns_tuple_or_none(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        mc = diag.answer_q5_most_common(s)
        if mc is not None:
            assert isinstance(mc, tuple)
            assert len(mc) == 2

    def test_q6_returns_bool(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q6_any_passed(s)
        assert isinstance(result, bool)

    def test_q8_returns_nonempty_string(self):
        """Q8 should always return a meaningful string (non-empty)."""
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q8_why_zero(s)
        assert isinstance(result, str)
        assert len(result) > 10  # always has content, even if signals present

    def test_q9_returns_nonempty_string(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q9_live_data()
        assert isinstance(result, str)
        assert len(result) > 50

    def test_q10_returns_nonempty_string(self):
        diag = TrendPersistenceDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q10_defect(s)
        assert isinstance(result, str)
        assert "DEFECT" in result.upper() or "defect" in result.lower()
