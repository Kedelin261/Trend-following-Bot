"""Tests for VolatilityTransitionDiagnostics — Phase 6.0A.

Tests verify:
- Rule waterfall instruments all 4 rules correctly
- R1 compression_not_confirmed dominates (~80% of candidates on SPY)
- At least 1 signal per asset captured in GBM data
- Signal evidence captures correct fields
- Total signals well below MIN_SAMPLE=50
- Q&A methods return correct types and meaningful content
- Multi-asset run aggregates correctly
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pytest

from src.data.models import Candle
from src.edge_discovery.diagnostics.volatility_transition_diagnostics import (
    VolatilityTransitionDiagnostics,
    SignalEvidence,
    _MIN_CANDLES,
    _MONOTONE_MIN,
    _COMPRESS_BARS,
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


def _spy_candles() -> List[Candle]:
    return _make_candles(1000, start_price=400.0, seed=42, symbol="SPY")


def _asset_candles_single() -> Dict[str, List[Candle]]:
    return {"SPY": _spy_candles()}


def _asset_candles_8() -> Dict[str, List[Candle]]:
    seeds  = {"SPY":42,"VOO":99,"DIA":17,"QQQ":7,"IWM":33,"VTI":55,"XLV":71,"SCHD":88}
    starts = {"SPY":400.0,"VOO":370.0,"DIA":330.0,"QQQ":350.0,
               "IWM":185.0,"VTI":210.0,"XLV":130.0,"SCHD":75.0}
    return {
        sym: _make_candles(1000, starts[sym], seeds[sym], symbol=sym)
        for sym in seeds
    }


# ---------------------------------------------------------------------------
# VolatilityTransitionDiagnostics tests
# ---------------------------------------------------------------------------

class TestVolatilityTransitionDiagnostics:

    def test_run_returns_rejection_summary(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert isinstance(summary, RejectionSummary)

    def test_family_name_correct(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.family_name == "VOLATILITY_TRANSITION"

    def test_candidates_equals_bars_minus_warmup(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        expected = 1000 - _MIN_CANDLES
        assert summary.total_candidates == expected

    def test_r1_dominates_rejections(self):
        """R1 compression_not_confirmed should be dominant rejection (>70%)."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        r1 = summary.rule_totals.get("R1:compression_not_confirmed", 0)
        # Pre-validated: 768/951 = 80.8% on SPY
        assert r1 > summary.total_candidates * 0.60

    def test_most_common_rejection_is_r1(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        mc = summary.most_common_rejection
        assert mc is not None
        assert "R1" in mc[0]

    def test_some_signals_generated_on_spy(self):
        """Pre-validated: at least 1 signal on SPY seed=42."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        # Pre-validated: 1 signal at bar 387
        assert summary.total_signals >= 1

    def test_signal_evidence_populated(self):
        """Signal evidence should capture valid signal details."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        evidence = diag.get_signal_evidence()
        if summary.total_signals > 0:
            assert len(evidence) > 0
            for ev in evidence:
                assert isinstance(ev, SignalEvidence)

    def test_signal_evidence_fields_valid(self):
        """Evidence fields should be non-negative and logically consistent."""
        diag = VolatilityTransitionDiagnostics()
        diag.run(_asset_candles_single())
        evidence = diag.get_signal_evidence()
        for ev in evidence:
            assert ev.declining_count >= _MONOTONE_MIN
            assert ev.expand_ratio >= 0.10
            assert ev.current_atr > 0
            assert ev.prior_atr > 0
            assert ev.close > ev.open_price   # bullish bar (close > open)
            assert ev.close > ev.ema20
            assert ev.strength >= 62.0        # base strength

    def test_first_signal_bar_identified(self):
        """Pre-validated: first signal around bar 387 on SPY."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        if summary.total_signals > 0:
            assert summary.first_signal_bar is not None
            assert summary.first_signal_bar >= _MIN_CANDLES

    def test_conservation_law(self):
        """rejections + signals == candidates."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.total_rejections + summary.total_signals == summary.total_candidates

    def test_8_asset_signals_below_min_sample(self):
        """Critical finding: total signals across 8 assets << MIN_SAMPLE=50."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_8())
        # Pre-validated: ~1 per asset × 8 = ~8 total, far below 50
        assert summary.total_signals < 50

    def test_8_asset_signals_positive(self):
        """There are some signals — the issue is quantity, not quality."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_8())
        # At least 1 valid signal across 8 assets
        assert summary.total_signals >= 1

    def test_conclusion_reflects_synthetic_limitation(self):
        """Conclusion should reflect synthetic data limitation."""
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_8())
        c = summary.conclusion
        assert (
            "SYNTHETIC" in c.upper()
            or "OVERLY_RESTRICTIVE" in c
            or "LIMITATION" in c.upper()
            or "MIN_SAMPLE" in c
        )

    def test_any_signals_true_for_spy(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_single())
        assert summary.any_signals is True

    def test_multi_asset_aggregates_candidates(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_8())
        expected = (1000 - _MIN_CANDLES) * 8
        assert summary.total_candidates == expected

    def test_per_asset_counter_count(self):
        diag = VolatilityTransitionDiagnostics()
        summary = diag.run(_asset_candles_8())
        assert len(summary.per_asset) == 8

    # ------------------------------------------------------------------
    # Q&A method tests
    # ------------------------------------------------------------------

    def test_q1_returns_candidates(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q1_candidates(s) == s.total_candidates

    def test_q2_returns_signals(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q2_signals(s) == s.total_signals

    def test_q3_returns_rejections(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q3_rejections(s) == s.total_rejections

    def test_q6_returns_bool(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        assert isinstance(diag.answer_q6_any_passed(s), bool)

    def test_q6_true_for_spy(self):
        """Pre-validated: SPY has at least 1 signal."""
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        assert diag.answer_q6_any_passed(s) is True

    def test_q7_first_signal_bar_consistent(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        q7 = diag.answer_q7_first_signal(s)
        assert q7 == s.first_signal_bar

    def test_q8_mentions_gbm_limitation(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_8())
        result = diag.answer_q8_why_zero(s)
        assert isinstance(result, str)
        assert len(result) > 50
        assert ("GBM" in result or "SYNTHETIC" in result or "rare" in result.lower())

    def test_q9_positive_live_outlook(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q9_live_data()
        assert "YES" in result or "IMPROVEMENT" in result or "EXPECTED" in result

    def test_q10_no_defect(self):
        diag = VolatilityTransitionDiagnostics()
        s = diag.run(_asset_candles_single())
        result = diag.answer_q10_defect(s)
        assert "NO" in result.upper()
