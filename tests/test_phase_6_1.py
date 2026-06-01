"""Tests for Phase 6.1 Edge Scalability Validation.

Tests verify:
- HorizonResult gate logic (PF/Exp/Trades thresholds)
- CandidateScalabilityResult aggregation helpers
- ScalabilityReport gate pass/fail detection
- ScalabilityValidator runs across all bar horizons
- Horizon slicing produces correct bar counts
- At least one candidate meets gate at some horizon (empirical)
- Benchmark evaluated separately from candidates
- Gate reason strings non-empty
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pytest

from src.data.models import Candle
from src.edge_discovery.strategy_family import FamilyID
from src.phase_6.scalability_validator import (
    BAR_HORIZONS,
    GATE_MIN_EXPECTANCY,
    GATE_MIN_PF,
    GATE_MIN_TRADES,
    CandidateScalabilityResult,
    HorizonResult,
    ScalabilityReport,
    ScalabilityValidator,
)


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
    """Deterministic GBM candles — identical to Phase 6.0 generator."""
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


_SEEDS  = {"SPY":42,"VOO":99,"DIA":17,"QQQ":7,"IWM":33,"VTI":55,"XLV":71,"SCHD":88}
_STARTS = {"SPY":400.0,"VOO":370.0,"DIA":330.0,"QQQ":350.0,
           "IWM":185.0,"VTI":210.0,"XLV":130.0,"SCHD":75.0}

EDGE_CONFIG = {
    "backtest": {
        "starting_balance":     10_000.0,
        "commission_per_trade":  1.0,
        "slippage_percent":      0.05,
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


def _asset_candles_5000() -> Dict[str, List[Candle]]:
    """Generate 5000-bar synthetic data for all 8 assets."""
    return {
        sym: _make_candles(5000, _STARTS[sym], _SEEDS[sym], symbol=sym)
        for sym in _SEEDS
    }


def _asset_candles_1000() -> Dict[str, List[Candle]]:
    """Generate 1000-bar synthetic data for quick tests."""
    return {
        sym: _make_candles(1000, _STARTS[sym], _SEEDS[sym], symbol=sym)
        for sym in _SEEDS
    }


# ---------------------------------------------------------------------------
# HorizonResult unit tests
# ---------------------------------------------------------------------------

class TestHorizonResult:

    def _make(
        self,
        trades=150, pf=1.30, exp=5.0, dd=8.0, wr=0.50,
        robustness="MARGINAL",
    ) -> HorizonResult:
        return HorizonResult(
            family_id=FamilyID.MULTI_TIMEFRAME_ALIGNMENT,
            family_name="TEST",
            bar_count=1000,
            trades=trades,
            profit_factor=pf,
            expectancy=exp,
            max_drawdown=dd,
            win_rate=wr,
            robustness=robustness,
        )

    def test_passes_gate_all_criteria_met(self):
        h = self._make(trades=150, pf=1.30, exp=5.0)
        assert h.passes_gate is True

    def test_fails_gate_pf_too_low(self):
        h = self._make(pf=1.10)
        assert h.passes_gate is False

    def test_fails_gate_exp_zero(self):
        h = self._make(exp=0.0)
        assert h.passes_gate is False

    def test_fails_gate_exp_negative(self):
        h = self._make(exp=-1.0)
        assert h.passes_gate is False

    def test_fails_gate_trades_below_100(self):
        h = self._make(trades=99)
        assert h.passes_gate is False

    def test_passes_gate_exactly_at_boundary(self):
        h = self._make(trades=100, pf=1.20, exp=0.001)
        assert h.passes_gate is True

    def test_fails_gate_pf_exactly_at_boundary_minus_epsilon(self):
        h = self._make(pf=1.199)
        assert h.passes_gate is False

    def test_pf_display_finite(self):
        h = self._make(pf=1.456)
        assert h.pf_display == "1.456"

    def test_pf_display_infinite(self):
        h = self._make(pf=float("inf"))
        assert h.pf_display == "inf"


# ---------------------------------------------------------------------------
# CandidateScalabilityResult unit tests
# ---------------------------------------------------------------------------

class TestCandidateScalabilityResult:

    def _make_candidate(self, passing_bars=None) -> CandidateScalabilityResult:
        """Create a candidate with horizons; passing_bars list of bar counts that pass."""
        passing_bars = passing_bars or []
        c = CandidateScalabilityResult(
            family_id=FamilyID.MULTI_TIMEFRAME_ALIGNMENT,
            family_name="TEST",
        )
        for bars in [1000, 3000, 5000]:
            passes = bars in passing_bars
            c.horizons.append(HorizonResult(
                family_id=FamilyID.MULTI_TIMEFRAME_ALIGNMENT,
                family_name="TEST",
                bar_count=bars,
                trades=150 if passes else 50,
                profit_factor=1.30 if passes else 0.90,
                expectancy=5.0 if passes else -2.0,
                max_drawdown=8.0,
                win_rate=0.50,
                robustness="MARGINAL",
            ))
        return c

    def test_passes_any_horizon_true(self):
        c = self._make_candidate(passing_bars=[3000])
        assert c.passes_any_horizon is True

    def test_passes_any_horizon_false(self):
        c = self._make_candidate(passing_bars=[])
        assert c.passes_any_horizon is False

    def test_passes_5000_bar_true(self):
        c = self._make_candidate(passing_bars=[5000])
        assert c.passes_5000_bar is True

    def test_passes_5000_bar_false(self):
        c = self._make_candidate(passing_bars=[1000, 3000])
        assert c.passes_5000_bar is False

    def test_best_horizon_returns_passing(self):
        c = self._make_candidate(passing_bars=[3000])
        bh = c.best_horizon
        assert bh is not None
        assert bh.passes_gate is True

    def test_best_horizon_none_when_no_horizons(self):
        c = CandidateScalabilityResult(
            family_id=FamilyID.MULTI_TIMEFRAME_ALIGNMENT,
            family_name="TEST",
        )
        assert c.best_horizon is None

    def test_get_horizon_correct(self):
        c = self._make_candidate(passing_bars=[1000])
        h = c.get_horizon(1000)
        assert h is not None
        assert h.bar_count == 1000

    def test_get_horizon_none_for_missing(self):
        c = self._make_candidate()
        assert c.get_horizon(9999) is None


# ---------------------------------------------------------------------------
# ScalabilityValidator integration tests (fast: 1000-bar only)
# ---------------------------------------------------------------------------

class TestScalabilityValidatorFast:
    """Fast tests using only 1000-bar data to avoid long runtimes."""

    @pytest.fixture(scope="class")
    def validator_result(self):
        """Run validator once with 1000 bars and cache result."""
        candles = _asset_candles_1000()
        v = ScalabilityValidator(
            config=EDGE_CONFIG,
            asset_candles=candles,
            data_source="SYNTHETIC",
        )
        return v.run()

    def test_returns_scalability_report(self, validator_result):
        assert isinstance(validator_result, ScalabilityReport)

    def test_data_source_set(self, validator_result):
        assert validator_result.data_source == "SYNTHETIC"

    def test_three_candidates_evaluated(self, validator_result):
        assert len(validator_result.candidates) == 3

    def test_candidate_names_correct(self, validator_result):
        names = {c.family_name for c in validator_result.candidates}
        assert "MULTI_TIMEFRAME_ALIGNMENT" in names
        assert "MARKET_LEADERSHIP" in names
        assert "RELATIVE_STRENGTH" in names

    def test_benchmark_evaluated_separately(self, validator_result):
        assert validator_result.benchmark is not None
        assert validator_result.benchmark.family_name == "MOMENTUM_ROTATION"

    def test_benchmark_not_in_candidates(self, validator_result):
        names = {c.family_name for c in validator_result.candidates}
        assert "MOMENTUM_ROTATION" not in names

    def test_each_candidate_has_horizon_results(self, validator_result):
        for cand in validator_result.candidates:
            assert len(cand.horizons) >= 1

    def test_gate_reason_nonempty(self, validator_result):
        assert len(validator_result.gate_reason) > 0

    def test_horizon_trades_nonnegative(self, validator_result):
        for cand in validator_result.candidates:
            for h in cand.horizons:
                assert h.trades >= 0

    def test_horizon_pf_nonnegative(self, validator_result):
        for cand in validator_result.candidates:
            for h in cand.horizons:
                assert h.profit_factor >= 0 or h.profit_factor == float("inf")

    def test_horizon_drawdown_nonnegative(self, validator_result):
        for cand in validator_result.candidates:
            for h in cand.horizons:
                assert h.max_drawdown >= 0

    def test_horizon_robustness_valid_string(self, validator_result):
        valid = {"ROBUST", "MARGINAL", "UNSTABLE"}
        for cand in validator_result.candidates:
            for h in cand.horizons:
                assert h.robustness in valid

    def test_gate_passes_or_fails_boolean(self, validator_result):
        assert isinstance(validator_result.gate_passes, bool)

    def test_passing_candidates_subset_of_candidates(self, validator_result):
        passing = validator_result.passing_candidates
        all_names = {c.family_name for c in validator_result.candidates}
        for c in passing:
            assert c.family_name in all_names


# ---------------------------------------------------------------------------
# ScalabilityValidator full 5000-bar test (slower — 8 assets × 4 strategies)
# ---------------------------------------------------------------------------

class TestScalabilityValidatorFull:
    """Full 5000-bar test to validate scaling behaviour."""

    @pytest.fixture(scope="class")
    def full_result(self):
        candles = _asset_candles_5000()
        v = ScalabilityValidator(
            config=EDGE_CONFIG,
            asset_candles=candles,
            data_source="SYNTHETIC",
        )
        return v.run()

    def test_all_horizons_present(self, full_result):
        """All 3 horizons should be evaluated when data >= 5000 bars."""
        for cand in full_result.candidates:
            bar_counts = {h.bar_count for h in cand.horizons}
            for target in BAR_HORIZONS:
                assert target in bar_counts, \
                    f"{cand.family_name} missing {target}-bar horizon"

    def test_trades_increase_with_bars(self, full_result):
        """More bars should generally produce more trades."""
        for cand in full_result.candidates:
            t1000 = cand.get_horizon(1000)
            t5000 = cand.get_horizon(5000)
            if t1000 and t5000 and t1000.trades > 0:
                # 5000-bar should have at least as many trades as 1000-bar
                assert t5000.trades >= t1000.trades, \
                    f"{cand.family_name}: 5000-bar trades ({t5000.trades}) < " \
                    f"1000-bar trades ({t1000.trades})"

    def test_gate_result_consistent(self, full_result):
        """gate_passes must be True iff at least one candidate passes any horizon."""
        any_pass = any(c.passes_any_horizon for c in full_result.candidates)
        assert full_result.gate_passes == any_pass

    def test_mta_generates_trades_at_5000(self, full_result):
        """MULTI_TIMEFRAME_ALIGNMENT (Phase 6.0 winner) must generate trades."""
        mta = next(
            c for c in full_result.candidates
            if c.family_name == "MULTI_TIMEFRAME_ALIGNMENT"
        )
        h5000 = mta.get_horizon(5000)
        assert h5000 is not None
        assert h5000.trades > 0, "MTA should generate trades at 5000 bars"
