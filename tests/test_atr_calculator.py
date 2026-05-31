"""Tests for ATRCalculator — True Range and Wilder's ATR."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.data.models import Candle
from src.risk.atr_calculator import ATRCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candle(
    close: float,
    high: float = None,
    low: float = None,
    i: int = 0,
    symbol: str = "SPY",
) -> Candle:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    h = high if high is not None else close * 1.01
    l = low  if low  is not None else close * 0.99
    return Candle(
        symbol=symbol, timeframe="D1",
        timestamp=base + timedelta(days=i),
        open=close, high=h, low=l, close=close,
        volume=1e8, provider="TEST",
    )


def flat_candles(
    n: int,
    price: float = 100.0,
    amplitude: float = 5.0,
) -> List[Candle]:
    """Candles with constant price (no gap) and fixed H-L range.

    TR for every bar (starting bar 2) = 2 × amplitude (high-low dominates).
    """
    return [
        _candle(price, high=price + amplitude, low=price - amplitude, i=i)
        for i in range(n)
    ]


def step_candles(
    n_low: int = 14,
    n_high: int = 5,
    low_tr: float = 2.0,
    high_tr: float = 6.0,
    base: float = 100.0,
) -> List[Candle]:
    """First n_low bars have TR=low_tr, then n_high bars have TR=high_tr."""
    candles = []
    for i in range(n_low + n_high):
        amp = low_tr / 2 if i < n_low else high_tr / 2
        candles.append(_candle(base, high=base + amp, low=base - amp, i=i))
    return candles


# ---------------------------------------------------------------------------
# calculate_true_range (single bar)
# ---------------------------------------------------------------------------

class TestCalculateTrueRange:

    def test_high_low_dominates_no_gap(self):
        atr = ATRCalculator()
        # H=105, L=95, prev_C=100 → max(10, 5, 5) = 10
        tr = atr.calculate_true_range(high=105.0, low=95.0, prev_close=100.0)
        assert tr == pytest.approx(10.0)

    def test_gap_up_makes_high_prev_close_dominant(self):
        atr = ATRCalculator()
        # H=115, L=110, prev_C=100 → max(5, 15, 10) = 15
        tr = atr.calculate_true_range(high=115.0, low=110.0, prev_close=100.0)
        assert tr == pytest.approx(15.0)

    def test_gap_down_makes_low_prev_close_dominant(self):
        atr = ATRCalculator()
        # H=92, L=88, prev_C=100 → max(4, 8, 12) = 12
        tr = atr.calculate_true_range(high=92.0, low=88.0, prev_close=100.0)
        assert tr == pytest.approx(12.0)

    def test_all_equal_returns_zero(self):
        atr = ATRCalculator()
        tr = atr.calculate_true_range(high=100.0, low=100.0, prev_close=100.0)
        assert tr == pytest.approx(0.0)

    def test_result_always_non_negative(self):
        atr = ATRCalculator()
        for h, l, pc in [(110, 90, 100), (95, 90, 100), (110, 105, 100)]:
            assert atr.calculate_true_range(h, l, pc) >= 0.0

    def test_high_low_large_bar(self):
        atr = ATRCalculator()
        # Huge intraday range dominates
        tr = atr.calculate_true_range(high=200.0, low=100.0, prev_close=150.0)
        assert tr == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# calculate_true_range_series
# ---------------------------------------------------------------------------

class TestCalculateTrueRangeSeries:

    def test_series_length_is_n_minus_1(self):
        atr = ATRCalculator()
        candles = flat_candles(n=10)
        trs = atr.calculate_true_range_series(candles)
        assert len(trs) == 9

    def test_returns_empty_for_single_candle(self):
        atr = ATRCalculator()
        assert atr.calculate_true_range_series([_candle(100.0)]) == []

    def test_returns_empty_for_empty_list(self):
        atr = ATRCalculator()
        assert atr.calculate_true_range_series([]) == []

    def test_flat_series_all_trs_equal_hl(self):
        atr = ATRCalculator()
        candles = flat_candles(n=10, amplitude=5.0)
        trs = atr.calculate_true_range_series(candles)
        for tr in trs:
            assert tr == pytest.approx(10.0)

    def test_all_values_non_negative(self):
        atr = ATRCalculator()
        candles = flat_candles(n=20)
        for tr in atr.calculate_true_range_series(candles):
            assert tr >= 0.0

    def test_two_candles_returns_one_tr(self):
        atr = ATRCalculator()
        candles = flat_candles(n=2, amplitude=5.0)
        trs = atr.calculate_true_range_series(candles)
        assert len(trs) == 1
        assert trs[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# calculate_atr — edge cases
# ---------------------------------------------------------------------------

class TestCalculateATREdgeCases:

    def test_returns_none_when_too_few_candles(self):
        atr = ATRCalculator(period=14)
        # Need 15+ candles (14 TRs), have 14 candles = 13 TRs
        assert atr.calculate_atr(flat_candles(n=14)) is None

    def test_returns_none_for_empty_list(self):
        assert ATRCalculator().calculate_atr([]) is None

    def test_returns_none_for_single_candle(self):
        assert ATRCalculator().calculate_atr([_candle(100.0)]) is None

    def test_exactly_period_plus_one_candles_returns_value(self):
        # 15 candles = 14 TRs = exactly the seed period — should succeed
        atr = ATRCalculator(period=14)
        result = atr.calculate_atr(flat_candles(n=15, amplitude=5.0))
        assert result is not None
        assert result == pytest.approx(10.0, rel=1e-5)

    def test_returns_positive_value(self):
        atr = ATRCalculator(period=14)
        result = atr.calculate_atr(flat_candles(n=20, amplitude=5.0))
        assert result is not None
        assert result > 0.0


# ---------------------------------------------------------------------------
# calculate_atr — correctness
# ---------------------------------------------------------------------------

class TestCalculateATRCorrectness:

    def test_flat_series_atr_equals_hl_range(self):
        """When TR is constant, ATR converges to that constant."""
        atr = ATRCalculator(period=14)
        candles = flat_candles(n=30, amplitude=5.0)   # TR = 10 for all bars
        result = atr.calculate_atr(candles)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_wilder_seed_equals_sma_of_first_period_trs(self):
        """With exactly period+1 candles, ATR is the simple average of the TRs."""
        atr3 = ATRCalculator(period=3)
        # 4 candles → 3 TRs, all TR=10 → seed ATR = (10+10+10)/3 = 10
        candles = flat_candles(n=4, amplitude=5.0)
        result = atr3.calculate_atr(candles)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_wilder_smoothing_after_step_change(self):
        """Verify Wilder update: ATR = (prev × (n-1) + TR) / n."""
        # Period=3, seed TRs=[2,2,2] → ATR_seed=2.0
        # Then TR=8: ATR = (2×2 + 8) / 3 = 12/3 = 4.0
        atr3 = ATRCalculator(period=3)
        # Build candles: 4 flat (amplitude=1 → TR=2), then 1 wide (amplitude=4 → TR=8)
        flat = flat_candles(n=4, price=100.0, amplitude=1.0)
        wide = _candle(100.0, high=104.0, low=96.0, i=4)
        result = atr3.calculate_atr(flat + [wide])
        assert result == pytest.approx(4.0, rel=1e-6)

    def test_multiple_wilder_steps(self):
        """Verify two consecutive Wilder updates against hand-calculated values."""
        # Period=3, seed=[2,2,2] → ATR=2.0
        # TR=8 → ATR=(2×2+8)/3=4.0
        # TR=2 → ATR=(4×2+2)/3=10/3≈3.333
        atr3 = ATRCalculator(period=3)
        flat = flat_candles(n=4, price=100.0, amplitude=1.0)   # 3 seed TRs = 2
        wide  = _candle(100.0, high=104.0, low=96.0, i=4)      # TR=8
        flat2 = _candle(100.0, high=101.0, low=99.0, i=5)      # TR=2
        result = atr3.calculate_atr(flat + [wide, flat2])
        assert result == pytest.approx(10 / 3, rel=1e-6)

    def test_atr_rises_on_volatility_expansion(self):
        atr = ATRCalculator(period=14)
        quiet = flat_candles(n=20, amplitude=1.0)   # small range
        noisy = flat_candles(n=20, amplitude=10.0)  # large range
        assert atr.calculate_atr(noisy) > atr.calculate_atr(quiet)

    def test_custom_period_accepted(self):
        atr5 = ATRCalculator(period=5)
        result = atr5.calculate_atr(flat_candles(n=10, amplitude=3.0))
        assert result == pytest.approx(6.0, rel=1e-6)

    def test_longer_series_same_result_when_flat(self):
        """Flat series: extra bars should not change ATR."""
        atr = ATRCalculator(period=14)
        r20  = atr.calculate_atr(flat_candles(n=20, amplitude=5.0))
        r100 = atr.calculate_atr(flat_candles(n=100, amplitude=5.0))
        assert r20  == pytest.approx(10.0, rel=1e-6)
        assert r100 == pytest.approx(10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestATRCalculatorInit:

    def test_period_zero_raises(self):
        with pytest.raises(ValueError):
            ATRCalculator(period=0)

    def test_negative_period_raises(self):
        with pytest.raises(ValueError):
            ATRCalculator(period=-5)

    def test_period_one_accepted(self):
        atr = ATRCalculator(period=1)
        assert atr.period == 1
