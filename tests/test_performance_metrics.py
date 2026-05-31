"""Tests for performance_metrics — verified against hand-calculated values."""

import math
import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.backtest import performance_metrics as pm
from src.signals.models import SignalType


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(pnl: float, i: int = 0) -> BacktestTrade:
    win = pnl > 0
    return BacktestTrade(
        symbol="SPY", entry_time=BASE + timedelta(days=i),
        exit_time=BASE + timedelta(days=i + 1),
        signal_type=SignalType.LONG,
        entry_price=100.0, exit_price=100.0 + pnl / 10,
        stop_price=90.0, target_price=115.0,
        position_size=10, pnl=pnl, return_percent=pnl / 10,
        holding_period=1, win_loss="WIN" if win else "LOSS",
        reason_closed=ClosingReason.TARGET if win else ClosingReason.STOP,
    )


def known_trades() -> List[BacktestTrade]:
    """5 trades: 3 wins (+100, +150, +80) and 2 losses (-50, -70).

    Hand-calculated expectations:
      wins = 3/5 = 60 %
      gross_wins = 330, gross_losses = 120
      profit_factor = 330/120 = 2.75
      avg_win = 110, avg_loss = 60
      expectancy = (0.6 × 110) − (0.4 × 60) = 66 − 24 = 42
      largest_win = 150, largest_loss = 70
    """
    return [
        _trade(100.0, i=0),
        _trade(150.0, i=1),
        _trade(-50.0, i=2),
        _trade(80.0,  i=3),
        _trade(-70.0, i=4),
    ]


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:

    def test_empty_trades_returns_zero(self):
        assert pm.win_rate([]) == 0.0

    def test_all_wins(self):
        trades = [_trade(100.0), _trade(200.0), _trade(50.0)]
        assert pm.win_rate(trades) == pytest.approx(1.0)

    def test_all_losses(self):
        trades = [_trade(-50.0), _trade(-30.0)]
        assert pm.win_rate(trades) == pytest.approx(0.0)

    def test_known_trades(self):
        assert pm.win_rate(known_trades()) == pytest.approx(0.60)

    def test_single_win(self):
        assert pm.win_rate([_trade(100.0)]) == pytest.approx(1.0)

    def test_single_loss(self):
        assert pm.win_rate([_trade(-50.0)]) == pytest.approx(0.0)

    def test_result_in_range(self):
        rate = pm.win_rate(known_trades())
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:

    def test_empty_trades_returns_zero(self):
        assert pm.profit_factor([]) == 0.0

    def test_no_losses_returns_infinity(self):
        trades = [_trade(100.0), _trade(200.0)]
        assert math.isinf(pm.profit_factor(trades))

    def test_no_wins_returns_zero(self):
        trades = [_trade(-50.0), _trade(-30.0)]
        assert pm.profit_factor(trades) == pytest.approx(0.0)

    def test_known_trades(self):
        # 330 / 120 = 2.75
        assert pm.profit_factor(known_trades()) == pytest.approx(2.75)

    def test_breakeven_is_one(self):
        trades = [_trade(100.0), _trade(-100.0)]
        assert pm.profit_factor(trades) == pytest.approx(1.0)

    def test_positive_for_profitable_strategy(self):
        trades = [_trade(100.0), _trade(-50.0)]
        assert pm.profit_factor(trades) > 1.0


# ---------------------------------------------------------------------------
# Expectancy
# ---------------------------------------------------------------------------

class TestExpectancy:

    def test_empty_trades_returns_zero(self):
        assert pm.expectancy([]) == 0.0

    def test_known_trades(self):
        # (0.6 × 110) − (0.4 × 60) = 66 − 24 = 42
        assert pm.expectancy(known_trades()) == pytest.approx(42.0)

    def test_negative_for_losing_strategy(self):
        trades = [_trade(10.0), _trade(-100.0)]
        assert pm.expectancy(trades) < 0.0

    def test_positive_for_winning_strategy(self):
        trades = [_trade(100.0), _trade(-20.0)]
        assert pm.expectancy(trades) > 0.0

    def test_all_wins(self):
        trades = [_trade(50.0), _trade(80.0)]
        # win_rate=1, loss_rate=0 → expectancy = avg_win
        assert pm.expectancy(trades) == pytest.approx(65.0)

    def test_formula_matches_manual(self):
        trades = [_trade(200.0), _trade(200.0), _trade(-50.0)]
        # win_rate = 2/3, avg_win = 200, avg_loss = 50
        # expectancy = (2/3 × 200) − (1/3 × 50) = 133.33 − 16.67 = 116.67
        assert pm.expectancy(trades) == pytest.approx(400 / 3 - 50 / 3, rel=1e-4)


# ---------------------------------------------------------------------------
# Average win / loss
# ---------------------------------------------------------------------------

class TestAverageWinLoss:

    def test_average_win_known(self):
        # (100 + 150 + 80) / 3 = 110
        assert pm.average_win(known_trades()) == pytest.approx(110.0)

    def test_average_loss_known(self):
        # (50 + 70) / 2 = 60
        assert pm.average_loss(known_trades()) == pytest.approx(60.0)

    def test_average_loss_is_positive(self):
        assert pm.average_loss([_trade(-50.0), _trade(-30.0)]) > 0.0

    def test_no_wins_returns_zero(self):
        assert pm.average_win([_trade(-50.0)]) == 0.0

    def test_no_losses_returns_zero(self):
        assert pm.average_loss([_trade(100.0)]) == 0.0


# ---------------------------------------------------------------------------
# Largest win / loss
# ---------------------------------------------------------------------------

class TestLargestWinLoss:

    def test_largest_win_known(self):
        assert pm.largest_win(known_trades()) == pytest.approx(150.0)

    def test_largest_loss_known(self):
        assert pm.largest_loss(known_trades()) == pytest.approx(70.0)

    def test_largest_loss_is_positive(self):
        assert pm.largest_loss([_trade(-50.0), _trade(-200.0)]) > 0.0

    def test_no_wins_returns_zero(self):
        assert pm.largest_win([_trade(-50.0)]) == 0.0

    def test_no_losses_returns_zero(self):
        assert pm.largest_loss([_trade(100.0)]) == 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_no_drawdown_returns_zero(self):
        # Monotonically rising equity
        assert pm.max_drawdown([10_000, 10_100, 10_200, 10_300]) == pytest.approx(0.0)

    def test_single_point_returns_zero(self):
        assert pm.max_drawdown([10_000]) == 0.0

    def test_known_drawdown(self):
        # Peak = 11000, trough = 9900 → DD = (11000 - 9900) / 11000 × 100 ≈ 10%
        equity = [10_000, 11_000, 9_900, 10_500]
        dd = pm.max_drawdown(equity)
        assert dd == pytest.approx(10.0, rel=0.01)

    def test_all_down_is_full_drawdown(self):
        # Peak = 10000, trough = 5000 → 50%
        dd = pm.max_drawdown([10_000, 8_000, 6_000, 5_000])
        assert dd == pytest.approx(50.0, rel=0.01)

    def test_recovery_does_not_reset_drawdown(self):
        # Peak=10000, goes to 8000 (20% DD), recovers to 11000 (new peak), then 9900
        equity = [10_000, 8_000, 11_000, 9_900]
        dd = pm.max_drawdown(equity)
        # Max DD from first drop: (10000-8000)/10000 = 20%
        assert dd == pytest.approx(20.0, rel=0.01)

    def test_empty_returns_zero(self):
        assert pm.max_drawdown([]) == 0.0


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:

    def test_insufficient_points_returns_zero(self):
        assert pm.sharpe_ratio([10_000, 10_100]) == pytest.approx(0.0)

    def test_zero_std_returns_zero(self):
        # Constant equity → returns all 0.0 → std = 0 → Sharpe = 0
        assert pm.sharpe_ratio([10_000, 10_000, 10_000, 10_000]) == pytest.approx(0.0)

    def test_positive_for_consistent_gains(self):
        # Alternating small returns with positive trend
        equity = [10_000, 10_100, 10_050, 10_200, 10_150, 10_300]
        assert pm.sharpe_ratio(equity) > 0.0

    def test_negative_for_consistent_losses(self):
        equity = [10_000, 9_900, 9_950, 9_800, 9_850, 9_700]
        assert pm.sharpe_ratio(equity) < 0.0

    def test_result_finite(self):
        equity = [10_000 + i * 10 + (i % 2) * (-5) for i in range(20)]
        result = pm.sharpe_ratio(equity)
        assert math.isfinite(result)
