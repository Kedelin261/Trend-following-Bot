"""Tests for BenchmarkEngine — buy-and-hold comparison."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.research.benchmark_engine import BenchmarkEngine, BenchmarkResult


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _candles(n=30, start=100.0, daily_change=0.01) -> List[Candle]:
    candles = []
    price = start
    for i in range(n):
        candles.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=BASE + timedelta(days=i),
            open=price, high=price * 1.005, low=price * 0.995,
            close=price, volume=1e8, provider="TEST",
        ))
        price *= (1.0 + daily_change)
    return candles


def _results(
    symbol="SPY",
    start_bal=10_000.0,
    end_bal=11_000.0,
    trades=20,
    wins=12,
    sharpe=1.2,
    dd=5.0,
) -> BacktestResults:
    return BacktestResults(
        symbol=symbol, timeframe="D1",
        starting_balance=start_bal, ending_balance=end_bal,
        net_profit=end_bal - start_bal,
        total_trades=trades, winning_trades=wins, losing_trades=trades - wins,
        win_rate=wins / trades,
        profit_factor=1.8, expectancy=50.0,
        max_drawdown=dd, sharpe_ratio=sharpe,
        average_win=100.0, average_loss=50.0,
        largest_win=200.0, largest_loss=80.0,
        equity_curve=[], trades=[],
    )


@pytest.fixture
def engine() -> BenchmarkEngine:
    return BenchmarkEngine()


class TestBenchmarkCompare:

    def test_returns_benchmark_result(self, engine):
        candles = _candles(n=30, daily_change=0.005)
        result = engine.compare(_results(), candles)
        assert isinstance(result, BenchmarkResult)

    def test_symbol_propagated(self, engine):
        candles = _candles(n=30)
        result = engine.compare(_results(symbol="QQQ"), candles)
        assert result.symbol == "QQQ"

    def test_buyhold_return_positive_for_rising_market(self, engine):
        candles = _candles(n=30, daily_change=0.01)  # rising market
        result = engine.compare(_results(), candles)
        assert result.buyhold_return_pct > 0.0

    def test_buyhold_return_negative_for_falling_market(self, engine):
        candles = _candles(n=30, daily_change=-0.01)  # falling market
        result = engine.compare(_results(), candles)
        assert result.buyhold_return_pct < 0.0

    def test_alpha_equals_strategy_minus_benchmark(self, engine):
        candles = _candles(n=30, daily_change=0.005)
        r = engine.compare(_results(start_bal=10_000, end_bal=11_000), candles)
        assert r.alpha_pct == pytest.approx(
            r.strategy_return_pct - r.buyhold_return_pct, rel=1e-4
        )

    def test_outperforms_true_when_strategy_beats_benchmark(self, engine):
        # Rising market +50% buy&hold; strategy returns +100%
        candles = _candles(n=30, daily_change=0.01)
        # Make strategy return very high
        r = engine.compare(_results(start_bal=10_000, end_bal=20_000), candles)
        assert r.strategy_outperforms is True

    def test_outperforms_false_when_strategy_lags_benchmark(self, engine):
        # Strong rising market; strategy barely returns anything
        candles = _candles(n=30, daily_change=0.02)
        r = engine.compare(_results(start_bal=10_000, end_bal=10_050), candles)
        # Buy&hold of 2%/day compounded over 30 days >> 0.5%
        assert r.strategy_outperforms is False

    def test_empty_candles_returns_result(self, engine):
        r = engine.compare(_results(), [])
        assert isinstance(r, BenchmarkResult)
        assert r.buyhold_return_pct == 0.0

    def test_notes_populated(self, engine):
        candles = _candles(n=30, daily_change=0.005)
        r = engine.compare(_results(), candles)
        assert len(r.notes) > 0

    def test_risk_adjusted_outperforms_when_sharpe_better(self, engine):
        candles = _candles(n=30, daily_change=0.001)  # low buy&hold return
        r = engine.compare(
            _results(start_bal=10_000, end_bal=12_000, sharpe=3.0), candles
        )
        if r.alpha_pct > 0 and r.strategy_sharpe > r.buyhold_sharpe:
            assert r.risk_adjusted_outperforms is True
