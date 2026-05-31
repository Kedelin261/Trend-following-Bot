"""Tests for BacktestEngine — orchestration, invariants, and broker-agnostic checks."""

import pathlib
import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults, StrategyHealth
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.risk.atr_calculator import ATRCalculator
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.risk.stop_loss_engine import StopLossEngine
from src.risk.take_profit_engine import TakeProfitEngine
from src.risk.trade_validator import TradeValidator
from src.signals.breakout_detector import BreakoutDetector
from src.signals.signal_engine import SignalEngine
from src.signals.support_resistance import SupportResistanceDetector
from src.signals.trend_detector import TrendDetector
from src.signals.volume_confirmation import VolumeConfirmation


BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candle(
    price: float,
    i: int,
    amplitude: float = 5.0,
    volume: float = 1e8,
    symbol: str = "SPY",
) -> Candle:
    return Candle(
        symbol=symbol, timeframe="D1",
        timestamp=BASE + timedelta(days=i),
        open=price, high=price + amplitude, low=price - amplitude,
        close=price, volume=volume, provider="TEST",
    )


def uptrend_candles(n: int = 100, start: float = 100.0, gain: float = 0.003) -> List[Candle]:
    """Rising price series designed to trigger signals with minimal engine."""
    candles = []
    price = start
    for i in range(n):
        candles.append(_candle(price, i))
        price *= (1.0 + gain)
    return candles


def flat_candles(n: int = 50, price: float = 100.0) -> List[Candle]:
    return [_candle(price, i) for i in range(n)]


def _minimal_engine(
    min_warmup: int = 12,
    account: float = 10_000.0,
) -> BacktestEngine:
    """Engine with very small periods so it works on 50–100 bar datasets."""
    profile = RiskProfile(
        account_size=account, cash_available=account, risk_per_trade_percent=1.0
    )
    signal_engine = SignalEngine(
        trend_detector=TrendDetector(fast_period=5, slow_period=10),
        sr_detector=SupportResistanceDetector(lookback=1),
        breakout_detector=BreakoutDetector(threshold=0.001),
        volume_confirmation=VolumeConfirmation(lookback=3),
        min_candles=12,
    )
    risk_engine = RiskEngine(
        risk_profile=profile,
        atr_calculator=ATRCalculator(period=3),
        stop_loss_engine=StopLossEngine(multiplier=2.0),
        take_profit_engine=TakeProfitEngine(multiplier=3.0),
        trade_validator=TradeValidator(
            minimum_signal_score=60.0, minimum_risk_reward=1.5
        ),
    )
    return BacktestEngine(
        signal_engine=signal_engine,
        risk_engine=risk_engine,
        portfolio=Portfolio(account),
        simulator=TradeSimulator(slippage_percent=0.0, commission_per_trade=0.0),
        min_warmup=min_warmup,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:

    def test_returns_backtest_results(self):
        engine = _minimal_engine()
        result = engine.run(flat_candles(n=50))
        assert isinstance(result, BacktestResults)

    def test_empty_candles_returns_results(self):
        engine = _minimal_engine()
        result = engine.run([])
        assert isinstance(result, BacktestResults)

    def test_fewer_than_warmup_candles_returns_results(self):
        engine = _minimal_engine(min_warmup=12)
        result = engine.run(flat_candles(n=5))
        assert isinstance(result, BacktestResults)
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# Accounting invariants (hold regardless of trade count)
# ---------------------------------------------------------------------------

class TestAccountingInvariants:

    def _run(self):
        engine = _minimal_engine()
        return engine.run(uptrend_candles(n=100))

    def test_total_trades_nonnegative(self):
        result = self._run()
        assert result.total_trades >= 0

    def test_total_trades_sum_of_wins_and_losses(self):
        result = self._run()
        assert result.total_trades == result.winning_trades + result.losing_trades

    def test_net_profit_equals_ending_minus_starting(self):
        result = self._run()
        assert result.net_profit == pytest.approx(
            result.ending_balance - result.starting_balance
        )

    def test_starting_balance_unchanged(self):
        engine = _minimal_engine(account=10_000)
        result = engine.run(flat_candles(n=50))
        assert result.starting_balance == 10_000.0

    def test_win_rate_in_valid_range(self):
        result = self._run()
        assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_starts_at_starting_balance(self):
        result = self._run()
        assert result.equity_curve[0][1] == pytest.approx(10_000.0)

    def test_equity_curve_length_is_trades_plus_one(self):
        result = self._run()
        assert len(result.equity_curve) == result.total_trades + 1

    def test_no_trades_means_no_profit_or_loss(self):
        engine = _minimal_engine(min_warmup=1000)
        result = engine.run(flat_candles(n=50))
        assert result.total_trades == 0
        assert result.net_profit == pytest.approx(0.0)
        assert result.ending_balance == pytest.approx(10_000.0)

    def test_profit_factor_nonnegative(self):
        result = self._run()
        assert result.profit_factor >= 0.0 or result.profit_factor == float("inf")

    def test_max_drawdown_in_valid_range(self):
        result = self._run()
        assert 0.0 <= result.max_drawdown <= 100.0


# ---------------------------------------------------------------------------
# No trades during warmup
# ---------------------------------------------------------------------------

class TestWarmupRespected:

    def test_no_trades_when_warmup_equals_candle_count(self):
        engine = _minimal_engine(min_warmup=50)
        result = engine.run(flat_candles(n=50))
        assert result.total_trades == 0

    def test_no_trades_when_warmup_exceeds_candle_count(self):
        engine = _minimal_engine(min_warmup=100)
        result = engine.run(flat_candles(n=50))
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# Strategy health evaluation
# ---------------------------------------------------------------------------

class TestStrategyHealth:

    def test_health_pass_conditions(self):
        from src.backtest.models import BacktestResults
        results = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=13_000,
            net_profit=3_000, total_trades=35,
            winning_trades=22, losing_trades=13,
            win_rate=0.628, profit_factor=2.1,
            expectancy=85.7, max_drawdown=8.5,
            sharpe_ratio=1.4,
            average_win=200.0, average_loss=90.0,
            largest_win=450.0, largest_loss=130.0,
            equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(results)
        assert health.passed is True
        assert health.expectancy_ok is True
        assert health.profit_factor_ok is True
        assert health.drawdown_ok is True
        assert health.min_trades_ok is True

    def test_health_fail_low_trade_count(self):
        from src.backtest.models import BacktestResults
        results = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=11_000,
            net_profit=1_000, total_trades=5,   # < 30
            winning_trades=3, losing_trades=2,
            win_rate=0.6, profit_factor=2.0,
            expectancy=50.0, max_drawdown=5.0,
            sharpe_ratio=1.2,
            average_win=200.0, average_loss=100.0,
            largest_win=300.0, largest_loss=120.0,
            equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(results)
        assert health.passed is False
        assert health.min_trades_ok is False

    def test_health_fail_negative_expectancy(self):
        from src.backtest.models import BacktestResults
        results = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=9_000,
            net_profit=-1_000, total_trades=40,
            winning_trades=15, losing_trades=25,
            win_rate=0.375, profit_factor=0.8,
            expectancy=-25.0, max_drawdown=12.0,
            sharpe_ratio=-0.5,
            average_win=100.0, average_loss=120.0,
            largest_win=200.0, largest_loss=180.0,
            equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(results)
        assert health.passed is False
        assert health.expectancy_ok is False

    def test_reasons_populated(self):
        from src.backtest.models import BacktestResults
        results = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=12_000,
            net_profit=2_000, total_trades=35,
            winning_trades=20, losing_trades=15,
            win_rate=0.571, profit_factor=1.8,
            expectancy=57.0, max_drawdown=10.0,
            sharpe_ratio=1.1,
            average_win=150.0, average_loss=95.0,
            largest_win=280.0, largest_loss=130.0,
            equity_curve=[], trades=[],
        )
        health = StrategyHealth.evaluate(results)
        assert len(health.reasons_passed) > 0


# ---------------------------------------------------------------------------
# from_config factory
# ---------------------------------------------------------------------------

class TestFromConfig:

    def test_builds_from_minimal_config(self):
        config = {
            "backtest": {"starting_balance": 5_000, "commission_per_trade": 0.5, "min_warmup": 50},
            "risk": {"risk_per_trade_percent": 0.5},
        }
        engine = BacktestEngine.from_config(config)
        assert isinstance(engine, BacktestEngine)

    def test_default_config_accepted(self):
        engine = BacktestEngine.from_config({})
        assert isinstance(engine, BacktestEngine)


# ---------------------------------------------------------------------------
# Broker-agnostic verification
# ---------------------------------------------------------------------------

class TestBrokerAgnostic:

    def test_no_broker_imports_in_backtest_modules(self):
        backtest_dir = pathlib.Path("src/backtest")
        for pyfile in backtest_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"  not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader" not in text, f"{pyfile} imports MetaTrader"

    def test_return_percent_property(self):
        from src.backtest.models import BacktestResults
        r = BacktestResults(
            symbol="SPY", timeframe="D1",
            starting_balance=10_000, ending_balance=12_000,
            net_profit=2_000, total_trades=10,
            winning_trades=7, losing_trades=3,
            win_rate=0.7, profit_factor=2.0,
            expectancy=200.0, max_drawdown=5.0,
            sharpe_ratio=1.5,
            average_win=300.0, average_loss=100.0,
            largest_win=500.0, largest_loss=120.0,
            equity_curve=[], trades=[],
        )
        assert r.return_percent == pytest.approx(20.0)
