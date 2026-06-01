"""Tests for Phase 5.2 risk overlay base class and OverlayBacktestEngine."""

import random
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.backtest.models import BacktestTrade, ClosingReason
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.risk_overlay.base_overlay import RiskOverlay
from src.risk_overlay.overlay_backtester import OverlayBacktestEngine, _scale_candidate
from src.risk.models import TradeCandidate
from src.signals.models import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG = {
    "backtest": {"starting_balance": 10_000.0, "commission_per_trade": 1.0, "slippage_percent": 0.05},
    "risk":     {"risk_per_trade_percent": 1.0, "atr_period": 14, "atr_stop_multiplier": 2.0,
                 "atr_target_multiplier": 3.0, "minimum_signal_score": 40.0, "minimum_risk_reward": 1.0},
}


def _make_candles(symbol: str = "SPY", n: int = 300) -> List[Candle]:
    rng   = random.Random(42)
    price = 400.0
    base  = datetime(2020, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        ret   = 0.08/252 + rng.gauss(0, 0.18/(252**0.5))
        price = max(1.0, price*(1+ret))
        vol   = price*0.18/(252**0.5)*0.6
        open_ = price*(1+rng.gauss(0, 0.18/(252**0.5)*0.25))
        high  = max(open_, price)+abs(rng.gauss(0, vol))
        low   = min(open_, price)-abs(rng.gauss(0, vol))
        candles.append(Candle(
            symbol=symbol, timeframe="D1",
            timestamp=base+timedelta(days=i),
            open=round(open_,4), high=round(high,4),
            low=round(low,4), close=round(price,4),
            volume=rng.uniform(30e6,200e6), provider="SYNTHETIC",
        ))
    return candles


def _make_trade(pnl: float, exit_year: int = 2020, exit_month: int = 1) -> BacktestTrade:
    ts = datetime(exit_year, exit_month, 15, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol="SPY", entry_time=ts, exit_time=ts,
        signal_type=SignalType.LONG,
        entry_price=400.0, exit_price=410.0 if pnl > 0 else 390.0,
        stop_price=390.0, target_price=430.0,
        position_size=10, pnl=pnl, return_percent=pnl/4000,
        holding_period=5, win_loss="WIN" if pnl > 0 else "LOSS",
        reason_closed=ClosingReason.TARGET if pnl > 0 else ClosingReason.STOP,
    )


# ---------------------------------------------------------------------------
# Test _scale_candidate
# ---------------------------------------------------------------------------

class TestScaleCandidate:
    def _make_candidate(self, size: int = 10) -> TradeCandidate:
        return TradeCandidate(
            symbol="SPY", timeframe="D1", signal_type=SignalType.LONG,
            entry_price=400.0, atr=8.0, stop_loss=384.0, take_profit=424.0,
            risk_per_share=16.0, reward_per_share=24.0, risk_reward_ratio=1.5,
            position_size=size, dollar_risk=100.0, signal_score=60.0,
            approved=True, rejection_reason=None,
        )

    def test_full_scale(self):
        c = self._make_candidate(10)
        scaled = _scale_candidate(c, 1.0)
        assert scaled.position_size == 10

    def test_half_scale(self):
        c = self._make_candidate(10)
        scaled = _scale_candidate(c, 0.5)
        assert scaled.position_size == 5

    def test_quarter_scale(self):
        c = self._make_candidate(10)
        scaled = _scale_candidate(c, 0.25)
        assert scaled.position_size == 2   # floor(10*0.25)=2

    def test_zero_scale(self):
        c = self._make_candidate(10)
        scaled = _scale_candidate(c, 0.0)
        assert scaled.position_size == 0

    def test_original_unchanged(self):
        c = self._make_candidate(10)
        _scale_candidate(c, 0.5)
        assert c.position_size == 10   # original not mutated

    def test_other_fields_preserved(self):
        c = self._make_candidate(10)
        scaled = _scale_candidate(c, 0.5)
        assert scaled.entry_price  == c.entry_price
        assert scaled.stop_loss    == c.stop_loss
        assert scaled.take_profit  == c.take_profit
        assert scaled.approved     == c.approved


# ---------------------------------------------------------------------------
# Test OverlayBacktestEngine produces results
# ---------------------------------------------------------------------------

class TestOverlayBacktestEngine:
    def _build_engine(self, overlay=None) -> OverlayBacktestEngine:
        from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
        strategy = MomentumRotationStrategy()
        bt_cfg = _CONFIG["backtest"]
        return OverlayBacktestEngine(
            signal_engine = strategy,
            risk_engine   = RiskEngine.from_config(_CONFIG),
            portfolio     = Portfolio(10_000.0),
            simulator     = TradeSimulator(0.05, 1.0),
            overlay       = overlay,
            min_warmup    = strategy.min_candles,
        )

    def test_runs_without_overlay(self):
        candles = _make_candles(n=300)
        engine  = self._build_engine(overlay=None)
        result  = engine.run(candles)
        assert result is not None
        assert result.total_trades >= 0
        assert result.starting_balance == 10_000.0

    def test_runs_with_no_overlay_class(self):
        from src.risk_overlay.profiles.no_overlay import NoOverlay
        candles = _make_candles(n=300)
        engine  = self._build_engine(overlay=NoOverlay())
        result  = engine.run(candles)
        assert result is not None

    def test_returns_valid_metrics(self):
        candles = _make_candles(n=300)
        engine  = self._build_engine(overlay=None)
        result  = engine.run(candles)
        assert result.profit_factor  >= 0.0
        assert result.max_drawdown   >= 0.0
        assert result.win_rate       >= 0.0
        assert result.win_rate       <= 1.0

    def test_portfolio_reset_between_runs(self):
        candles = _make_candles(n=300)
        engine  = self._build_engine()
        r1 = engine.run(candles)
        r2 = engine.run(candles)
        # Results should be identical (deterministic)
        assert r1.total_trades == r2.total_trades
        assert abs(r1.profit_factor - r2.profit_factor) < 1e-6
