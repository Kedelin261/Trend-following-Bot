"""Tests for TradeSimulator — entry, exit, slippage, commission, PnL."""

import pytest
from datetime import datetime, timedelta, timezone

from src.backtest.models import ClosingReason
from src.backtest.trade_simulator import ActiveTrade, TradeSimulator
from src.data.models import Candle
from src.risk.models import TradeCandidate
from src.signals.models import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    i: int = 0,
    symbol: str = "SPY",
) -> Candle:
    return Candle(
        symbol=symbol, timeframe="D1",
        timestamp=BASE_TIME + timedelta(days=i),
        open=open_, high=high, low=low, close=close,
        volume=1e8, provider="TEST",
    )


def _candidate(
    symbol: str = "SPY",
    signal_type: SignalType = SignalType.LONG,
    stop: float = 90.0,
    target: float = 115.0,
    size: int = 10,
    entry_hint: float = 100.0,
) -> TradeCandidate:
    return TradeCandidate(
        symbol=symbol, timeframe="D1",
        signal_type=signal_type,
        entry_price=entry_hint,
        atr=5.0,
        stop_loss=stop,
        take_profit=target,
        risk_per_share=abs(entry_hint - stop),
        reward_per_share=abs(target - entry_hint),
        risk_reward_ratio=1.5,
        position_size=size,
        dollar_risk=100.0,
        signal_score=85.0,
        approved=True,
    )


@pytest.fixture
def sim() -> TradeSimulator:
    return TradeSimulator(slippage_percent=0.0, commission_per_trade=0.0)


@pytest.fixture
def sim_fees() -> TradeSimulator:
    return TradeSimulator(slippage_percent=0.1, commission_per_trade=1.0)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

class TestEnterTrade:

    def test_entry_at_bar_open(self, sim):
        bar = _bar(open_=100.0, high=102.0, low=98.0, close=101.0)
        trade = sim.enter_trade(_candidate(), bar, 0)
        assert trade.entry_price == pytest.approx(100.0)

    def test_long_entry_slippage_adds_to_price(self):
        s = TradeSimulator(slippage_percent=0.1, commission_per_trade=0.0)
        bar = _bar(open_=100.0, high=102.0, low=98.0, close=101.0)
        trade = s.enter_trade(_candidate(signal_type=SignalType.LONG), bar, 0)
        assert trade.entry_price == pytest.approx(100.10)

    def test_short_entry_slippage_subtracts_from_price(self):
        s = TradeSimulator(slippage_percent=0.1, commission_per_trade=0.0)
        bar = _bar(open_=100.0, high=102.0, low=98.0, close=101.0)
        trade = s.enter_trade(_candidate(signal_type=SignalType.SHORT, stop=110.0, target=85.0), bar, 0)
        assert trade.entry_price == pytest.approx(99.90)

    def test_active_trade_fields_set(self, sim):
        bar = _bar(open_=100.0, high=102.0, low=98.0, close=101.0, i=5)
        trade = sim.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 5)
        assert trade.symbol         == "SPY"
        assert trade.signal_type    == SignalType.LONG
        assert trade.stop_price     == 90.0
        assert trade.target_price   == 115.0
        assert trade.position_size  == 10
        assert trade.entry_bar_index == 5
        assert trade.entry_time     == bar.timestamp


# ---------------------------------------------------------------------------
# LONG exit logic
# ---------------------------------------------------------------------------

class TestLongExitLogic:

    def _open_trade(self, sim, stop=90.0, target=115.0, size=10) -> ActiveTrade:
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        return sim.enter_trade(_candidate(stop=stop, target=target, size=size), bar, 0)

    def test_long_stop_hit_when_low_lte_stop(self, sim):
        trade = self._open_trade(sim)
        bar = _bar(open_=98.0, high=100.0, low=89.0, close=91.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result is not None
        assert result.reason_closed == ClosingReason.STOP

    def test_long_stop_exit_price_at_stop_level(self, sim):
        trade = self._open_trade(sim, stop=90.0)
        bar = _bar(open_=98.0, high=100.0, low=89.0, close=91.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result.exit_price == pytest.approx(90.0)

    def test_long_target_hit_when_high_gte_target(self, sim):
        trade = self._open_trade(sim)
        bar = _bar(open_=112.0, high=116.0, low=111.0, close=114.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result is not None
        assert result.reason_closed == ClosingReason.TARGET

    def test_long_target_exit_price_at_target_level(self, sim):
        trade = self._open_trade(sim, target=115.0)
        bar = _bar(open_=112.0, high=116.0, low=111.0, close=114.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result.exit_price == pytest.approx(115.0)

    def test_long_both_hit_stop_wins(self, sim):
        """Conservative: if both stop and target breached same bar, stop wins."""
        trade = self._open_trade(sim, stop=90.0, target=115.0)
        bar = _bar(open_=100.0, high=120.0, low=85.0, close=100.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result.reason_closed == ClosingReason.STOP

    def test_long_neither_hit_returns_none(self, sim):
        trade = self._open_trade(sim)
        bar = _bar(open_=98.0, high=105.0, low=95.0, close=101.0, i=1)
        assert sim.check_exit(trade, bar, 1) is None

    def test_long_exactly_at_stop_is_hit(self, sim):
        trade = self._open_trade(sim, stop=90.0)
        bar = _bar(open_=95.0, high=98.0, low=90.0, close=93.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result is not None
        assert result.reason_closed == ClosingReason.STOP


# ---------------------------------------------------------------------------
# SHORT exit logic
# ---------------------------------------------------------------------------

class TestShortExitLogic:

    def _open_short(self, sim, stop=110.0, target=85.0, size=10) -> ActiveTrade:
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        return sim.enter_trade(
            _candidate(signal_type=SignalType.SHORT, stop=stop, target=target, size=size),
            bar, 0,
        )

    def test_short_stop_hit_when_high_gte_stop(self, sim):
        trade = self._open_short(sim, stop=110.0)
        bar = _bar(open_=108.0, high=112.0, low=106.0, close=110.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result is not None
        assert result.reason_closed == ClosingReason.STOP

    def test_short_target_hit_when_low_lte_target(self, sim):
        trade = self._open_short(sim, target=85.0)
        bar = _bar(open_=88.0, high=90.0, low=83.0, close=86.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result is not None
        assert result.reason_closed == ClosingReason.TARGET

    def test_short_both_hit_stop_wins(self, sim):
        trade = self._open_short(sim, stop=110.0, target=85.0)
        bar = _bar(open_=100.0, high=115.0, low=80.0, close=100.0, i=1)
        result = sim.check_exit(trade, bar, 1)
        assert result.reason_closed == ClosingReason.STOP

    def test_short_neither_hit_returns_none(self, sim):
        trade = self._open_short(sim)
        bar = _bar(open_=99.0, high=108.0, low=95.0, close=100.0, i=1)
        assert sim.check_exit(trade, bar, 1) is None


# ---------------------------------------------------------------------------
# PnL calculation
# ---------------------------------------------------------------------------

class TestPnLCalculation:

    def test_long_win_pnl(self, sim):
        # Buy 10 @ 100, sell @ 115 = $150 gross; no commission/slippage
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = sim.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 0)
        exit_bar = _bar(112.0, 116.0, 111.0, 114.0, i=1)
        result = sim.check_exit(trade, exit_bar, 1)
        assert result.pnl == pytest.approx(150.0)   # (115-100) × 10

    def test_long_loss_pnl(self, sim):
        # Buy 10 @ 100, stop @ 90 = -$100
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = sim.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 0)
        exit_bar = _bar(95.0, 95.0, 88.0, 90.0, i=1)
        result = sim.check_exit(trade, exit_bar, 1)
        assert result.pnl == pytest.approx(-100.0)  # (90-100) × 10

    def test_commission_deducted_round_trip(self):
        s = TradeSimulator(slippage_percent=0.0, commission_per_trade=5.0)
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = s.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 0)
        exit_bar = _bar(112.0, 116.0, 111.0, 114.0, i=1)
        result = s.check_exit(trade, exit_bar, 1)
        # Gross = 150, commission = 2 × 5 = 10
        assert result.pnl == pytest.approx(140.0)

    def test_slippage_reduces_long_pnl(self):
        s = TradeSimulator(slippage_percent=0.1, commission_per_trade=0.0)
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = s.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 0)
        exit_bar = _bar(112.0, 116.0, 111.0, 114.0, i=1)
        result = s.check_exit(trade, exit_bar, 1)
        # Entry at 100.10, exit at 114.885 (115 × 0.999)
        # Gross = (114.885 - 100.10) × 10 = 147.85
        assert result.pnl < 150.0   # slippage reduces vs no-slippage

    def test_holding_period_calculated(self, sim):
        bar = _bar(100.0, 102.0, 98.0, 101.0, i=3)
        trade = sim.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 3)
        exit_bar = _bar(112.0, 116.0, 111.0, 114.0, i=8)
        result = sim.check_exit(trade, exit_bar, 8)
        assert result.holding_period == 5

    def test_win_loss_flag_correct(self, sim):
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = sim.enter_trade(_candidate(stop=90.0, target=115.0, size=10), bar, 0)
        exit_bar = _bar(112.0, 116.0, 111.0, 114.0, i=1)
        result = sim.check_exit(trade, exit_bar, 1)
        assert result.win_loss == "WIN"


# ---------------------------------------------------------------------------
# Force close
# ---------------------------------------------------------------------------

class TestForceClose:

    def test_force_close_at_bar_close(self, sim):
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = sim.enter_trade(_candidate(), bar, 0)
        last_bar = _bar(105.0, 107.0, 103.0, 106.0, i=10)
        result = sim.force_close(trade, last_bar, 10)
        assert result.reason_closed == ClosingReason.TIME_EXIT
        assert result.exit_price == pytest.approx(106.0)  # close price

    def test_force_close_pnl_positive_when_profitable(self, sim):
        bar = _bar(100.0, 102.0, 98.0, 101.0)
        trade = sim.enter_trade(_candidate(size=10), bar, 0)
        last_bar = _bar(110.0, 112.0, 108.0, 110.0, i=5)
        result = sim.force_close(trade, last_bar, 5)
        assert result.pnl == pytest.approx(100.0)  # (110-100)×10
