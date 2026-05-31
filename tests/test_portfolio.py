"""Tests for Portfolio — balance tracking, trade history, equity curve."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from src.backtest.models import BacktestTrade, ClosingReason
from src.backtest.portfolio import Portfolio
from src.risk.models import RiskProfile
from src.signals.models import SignalType


BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _trade(
    pnl: float,
    i: int = 0,
    symbol: str = "SPY",
    signal_type: SignalType = SignalType.LONG,
    reason: ClosingReason = ClosingReason.TARGET,
) -> BacktestTrade:
    entry = BASE_TIME + timedelta(days=i)
    exit_ = entry + timedelta(days=1)
    win = pnl > 0
    return BacktestTrade(
        symbol=symbol,
        entry_time=entry,
        exit_time=exit_,
        signal_type=signal_type,
        entry_price=100.0,
        exit_price=100.0 + pnl / 10,
        stop_price=90.0,
        target_price=115.0,
        position_size=10,
        pnl=pnl,
        return_percent=pnl / 1000 * 100,
        holding_period=1,
        win_loss="WIN" if win else "LOSS",
        reason_closed=reason,
    )


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(starting_balance=10_000.0)


class TestPortfolioInit:

    def test_initial_balance(self, portfolio):
        assert portfolio.balance == 10_000.0

    def test_starting_balance_stored(self, portfolio):
        assert portfolio.starting_balance == 10_000.0

    def test_no_trades_initially(self, portfolio):
        assert portfolio.trades == []

    def test_equity_history_starts_with_opening_balance(self, portfolio):
        history = portfolio.equity_history
        assert len(history) == 1
        assert history[0][1] == 10_000.0

    def test_net_profit_zero_initially(self, portfolio):
        assert portfolio.net_profit == 0.0


class TestProcessTrade:

    def test_balance_increases_on_win(self, portfolio):
        portfolio.process_trade(_trade(pnl=100.0))
        assert portfolio.balance == pytest.approx(10_100.0)

    def test_balance_decreases_on_loss(self, portfolio):
        portfolio.process_trade(_trade(pnl=-50.0))
        assert portfolio.balance == pytest.approx(9_950.0)

    def test_multiple_trades_accumulate(self, portfolio):
        portfolio.process_trade(_trade(pnl=200.0, i=0))
        portfolio.process_trade(_trade(pnl=-80.0,  i=1))
        portfolio.process_trade(_trade(pnl=150.0, i=2))
        assert portfolio.balance == pytest.approx(10_270.0)

    def test_trade_added_to_history(self, portfolio):
        t = _trade(pnl=100.0)
        portfolio.process_trade(t)
        assert t in portfolio.trades

    def test_equity_history_grows(self, portfolio):
        portfolio.process_trade(_trade(pnl=100.0))
        portfolio.process_trade(_trade(pnl=50.0))
        assert len(portfolio.equity_history) == 3  # start + 2 trades

    def test_equity_values_match_balance_progression(self, portfolio):
        portfolio.process_trade(_trade(pnl=100.0))
        portfolio.process_trade(_trade(pnl=-30.0))
        vals = portfolio.equity_values
        assert vals == pytest.approx([10_000.0, 10_100.0, 10_070.0])

    def test_net_profit_after_trades(self, portfolio):
        portfolio.process_trade(_trade(pnl=300.0))
        portfolio.process_trade(_trade(pnl=-100.0))
        assert portfolio.net_profit == pytest.approx(200.0)


class TestReset:

    def test_reset_restores_starting_balance(self, portfolio):
        portfolio.process_trade(_trade(pnl=500.0))
        portfolio.reset()
        assert portfolio.balance == 10_000.0

    def test_reset_clears_trade_history(self, portfolio):
        portfolio.process_trade(_trade(pnl=100.0))
        portfolio.reset()
        assert portfolio.trades == []

    def test_reset_clears_equity_history_except_start(self, portfolio):
        portfolio.process_trade(_trade(pnl=100.0))
        portfolio.reset()
        assert len(portfolio.equity_history) == 1

    def test_can_process_trades_after_reset(self, portfolio):
        portfolio.process_trade(_trade(pnl=100.0))
        portfolio.reset()
        portfolio.process_trade(_trade(pnl=200.0))
        assert portfolio.balance == pytest.approx(10_200.0)


class TestFromRiskProfile:

    def test_balance_from_cash_available(self):
        profile = RiskProfile(account_size=20_000, cash_available=15_000)
        p = Portfolio.from_risk_profile(profile)
        assert p.starting_balance == 15_000.0

    def test_trades_empty_initially(self):
        profile = RiskProfile(account_size=10_000, cash_available=10_000)
        p = Portfolio.from_risk_profile(profile)
        assert p.trades == []
