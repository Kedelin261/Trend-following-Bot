"""Trade simulator — converts approved TradeCandidates into BacktestTrades.

Execution rules:
  - Entry at the OPEN of the bar following the signal bar (no lookahead)
  - Stop/target checked against each bar's HIGH and LOW
  - If both stop and target are breached in the same bar: STOP wins (conservative)
  - Slippage applied to every entry and every exit separately
  - Commission deducted round-trip (2× per completed trade)
  - No partial fills, no scaling, no pyramiding

No broker code. No API calls. Pure OHLC simulation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.backtest.models import BacktestTrade, ClosingReason
from src.data.models import Candle
from src.risk.models import TradeCandidate
from src.signals.models import SignalType

logger = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    """Open simulated position awaiting exit.

    Public so downstream components (portfolio, tests) can inspect it.
    """

    symbol:          str
    signal_type:     SignalType
    entry_price:     float       # actual entry price after slippage
    entry_time:      datetime
    entry_bar_index: int
    stop_price:      float       # original stop from TradeCandidate
    target_price:    float       # original target from TradeCandidate
    position_size:   int


class TradeSimulator:
    """Simulates trade entries and exits against historical OHLC data.

    Parameters
    ----------
    slippage_percent     : one-way slippage as % of price (default 0.05 %)
    commission_per_trade : flat fee per direction — total 2× per round trip ($)
    """

    def __init__(
        self,
        slippage_percent:     float = 0.05,
        commission_per_trade: float = 1.0,
    ) -> None:
        self.slippage_pct = slippage_percent
        self.commission   = commission_per_trade

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def enter_trade(
        self,
        candidate: TradeCandidate,
        entry_bar: Candle,
        bar_index: int,
    ) -> ActiveTrade:
        """Open a position at entry_bar's open price with slippage applied."""
        entry_price = self._entry_price(entry_bar.open, candidate.signal_type)

        logger.debug(
            "trade_open: %s %s entry=%.5f (raw=%.5f) stop=%.5f target=%.5f size=%d",
            candidate.symbol,
            candidate.signal_type.value,
            entry_price,
            entry_bar.open,
            candidate.stop_loss,
            candidate.take_profit,
            candidate.position_size,
        )

        return ActiveTrade(
            symbol          = candidate.symbol,
            signal_type     = candidate.signal_type,
            entry_price     = entry_price,
            entry_time      = entry_bar.timestamp,
            entry_bar_index = bar_index,
            stop_price      = candidate.stop_loss,
            target_price    = candidate.take_profit,
            position_size   = candidate.position_size,
        )

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade:     ActiveTrade,
        bar:       Candle,
        bar_index: int,
    ) -> Optional[BacktestTrade]:
        """Check whether stop or target is breached on *bar*.

        Returns a completed BacktestTrade if closed, else None.
        If both stop and target are hit in the same bar, stop wins.
        """
        if trade.signal_type == SignalType.LONG:
            stop_hit   = bar.low  <= trade.stop_price
            target_hit = bar.high >= trade.target_price

            if stop_hit:
                exit_px = self._exit_price(trade.stop_price, SignalType.LONG)
                return self._close(trade, bar, bar_index, exit_px, ClosingReason.STOP)
            if target_hit:
                exit_px = self._exit_price(trade.target_price, SignalType.LONG)
                return self._close(trade, bar, bar_index, exit_px, ClosingReason.TARGET)

        else:  # SHORT
            stop_hit   = bar.high >= trade.stop_price
            target_hit = bar.low  <= trade.target_price

            if stop_hit:
                exit_px = self._exit_price(trade.stop_price, SignalType.SHORT)
                return self._close(trade, bar, bar_index, exit_px, ClosingReason.STOP)
            if target_hit:
                exit_px = self._exit_price(trade.target_price, SignalType.SHORT)
                return self._close(trade, bar, bar_index, exit_px, ClosingReason.TARGET)

        return None

    def force_close(
        self,
        trade:     ActiveTrade,
        bar:       Candle,
        bar_index: int,
    ) -> BacktestTrade:
        """Close a trade at bar's close price — used at end of backtest window."""
        exit_px = self._exit_price(bar.close, trade.signal_type)
        return self._close(trade, bar, bar_index, exit_px, ClosingReason.TIME_EXIT)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry_price(self, price: float, signal_type: SignalType) -> float:
        """Apply entry slippage: LONG pays more; SHORT receives less."""
        slip = price * self.slippage_pct / 100.0
        return price + slip if signal_type == SignalType.LONG else price - slip

    def _exit_price(self, price: float, signal_type: SignalType) -> float:
        """Apply exit slippage: LONG receives less; SHORT pays more."""
        slip = price * self.slippage_pct / 100.0
        return price - slip if signal_type == SignalType.LONG else price + slip

    def _gross_pnl(
        self,
        trade:      ActiveTrade,
        exit_price: float,
    ) -> float:
        if trade.signal_type == SignalType.LONG:
            return (exit_price - trade.entry_price) * trade.position_size
        return (trade.entry_price - exit_price) * trade.position_size

    def _close(
        self,
        trade:      ActiveTrade,
        bar:        Candle,
        bar_index:  int,
        exit_price: float,
        reason:     ClosingReason,
    ) -> BacktestTrade:
        gross   = self._gross_pnl(trade, exit_price)
        net_pnl = gross - 2.0 * self.commission   # round-trip commission
        notional = trade.entry_price * trade.position_size
        ret_pct  = (net_pnl / notional * 100.0) if notional > 0 else 0.0
        holding  = bar_index - trade.entry_bar_index

        logger.debug(
            "trade_closed: %s %s gross=%.2f net=%.2f reason=%s bars=%d",
            trade.symbol,
            trade.signal_type.value,
            gross,
            net_pnl,
            reason.value,
            holding,
        )

        return BacktestTrade(
            symbol         = trade.symbol,
            entry_time     = trade.entry_time,
            exit_time      = bar.timestamp,
            signal_type    = trade.signal_type,
            entry_price    = trade.entry_price,
            exit_price     = exit_price,
            stop_price     = trade.stop_price,
            target_price   = trade.target_price,
            position_size  = trade.position_size,
            pnl            = net_pnl,
            return_percent = ret_pct,
            holding_period = holding,
            win_loss       = "WIN" if net_pnl > 0 else "LOSS",
            reason_closed  = reason,
        )
