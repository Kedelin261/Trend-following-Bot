"""Overlay-Aware Backtest Engine — Phase 5.2.

Wraps the existing BacktestEngine bar-by-bar loop and injects an overlay
decision at the point where a new trade entry would be opened.

CONSTRAINTS (fairness guarantee):
    - Identical signal engine (MomentumRotationStrategy)
    - Identical risk engine (RiskEngine.from_config)
    - Identical TradeSimulator (same slippage, same commission)
    - Identical Portfolio
    - Identical warmup period
    - Only the overlay gate is added

The overlay intercepts BEFORE entering a pending trade:
    Step 1: Signal fires → candidate produced (no change)
    Step 2: Overlay.evaluate() called → may veto or scale
    Step 3: If allowed, entry proceeds with scaled position size

Position size scaling is applied by creating a modified TradeCandidate
with a reduced position_size.  The risk engine itself is not modified.

No execution code. No broker code. No live trading.
"""

import logging
from copy import copy
from typing import List, Optional

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.equity_curve import EquityCurve
from src.backtest.models import BacktestResults, BacktestTrade
from src.backtest.performance_metrics import (
    average_loss, average_win, expectancy, largest_loss, largest_win,
    losing_trades, max_drawdown, profit_factor, sharpe_ratio,
    win_rate, winning_trades,
)
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import ActiveTrade, TradeSimulator
from src.data.models import Candle
from src.risk.models import RiskProfile, TradeCandidate
from src.risk.risk_engine import RiskEngine
from src.signals.signal_engine import SignalEngine
from src.risk_overlay.base_overlay import RiskOverlay

logger = logging.getLogger(__name__)


class OverlayBacktestEngine:
    """Bar-by-bar backtest with an injected risk overlay gate.

    Parameters
    ----------
    signal_engine : strategy implementing generate_signal(candles) → Signal
    risk_engine   : Phase 3 RiskEngine
    portfolio     : Portfolio instance
    simulator     : TradeSimulator instance
    overlay       : RiskOverlay instance (or None for baseline)
    min_warmup    : bars to skip before first signal attempt
    symbol        : display label
    timeframe     : display label
    """

    def __init__(
        self,
        signal_engine: SignalEngine,
        risk_engine:   RiskEngine,
        portfolio:     Portfolio,
        simulator:     TradeSimulator,
        overlay:       Optional[RiskOverlay] = None,
        min_warmup:    int = 210,
        symbol:        str = "",
        timeframe:     str = "",
    ) -> None:
        self._signal    = signal_engine
        self._risk      = risk_engine
        self._portfolio = portfolio
        self._simulator = simulator
        self._overlay   = overlay
        self._min_warmup = min_warmup
        self._symbol    = symbol
        self._timeframe = timeframe

    def run(self, candles: List[Candle]) -> BacktestResults:
        """Execute a full bar-by-bar backtest with overlay gate."""
        self._portfolio.reset()
        if self._overlay is not None:
            self._overlay.reset()

        symbol    = candles[0].symbol    if candles else self._symbol
        timeframe = candles[0].timeframe if candles else self._timeframe

        pending:      Optional[object]      = None
        active_trade: Optional[ActiveTrade] = None

        for i, bar in enumerate(candles):

            # ---- Step 1: Execute pending entry at this bar's open --------
            if pending is not None and active_trade is None:
                active_trade = self._simulator.enter_trade(pending, bar, i)
                pending = None

            # ---- Step 2: Check if open trade exits on this bar ----------
            if active_trade is not None:
                closed = self._simulator.check_exit(active_trade, bar, i)
                if closed is not None:
                    self._portfolio.process_trade(closed)
                    active_trade = None

            # ---- Step 3: Generate signal at end of this bar -------------
            if (
                i >= self._min_warmup
                and active_trade is None
                and pending is None
            ):
                history = candles[: i + 1]
                signal  = self._signal.generate_signal(history)

                if signal.is_actionable:
                    candidate = self._risk.evaluate(signal, history)

                    if candidate.approved:
                        # ---- Overlay gate ----------------------------
                        trade_allowed = True
                        size_scale    = 1.0

                        if self._overlay is not None:
                            allowed, size_scale, reason = self._overlay.evaluate(
                                bar_index     = i,
                                candles       = history,
                                equity_values = self._portfolio.equity_values,
                                closed_trades = self._portfolio.trades,
                            )
                            trade_allowed = allowed and size_scale > 0.0
                            if not trade_allowed:
                                logger.debug(
                                    "overlay_gate: BLOCKED bar=%d overlay=%s reason=%s",
                                    i, self._overlay.name, reason,
                                )

                        if trade_allowed:
                            if size_scale < 1.0 and size_scale > 0.0:
                                candidate = _scale_candidate(candidate, size_scale)
                            if candidate.position_size > 0:
                                pending = candidate

        # ---- Force-close any still-open position at end of window ------
        if active_trade is not None and candles:
            closed = self._simulator.force_close(
                active_trade, candles[-1], len(candles) - 1
            )
            self._portfolio.process_trade(closed)

        return self._compile(symbol, timeframe)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile(self, symbol: str, timeframe: str) -> BacktestResults:
        trades    = self._portfolio.trades
        eq_vals   = self._portfolio.equity_values
        start_bal = self._portfolio.starting_balance
        end_bal   = self._portfolio.balance

        curve = EquityCurve(start_bal, trades)

        return BacktestResults(
            symbol           = symbol,
            timeframe        = timeframe,
            starting_balance = start_bal,
            ending_balance   = end_bal,
            net_profit       = end_bal - start_bal,
            total_trades     = len(trades),
            winning_trades   = winning_trades(trades),
            losing_trades    = losing_trades(trades),
            win_rate         = win_rate(trades),
            profit_factor    = profit_factor(trades),
            expectancy       = expectancy(trades),
            max_drawdown     = max_drawdown(eq_vals),
            sharpe_ratio     = sharpe_ratio(eq_vals),
            average_win      = average_win(trades),
            average_loss     = average_loss(trades),
            largest_win      = largest_win(trades),
            largest_loss     = largest_loss(trades),
            equity_curve     = curve.as_tuples(),
            trades           = trades,
        )


def _scale_candidate(candidate: TradeCandidate, scale: float) -> TradeCandidate:
    """Return a new TradeCandidate with position_size reduced by scale factor.

    All other fields (stop, target, risk levels) remain unchanged — only
    the position size is reduced. This preserves all risk engine logic.
    """
    new_size = max(0, int(candidate.position_size * scale))
    scaled   = copy(candidate)
    scaled.position_size = new_size
    return scaled
