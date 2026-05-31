"""Backtest Engine — Phase 4 master orchestrator.

Drives a bar-by-bar simulation using the exact signal and risk logic
already built in Phases 2 and 3.  Nothing in this file is modified to
make results look better — the engine tests the strategy as-is.

Signal rule (no lookahead bias):
  Bar N close  → signal generated
  Bar N+1 open → entry executed
  Bar N+1 onwards → stop/target checked each bar

One trade open at a time (no pyramiding).
Force-close any open position at the end of the backtest window.

No live trading. No paper trading. No broker mutations. No order placement.
"""

import logging
from typing import List, Optional

from src.backtest.equity_curve import EquityCurve
from src.backtest.models import BacktestResults, StrategyHealth
from src.backtest.performance_metrics import (
    average_loss,
    average_win,
    expectancy,
    largest_loss,
    largest_win,
    losing_trades,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
    winning_trades,
)
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import ActiveTrade, TradeSimulator
from src.data.models import Candle
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.signals.signal_engine import SignalEngine

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Drives a complete bar-by-bar backtest simulation.

    All components are injected for testability and future swapping.

    Parameters
    ----------
    signal_engine : Phase 2 SignalEngine instance
    risk_engine   : Phase 3 RiskEngine instance
    portfolio     : Portfolio tracking cash and trades
    simulator     : TradeSimulator for entry/exit mechanics
    min_warmup    : bars consumed before the first signal attempt
                    (should match SignalEngine.min_candles, default 210)
    symbol        : display name (overridden by candle data when available)
    timeframe     : display name (overridden by candle data when available)
    """

    def __init__(
        self,
        signal_engine: SignalEngine,
        risk_engine:   RiskEngine,
        portfolio:     Portfolio,
        simulator:     TradeSimulator,
        min_warmup:    int = 210,
        symbol:        str = "",
        timeframe:     str = "",
    ) -> None:
        self._signal      = signal_engine
        self._risk        = risk_engine
        self._portfolio   = portfolio
        self._simulator   = simulator
        self._min_warmup  = min_warmup
        self._symbol      = symbol
        self._timeframe   = timeframe

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config:    dict,
        symbol:    str = "SPY",
        timeframe: str = "D1",
    ) -> "BacktestEngine":
        """Build a fully-configured BacktestEngine from settings.yaml."""
        bt_cfg   = config.get("backtest", {})
        risk_cfg = config.get("risk", {})

        starting_balance = float(bt_cfg.get("starting_balance", 10_000.0))
        commission       = float(bt_cfg.get("commission_per_trade", 1.0))
        slippage         = float(bt_cfg.get("slippage_percent", 0.05))
        min_warmup       = int(bt_cfg.get("min_warmup", 210))

        risk_profile = RiskProfile(
            account_size           = starting_balance,
            cash_available         = starting_balance,
            risk_per_trade_percent = float(risk_cfg.get("risk_per_trade_percent", 1.0)),
        )

        return cls(
            signal_engine = SignalEngine(),
            risk_engine   = RiskEngine.from_config(config),
            portfolio     = Portfolio(starting_balance),
            simulator     = TradeSimulator(
                slippage_percent     = slippage,
                commission_per_trade = commission,
            ),
            min_warmup    = min_warmup,
            symbol        = symbol,
            timeframe     = timeframe,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, candles: List[Candle]) -> BacktestResults:
        """Execute a full bar-by-bar backtest over *candles*.

        Always returns a valid BacktestResults — never raises.
        Bars before *min_warmup* are used only for indicator warm-up;
        no trades are taken during that period.
        """
        self._portfolio.reset()

        symbol    = candles[0].symbol    if candles else self._symbol
        timeframe = candles[0].timeframe if candles else self._timeframe

        pending:      Optional[object] = None    # TradeCandidate awaiting entry
        active_trade: Optional[ActiveTrade] = None

        logger.info(
            "backtest_engine.run: %s/%s candles=%d warmup=%d balance=%.0f",
            symbol, timeframe,
            len(candles), self._min_warmup,
            self._portfolio.starting_balance,
        )

        for i, bar in enumerate(candles):

            # ---- Step 1: Execute pending entry at this bar's open --------
            if pending is not None and active_trade is None:
                active_trade = self._simulator.enter_trade(pending, bar, i)  # type: ignore[arg-type]
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
                        pending = candidate
                        logger.debug("backtest: signal@%d → entry@%d", i, i + 1)

        # ---- Force-close any still-open position at end of window ------
        if active_trade is not None and candles:
            closed = self._simulator.force_close(
                active_trade, candles[-1], len(candles) - 1
            )
            self._portfolio.process_trade(closed)
            logger.info(
                "backtest: force-close at end | pnl=%.2f", closed.pnl
            )

        result = self._compile(symbol, timeframe)

        logger.info(
            "backtest_complete: %s/%s trades=%d win_rate=%.0f%% "
            "pf=%.2f exp=%.2f dd=%.1f%% net=%.2f",
            symbol, timeframe,
            result.total_trades,
            result.win_rate * 100,
            result.profit_factor,
            result.expectancy,
            result.max_drawdown,
            result.net_profit,
        )
        return result

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
