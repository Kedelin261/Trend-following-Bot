"""CONSECUTIVE_LOSS_COOLDOWN — pause after 5 consecutive losing trades.

Logic:
    1. Track consecutive losing trades.
    2. After 5 consecutive losses, enter cooldown of 10 trading days
       (10 bars on D1 data).
    3. After cooldown expires, resume normal trading.
    4. Consecutive loss counter resets on any winning trade.

Parameters: 5 losses, 10-bar cooldown — fixed, not tuned.

Research only. No execution. No broker code. No live trading.
"""

from typing import List, Optional

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.risk_overlay.base_overlay import RiskOverlay

# Fixed parameters — not tuned
CONSECUTIVE_LOSS_TRIGGER = 5    # losses to trigger cooldown
COOLDOWN_BARS            = 10   # trading days (D1 bars)


class ConsecutiveLossCooldown(RiskOverlay):
    """Pause trading for 10 bars after 5 consecutive losing trades.

    Parameters
    ----------
    loss_trigger   : number of consecutive losses before cooldown (default 5)
    cooldown_bars  : number of bars to pause (default 10)
    """

    def __init__(
        self,
        loss_trigger:  int = CONSECUTIVE_LOSS_TRIGGER,
        cooldown_bars: int = COOLDOWN_BARS,
    ) -> None:
        self._trigger      = loss_trigger
        self._cooldown     = cooldown_bars
        self._cooldown_end_bar: Optional[int] = None   # bar index when cooldown ends
        self._last_n_trades: int = 0   # how many trades we've seen last evaluation

    @property
    def name(self) -> str:
        return "CONSECUTIVE_LOSS_COOLDOWN"

    def reset(self) -> None:
        self._cooldown_end_bar = None
        self._last_n_trades    = 0

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Allow trade only if not in cooldown after consecutive losses."""
        # Check if we're still in a cooldown period
        if self._cooldown_end_bar is not None:
            if bar_index < self._cooldown_end_bar:
                remaining = self._cooldown_end_bar - bar_index
                return (
                    False, 0.0,
                    f"cooldown: {remaining} bars remaining after "
                    f"{self._trigger} consecutive losses",
                )
            else:
                # Cooldown expired
                self._cooldown_end_bar = None

        # Count consecutive losses from the end of the trade list
        if not closed_trades:
            return True, 1.0, "no trades yet"

        consecutive = 0
        for trade in reversed(closed_trades):
            if trade.is_loss:
                consecutive += 1
            else:
                break  # win resets streak

        if consecutive >= self._trigger:
            self._cooldown_end_bar = bar_index + self._cooldown
            return (
                False, 0.0,
                f"cooldown triggered: {consecutive} consecutive losses — "
                f"pausing {self._cooldown} bars",
            )

        return True, 1.0, f"consecutive losses: {consecutive} < {self._trigger}"
