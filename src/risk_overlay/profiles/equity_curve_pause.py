"""EQUITY_CURVE_PAUSE — pause new trades after equity falls 10% from peak.

Logic:
    1. Track the rolling peak of the equity curve.
    2. If current equity has declined >= PAUSE_THRESHOLD from the peak,
       pause all new trade entries.
    3. Resume once equity recovers back to the peak level.

Default threshold: 10% (fixed — not optimised).
No parameter sweeps. No tuning.

Research only. No execution. No broker code. No live trading.
"""

from typing import List

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.risk_overlay.base_overlay import RiskOverlay

PAUSE_THRESHOLD = 0.10   # 10% equity drawdown from peak — fixed default


class EquityCurvePause(RiskOverlay):
    """Pause new trade entries when equity drawdown from peak >= 10%.

    Resumes immediately when equity recovers to the recorded peak.
    """

    def __init__(self, pause_threshold: float = PAUSE_THRESHOLD) -> None:
        self._threshold = pause_threshold
        self._peak:     float = 0.0
        self._paused:   bool  = False

    @property
    def name(self) -> str:
        return "EQUITY_CURVE_PAUSE"

    def reset(self) -> None:
        self._peak   = 0.0
        self._paused = False

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Allow trade only if equity has not fallen >= threshold from peak."""
        if not equity_values:
            return True, 1.0, "no equity history"

        current = equity_values[-1]

        # Update rolling peak
        if current > self._peak:
            self._peak = current

        if self._peak <= 0:
            return True, 1.0, "peak not established"

        dd_pct = (self._peak - current) / self._peak

        if dd_pct >= self._threshold:
            self._paused = True
            return (
                False, 0.0,
                f"equity curve pause: DD={dd_pct:.1%} >= {self._threshold:.1%} threshold",
            )

        # Recovery: drawdown below threshold — resume
        self._paused = False
        return True, 1.0, f"equity OK: DD={dd_pct:.1%} < {self._threshold:.1%}"
