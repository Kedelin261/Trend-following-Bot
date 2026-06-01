"""NO_OVERLAY — baseline reference overlay.

Always permits trades with full position size.
Used to confirm that the overlay-aware backtester
produces identical results to the Phase 5.1 baseline.

No filtering. No scaling. Research only.
"""

from typing import List

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.risk_overlay.base_overlay import RiskOverlay


class NoOverlay(RiskOverlay):
    """Pass-through overlay — no filtering, no size scaling."""

    @property
    def name(self) -> str:
        return "NO_OVERLAY"

    def reset(self) -> None:
        pass  # No state to reset

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Always allow trade at full size."""
        return True, 1.0, "no overlay active"
