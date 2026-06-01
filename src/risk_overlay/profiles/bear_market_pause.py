"""BEAR_MARKET_PAUSE — pause new trades during confirmed bear regimes.

Uses the existing MacroRegimeDetector (no new regime definitions).
Bear regimes are: CONTRACTION and CRISIS.
Non-bear regimes (EXPANSION, RECOVERY, UNKNOWN) permit trading.

No parameter sweeps. No tuning. No new regime definitions.
Research only. No execution. No broker code. No live trading.
"""

from typing import List

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.regime.macro_regime_detector import MacroRegime, MacroRegimeDetector
from src.risk_overlay.base_overlay import RiskOverlay

# Bear regimes from existing framework — not modified
_BEAR_REGIMES = {MacroRegime.CONTRACTION, MacroRegime.CRISIS}


class BearMarketPause(RiskOverlay):
    """Pause new trade entries during CONTRACTION or CRISIS macro regimes.

    Uses the existing MacroRegimeDetector.  No new regime definitions.
    """

    def __init__(self) -> None:
        self._detector = MacroRegimeDetector()

    @property
    def name(self) -> str:
        return "BEAR_MARKET_PAUSE"

    def reset(self) -> None:
        pass  # MacroRegimeDetector is stateless

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Allow trade only when macro regime is not bear (CONTRACTION/CRISIS)."""
        if len(candles) < self._detector.ema_period:
            # Not enough history to classify — allow trading
            return True, 1.0, "insufficient history for regime classification"

        regime = self._detector.classify(candles)

        if regime in _BEAR_REGIMES:
            return (
                False, 0.0,
                f"bear market pause: regime={regime.value}",
            )

        return True, 1.0, f"regime={regime.value} — trading permitted"
