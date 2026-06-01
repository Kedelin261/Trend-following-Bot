"""VOLATILITY_RISK_SCALING — reduce position size in HIGH or EXTREME volatility.

Uses the existing VolatilityRegimeDetector (no new volatility definitions).

Size scaling rules (fixed — not optimised):
    LOW_VOL    : 100% size (full)
    NORMAL_VOL : 100% size (full)
    HIGH_VOL   :  50% size (reduced risk)
    EXTREME_VOL:  25% size (heavily reduced risk)
    UNKNOWN    : 100% size (no data — conservative pass-through)

Trades are never blocked — only size-reduced.
The entry signal is still taken; position size is scaled down.

Research only. No execution. No broker code. No live trading.
"""

from typing import List

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.regime.volatility_regime_detector import VolatilityRegime, VolatilityRegimeDetector
from src.risk_overlay.base_overlay import RiskOverlay

# Fixed scaling map — not tuned
_SCALE_MAP = {
    VolatilityRegime.LOW_VOL:     1.00,
    VolatilityRegime.NORMAL_VOL:  1.00,
    VolatilityRegime.HIGH_VOL:    0.50,
    VolatilityRegime.EXTREME_VOL: 0.25,
    VolatilityRegime.UNKNOWN:     1.00,
}


class VolatilityRiskScaling(RiskOverlay):
    """Reduce position size during HIGH or EXTREME volatility regimes.

    Uses existing VolatilityRegimeDetector — no new volatility definitions.
    """

    def __init__(self) -> None:
        self._detector = VolatilityRegimeDetector()

    @property
    def name(self) -> str:
        return "VOLATILITY_RISK_SCALING"

    def reset(self) -> None:
        pass  # VolatilityRegimeDetector is stateless

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Scale position size based on current volatility regime."""
        if len(candles) < 15:
            return True, 1.0, "insufficient history for volatility classification"

        regime = self._detector.classify(candles)
        scale  = _SCALE_MAP.get(regime, 1.0)

        if scale < 1.0:
            return (
                True, scale,
                f"volatility scaling: regime={regime.value} → size={scale:.0%}",
            )

        return True, 1.0, f"volatility regime={regime.value} → full size"
