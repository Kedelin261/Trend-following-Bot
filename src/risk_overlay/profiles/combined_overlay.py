"""COMBINED_OVERLAY — combine maximum 2 overlays.

Combines EQUITY_CURVE_PAUSE (dd protection) and VOLATILITY_RISK_SCALING
(position sizing) because these address orthogonal risk dimensions:
    - EQUITY_CURVE_PAUSE: stops trading during equity curve deterioration
    - VOLATILITY_RISK_SCALING: reduces size during high/extreme volatility

Combination rules:
    - A trade is blocked if ANY component overlay blocks it.
    - If all overlays allow, the size_scale is the MINIMUM of all scales.
    - Maximum 2 overlays combined (anti-overfitting rule).

No parameter sweeps. No additional tuning beyond individual overlay defaults.
Research only. No execution. No broker code. No live trading.
"""

from typing import List

from src.backtest.models import BacktestTrade
from src.data.models import Candle
from src.risk_overlay.base_overlay import RiskOverlay
from src.risk_overlay.profiles.equity_curve_pause import EquityCurvePause
from src.risk_overlay.profiles.volatility_risk_scaling import VolatilityRiskScaling


class CombinedOverlay(RiskOverlay):
    """Combine EQUITY_CURVE_PAUSE + VOLATILITY_RISK_SCALING (max 2 overlays).

    Selection rationale (no parameter sweeps):
        - These two overlays address orthogonal risk dimensions.
        - Both individually reduce drawdown without excessive trade filtering.
        - Combining them applies both protections simultaneously.
    """

    def __init__(self) -> None:
        # Exactly 2 overlays — anti-overfitting rule enforced
        self._overlays: List[RiskOverlay] = [
            EquityCurvePause(),
            VolatilityRiskScaling(),
        ]

    @property
    def name(self) -> str:
        return "COMBINED_OVERLAY"

    @property
    def component_names(self) -> List[str]:
        return [o.name for o in self._overlays]

    def reset(self) -> None:
        for overlay in self._overlays:
            overlay.reset()

    def evaluate(
        self,
        bar_index:     int,
        candles:       List[Candle],
        equity_values: List[float],
        closed_trades: List[BacktestTrade],
    ) -> tuple:
        """Apply all component overlays; block if any blocks; take minimum scale."""
        results = []
        for overlay in self._overlays:
            allowed, scale, reason = overlay.evaluate(
                bar_index     = bar_index,
                candles       = candles,
                equity_values = equity_values,
                closed_trades = closed_trades,
            )
            results.append((allowed, scale, reason, overlay.name))

        # Block if any overlay blocks
        for allowed, scale, reason, oname in results:
            if not allowed:
                return (
                    False, 0.0,
                    f"combined blocked by {oname}: {reason}",
                )

        # All allowed — use minimum scale
        min_scale = min(s for _, s, _, _ in results)
        reasons   = " | ".join(f"{n}:{s:.0%}" for _, s, _, n in results)

        return True, min_scale, f"combined allowed: {reasons}"
