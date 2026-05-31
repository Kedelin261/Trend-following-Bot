"""Robustness validator — wraps existing EdgeStabilityAnalyzer for Phase 5.1.

Reuses EdgeStabilityAnalyzer from Phase 5.0. No duplicate logic.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.data.models import Candle
from src.edge_lab.edge_engine import EdgeEngine
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer, StabilityWindow
from src.edge_lab.strategy_interface import StrategyInterface


@dataclass
class RobustnessValidationResult:
    strategy_name:   str
    rating:          str     # ROBUST / MARGINAL / UNSTABLE
    windows_passing: int
    windows:         List[StabilityWindow]
    verdict:         str

    @property
    def passes(self) -> bool:
        return self.rating in ("ROBUST", "MARGINAL")


class RobustnessValidator:
    """Validates strategy consistency across three historical time windows."""

    def __init__(self, config: dict) -> None:
        self._engine    = EdgeEngine(config)
        self._stability = EdgeStabilityAnalyzer()

    def validate(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
    ) -> RobustnessValidationResult:
        build_fn = lambda s: self._engine._build_engine(s)
        rating, windows = self._stability.analyze(strategy, asset_candles, build_fn)

        passing = sum(1 for w in windows if w.passes)
        verdict = (
            f"Passes in {passing}/3 windows — "
            + ("all windows profitable" if rating == "ROBUST"
               else "some windows fail" if rating == "MARGINAL"
               else "majority of windows fail")
        )

        return RobustnessValidationResult(
            strategy_name   = strategy.name,
            rating          = rating,
            windows_passing = passing,
            windows         = windows,
            verdict         = verdict,
        )
