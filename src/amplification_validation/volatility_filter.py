"""Volatility Filter — Phase 5.4.

Gates trade candidates by current volatility regime.
If the regime at signal time is HIGH_VOL, the trade is rejected.

This validates the Phase 5.3 volatility finding:
  HIGH_VOL: PF=0.93, Exp=-$3.11 (negative expectancy)

CRITICAL CONSTRAINTS:
  - Does NOT modify entry logic
  - Does NOT modify signal generation
  - Does NOT modify risk engine
  - Does NOT modify VolatilityRegimeDetector thresholds
  - No threshold tuning — uses existing detector as-is
  - Only validates the exact finding from Phase 5.3
  - Classifies at signal bar (no lookahead)

Research only. No execution. No broker code.
"""

import logging
from typing import List, Optional

from src.data.models import Candle
from src.regime.volatility_regime_detector import (
    VolatilityRegimeDetector,
    VolatilityRegime,
)

logger = logging.getLogger(__name__)


class VolatilityFilter:
    """Rejects trade candidates during HIGH_VOL regimes.

    The regime is classified at the current bar's history (no lookahead).
    Uses the same VolatilityRegimeDetector as Phase 5.3 — no threshold changes.

    Parameters
    ----------
    detector : VolatilityRegimeDetector instance (or None to create default)
    """

    BLOCKED_REGIMES = frozenset({VolatilityRegime.HIGH_VOL})

    def __init__(
        self,
        detector: Optional[VolatilityRegimeDetector] = None,
    ) -> None:
        self._detector = detector or VolatilityRegimeDetector()

    @property
    def detector(self) -> VolatilityRegimeDetector:
        return self._detector

    def allows(self, candles: List[Candle]) -> bool:
        """Return True if current volatility regime is NOT HIGH_VOL.

        Parameters
        ----------
        candles : history up to and including current bar (no lookahead)
        """
        if not candles:
            return True  # no data → allow (cannot classify)

        regime = self._detector.classify(candles)

        if regime in self.BLOCKED_REGIMES:
            logger.debug(
                "volatility_filter: BLOCKED regime=%s", regime.value
            )
            return False

        return True

    def classify(self, candles: List[Candle]) -> VolatilityRegime:
        """Expose regime for logging / reporting."""
        return self._detector.classify(candles) if candles else VolatilityRegime.UNKNOWN

    def __repr__(self) -> str:
        return (
            f"VolatilityFilter(blocked={sorted(r.value for r in self.BLOCKED_REGIMES)})"
        )
