"""Quality Band Filter — Phase 5.4.

Gates trade candidates by signal strength score.
If a score falls within the excluded band [lo, hi], the trade is rejected.

This validates the Phase 5.3 quality finding:
  Score band 60-69: PF=0.91, Exp=-$4.76 (NO EDGE)

CRITICAL CONSTRAINTS:
  - Does NOT modify entry logic
  - Does NOT modify signal generation (score is READ, not changed)
  - Does NOT modify risk engine
  - Does NOT modify position sizing
  - No threshold tuning — exactly [60.0, 69.9] as found in Phase 5.3
  - No optimization of score cutoffs
  - Only validates the exact band found in Phase 5.3

Research only. No execution. No broker code.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QualityBandFilter:
    """Rejects trade candidates whose signal score falls within [lo, hi].

    Parameters
    ----------
    score_lo : Lower bound of excluded band (inclusive)
    score_hi : Upper bound of excluded band (inclusive)
              Pass None for both to disable (allow all scores).
    """

    def __init__(
        self,
        score_lo: Optional[float],
        score_hi: Optional[float],
    ) -> None:
        if (score_lo is None) != (score_hi is None):
            raise ValueError(
                "QualityBandFilter: score_lo and score_hi must both be set or both be None"
            )
        if score_lo is not None and score_lo > score_hi:  # type: ignore[operator]
            raise ValueError(
                f"QualityBandFilter: score_lo ({score_lo}) > score_hi ({score_hi})"
            )
        self._lo = score_lo
        self._hi = score_hi

    @property
    def score_lo(self) -> Optional[float]:
        return self._lo

    @property
    def score_hi(self) -> Optional[float]:
        return self._hi

    @property
    def is_active(self) -> bool:
        return self._lo is not None

    def allows(self, score: float) -> bool:
        """Return True if the score is allowed (not in the excluded band).

        Parameters
        ----------
        score : signal strength_score (0–100)
        """
        if not self.is_active:
            return True
        # Exclude if score falls within [lo, hi] inclusive
        in_band = self._lo <= score <= self._hi  # type: ignore[operator]
        if in_band:
            logger.debug(
                "quality_band_filter: BLOCKED score=%.1f in [%.1f, %.1f]",
                score, self._lo, self._hi,
            )
        return not in_band

    def __repr__(self) -> str:
        if not self.is_active:
            return "QualityBandFilter(disabled)"
        return f"QualityBandFilter(excluded=[{self._lo}, {self._hi}])"
