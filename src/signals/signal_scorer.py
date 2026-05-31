"""Signal scoring: combines trend, breakout, and volume into a 0–100 score.

Weight allocation
-----------------
Trend     40 %
Breakout  40 %
Volume    20 %

Volume UNKNOWN (Forex / missing data) receives 50 % credit rather than 0 %,
so Forex signals are not unfairly penalised for missing exchange volume.

Score categories
----------------
STRONG   ≥ 70
MODERATE ≥ 40
WEAK     <  40

No broker code. No API calls. No candle data — pure scoring logic.
"""

import logging
from typing import Optional

from src.signals.models import SignalStrength, SignalType, TrendDirection

logger = logging.getLogger(__name__)

_TREND_WEIGHT    = 0.40
_BREAKOUT_WEIGHT = 0.40
_VOLUME_WEIGHT   = 0.20


class SignalScorer:
    """Produces a composite quality score for a signal.

    Designed to be stateless — all inputs are passed to ``score_signal``.
    """

    def score_signal(
        self,
        trend_direction: TrendDirection,
        trend_strength: float,
        breakout_detected: bool,
        volume_confirmed: Optional[bool],
        signal_type: SignalType,
    ) -> float:
        """Return a composite quality score in [0, 100].

        Parameters
        ----------
        trend_direction   : BULLISH / BEARISH / NEUTRAL
        trend_strength    : raw strength from TrendDetector (0–100)
        breakout_detected : True if a confirmed breakout was found
        volume_confirmed  : True=confirmed, False=not confirmed, None=unknown
        signal_type       : LONG / SHORT / NONE
        """
        if signal_type == SignalType.NONE:
            return 0.0

        # -- Trend component --------------------------------------------------
        if trend_direction in (TrendDirection.BULLISH, TrendDirection.BEARISH):
            # Floor at 40 when trend is directional (reward aligned direction)
            trend_score = max(40.0, min(100.0, trend_strength))
        else:
            trend_score = 0.0

        # -- Breakout component -----------------------------------------------
        breakout_score = 100.0 if breakout_detected else 0.0

        # -- Volume component --------------------------------------------------
        if volume_confirmed is True:
            volume_score = 100.0
        elif volume_confirmed is None:
            # UNKNOWN: neutral — give 50 % so Forex isn't penalised
            volume_score = 50.0
        else:
            volume_score = 0.0

        raw = (
            trend_score    * _TREND_WEIGHT +
            breakout_score * _BREAKOUT_WEIGHT +
            volume_score   * _VOLUME_WEIGHT
        )

        score = round(min(100.0, max(0.0, raw)), 1)

        logger.debug(
            "signal_scorer: type=%s trend=%.1f breakout=%.1f volume=%.1f → %.1f",
            signal_type,
            trend_score * _TREND_WEIGHT,
            breakout_score * _BREAKOUT_WEIGHT,
            volume_score * _VOLUME_WEIGHT,
            score,
        )
        return score

    def categorize_score(self, score: float) -> SignalStrength:
        """Map a numeric score to a STRONG / MODERATE / WEAK category."""
        if score >= 70.0:
            return SignalStrength.STRONG
        if score >= 40.0:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK
