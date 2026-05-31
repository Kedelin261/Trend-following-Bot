"""Breakout detection above resistance and below support.

False-breakout filter: price must clear the level by at least
``threshold`` (default 0.25 %) before a breakout is confirmed.

Example
-------
Resistance = 100.00, threshold = 0.0025
Breakout is valid only when close > 100.25

No broker code. No API calls. Candle data only.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class BreakoutDetector:
    """Detects confirmed bullish and bearish price breakouts.

    Parameters
    ----------
    threshold : fractional clearance required above resistance / below support
                to confirm a breakout (default 0.0025 = 0.25 %)
    """

    def __init__(self, threshold: float = 0.0025) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        self.threshold = threshold

    def detect_bullish_breakout(
        self,
        current_close: float,
        resistance_levels: List[float],
    ) -> Tuple[bool, Optional[float]]:
        """Detect whether price has broken above a resistance level.

        A bullish breakout is confirmed when:
            current_close > resistance × (1 + threshold)

        Returns
        -------
        (True, breached_resistance)  when breakout is detected
        (False, None)                otherwise

        The breached level returned is the highest resistance that has
        been cleared — i.e., the most recently relevant barrier.
        """
        if not resistance_levels or current_close <= 0:
            return False, None

        broken = [
            r for r in resistance_levels
            if r > 0 and current_close > r * (1.0 + self.threshold)
        ]

        if not broken:
            return False, None

        # Return the highest breached level (most significant recent barrier)
        return True, max(broken)

    def detect_bearish_breakout(
        self,
        current_close: float,
        support_levels: List[float],
    ) -> Tuple[bool, Optional[float]]:
        """Detect whether price has broken below a support level.

        A bearish breakout is confirmed when:
            current_close < support × (1 − threshold)

        Returns
        -------
        (True, broken_support)  when breakdown is detected
        (False, None)           otherwise

        The broken level returned is the lowest support that has
        been pierced — i.e., the most recently relevant floor.
        """
        if not support_levels or current_close <= 0:
            return False, None

        broken = [
            s for s in support_levels
            if s > 0 and current_close < s * (1.0 - self.threshold)
        ]

        if not broken:
            return False, None

        # Return the lowest broken level (most significant recent floor)
        return True, min(broken)
