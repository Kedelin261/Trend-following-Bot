"""Volume confirmation for breakout signals.

Rule: current volume > N-period average volume.

Forex and many ETF data sources report zero or near-zero tick volume,
which is not meaningful for confirmation. In those cases this module
returns None (UNKNOWN) rather than False — callers must treat None
as "cannot confirm, do not penalise."

No broker code. No API calls. Candle data only.
"""

import logging
from typing import List, Optional

from src.data.models import Candle

logger = logging.getLogger(__name__)

_ZERO_VOLUME_THRESHOLD = 1.0  # volumes ≤ this are treated as "no data"


class VolumeConfirmation:
    """Validates breakout signals using relative volume analysis.

    Parameters
    ----------
    lookback : rolling window for average volume (default 20 periods)
    """

    def __init__(self, lookback: int = 20) -> None:
        if lookback < 2:
            raise ValueError(f"lookback must be ≥ 2, got {lookback}")
        self.lookback = lookback

    def average_volume(self, candles: List[Candle]) -> Optional[float]:
        """Return the mean volume over the previous *lookback* completed bars.

        Excludes the current (last) candle so the comparison is fair.
        Returns None when data is insufficient or all volumes are zero.
        """
        # Need at least lookback + 1 candles (lookback completed + current)
        if len(candles) < self.lookback + 1:
            return None

        reference = candles[-(self.lookback + 1) : -1]  # lookback completed bars
        volumes = [c.volume for c in reference]

        meaningful = [v for v in volumes if v > _ZERO_VOLUME_THRESHOLD]
        if not meaningful:
            return None

        return sum(meaningful) / len(meaningful)

    def is_volume_confirmed(self, candles: List[Candle]) -> Optional[bool]:
        """Return volume confirmation status for the latest candle.

        Returns
        -------
        True   : current volume > average (confirmed)
        False  : current volume ≤ average (not confirmed)
        None   : volume data is unavailable or meaningless (UNKNOWN)
                 — treat as neutral, do not penalise the signal

        Assets that commonly return UNKNOWN: Forex tick volume,
        crypto on some brokers, after-hours equity data.
        """
        if not candles:
            return None

        current_volume = candles[-1].volume

        # Zero / missing volume → UNKNOWN
        if current_volume <= _ZERO_VOLUME_THRESHOLD:
            logger.debug(
                "volume_confirmation: zero volume for %s on %s — returning UNKNOWN",
                candles[-1].symbol,
                candles[-1].timeframe,
            )
            return None

        avg = self.average_volume(candles)
        if avg is None:
            return None

        confirmed = current_volume > avg
        logger.debug(
            "volume_confirmation: %s current=%.0f avg=%.0f confirmed=%s",
            candles[-1].symbol,
            current_volume,
            avg,
            confirmed,
        )
        return confirmed
