"""Volatility gate for Strategy V2 — only trade in optimal ATR% environments.

Research finding: strategy performs best in MEDIUM volatility.
This filter blocks trades in LOW and HIGH volatility conditions.

Filter modes:
  MEDIUM_ONLY    : only ATR% in [low_threshold, high_threshold]
  LOW_AND_MEDIUM : block only HIGH volatility
  NONE           : no filter (all conditions pass)

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.research.volatility_filter import VolatilityCategory, VolatilityFilter

logger = logging.getLogger(__name__)


class VolatilityFilterMode(str, Enum):
    MEDIUM_ONLY     = "MEDIUM_ONLY"
    LOW_AND_MEDIUM  = "LOW_AND_MEDIUM"
    MEDIUM_AND_HIGH = "MEDIUM_AND_HIGH"   # added for Phase 4.7 density research
    NONE            = "NONE"


class VolatilityTradeFilter:
    """Gates trades by ATR-as-percentage-of-price.

    Parameters
    ----------
    mode           : which volatility categories to allow
    atr_period     : ATR lookback window (default 14)
    low_threshold  : ATR% below which market is LOW vol (default 0.5 %)
    high_threshold : ATR% above which market is HIGH vol (default 2.0 %)
    """

    def __init__(
        self,
        mode:           VolatilityFilterMode = VolatilityFilterMode.MEDIUM_ONLY,
        atr_period:     int   = 14,
        low_threshold:  float = 0.5,
        high_threshold: float = 2.0,
    ) -> None:
        self.mode           = mode
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self._vf = VolatilityFilter(
            atr_period      = atr_period,
            low_threshold   = low_threshold,
            high_threshold  = high_threshold,
        )

    def passes(self, candles: List[Candle]) -> bool:
        """Return True when current volatility is in the allowed range."""
        if self.mode == VolatilityFilterMode.NONE:
            return True

        cat = self._vf.categorize(candles)

        if cat == VolatilityCategory.UNKNOWN:
            logger.debug("volatility_filter: BLOCKED — insufficient data")
            return False

        if self.mode == VolatilityFilterMode.MEDIUM_ONLY:
            allowed = cat == VolatilityCategory.MEDIUM
        elif self.mode == VolatilityFilterMode.LOW_AND_MEDIUM:
            allowed = cat in (VolatilityCategory.LOW, VolatilityCategory.MEDIUM)
        elif self.mode == VolatilityFilterMode.MEDIUM_AND_HIGH:
            allowed = cat in (VolatilityCategory.MEDIUM, VolatilityCategory.HIGH)
        else:
            allowed = True

        if not allowed:
            atr_pct = self._vf.atr_pct(candles)
            logger.debug(
                "volatility_filter: BLOCKED — cat=%s atr_pct=%.2f%% | %s",
                cat.value,
                atr_pct or 0.0,
                candles[-1].symbol if candles else "?",
            )
        return allowed

    def current_category(self, candles: List[Candle]) -> VolatilityCategory:
        return self._vf.categorize(candles)

    def current_atr_pct(self, candles: List[Candle]) -> Optional[float]:
        return self._vf.atr_pct(candles)
