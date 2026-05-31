"""Drawdown environment detector — BULL_RECOVERY / CORRECTION / BEAR_MARKET / CRASH.

Detects market stress level by measuring the current drawdown from the
recent peak within a rolling lookback window.

Thresholds (drawdown from recent peak):
  BULL_RECOVERY : < 5 %    (near highs, no meaningful correction)
  CORRECTION    : 5–15 %   (normal pullback within bull market)
  BEAR_MARKET   : 15–30 %  (extended decline, trend broken)
  CRASH         : > 30 %   (crisis environment — 2008/2020 style)

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle

logger = logging.getLogger(__name__)


class DrawdownEnvironment(str, Enum):
    BULL_RECOVERY = "BULL_RECOVERY"
    CORRECTION    = "CORRECTION"
    BEAR_MARKET   = "BEAR_MARKET"
    CRASH         = "CRASH"
    UNKNOWN       = "UNKNOWN"


class DrawdownEnvironmentDetector:
    """Detects market stress level from rolling peak-to-trough drawdown.

    Parameters
    ----------
    lookback         : bars to scan for the recent peak (default 252 = ~1 year D1)
    correction_thresh: drawdown % above which market is in CORRECTION
    bear_thresh      : drawdown % above which market is in BEAR_MARKET
    crash_thresh     : drawdown % above which market is in CRASH
    """

    def __init__(
        self,
        lookback:          int   = 252,
        correction_thresh: float = 0.05,
        bear_thresh:       float = 0.15,
        crash_thresh:      float = 0.30,
    ) -> None:
        self.lookback          = lookback
        self.correction_thresh = correction_thresh
        self.bear_thresh       = bear_thresh
        self.crash_thresh      = crash_thresh

    def current_drawdown_pct(self, candles: List[Candle]) -> Optional[float]:
        """Return current drawdown from recent peak as a fraction (0.0–1.0)."""
        if not candles:
            return None
        closes  = [c.close for c in candles]
        current = closes[-1]
        n       = min(self.lookback, len(closes))
        peak    = max(closes[-n:])
        if peak == 0:
            return None
        return (peak - current) / peak

    def classify(self, candles: List[Candle]) -> DrawdownEnvironment:
        """Return the drawdown environment at the most recent bar."""
        dd = self.current_drawdown_pct(candles)
        if dd is None:
            return DrawdownEnvironment.UNKNOWN
        if dd < self.correction_thresh:
            return DrawdownEnvironment.BULL_RECOVERY
        if dd < self.bear_thresh:
            return DrawdownEnvironment.CORRECTION
        if dd < self.crash_thresh:
            return DrawdownEnvironment.BEAR_MARKET
        return DrawdownEnvironment.CRASH

    def classify_at_timestamp(
        self, candles: List[Candle], timestamp
    ) -> DrawdownEnvironment:
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.classify(history) if history else DrawdownEnvironment.UNKNOWN
