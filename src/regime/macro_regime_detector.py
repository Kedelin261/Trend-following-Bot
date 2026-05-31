"""Macro regime detector — EXPANSION / RECOVERY / CONTRACTION / CRISIS.

Infers the economic environment from price structure alone.
No external economic data APIs are required.

Classification logic (in priority order):
  1. CRISIS     : price dropped > 20 % in the last 60 bars (rapid crash)
  2. EXPANSION  : price above EMA200 AND EMA200 slope is positive
  3. RECOVERY   : price above EMA200 BUT EMA200 slope is flat or negative
  4. CONTRACTION: price below EMA200

Rationale:
  - Post-2020 bull run → EXPANSION
  - 2019 late-cycle → RECOVERY
  - 2022 rate-hike bear → CONTRACTION
  - 2008/2020 crash phase → CRISIS

No broker code. No API calls. Candle data only.
"""

import logging
from enum import Enum
from typing import List, Optional

from src.data.models import Candle
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)

CRISIS_DECLINE_THRESHOLD = 0.20   # 20 % rapid decline → CRISIS
CRISIS_LOOKBACK          = 60     # bars for rapid-decline check
EMA_SLOPE_LOOKBACK       = 20     # bars for EMA200 slope calculation


class MacroRegime(str, Enum):
    EXPANSION   = "EXPANSION"
    RECOVERY    = "RECOVERY"
    CONTRACTION = "CONTRACTION"
    CRISIS      = "CRISIS"
    UNKNOWN     = "UNKNOWN"


class MacroRegimeDetector:
    """Infers macro economic environment from long-term price structure.

    Parameters
    ----------
    ema_period     : long-term EMA period (default 200)
    slope_lookback : bars for EMA slope computation (default 20)
    crisis_decline : rapid decline threshold for CRISIS detection (default 0.20)
    crisis_lookback: bar window for rapid-decline check (default 60)
    """

    def __init__(
        self,
        ema_period:      int   = 200,
        slope_lookback:  int   = EMA_SLOPE_LOOKBACK,
        crisis_decline:  float = CRISIS_DECLINE_THRESHOLD,
        crisis_lookback: int   = CRISIS_LOOKBACK,
    ) -> None:
        self.ema_period      = ema_period
        self.slope_lookback  = slope_lookback
        self.crisis_decline  = crisis_decline
        self.crisis_lookback = crisis_lookback
        # fast_period must be strictly less than ema_period for TrendDetector
        fast = max(1, ema_period // 4)
        self._det = TrendDetector(fast_period=fast, slow_period=ema_period)

    def classify(self, candles: List[Candle]) -> MacroRegime:
        """Return the macro regime at the most recent bar."""
        n = len(candles)
        if n < self.ema_period:
            return MacroRegime.UNKNOWN

        closes  = [c.close for c in candles]
        current = closes[-1]
        ema200  = self._det.calculate_ema(closes, self.ema_period)

        if ema200 is None:
            return MacroRegime.UNKNOWN

        # Priority 1: CRISIS — rapid significant decline
        lookback_n   = min(self.crisis_lookback, n)
        recent_high  = max(closes[-lookback_n:])
        if recent_high > 0 and (recent_high - current) / recent_high >= self.crisis_decline:
            return MacroRegime.CRISIS

        # EMA200 slope
        if n > self.ema_period + self.slope_lookback:
            prev_ema = self._det.calculate_ema(
                closes[: -self.slope_lookback], self.ema_period
            )
            slope = ((ema200 - prev_ema) / prev_ema) if prev_ema else 0.0
        else:
            slope = 0.0

        # Priority 2/3/4: based on price vs EMA200 and slope direction
        if current > ema200:
            return MacroRegime.EXPANSION if slope >= 0 else MacroRegime.RECOVERY
        return MacroRegime.CONTRACTION

    def classify_at_timestamp(
        self, candles: List[Candle], timestamp
    ) -> MacroRegime:
        history = [c for c in candles if c.timestamp <= timestamp]
        return self.classify(history) if history else MacroRegime.UNKNOWN
