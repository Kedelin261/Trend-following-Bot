"""Signal data model for Phase 2 Signal Engine.

The Signal object is the sole output of the signal engine.
No execution code. No broker references. No order types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class SignalType(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    NONE  = "NONE"


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(str, Enum):
    STRONG   = "STRONG"
    MODERATE = "MODERATE"
    WEAK     = "WEAK"


@dataclass
class Signal:
    """Fully-populated signal object produced by the SignalEngine.

    Downstream consumers (Phase 3 Risk Engine, Phase 4 Backtester, etc.)
    read this object and never interact with the signal engine internals.
    """

    symbol:             str
    timeframe:          str
    timestamp:          datetime
    signal_type:        SignalType
    trend_direction:    TrendDirection
    strength_score:     float           # 0–100
    breakout_detected:  bool
    volume_confirmed:   Optional[bool]  # None = data unavailable (e.g. Forex)
    support_level:      Optional[float]
    resistance_level:   Optional[float]
    close_price:        float
    reasoning:          List[str] = field(default_factory=list)
    ema50:              Optional[float] = field(default=None)
    ema200:             Optional[float] = field(default=None)

    @property
    def strength_category(self) -> SignalStrength:
        """Derive STRONG / MODERATE / WEAK from the numeric score."""
        if self.strength_score >= 70:
            return SignalStrength.STRONG
        if self.strength_score >= 40:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def is_actionable(self) -> bool:
        """True when signal is LONG or SHORT (not NONE)."""
        return self.signal_type != SignalType.NONE
