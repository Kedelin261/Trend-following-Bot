"""Strategy family taxonomy for Phase 6.0 Edge Discovery.

Defines the six new strategy families under research plus the
MOMENTUM_ROTATION benchmark (Phase 5.x rejected candidate).

No execution logic lives here — this is metadata only.
"""

from enum import Enum
from typing import Dict


class FamilyID(str, Enum):
    """Canonical identifiers for each strategy family under research."""

    TREND_PERSISTENCE        = "TREND_PERSISTENCE"
    BREAKOUT_CONTINUATION    = "BREAKOUT_CONTINUATION"
    RELATIVE_STRENGTH        = "RELATIVE_STRENGTH"
    MARKET_LEADERSHIP        = "MARKET_LEADERSHIP"
    VOLATILITY_TRANSITION    = "VOLATILITY_TRANSITION"
    MULTI_TIMEFRAME_ALIGNMENT = "MULTI_TIMEFRAME_ALIGNMENT"
    MOMENTUM_ROTATION        = "MOMENTUM_ROTATION"  # benchmark only


FAMILY_DESCRIPTIONS: Dict[FamilyID, str] = {
    FamilyID.TREND_PERSISTENCE: (
        "Enter after ≥20 consecutive bars above EMA50 with a proximity "
        "pullback. Hypothesis: sustained trend persistence above a key "
        "moving average signals continuation, not exhaustion."
    ),
    FamilyID.BREAKOUT_CONTINUATION: (
        "Enter breakouts preceded by ≥10-bar ATR compression. Hypothesis: "
        "consolidation before breakout improves continuation quality vs "
        "random breakouts."
    ),
    FamilyID.RELATIVE_STRENGTH: (
        "Enter assets ranking top-2 by rolling 20-bar return vs the 8-asset "
        "universe. Hypothesis: relative strength rank creates persistent edge "
        "through cross-asset momentum."
    ),
    FamilyID.MARKET_LEADERSHIP: (
        "Enter when 10-bar ROC exceeds median universe ROC by a threshold. "
        "Hypothesis: leadership assets move first and continuation follows "
        "the sector/market leader."
    ),
    FamilyID.VOLATILITY_TRANSITION: (
        "Enter on expansion signal following confirmed compression state. "
        "Hypothesis: expansion preceded by compression is directional; "
        "random expansion is not."
    ),
    FamilyID.MULTI_TIMEFRAME_ALIGNMENT: (
        "Enter when weekly, monthly, and quarterly trend proxies all confirm "
        "bullish direction from daily data. Hypothesis: higher-timeframe "
        "alignment produces more durable edge."
    ),
    FamilyID.MOMENTUM_ROTATION: (
        "BENCHMARK — Enter when short/medium ROC positive and EMA20>EMA50. "
        "Researched and rejected in Phases 5.0-5.4."
    ),
}
