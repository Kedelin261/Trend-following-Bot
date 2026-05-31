"""Timeframe profiles for Phase 4.8 expansion research.

A TimeframeProfile bundles the timeframe identifier with frequency
metadata.  Predefined profiles cover the four research targets:
D1, H4, H1, W1.

The best density-optimised strategy (EMA20/50, ADX≥27, Brk≥1%,
MEDIUM+HIGH vol) runs unchanged — only the candle timeframe varies.

No broker code. No API calls. Pure configuration.
"""

from dataclasses import dataclass, field
from typing import List

from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE
from src.refinement.volatility_trade_filter import VolatilityFilterMode

# ---------------------------------------------------------------------------
# Best density-optimised profile from Phase 4.7
# ---------------------------------------------------------------------------
from dataclasses import replace as _replace

BEST_DENSITY_PROFILE: StrategyProfile = _replace(
    V2_PROFILE,
    name             = "Best-Density-V2",
    description      = (
        "Phase 4.7 best profile: EMA20/50, Bull filter, ADX≥27, "
        "Brk≥1.00%, Medium+High volatility"
    ),
    adx_threshold    = 27.0,
    volatility_mode  = VolatilityFilterMode.MEDIUM_AND_HIGH,
    breakout_threshold = 0.0100,
)


# ---------------------------------------------------------------------------
# TimeframeProfile
# ---------------------------------------------------------------------------

@dataclass
class TimeframeProfile:
    """Metadata describing a single candle timeframe for research.

    Parameters
    ----------
    name          : human-readable label (e.g. "D1", "H4")
    timeframe     : standard registry key (must match candle.timeframe)
    bars_per_year : approximate trading bars in a calendar year
    min_warmup    : bars required before first signal attempt
    description   : optional detail
    """

    name:         str
    timeframe:    str
    bars_per_year: int
    min_warmup:   int = 210
    description:  str = ""

    @property
    def is_intraday(self) -> bool:
        return self.timeframe in ("M1", "M5", "M15", "M30", "H1", "H4")

    @property
    def is_daily_or_higher(self) -> bool:
        return self.timeframe in ("D1", "W1")

    def years_from_bars(self, bar_count: int) -> float:
        """Convert a candle count to approximate calendar years."""
        return bar_count / max(self.bars_per_year, 1)


# ---------------------------------------------------------------------------
# MultiTimeframeProfile — a combination of timeframes
# ---------------------------------------------------------------------------

@dataclass
class MultiTimeframeProfile:
    """Defines a research combination of two or more timeframes.

    Used by MultiTimeframeResearcher to run a composite backtest.
    """

    name:     str
    profiles: List[TimeframeProfile]
    description: str = ""

    @property
    def timeframes(self) -> List[str]:
        return [p.timeframe for p in self.profiles]

    @property
    def label(self) -> str:
        return " + ".join(p.name for p in self.profiles)


# ---------------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------------

#   bars_per_year approximations (US equity session):
#   D1  : 252 trading days
#   H4  : 252 × 1.625 ≈ 409 (6.5h session / 4h)
#   H1  : 252 × 6.5   ≈ 1638 (6.5h session / 1h)
#   W1  : 52 calendar weeks

D1_PROFILE = TimeframeProfile(
    name="D1", timeframe="D1", bars_per_year=252, min_warmup=210,
    description="Daily bars — primary timeframe"
)
H4_PROFILE = TimeframeProfile(
    name="H4", timeframe="H4", bars_per_year=409, min_warmup=210,
    description="4-hour bars — intermediate frequency"
)
H1_PROFILE = TimeframeProfile(
    name="H1", timeframe="H1", bars_per_year=1638, min_warmup=210,
    description="1-hour bars — high frequency"
)
W1_PROFILE = TimeframeProfile(
    name="W1", timeframe="W1", bars_per_year=52, min_warmup=210,
    description="Weekly bars — low frequency, strong trend filter"
)

ALL_SINGLE_PROFILES = [D1_PROFILE, H4_PROFILE, H1_PROFILE, W1_PROFILE]

# Multi-timeframe combinations to research
COMBO_D1_H4   = MultiTimeframeProfile("D1+H4",    [D1_PROFILE, H4_PROFILE])
COMBO_D1_H1   = MultiTimeframeProfile("D1+H1",    [D1_PROFILE, H1_PROFILE])
COMBO_D1_H4_H1 = MultiTimeframeProfile("D1+H4+H1", [D1_PROFILE, H4_PROFILE, H1_PROFILE])
ALL_COMBOS    = [COMBO_D1_H4, COMBO_D1_H1, COMBO_D1_H4_H1]
