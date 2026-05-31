"""Filter profiles for Phase 4.11 regime-aware entry filtering.

Each FilterProfile is immutable (frozen dataclass) and describes which
market environments to block.  Profiles use frozensets so they are
hashable and safe to use as dict keys.

Predefined profiles implement the exclusions identified in Phase 4.10:
  - STRONG_BULL generates disproportionate losses
  - EXTREME_VOL increases whipsaw risk
  - CRISIS/CRASH are clearly destructive environments

No broker code.  No strategy parameter changes.  Filter only.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from src.regime.drawdown_environment_detector import DrawdownEnvironment
from src.regime.macro_regime_detector import MacroRegime
from src.regime.market_regime_classifier import MarketRegime
from src.regime.trend_regime_detector import TrendRegime
from src.regime.volatility_regime_detector import VolatilityRegime


@dataclass(frozen=True)
class FilterProfile:
    """Immutable set of regime-exclusion rules.

    Block rules remove trades in specific bad environments.
    Allow rules (whitelists) restrict trades to specific good environments.
    If both are set for the same dimension, block rules take priority.

    Parameters
    ----------
    name                        : unique identifier
    description                 : human-readable explanation
    blocked_market_regimes      : set of MarketRegime values to block
    blocked_volatility_regimes  : set of VolatilityRegime values to block
    blocked_macro_regimes       : set of MacroRegime values to block
    blocked_drawdown_envs       : set of DrawdownEnvironment values to block
    blocked_trend_regimes       : set of TrendRegime values to block
    allowed_market_regimes      : whitelist — if non-empty, only these are allowed
    """

    name:                       str
    description:                str
    blocked_market_regimes:     FrozenSet[MarketRegime]     = field(default_factory=frozenset)
    blocked_volatility_regimes: FrozenSet[VolatilityRegime]  = field(default_factory=frozenset)
    blocked_macro_regimes:      FrozenSet[MacroRegime]       = field(default_factory=frozenset)
    blocked_drawdown_envs:      FrozenSet[DrawdownEnvironment] = field(default_factory=frozenset)
    blocked_trend_regimes:      FrozenSet[TrendRegime]       = field(default_factory=frozenset)
    allowed_market_regimes:     FrozenSet[MarketRegime]      = field(default_factory=frozenset)

    @property
    def has_any_filter(self) -> bool:
        """True when at least one block or allow rule is defined."""
        return bool(
            self.blocked_market_regimes
            or self.blocked_volatility_regimes
            or self.blocked_macro_regimes
            or self.blocked_drawdown_envs
            or self.blocked_trend_regimes
            or self.allowed_market_regimes
        )


# ---------------------------------------------------------------------------
# Predefined research profiles
# ---------------------------------------------------------------------------

NO_FILTER = FilterProfile(
    name        = "NO_FILTER",
    description = "Baseline — no regime filtering applied",
)

AVOID_STRONG_BULL = FilterProfile(
    name                    = "AVOID_STRONG_BULL",
    description             = "Exclude STRONG_BULL: high-extension bull markets",
    blocked_market_regimes  = frozenset({MarketRegime.STRONG_BULL}),
)

AVOID_BEAR = FilterProfile(
    name                    = "AVOID_BEAR",
    description             = "Exclude BEAR and STRONG_BEAR market regimes",
    blocked_market_regimes  = frozenset({MarketRegime.BEAR, MarketRegime.STRONG_BEAR}),
)

AVOID_EXTREME_VOL = FilterProfile(
    name                       = "AVOID_EXTREME_VOL",
    description                = "Exclude EXTREME_VOL: whipsaw environments",
    blocked_volatility_regimes = frozenset({VolatilityRegime.EXTREME_VOL}),
)

AVOID_STRONG_BULL_AND_EXTREME_VOL = FilterProfile(
    name                       = "AVOID_STRONG_BULL_AND_EXTREME_VOL",
    description                = "Exclude STRONG_BULL + EXTREME_VOL",
    blocked_market_regimes     = frozenset({MarketRegime.STRONG_BULL}),
    blocked_volatility_regimes = frozenset({VolatilityRegime.EXTREME_VOL}),
)

BULL_ONLY = FilterProfile(
    name                   = "BULL_ONLY",
    description            = "Whitelist: only trade in BULL regime",
    allowed_market_regimes = frozenset({MarketRegime.BULL}),
)

BULL_AND_STRONG_BULL = FilterProfile(
    name                   = "BULL_AND_STRONG_BULL",
    description            = "Whitelist: BULL and STRONG_BULL only",
    allowed_market_regimes = frozenset({MarketRegime.BULL, MarketRegime.STRONG_BULL}),
)

AVOID_CRISIS_AND_CRASH = FilterProfile(
    name                  = "AVOID_CRISIS_AND_CRASH",
    description           = "Exclude CRISIS macro + CRASH drawdown environments",
    blocked_macro_regimes = frozenset({MacroRegime.CRISIS}),
    blocked_drawdown_envs = frozenset({DrawdownEnvironment.CRASH}),
)

AVOID_WEAK_TREND = FilterProfile(
    name                  = "AVOID_WEAK_TREND",
    description           = "Exclude WEAK_TREND: low-conviction trend environments",
    blocked_trend_regimes = frozenset({TrendRegime.WEAK_TREND}),
)

COMPREHENSIVE_FILTER = FilterProfile(
    name                       = "COMPREHENSIVE_FILTER",
    description                = "Avoid STRONG_BULL + EXTREME_VOL + CRISIS + CRASH",
    blocked_market_regimes     = frozenset({MarketRegime.STRONG_BULL}),
    blocked_volatility_regimes = frozenset({VolatilityRegime.EXTREME_VOL}),
    blocked_macro_regimes      = frozenset({MacroRegime.CRISIS}),
    blocked_drawdown_envs      = frozenset({DrawdownEnvironment.CRASH}),
)

# Research universe — ordered from least to most restrictive
RESEARCH_PROFILES = [
    NO_FILTER,
    AVOID_STRONG_BULL,
    AVOID_BEAR,
    AVOID_EXTREME_VOL,
    AVOID_STRONG_BULL_AND_EXTREME_VOL,
    BULL_ONLY,
    BULL_AND_STRONG_BULL,
    AVOID_CRISIS_AND_CRASH,
    AVOID_WEAK_TREND,
    COMPREHENSIVE_FILTER,
]
