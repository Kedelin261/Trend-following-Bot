"""Filter profiles — Phase 5.4 validation scenarios.

Defines the 5 scenarios to validate Phase 5.3 edge-destroyer findings.
Each FilterProfile is a plain data container describing what to exclude.

CRITICAL: These profiles are read-only descriptors.
They do NOT modify entry logic, signal generation, or risk engine.
They only gate which already-generated trade candidates are accepted.

Phase 5.3 findings being validated:
  - SCHD:          PF=0.95, Exp=-$2.52 (negative expectancy)
  - Quality 60-69: PF=0.91, Exp=-$4.76 (no edge)
  - HIGH_VOL:      PF=0.93, Exp=-$3.11 (negative expectancy)

Anti-curve-fitting: these exact three findings and nothing else.
No threshold tuning. No searching for better cutoffs.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class FilterProfile:
    """Immutable descriptor for one validation scenario.

    Parameters
    ----------
    name              : Human-readable scenario name (e.g. "REMOVE_SCHD")
    excluded_assets   : frozenset of ticker symbols to exclude entirely
    excluded_score_lo : Low end of quality score band to exclude (inclusive)
    excluded_score_hi : High end of quality score band to exclude (inclusive)
                        Set both to None to skip quality filter
    exclude_high_vol  : True to reject trades in HIGH_VOL regime
    description       : One-line description for reports
    """

    name:              str
    excluded_assets:   FrozenSet[str]
    excluded_score_lo: Optional[float]
    excluded_score_hi: Optional[float]
    exclude_high_vol:  bool
    description:       str

    @property
    def has_asset_filter(self) -> bool:
        return bool(self.excluded_assets)

    @property
    def has_quality_filter(self) -> bool:
        return (
            self.excluded_score_lo is not None
            and self.excluded_score_hi is not None
        )

    @property
    def has_vol_filter(self) -> bool:
        return self.exclude_high_vol

    @property
    def filter_summary(self) -> str:
        parts = []
        if self.has_asset_filter:
            parts.append(f"excl_assets={sorted(self.excluded_assets)}")
        if self.has_quality_filter:
            parts.append(
                f"excl_score=[{self.excluded_score_lo},{self.excluded_score_hi}]"
            )
        if self.has_vol_filter:
            parts.append("excl_HIGH_VOL")
        return "; ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# The five validation scenarios — exactly as specified in Phase 5.4 brief
# ---------------------------------------------------------------------------

SCENARIO_BASELINE = FilterProfile(
    name              = "BASELINE",
    excluded_assets   = frozenset(),
    excluded_score_lo = None,
    excluded_score_hi = None,
    exclude_high_vol  = False,
    description       = "No changes — current MOMENTUM_ROTATION, all assets",
)

SCENARIO_REMOVE_SCHD = FilterProfile(
    name              = "REMOVE_SCHD",
    excluded_assets   = frozenset({"SCHD"}),
    excluded_score_lo = None,
    excluded_score_hi = None,
    exclude_high_vol  = False,
    description       = "Exclude SCHD entirely (Phase 5.3: PF=0.95, Exp=-$2.52)",
)

SCENARIO_REMOVE_QUALITY_60_69 = FilterProfile(
    name              = "REMOVE_QUALITY_60_69",
    excluded_assets   = frozenset(),
    excluded_score_lo = 60.0,
    excluded_score_hi = 69.9,
    exclude_high_vol  = False,
    description       = (
        "Reject trades with signal score 60-69 "
        "(Phase 5.3: PF=0.91, Exp=-$4.76, NO EDGE)"
    ),
)

SCENARIO_REMOVE_HIGH_VOL = FilterProfile(
    name              = "REMOVE_HIGH_VOL",
    excluded_assets   = frozenset(),
    excluded_score_lo = None,
    excluded_score_hi = None,
    exclude_high_vol  = True,
    description       = (
        "Reject trades in HIGH_VOL regime "
        "(Phase 5.3: PF=0.93, Exp=-$3.11)"
    ),
)

SCENARIO_COMBINED = FilterProfile(
    name              = "COMBINED_FILTERS",
    excluded_assets   = frozenset({"SCHD"}),
    excluded_score_lo = 60.0,
    excluded_score_hi = 69.9,
    exclude_high_vol  = True,
    description       = (
        "All three Phase 5.3 filters: "
        "excl SCHD + excl score 60-69 + excl HIGH_VOL"
    ),
)

ALL_SCENARIOS: Tuple[FilterProfile, ...] = (
    SCENARIO_BASELINE,
    SCENARIO_REMOVE_SCHD,
    SCENARIO_REMOVE_QUALITY_60_69,
    SCENARIO_REMOVE_HIGH_VOL,
    SCENARIO_COMBINED,
)
