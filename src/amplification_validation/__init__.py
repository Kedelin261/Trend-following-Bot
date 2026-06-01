"""Phase 5.4 — Amplification Validation package.

Validates Phase 5.3 edge-destroyer findings by running 5 scenarios:
  1. BASELINE          — no changes
  2. REMOVE_SCHD       — exclude SCHD asset
  3. REMOVE_QUALITY_60_69 — reject trades with signal score 60-69
  4. REMOVE_HIGH_VOL   — reject trades classified HIGH_VOL
  5. COMBINED          — all three filters applied

Research only. No execution. No broker code. No live trading.
No entry logic modifications. No signal engine changes.
No parameter optimization. No curve fitting.
"""

from src.amplification_validation.filter_profiles import (
    FilterProfile,
    SCENARIO_BASELINE,
    SCENARIO_REMOVE_SCHD,
    SCENARIO_REMOVE_QUALITY_60_69,
    SCENARIO_REMOVE_HIGH_VOL,
    SCENARIO_COMBINED,
    ALL_SCENARIOS,
)
from src.amplification_validation.asset_exclusion_filter import AssetExclusionFilter
from src.amplification_validation.quality_band_filter import QualityBandFilter
from src.amplification_validation.volatility_filter import VolatilityFilter
from src.amplification_validation.amplification_validator import (
    ScenarioResult,
    AmplificationValidator,
)
from src.amplification_validation.amplification_comparator import (
    ScenarioRanking,
    AmplificationComparator,
)
from src.amplification_validation.amplification_report import AmplificationValidationReport

__all__ = [
    "FilterProfile",
    "SCENARIO_BASELINE",
    "SCENARIO_REMOVE_SCHD",
    "SCENARIO_REMOVE_QUALITY_60_69",
    "SCENARIO_REMOVE_HIGH_VOL",
    "SCENARIO_COMBINED",
    "ALL_SCENARIOS",
    "AssetExclusionFilter",
    "QualityBandFilter",
    "VolatilityFilter",
    "ScenarioResult",
    "AmplificationValidator",
    "ScenarioRanking",
    "AmplificationComparator",
    "AmplificationValidationReport",
]
