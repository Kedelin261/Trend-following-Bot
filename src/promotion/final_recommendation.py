"""Final recommendation — synthesises all Phase 4.9 evidence into a decision.

Decision logic:
  PROMOTE_TO_PHASE_5  : promotion validation passed AND robustness ≥ MARGINAL
  RETURN_TO_RESEARCH  : promotion validation failed OR robustness = UNSTABLE

A MARGINAL robustness with a PROMOTE outcome includes an explicit note
to monitor the strategy closely after promotion.

No broker code. No API calls. Pure logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.promotion.promotion_validator import PromotionValidationResult
from src.promotion.robustness_checker import RobustnessResult


class PromotionStatus(str, Enum):
    PROMOTE = "PROMOTE_TO_PHASE_5"
    RETURN  = "RETURN_TO_RESEARCH"


@dataclass
class FinalRecommendationResult:
    """The final PROMOTE / RETURN decision with full reasoning."""

    status:          PromotionStatus
    headline:        str
    primary_reasons: List[str]
    caveats:         List[str] = field(default_factory=list)
    next_steps:      List[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.status == PromotionStatus.PROMOTE


class FinalRecommendationEngine:
    """Generates a final promotion recommendation from research evidence."""

    def generate(
        self,
        validation:  PromotionValidationResult,
        robustness:  RobustnessResult,
        history_candle_count: int,
        strategy_name: str = "Best-Density-V2",
    ) -> FinalRecommendationResult:
        """Combine validation and robustness to reach a final decision."""

        if not validation.all_passed:
            return self._return_decision(validation, robustness)

        if robustness.rating == "UNSTABLE":
            return FinalRecommendationResult(
                status   = PromotionStatus.RETURN,
                headline = "RETURN TO RESEARCH — Strategy unstable across market periods",
                primary_reasons = [
                    f"Promotion criteria passed but strategy rated UNSTABLE",
                    f"Only {robustness.windows_passing}/3 historical windows pass quality",
                ] + robustness.notes,
                next_steps = [
                    "Investigate which market regimes cause underperformance",
                    "Consider adding regime-specific filters",
                    "Extend data history before re-testing",
                ],
            )

        # PROMOTE (ROBUST or MARGINAL)
        caveats = []
        if robustness.rating == "MARGINAL":
            caveats.append(
                "Robustness is MARGINAL (2/3 windows pass) — monitor strategy "
                "performance carefully after Phase 5 deployment"
            )

        return FinalRecommendationResult(
            status   = PromotionStatus.PROMOTE,
            headline = f"PROMOTE {strategy_name} TO PHASE 5 — Trade Journal & Analytics",
            primary_reasons = validation.pass_reasons + [
                f"Robustness: {robustness.rating} ({robustness.windows_passing}/3 windows pass)",
                f"History length: {history_candle_count:,} candles",
            ],
            caveats = caveats,
            next_steps = [
                "Proceed to Phase 5: Trade Journal & Analytics",
                "Record every signal with full metadata",
                "Track realised vs expected expectancy",
                "Review after first 30 live/paper signals",
            ],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _return_decision(
        self,
        validation: PromotionValidationResult,
        robustness: RobustnessResult,
    ) -> FinalRecommendationResult:
        reasons = validation.failure_reasons.copy()
        if robustness.rating == "UNSTABLE":
            reasons += robustness.notes

        # Targeted next steps based on which criteria failed
        next_steps = []
        if not validation.trades_ok:
            next_steps.append(
                f"Trade count is {validation.actual_trades} — need {100}. "
                "Options: add XLV/SCHD/VTI, extend historical data, or relax ADX/volatility filter."
            )
        if not validation.pf_ok:
            next_steps.append(
                "Profit factor below 1.50 — return to Phase 4.7/4.8 density research."
            )
        if not validation.expectancy_ok:
            next_steps.append(
                "Negative expectancy — strategy is not viable. Return to Phase 4.5."
            )
        if not validation.drawdown_ok:
            next_steps.append(
                f"Drawdown {validation.actual_dd:.1f}% exceeds 15% — "
                "tighten stop-loss multiplier or add DD protection."
            )
        if not next_steps:
            next_steps.append("Review all Phase 4.x research modules.")

        return FinalRecommendationResult(
            status          = PromotionStatus.RETURN,
            headline        = "RETURN TO RESEARCH — Promotion criteria not met",
            primary_reasons = reasons,
            next_steps      = next_steps,
        )
