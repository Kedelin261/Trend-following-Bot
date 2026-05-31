"""Validation Engine — Phase 5.1 master orchestrator.

Coordinates history expansion, asset expansion, scalability analysis,
robustness validation, survivability, and promotion evaluation for each
candidate strategy.

Reuses: EdgeEngine, EdgeStabilityAnalyzer, all Phase 5.0 infrastructure.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface
from src.edge_validation.asset_expansion import AssetExpansionResearcher, AssetExpansionResult
from src.edge_validation.history_expansion import HistoryExpansionResearcher, HistorySliceResult
from src.edge_validation.promotion_candidate_evaluator import (
    PromotionCandidateEvaluator,
    PromotionCandidateResult,
)
from src.edge_validation.robustness_validator import RobustnessValidationResult, RobustnessValidator
from src.edge_validation.scalability_analyzer import ScalabilityResult, analyze_scalability
from src.edge_validation.strategy_survivability import SurvivabilityResult, evaluate_survivability

logger = logging.getLogger(__name__)


@dataclass
class StrategyValidationResult:
    """Complete validation outcome for one strategy."""

    strategy:        StrategyInterface
    history_slices:  List[HistorySliceResult]
    asset_expansion: AssetExpansionResult
    scalability:     ScalabilityResult
    robustness:      RobustnessValidationResult
    survivability:   SurvivabilityResult
    promotion:       PromotionCandidateResult

    @property
    def name(self) -> str:
        return self.strategy.name


@dataclass
class ValidationReport:
    """Top-level Phase 5.1 validation report."""

    strategy_results: List[StrategyValidationResult]
    best_candidate:   Optional[PromotionCandidateResult]
    recommendation:   str
    notes:            List[str] = field(default_factory=list)


class ValidationEngine:
    """Runs all Phase 5.1 validation modules for a set of strategy candidates."""

    def __init__(self, config: dict) -> None:
        self._config       = config
        self._hist_res     = HistoryExpansionResearcher(config)
        self._asset_res    = AssetExpansionResearcher(config)
        self._rob_val      = RobustnessValidator(config)
        self._promo_eval   = PromotionCandidateEvaluator()

    def run(
        self,
        strategies:    List[StrategyInterface],
        asset_candles: Dict[str, List[Candle]],
    ) -> ValidationReport:
        results = []
        for strategy in strategies:
            logger.info("validation_engine: validating %s ...", strategy.name)
            result = self._validate_one(strategy, asset_candles)
            results.append(result)

        candidates = [r.promotion for r in results if r.promotion.is_candidate]
        if candidates:
            best = max(candidates, key=lambda c: c.expectancy)
            recommendation = (
                f"PROMOTE {best.strategy_name} TO PHASE 5.2\n"
                f"  Strategy passed all {len(best.pass_criteria)} promotion criteria."
            )
        else:
            best = None
            top  = max(results, key=lambda r: r.promotion.expectancy) if results else None
            recommendation = "RETURN TO EDGE RESEARCH"
            if top:
                missing = top.promotion.fail_criteria
                recommendation += f"\n  {top.name} closest — failing: {', '.join(missing[:2])}"

        notes = self._compile_notes(results)

        return ValidationReport(
            strategy_results = results,
            best_candidate   = best,
            recommendation   = recommendation,
            notes            = notes,
        )

    # ------------------------------------------------------------------

    def _validate_one(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
    ) -> StrategyValidationResult:
        logger.info("  %s: history expansion...", strategy.name)
        history = self._hist_res.research(strategy, asset_candles)

        logger.info("  %s: asset expansion...", strategy.name)
        assets = self._asset_res.research(strategy, asset_candles)

        logger.info("  %s: scalability...", strategy.name)
        scalability = analyze_scalability(strategy.name, history)

        logger.info("  %s: robustness...", strategy.name)
        robustness = self._rob_val.validate(strategy, asset_candles)

        logger.info("  %s: survivability...", strategy.name)
        survivability = evaluate_survivability(
            strategy.name, history, assets, robustness
        )

        # Use best (largest) history result for promotion evaluation
        max_slice = max(history, key=lambda s: s.actual_bars)
        logger.info("  %s: promotion eval (%d bars)...", strategy.name, max_slice.actual_bars)
        promotion = self._promo_eval.evaluate(
            max_slice.profile, survivability, max_slice.actual_bars
        )

        return StrategyValidationResult(
            strategy        = strategy,
            history_slices  = history,
            asset_expansion = assets,
            scalability     = scalability,
            robustness      = robustness,
            survivability   = survivability,
            promotion       = promotion,
        )

    @staticmethod
    def _compile_notes(results: List[StrategyValidationResult]) -> List[str]:
        notes = []
        for r in results:
            if r.promotion.trades < 100 and r.promotion.profit_factor >= 1.5:
                notes.append(
                    f"{r.name}: quality ✓ but only {r.promotion.trades} trades — "
                    "extend history or add assets to reach 100"
                )
        return notes
