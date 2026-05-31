"""Promotion Engine — Phase 4.9 master orchestrator.

This is the final gate before Phase 5 (Trade Journal & Analytics).
The strategy is LOCKED.  No parameter changes are permitted.

Workflow:
  1. History expansion — test with more candles
  2. Candidate asset validation — test XLV/SCHD/VTI one at a time
  3. Robustness checks — verify stability across market periods
  4. Promotion validation — apply the five promotion criteria
  5. Final recommendation — PROMOTE or RETURN_TO_RESEARCH

LOCKED STRATEGY:
  EMA 20/50 | Bull filter | ADX ≥ 27 | Breakout ≥ 1% | MEDIUM+HIGH vol

Verification: if the strategy profile parameters have been changed,
the engine raises StrategyTamperedError.

No broker code. No API calls. Research only.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.promotion.asset_validation import AssetValidator, AssetValidationResult
from src.promotion.final_recommendation import (
    FinalRecommendationEngine,
    FinalRecommendationResult,
    PromotionStatus,
)
from src.promotion.history_expansion import HistoryExpansionResearcher, HistoryExpansionResult
from src.promotion.promotion_validator import PromotionValidationResult, PromotionValidator
from src.promotion.robustness_checker import RobustnessChecker, RobustnessResult
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locked strategy parameters — any deviation raises StrategyTamperedError
# ---------------------------------------------------------------------------

LOCKED_PARAMS = {
    "ema_fast":            20,
    "ema_slow":            50,
    "adx_threshold":       27.0,
    "breakout_threshold":  0.0100,
    "volatility_mode":     VolatilityFilterMode.MEDIUM_AND_HIGH,
    "require_bull_regime": True,
}


class StrategyTamperedError(RuntimeError):
    """Raised when the strategy profile has been modified from locked values."""


def verify_strategy_locked(profile: StrategyProfile) -> None:
    """Raise StrategyTamperedError if profile deviates from LOCKED_PARAMS."""
    mismatches = []
    for param, expected in LOCKED_PARAMS.items():
        actual = getattr(profile, param, None)
        if actual != expected:
            mismatches.append(f"{param}: expected={expected!r} got={actual!r}")
    if mismatches:
        raise StrategyTamperedError(
            "Strategy parameters have been modified. "
            "Phase 4.9 only validates the locked strategy. "
            f"Mismatches: {'; '.join(mismatches)}"
        )


# ---------------------------------------------------------------------------
# PromotionReport
# ---------------------------------------------------------------------------

@dataclass
class PromotionReport:
    """Complete Phase 4.9 promotion validation report."""

    strategy_name:               str
    trades:                      int
    profit_factor:               float
    expectancy:                  float
    drawdown:                    float
    assets_passing:              int
    history_length:              int
    robustness_rating:           str
    promotion_status:            str      # PromotionStatus value
    promotion_reason:            str

    history_expansion_results:   List[HistoryExpansionResult]
    asset_validation_results:    List[AssetValidationResult]
    robustness_result:           RobustnessResult
    validation_result:           PromotionValidationResult
    final_recommendation:        FinalRecommendationResult

    approved_candidate_assets:   List[str] = field(default_factory=list)
    notes:                       List[str] = field(default_factory=list)
    warnings:                    List[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.promotion_status == PromotionStatus.PROMOTE.value


# ---------------------------------------------------------------------------
# PromotionEngine
# ---------------------------------------------------------------------------

class PromotionEngine:
    """Orchestrates all Phase 4.9 validation steps.

    Parameters
    ----------
    config          : settings dict (backtest, risk sections)
    strategy_profile: must equal BEST_DENSITY_PROFILE (verified before run)
    """

    def __init__(
        self,
        config:           dict,
        strategy_profile: StrategyProfile = None,
    ) -> None:
        self._config   = config
        self._profile  = strategy_profile or BEST_DENSITY_PROFILE

    def run(
        self,
        asset_candles:      Dict[str, List[Candle]],
        candidate_candles:  Dict[str, List[Candle]] = None,
        skip_lock_check:    bool = False,
    ) -> PromotionReport:
        """Execute all validation steps and return a PromotionReport.

        Parameters
        ----------
        asset_candles      : core portfolio candles {symbol: candles}
        candidate_candles  : optional expansion candidates {symbol: candles}
        skip_lock_check    : set True only in tests with modified mini-profiles
        """
        if not asset_candles:
            raise ValueError("asset_candles must not be empty")

        if not skip_lock_check:
            verify_strategy_locked(self._profile)

        candidate_candles = candidate_candles or {}

        # ---- 1. History expansion -----------------------------------
        logger.info("promotion_engine: step 1/5 — history expansion")
        history_res = HistoryExpansionResearcher(self._config, self._profile).research(
            asset_candles
        )
        # Use the largest quality-passing window
        best_history = HistoryExpansionResearcher(
            self._config, self._profile
        ).best_window(history_res)
        best_candles = asset_candles  # default to all available
        history_length = max(len(c) for c in asset_candles.values()) if asset_candles else 0

        if best_history:
            count = best_history.candle_count
            best_candles = {
                sym: c[-count:] if len(c) >= count else c
                for sym, c in asset_candles.items()
            }
            history_length = count

        # ---- 2. Candidate asset validation -------------------------
        logger.info("promotion_engine: step 2/5 — candidate asset validation")
        asset_validator = AssetValidator(self._config, self._profile)
        asset_validation_results = asset_validator.validate_all(
            best_candles, candidate_candles
        )
        approved_candidates = asset_validator.approved_candidates(asset_validation_results)

        # Build final asset universe
        final_candles = {**best_candles}
        for sym in approved_candidates:
            if sym in candidate_candles:
                final_candles[sym] = candidate_candles[sym]

        # ---- 3. Robustness checks ----------------------------------
        logger.info("promotion_engine: step 3/5 — robustness checks")
        robustness = RobustnessChecker(self._config, self._profile).check(final_candles)

        # ---- 4. Promotion validation -------------------------------
        logger.info("promotion_engine: step 4/5 — promotion validation")
        final_bt     = self._run_portfolio(final_candles)
        total_trades = sum(bt.total_trades for bt in final_bt.values())
        pf, exp      = self._aggregate(final_bt)
        worst_dd     = max((bt.max_drawdown for bt in final_bt.values()), default=0.0)
        assets_pass  = sum(
            1 for bt in final_bt.values()
            if bt.profit_factor >= 1.5
            and bt.expectancy > 0
            and bt.max_drawdown <= 15.0
        )

        validator   = PromotionValidator()
        validation  = validator.validate(total_trades, pf, exp, worst_dd, assets_pass)

        # ---- 5. Final recommendation -------------------------------
        logger.info("promotion_engine: step 5/5 — final recommendation")
        rec_engine = FinalRecommendationEngine()
        final_rec  = rec_engine.generate(
            validation   = validation,
            robustness   = robustness,
            history_candle_count = history_length,
            strategy_name        = self._profile.name,
        )

        # Notes and warnings
        notes, warnings = self._compile_notes(
            history_res, asset_validation_results, robustness, validation
        )

        logger.info(
            "promotion_engine: complete | status=%s trades=%d pf=%.2f exp=%.2f",
            final_rec.status.value, total_trades, pf if not math.isinf(pf) else 99.0, exp,
        )

        return PromotionReport(
            strategy_name             = self._profile.name,
            trades                    = total_trades,
            profit_factor             = pf if not math.isinf(pf) else 99.0,
            expectancy                = exp,
            drawdown                  = worst_dd,
            assets_passing            = assets_pass,
            history_length            = history_length,
            robustness_rating         = robustness.rating,
            promotion_status          = final_rec.status.value,
            promotion_reason          = "\n".join(final_rec.primary_reasons),
            history_expansion_results = history_res,
            asset_validation_results  = asset_validation_results,
            robustness_result         = robustness,
            validation_result         = validation,
            final_recommendation      = final_rec,
            approved_candidate_assets = approved_candidates,
            notes                     = notes,
            warnings                  = warnings,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_portfolio(
        self, asset_candles: Dict[str, List[Candle]]
    ) -> Dict[str, BacktestResults]:
        return {
            sym: self._profile.build_backtest_engine(self._config).run(c)
            for sym, c in asset_candles.items()
            if c
        }

    @staticmethod
    def _aggregate(asset_results: Dict[str, BacktestResults]):
        all_trades = [t for bt in asset_results.values() for t in bt.trades]
        if not all_trades:
            return 0.0, 0.0
        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if t.is_loss]
        gross_wins   = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        exp = sum(t.pnl for t in all_trades) / len(all_trades)
        return pf, exp

    @staticmethod
    def _compile_notes(history_res, asset_res, robustness, validation):
        notes, warnings = [], []

        # History expansion
        passing_windows = [r for r in history_res if r.meets_threshold and r.meets_quality]
        if passing_windows:
            best = max(passing_windows, key=lambda r: r.candle_count)
            notes.append(
                f"History window of {best.candle_count:,} candles "
                f"produces {best.total_trades} trades with PF={best.pf_str}"
            )

        # Approved candidates
        approved = [r.candidate_symbol for r in asset_res if r.approved]
        if approved:
            notes.append(f"Approved expansion assets: {', '.join(approved)}")

        # Robustness
        notes.append(f"Strategy robustness: {robustness.rating}")

        # Warnings
        if robustness.rating == "MARGINAL":
            warnings.append(
                "Robustness is MARGINAL — monitor strategy performance closely after promotion"
            )
        if not validation.trades_ok:
            warnings.append(
                f"Only {validation.actual_trades} trades "
                f"(need {100}) — consider adding more history or expansion assets"
            )

        return notes, warnings
