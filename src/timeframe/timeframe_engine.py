"""Timeframe Engine — Phase 4.8 master orchestrator.

Runs the best density-optimised strategy (EMA20/50, ADX≥27, Brk≥1%,
MEDIUM+HIGH vol) across D1, H4, H1, W1 and their combinations.

Promotion criteria:
  Aggregate Unique Trades ≥ 100
  Combined PF ≥ 1.50
  Combined Expectancy > $0
  Combined DD < 15 %
  Signal Overlap < 50 %
  ≥ 2 assets passing individually

No broker code. No API calls. Research only.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.timeframe.multi_timeframe_research import (
    MultiTimeframeResearcher,
    MultiTimeframeResult,
)
from src.timeframe.opportunity_analyzer import (
    OpportunityAnalyzer,
    PortfolioOpportunityMetrics,
)
from src.timeframe.signal_overlap_analyzer import SignalOverlapAnalyzer
from src.timeframe.timeframe_backtester import TimeframeBacktestResult, TimeframeBacktester
from src.timeframe.timeframe_comparator import TimeframeComparator, TimeframeComparisonResult
from src.timeframe.timeframe_profile import (
    ALL_COMBOS,
    ALL_SINGLE_PROFILES,
    BEST_DENSITY_PROFILE,
    D1_PROFILE,
    MultiTimeframeProfile,
    TimeframeProfile,
)

logger = logging.getLogger(__name__)

PORTFOLIO_MIN_TRADES   = 100
PORTFOLIO_MIN_PF       = 1.50
PORTFOLIO_MIN_EXP      = 0.0
PORTFOLIO_MAX_DD       = 15.0
PORTFOLIO_MIN_ASSETS   = 2
MAX_ACCEPTABLE_OVERLAP = 0.50


@dataclass
class TimeframeReport:
    """Complete timeframe expansion research report."""

    # Single-timeframe results: {timeframe: {symbol: result}}
    single_tf_results:      Dict[str, Dict[str, TimeframeBacktestResult]]

    # Multi-timeframe combination results
    combo_results:          List[MultiTimeframeResult]

    # D1 vs each other TF comparisons per symbol
    comparisons:            Dict[str, List[TimeframeComparisonResult]]

    # Opportunity flow per timeframe
    opportunity_metrics:    Dict[str, PortfolioOpportunityMetrics]

    # Best single timeframe and best combination
    best_single_tf:         Optional[str]
    best_combo:             Optional[MultiTimeframeResult]

    # Promotion decision
    promoted:               bool
    promotion_reason:       str
    recommendation:         str

    notes:                  List[str] = field(default_factory=list)
    warnings:               List[str] = field(default_factory=list)


class TimeframeEngine:
    """Orchestrates all Phase 4.8 timeframe expansion research.

    Parameters
    ----------
    config : settings dict
    """

    def __init__(self, config: dict) -> None:
        self._config      = config
        self._backtester  = TimeframeBacktester(config, BEST_DENSITY_PROFILE)
        self._comparator  = TimeframeComparator()
        self._overlap_ana = SignalOverlapAnalyzer(MAX_ACCEPTABLE_OVERLAP)
        self._oppo_ana    = OpportunityAnalyzer()
        self._multi_res   = MultiTimeframeResearcher(
            config, self._backtester, self._overlap_ana
        )

    def run(
        self,
        symbol_candles_by_tf: Dict[str, Dict[str, List[Candle]]],
        single_profiles:      List[TimeframeProfile] = None,
        combo_profiles:       List[MultiTimeframeProfile] = None,
    ) -> TimeframeReport:
        """Execute the full timeframe research suite.

        Parameters
        ----------
        symbol_candles_by_tf : {symbol: {timeframe: [candles]}}
        single_profiles      : timeframes to test individually (default: D1/H4/H1/W1)
        combo_profiles       : combinations to test (default: D1+H4, D1+H1, D1+H4+H1)
        """
        if not symbol_candles_by_tf:
            raise ValueError("symbol_candles_by_tf must not be empty")

        singles = single_profiles or ALL_SINGLE_PROFILES
        combos  = combo_profiles  or ALL_COMBOS

        # ---- 1. Single-timeframe backtests --------------------------
        logger.info("timeframe_engine: step 1/5 — single-TF backtests")
        single_tf: Dict[str, Dict[str, TimeframeBacktestResult]] = {}
        for tf_profile in singles:
            tf = tf_profile.timeframe
            logger.info("  running %s...", tf)
            single_tf[tf] = {}
            for sym, tf_candles in symbol_candles_by_tf.items():
                candles = tf_candles.get(tf, [])
                if candles:
                    single_tf[tf][sym] = self._backtester.backtest(sym, candles, tf_profile)

        # ---- 2. Comparisons (D1 baseline) ---------------------------
        logger.info("timeframe_engine: step 2/5 — D1 comparisons")
        comparisons: Dict[str, List[TimeframeComparisonResult]] = {}
        d1_results  = single_tf.get("D1", {})
        for sym in d1_results:
            d1_res   = d1_results[sym]
            others   = [
                single_tf[tf][sym]
                for tf in single_tf if tf != "D1" and sym in single_tf.get(tf, {})
            ]
            comparisons[sym] = self._comparator.compare_all_to_baseline(d1_res, others)

        # ---- 3. Multi-timeframe combinations ------------------------
        logger.info("timeframe_engine: step 3/5 — multi-TF combinations")
        combo_results = self._multi_res.research_all(symbol_candles_by_tf, combos)

        # ---- 4. Opportunity analysis --------------------------------
        logger.info("timeframe_engine: step 4/5 — opportunity analysis")
        oppo: Dict[str, PortfolioOpportunityMetrics] = {}
        for tf_profile in singles:
            tf = tf_profile.timeframe
            if tf in single_tf and single_tf[tf]:
                bt_map  = {sym: r.results for sym, r in single_tf[tf].items()}
                cnt_map = {
                    sym: len(symbol_candles_by_tf[sym].get(tf, []))
                    for sym in bt_map
                }
                oppo[tf] = self._oppo_ana.analyze_portfolio(bt_map, tf_profile, cnt_map)

        # ---- 5. Promotion decision ----------------------------------
        logger.info("timeframe_engine: step 5/5 — promotion evaluation")
        best_combo = self._multi_res.best_combo(combo_results)
        best_single_tf = self._best_single(single_tf, oppo)

        promoted, reason = self._evaluate_promotion(
            single_tf, best_combo, oppo
        )
        recommendation = (
            "PROMOTE to Phase 5 (Trade Journal & Analytics) — "
            "multi-timeframe approach provides sufficient opportunities."
            if promoted else
            "DO NOT PROMOTE — Return to research. "
            "Insufficient unique trades or quality not maintained across timeframes."
        )

        notes, warnings = self._compile_notes(single_tf, combo_results, best_combo)

        return TimeframeReport(
            single_tf_results   = single_tf,
            combo_results       = combo_results,
            comparisons         = comparisons,
            opportunity_metrics = oppo,
            best_single_tf      = best_single_tf,
            best_combo          = best_combo,
            promoted            = promoted,
            promotion_reason    = reason,
            recommendation      = recommendation,
            notes               = notes,
            warnings            = warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_promotion(
        self,
        single_tf:  Dict[str, Dict[str, TimeframeBacktestResult]],
        best_combo: Optional[MultiTimeframeResult],
        oppo:       Dict[str, PortfolioOpportunityMetrics],
    ) -> tuple:
        # Option A: best combination meets all criteria
        if best_combo and best_combo.is_viable:
            trades  = best_combo.unique_trades
            pf      = best_combo.combined_pf
            exp     = best_combo.combined_expectancy
            dd      = best_combo.max_drawdown
            overlap = best_combo.overlap.average_overlap_pct

            assets_ok = sum(
                1
                for tf_sym_res in best_combo.per_tf_results.values()
                for r in tf_sym_res.values()
                if r.meets_quality
            ) >= PORTFOLIO_MIN_ASSETS

            if trades >= PORTFOLIO_MIN_TRADES:
                if pf < PORTFOLIO_MIN_PF:
                    return False, f"Combined PF {pf:.2f} < {PORTFOLIO_MIN_PF}"
                if exp <= PORTFOLIO_MIN_EXP:
                    return False, f"Combined expectancy ${exp:.2f} ≤ $0"
                if dd > PORTFOLIO_MAX_DD:
                    return False, f"Max drawdown {dd:.1f}% > {PORTFOLIO_MAX_DD:.1f}%"
                if overlap >= MAX_ACCEPTABLE_OVERLAP:
                    return False, f"Overlap {overlap*100:.0f}% ≥ {MAX_ACCEPTABLE_OVERLAP*100:.0f}%"
                if not assets_ok:
                    return False, f"Fewer than {PORTFOLIO_MIN_ASSETS} assets pass quality"
                return True, (
                    f"Best combo '{best_combo.label}': "
                    f"{trades} unique trades, PF={pf:.2f}, "
                    f"Exp=${exp:.2f}, DD={dd:.1f}%, Overlap={overlap*100:.0f}%"
                )

        # Option B: a single timeframe alone meets criteria
        for tf, sym_res in single_tf.items():
            if not sym_res:
                continue
            total = sum(r.trade_count for r in sym_res.values())
            if total >= PORTFOLIO_MIN_TRADES:
                passing = sum(1 for r in sym_res.values() if r.meets_quality)
                if passing >= PORTFOLIO_MIN_ASSETS:
                    return True, f"Single timeframe {tf} alone meets all criteria"

        total_best = best_combo.unique_trades if best_combo else 0
        return False, (
            f"Best combo has only {total_best} unique trades "
            f"(need ≥ {PORTFOLIO_MIN_TRADES}). "
            "Consider adding more assets or extending data history."
        )

    @staticmethod
    def _best_single(
        single_tf: Dict[str, Dict[str, TimeframeBacktestResult]],
        oppo:      Dict[str, PortfolioOpportunityMetrics],
    ) -> Optional[str]:
        """Return the single timeframe with the most quality-passing trades."""
        best_tf, best_count = None, 0
        for tf, sym_res in single_tf.items():
            count = sum(
                r.trade_count for r in sym_res.values() if r.meets_quality
            )
            if count > best_count:
                best_count, best_tf = count, tf
        return best_tf

    @staticmethod
    def _compile_notes(single_tf, combo_results, best_combo):
        notes, warnings = [], []

        viable = [r for r in combo_results if r.is_viable]
        if viable:
            best = max(viable, key=lambda r: r.unique_trades)
            notes.append(
                f"Best viable combo: {best.label} — "
                f"{best.unique_trades} unique trades, "
                f"PF={best.combined_pf:.2f}, Exp=${best.combined_expectancy:.2f}"
            )

        for tf, sym_res in single_tf.items():
            passing = sum(1 for r in sym_res.values() if r.meets_quality)
            if passing == 0:
                warnings.append(
                    f"{tf}: no assets pass quality thresholds — "
                    "exclude from multi-TF combinations"
                )

        if not viable:
            warnings.append(
                "No viable multi-timeframe combination found. "
                "Consider relaxing density parameters or adding more assets."
            )

        return notes, warnings
