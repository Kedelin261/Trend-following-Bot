"""Regime Stability Engine — Phase 4.10 master orchestrator.

Runs the locked strategy on historical candle data, labels every closed
trade with five regime dimensions, then analyses performance per regime
and simulates hypothetical filters.

The strategy is NOT changed.  Only the analysis layer changes.

Answers the fundamental question:
  'When does the strategy work — and when should it NOT trade?'

No broker code. No API calls. Research only.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.promotion.promotion_engine import verify_strategy_locked
from src.regime.drawdown_environment_detector import DrawdownEnvironmentDetector
from src.regime.macro_regime_detector import MacroRegimeDetector
from src.regime.market_regime_classifier import MarketRegimeClassifier
from src.regime.regime_performance_analyzer import (
    LabelledTrade,
    RegimePerformance,
    RegimePerformanceAnalyzer,
)
from src.regime.regime_trade_filter import (
    FilterSimulationResult,
    RegimeTradeFilter,
)
from src.regime.trend_regime_detector import TrendRegimeDetector
from src.regime.volatility_regime_detector import VolatilityRegimeDetector
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime Stability Report
# ---------------------------------------------------------------------------

@dataclass
class RegimeStabilityReport:
    """Complete Phase 4.10 regime stability research report."""

    labelled_trades:               List[LabelledTrade]
    total_trades_analyzed:         int

    # Per-dimension performance
    market_regime_performance:     Dict[str, RegimePerformance]
    trend_regime_performance:      Dict[str, RegimePerformance]
    volatility_regime_performance: Dict[str, RegimePerformance]
    drawdown_env_performance:      Dict[str, RegimePerformance]
    macro_regime_performance:      Dict[str, RegimePerformance]

    # Best / worst market regime
    best_market_regime:            Optional[str]
    worst_market_regime:           Optional[str]

    # Filter simulations
    filter_simulations:            List[FilterSimulationResult]
    best_filter:                   Optional[FilterSimulationResult]

    # Concentration metrics
    top_profit_regime:             Optional[str]
    top_loss_regime:               Optional[str]
    profit_in_best_regime_pct:     float
    loss_in_worst_regime_pct:      float

    # Assessment
    edge_is_regime_dependent:      bool
    stability_assessment:          str
    recommendations:               List[str] = field(default_factory=list)
    warnings:                      List[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return False  # Phase 4.10 never promotes — research only


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RegimeStabilityEngine:
    """Orchestrates all Phase 4.10 regime stability research.

    Parameters
    ----------
    config           : settings dict (backtest, risk sections)
    strategy_profile : must be BEST_DENSITY_PROFILE (verified before run)
    skip_lock_check  : set True only in unit tests with mini-profiles
    """

    def __init__(
        self,
        config:           dict,
        strategy_profile: StrategyProfile = None,
        skip_lock_check:  bool = False,
    ) -> None:
        self._config   = config
        self._profile  = strategy_profile or BEST_DENSITY_PROFILE
        self._skip_lock = skip_lock_check

        # Regime detectors
        self._market_clf = MarketRegimeClassifier()
        self._trend_det  = TrendRegimeDetector()
        self._vol_det    = VolatilityRegimeDetector()
        self._dd_det     = DrawdownEnvironmentDetector()
        self._macro_det  = MacroRegimeDetector()

        # Analytics
        self._perf_ana   = RegimePerformanceAnalyzer()
        self._trade_filt = RegimeTradeFilter()

    def run(
        self,
        asset_candles: Dict[str, List[Candle]],
    ) -> RegimeStabilityReport:
        """Run the full regime stability research pipeline.

        1. Backtest the locked strategy on all assets
        2. Label every closed trade with its five regime dimensions
        3. Compute per-regime performance metrics
        4. Simulate hypothetical regime filters
        5. Compile and return the RegimeStabilityReport
        """
        if not asset_candles:
            raise ValueError("asset_candles must not be empty")

        if not self._skip_lock:
            verify_strategy_locked(self._profile)

        # ---- 1. Run backtests and label trades ----------------------
        logger.info("regime_engine: step 1/4 — backtests + trade labelling")
        all_labelled: List[LabelledTrade] = []

        for sym, candles in asset_candles.items():
            if not candles:
                continue
            logger.info("regime_engine: backtesting %s (%d candles)", sym, len(candles))
            bt = self._run_backtest(candles)
            for trade in bt.trades:
                all_labelled.append(self._label_trade(trade, candles))

        logger.info(
            "regime_engine: labelled %d trades across %d assets",
            len(all_labelled), len(asset_candles),
        )

        # ---- 2. Per-regime performance --------------------------------
        logger.info("regime_engine: step 2/4 — regime performance analysis")
        all_perf = self._perf_ana.analyze_all(all_labelled)

        market_perf  = all_perf["market_regime"]
        trend_perf   = all_perf["trend_regime"]
        vol_perf     = all_perf["volatility_regime"]
        dd_perf      = all_perf["drawdown_env"]
        macro_perf   = all_perf["macro_regime"]

        # ---- 3. Filter simulations ------------------------------------
        logger.info("regime_engine: step 3/4 — filter simulations")
        filter_results = self._trade_filt.simulate_all(all_labelled)
        best_filt      = self._trade_filt.best_filter(filter_results)

        # ---- 4. Compile report ----------------------------------------
        logger.info("regime_engine: step 4/4 — compiling report")
        report = self._compile_report(
            all_labelled, market_perf, trend_perf, vol_perf, dd_perf, macro_perf,
            filter_results, best_filt,
        )

        logger.info(
            "regime_engine: complete | trades=%d regime_dependent=%s best=%s worst=%s",
            len(all_labelled),
            report.edge_is_regime_dependent,
            report.best_market_regime,
            report.worst_market_regime,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_backtest(self, candles: List[Candle]) -> BacktestResults:
        engine = self._profile.build_backtest_engine(self._config)
        return engine.run(candles)

    def _label_trade(
        self,
        trade,
        candles: List[Candle],
    ) -> LabelledTrade:
        ts = trade.entry_time
        return LabelledTrade(
            trade             = trade,
            market_regime     = self._market_clf.classify_at_timestamp(candles, ts),
            trend_regime      = self._trend_det.classify_at_timestamp(candles, ts),
            volatility_regime = self._vol_det.classify_at_timestamp(candles, ts),
            drawdown_env      = self._dd_det.classify_at_timestamp(candles, ts),
            macro_regime      = self._macro_det.classify_at_timestamp(candles, ts),
        )

    def _compile_report(
        self,
        labelled_trades: List[LabelledTrade],
        market_perf:     Dict[str, RegimePerformance],
        trend_perf:      Dict[str, RegimePerformance],
        vol_perf:        Dict[str, RegimePerformance],
        dd_perf:         Dict[str, RegimePerformance],
        macro_perf:      Dict[str, RegimePerformance],
        filter_results:  List[FilterSimulationResult],
        best_filt:       Optional[FilterSimulationResult],
    ) -> RegimeStabilityReport:

        best_mr  = self._perf_ana.best_regime(market_perf)
        worst_mr = self._perf_ana.worst_regime(market_perf)

        # Regime dependency: best/worst have significantly different expectancy
        edge_dependent = False
        if best_mr and worst_mr and best_mr != worst_mr:
            best_exp  = market_perf[best_mr].expectancy
            worst_exp = market_perf[worst_mr].expectancy
            edge_dependent = (best_exp > 0 and worst_exp < 0) or (
                abs(best_exp - worst_exp) > 20.0 and worst_exp < 0
            )

        # Concentration
        top_profit = max(market_perf.items(),
                         key=lambda x: x[1].profit_concentration,
                         default=(None, None))
        top_loss   = max(market_perf.items(),
                         key=lambda x: x[1].loss_concentration,
                         default=(None, None))

        profit_in_best = market_perf[best_mr].profit_concentration if best_mr else 0.0
        loss_in_worst  = market_perf[worst_mr].loss_concentration  if worst_mr else 0.0

        # Assessment
        if edge_dependent:
            assessment = (
                "EDGE IS REGIME DEPENDENT — strategy performance varies "
                "significantly across market regimes. Research regime filters "
                "before considering promotion."
            )
        elif all(
            p.is_net_positive for p in market_perf.values() if p.trade_count >= 5
        ):
            assessment = (
                "EDGE IS ROBUST — strategy generates positive expectancy "
                "across all sufficiently-sampled market regimes."
            )
        else:
            assessment = (
                "EDGE IS MARGINAL — some regimes are profitable, others are not. "
                "Regime filtering could materially improve robustness."
            )

        # Recommendations
        recs: List[str] = []
        warnings: List[str] = []

        if worst_mr and market_perf[worst_mr].expectancy < 0:
            recs.append(
                f"Implement regime filter to avoid '{worst_mr}' — "
                f"expected loss: ${market_perf[worst_mr].expectancy:.2f}/trade"
            )
        if best_mr:
            recs.append(
                f"'{best_mr}' is the highest-quality environment "
                f"({profit_in_best*100:.0f}% of profits) — "
                "consider prioritising this regime"
            )
        if best_filt and best_filt.improves_quality:
            recs.append(
                f"Simulated filter '{best_filt.filter_description}' "
                f"projects PF={best_filt.pf_str}, "
                f"Exp=${best_filt.projected_expectancy:.2f} "
                f"({best_filt.filtered_trades}/{best_filt.original_trades} trades retained)"
            )
        if len(labelled_trades) < 30:
            warnings.append(
                f"Only {len(labelled_trades)} trades labelled — "
                "regime analysis reliability is low. Extend history."
            )

        return RegimeStabilityReport(
            labelled_trades               = labelled_trades,
            total_trades_analyzed         = len(labelled_trades),
            market_regime_performance     = market_perf,
            trend_regime_performance      = trend_perf,
            volatility_regime_performance = vol_perf,
            drawdown_env_performance      = dd_perf,
            macro_regime_performance      = macro_perf,
            best_market_regime            = best_mr,
            worst_market_regime           = worst_mr,
            filter_simulations            = filter_results,
            best_filter                   = best_filt,
            top_profit_regime             = top_profit[0] if top_profit[0] else None,
            top_loss_regime               = top_loss[0]   if top_loss[0]   else None,
            profit_in_best_regime_pct     = profit_in_best,
            loss_in_worst_regime_pct      = loss_in_worst,
            edge_is_regime_dependent      = edge_dependent,
            stability_assessment          = assessment,
            recommendations               = recs,
            warnings                      = warnings,
        )
