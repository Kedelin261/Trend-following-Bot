"""Density Engine — Phase 4.7 master orchestrator.

Runs ADX, volatility, breakout, and asset-expansion research to find
the optimal configuration that maximises trade frequency while
preserving edge quality.

DensityProfile captures the result so future phases can test multiple
configurations without rewriting research code.

Promotion criteria (portfolio-level):
  Aggregate Trades ≥ 100
  Profit Factor   ≥ 1.50
  Expectancy      > $0
  Max Drawdown    < 15 %
  ≥ 2 assets individually pass

No broker code. No API calls. Research only.
"""

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.density.adx_density_research import ADXDensityResearcher, ADXDensityResult
from src.density.asset_expansion import AssetExpansionResearcher, AssetExpansionResult
from src.density.breakout_density_research import BreakoutDensityResearcher, BreakoutDensityResult
from src.density.portfolio_density_analyzer import PortfolioDensityAnalyzer, PortfolioDensityReport
from src.density.signal_density_analyzer import SignalDensityAnalyzer, SignalDensityMetrics
from src.density.volatility_density_research import VolatilityDensityResearcher, VolatilityDensityResult
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE
from src.refinement.volatility_trade_filter import VolatilityFilterMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DensityProfile
# ---------------------------------------------------------------------------

@dataclass
class DensityProfile:
    """Configuration for a density-optimised strategy variant.

    Used to communicate research findings to future phases without
    requiring any code modifications in downstream modules.
    """

    name:                str
    adx_threshold:       float
    breakout_threshold:  float
    volatility_mode:     VolatilityFilterMode
    assets:              List[str]
    description:         str = ""

    def to_strategy_profile(self, base: StrategyProfile = None) -> StrategyProfile:
        """Create a StrategyProfile from this density config and an optional base."""
        src = base or V2_PROFILE
        return replace(
            src,
            name                = self.name,
            description         = self.description or f"Density: {self.name}",
            adx_threshold       = self.adx_threshold,
            breakout_threshold  = self.breakout_threshold,
            volatility_mode     = self.volatility_mode,
        )

    def run_backtest(
        self,
        symbol:  str,
        candles: List[Candle],
        config:  dict,
    ) -> BacktestResults:
        """Backtest this density profile on a single asset."""
        profile = self.to_strategy_profile()
        engine  = profile.build_backtest_engine(config)
        return engine.run(candles)


# V2 baseline as a DensityProfile (for comparison)
V2_DENSITY_BASELINE = DensityProfile(
    name               = "V2-Baseline",
    adx_threshold      = 25.0,
    breakout_threshold = 0.0100,
    volatility_mode    = VolatilityFilterMode.MEDIUM_ONLY,
    assets             = ["SPY", "VOO", "DIA"],
    description        = "V2 strategy with original filter settings",
)


# ---------------------------------------------------------------------------
# DensityReport
# ---------------------------------------------------------------------------

@dataclass
class DensityReport:
    """Complete density optimisation findings."""

    baseline_asset_results:    Dict[str, BacktestResults]
    adx_results:               List[ADXDensityResult]
    volatility_results:        List[VolatilityDensityResult]
    breakout_results:          List[BreakoutDensityResult]
    asset_expansion_results:   List[AssetExpansionResult]
    density_metrics:           List[SignalDensityMetrics]
    portfolio_report:          PortfolioDensityReport

    best_density_profile:      DensityProfile
    best_asset_results:        Dict[str, BacktestResults]
    best_portfolio_report:     PortfolioDensityReport

    promoted:                  bool
    recommendation:            str
    notes:                     List[str] = field(default_factory=list)
    warnings:                  List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DensityEngine
# ---------------------------------------------------------------------------

class DensityEngine:
    """Orchestrates all density research and produces a DensityReport."""

    def __init__(
        self,
        config:         dict,
        base_profile:   StrategyProfile = None,
        primary_symbol: str = "SPY",
    ) -> None:
        self._config        = config
        self._base          = base_profile or V2_PROFILE
        self._primary       = primary_symbol
        self._portfolio_ana = PortfolioDensityAnalyzer()
        self._density_ana   = SignalDensityAnalyzer()

    def run(
        self,
        asset_candles:   Dict[str, List[Candle]],
        run_adx:         bool = True,
        run_vol:         bool = True,
        run_breakout:    bool = True,
        run_expansion:   bool = True,
    ) -> DensityReport:
        """Execute full density research suite."""
        if not asset_candles:
            raise ValueError("asset_candles must not be empty")

        primary_candles = asset_candles.get(
            self._primary, next(iter(asset_candles.values()))
        )
        candle_counts = {s: len(c) for s, c in asset_candles.items()}

        # ---- 1. Baseline -------------------------------------------------
        logger.info("density_engine: step 1/6 — baseline backtest")
        baseline_results = self._run_all_assets(asset_candles, self._base)
        density_metrics  = self._density_ana.analyze_multi(baseline_results, candle_counts)
        portfolio_report = self._portfolio_ana.analyze(baseline_results, candle_counts)

        # ---- 2. ADX research -------------------------------------------
        adx_results: List[ADXDensityResult] = []
        if run_adx:
            logger.info("density_engine: step 2/6 — ADX density research")
            adx_results = ADXDensityResearcher(self._config, self._base).research(
                primary_candles
            )

        # ---- 3. Volatility research -------------------------------------
        vol_results: List[VolatilityDensityResult] = []
        if run_vol:
            logger.info("density_engine: step 3/6 — volatility density research")
            vol_results = VolatilityDensityResearcher(self._config, self._base).research(
                primary_candles
            )

        # ---- 4. Breakout research ----------------------------------------
        brk_results: List[BreakoutDensityResult] = []
        if run_breakout:
            logger.info("density_engine: step 4/6 — breakout density research")
            brk_results = BreakoutDensityResearcher(self._config, self._base).research(
                primary_candles
            )

        # ---- 5. Asset expansion ------------------------------------------
        exp_results: List[AssetExpansionResult] = []
        if run_expansion:
            logger.info("density_engine: step 5/6 — asset expansion research")
            exp_results = AssetExpansionResearcher(self._config, self._base).research(
                asset_candles
            )

        # ---- 6. Best profile selection -----------------------------------
        logger.info("density_engine: step 6/6 — selecting best density profile")
        best_profile = self._select_best_profile(
            adx_results, vol_results, brk_results, exp_results, asset_candles
        )

        # Run backtest with best profile on all assets
        best_asset_results   = self._run_best_profile(best_profile, asset_candles)
        best_portfolio_report = self._portfolio_ana.analyze(best_asset_results, candle_counts)

        promoted    = best_portfolio_report.promoted
        recommendation = (
            "PROMOTE to Phase 5 (Trade Journal & Analytics) — "
            "portfolio generates sufficient high-quality trade opportunities."
            if promoted else
            "DO NOT PROMOTE — Return to research. "
            "Portfolio does not yet generate enough quality trades."
        )

        # Notes & warnings
        notes, warnings = self._compile_notes(
            baseline_results, best_portfolio_report, adx_results, vol_results, brk_results
        )

        logger.info(
            "density_engine: complete | promoted=%s best=%s portfolio_trades=%d",
            promoted,
            best_profile.name,
            best_portfolio_report.total_trades,
        )

        return DensityReport(
            baseline_asset_results  = baseline_results,
            adx_results             = adx_results,
            volatility_results      = vol_results,
            breakout_results        = brk_results,
            asset_expansion_results = exp_results,
            density_metrics         = density_metrics,
            portfolio_report        = portfolio_report,
            best_density_profile    = best_profile,
            best_asset_results      = best_asset_results,
            best_portfolio_report   = best_portfolio_report,
            promoted                = promoted,
            recommendation          = recommendation,
            notes                   = notes,
            warnings                = warnings,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_all_assets(
        self,
        asset_candles: Dict[str, List[Candle]],
        profile:       StrategyProfile,
    ) -> Dict[str, BacktestResults]:
        results = {}
        for sym, candles in asset_candles.items():
            if candles:
                engine = profile.build_backtest_engine(self._config)
                results[sym] = engine.run(candles)
        return results

    def _select_best_profile(
        self,
        adx_results:  List[ADXDensityResult],
        vol_results:  List[VolatilityDensityResult],
        brk_results:  List[BreakoutDensityResult],
        exp_results:  List[AssetExpansionResult],
        asset_candles: Dict[str, List[Candle]],
    ) -> DensityProfile:
        """Select the best filter combination from research results."""
        # ADX: lowest quality-preserving threshold
        adx_researcher = ADXDensityResearcher(self._config, self._base)
        best_adx = adx_researcher.best_threshold(adx_results) or self._base.adx_threshold

        # Volatility: most permissive quality-preserving mode
        vol_researcher = VolatilityDensityResearcher(self._config, self._base)
        best_vol = vol_researcher.best_mode(vol_results) or self._base.volatility_mode

        # Breakout: lowest quality-preserving threshold
        brk_researcher = BreakoutDensityResearcher(self._config, self._base)
        best_brk = brk_researcher.best_threshold(brk_results) or self._base.breakout_threshold

        # Assets: base universe + approved expansions
        base_assets    = ["SPY", "VOO", "DIA"]
        expansion_rec  = [r.symbol for r in exp_results if r.recommended]
        best_assets    = base_assets + [s for s in expansion_rec if s not in base_assets]

        return DensityProfile(
            name               = "Density-Optimised",
            adx_threshold      = best_adx,
            breakout_threshold = best_brk,
            volatility_mode    = best_vol,
            assets             = best_assets,
            description        = (
                f"ADX≥{best_adx:.0f} | {best_vol.value} | "
                f"Brk{best_brk*100:.2f}% | {len(best_assets)} assets"
            ),
        )

    def _run_best_profile(
        self,
        profile:       DensityProfile,
        asset_candles: Dict[str, List[Candle]],
    ) -> Dict[str, BacktestResults]:
        results = {}
        strat   = profile.to_strategy_profile()
        for sym in profile.assets:
            candles = asset_candles.get(sym, [])
            if candles:
                engine     = strat.build_backtest_engine(self._config)
                results[sym] = engine.run(candles)
        return results

    @staticmethod
    def _compile_notes(baseline, best_portfolio, adx_res, vol_res, brk_res):
        notes, warnings = [], []

        baseline_trades = sum(bt.total_trades for bt in baseline.values())
        if best_portfolio.total_trades > baseline_trades:
            notes.append(
                f"Best profile increased portfolio trades: "
                f"{baseline_trades} → {best_portfolio.total_trades}"
            )

        if adx_res:
            best_adx = max(adx_res, key=lambda r: r.density_score)
            if best_adx.meets_quality:
                notes.append(
                    f"ADX ≥ {best_adx.threshold:.0f} gives best density "
                    f"({best_adx.trade_count} trades)"
                )

        if vol_res:
            best_vol = max(vol_res, key=lambda r: r.density_score)
            if best_vol.meets_quality:
                notes.append(
                    f"Volatility mode {best_vol.mode.value} gives best density "
                    f"({best_vol.trade_count} trades)"
                )

        if brk_res:
            best_brk = max(brk_res, key=lambda r: r.density_score)
            if best_brk.meets_quality:
                notes.append(
                    f"Breakout ≥ {best_brk.threshold*100:.2f}% gives best density "
                    f"({best_brk.trade_count} trades)"
                )

        if not best_portfolio.promoted:
            warnings.append(
                "Portfolio trade count still insufficient for statistically reliable results. "
                "Consider adding more assets or extending historical data."
            )

        return notes, warnings
