"""Research Engine — Phase 4.5 master orchestrator.

Coordinates all research modules to answer the core question:
'Under what conditions does this strategy have a repeatable edge?'

The engine is designed to be run once per research cycle, not in
production.  It processes pre-fetched candle data (no live connections).

No broker code. No API calls. Research only.
"""

import logging
from typing import Dict, List, Optional

from src.data.models import Candle
from src.research.adx_filter import ADXFilter
from src.research.benchmark_engine import BenchmarkEngine
from src.research.breakout_quality import BreakoutQualityResearcher
from src.research.market_regime import MarketRegimeDetector
from src.research.multi_asset_runner import MultiAssetRunner
from src.research.parameter_sweep import ParameterSweep
from src.research.strategy_analyzer import ResearchReport, StrategyAnalyzer
from src.research.volatility_filter import VolatilityFilter

logger = logging.getLogger(__name__)

# Default multi-asset universe
DEFAULT_ASSETS = [
    "SPY", "QQQ", "VOO", "VTI",
    "DIA", "IWM", "XLK", "XLF", "XLE",
]


class ResearchEngine:
    """Orchestrates all Phase 4.5 research modules.

    Parameters
    ----------
    config     : settings dict (backtest, risk, research sections)
    min_warmup : minimum bars before first signal (default 210 for EMA200)
    """

    def __init__(
        self,
        config:     dict,
        min_warmup: int = 210,
    ) -> None:
        self._config     = config
        self._min_warmup = min_warmup

        # Instantiate sub-modules
        self._regime_detector  = MarketRegimeDetector()
        self._adx_filter       = ADXFilter()
        self._vol_filter       = VolatilityFilter()
        self._breakout_res     = BreakoutQualityResearcher(config, min_warmup)
        self._param_sweep      = ParameterSweep(config, min_warmup)
        self._multi_runner     = MultiAssetRunner(config, min_warmup)
        self._bench_engine     = BenchmarkEngine()
        self._analyzer         = StrategyAnalyzer()

    def run(
        self,
        asset_candles:  Dict[str, List[Candle]],
        primary_symbol: Optional[str] = None,
        run_sweep:      bool = True,
        run_breakout:   bool = True,
    ) -> ResearchReport:
        """Execute the full research suite on provided candle data.

        Parameters
        ----------
        asset_candles  : {symbol: candle_list} for all assets to analyse
        primary_symbol : the symbol used for regime/ADX/vol/sweep analysis
                         (defaults to the first key in asset_candles)
        run_sweep      : whether to run parameter sweep (slow)
        run_breakout   : whether to run breakout quality research (slow)
        """
        if not asset_candles:
            raise ValueError("asset_candles must not be empty")

        primary = primary_symbol or next(iter(asset_candles))
        primary_candles = asset_candles[primary]

        logger.info(
            "research_engine: starting | primary=%s assets=%d",
            primary,
            len(asset_candles),
        )

        # ---- 1. Multi-asset backtest --------------------------------
        logger.info("research_engine: step 1/7 — multi-asset backtest")
        asset_results = self._multi_runner.run(asset_candles)

        # ---- 2. Regime analysis ------------------------------------
        logger.info("research_engine: step 2/7 — regime analysis")
        # Use trades from primary asset for regime slicing
        primary_result = next(
            (r for r in asset_results if r.symbol == primary), None
        )
        regime_analysis = {}
        if primary_result and primary_result.results.trades:
            regime_analysis = self._regime_detector.analyze_trades_by_regime(
                primary_result.results.trades, primary_candles
            )

        # ---- 3. ADX filter analysis --------------------------------
        logger.info("research_engine: step 3/7 — ADX filter analysis")
        adx_results = []
        if primary_result and primary_result.results.trades:
            adx_results = self._adx_filter.compare_thresholds(
                primary_result.results.trades, primary_candles
            )

        # ---- 4. Volatility analysis --------------------------------
        logger.info("research_engine: step 4/7 — volatility analysis")
        vol_results = {}
        if primary_result and primary_result.results.trades:
            vol_results = self._vol_filter.analyze_trades_by_volatility(
                primary_result.results.trades, primary_candles
            )

        # ---- 5. Breakout quality -----------------------------------
        breakout_results = []
        if run_breakout:
            logger.info("research_engine: step 5/7 — breakout quality research")
            breakout_results = self._breakout_res.research(primary_candles)
        else:
            logger.info("research_engine: step 5/7 — breakout research skipped")

        # ---- 6. Parameter sweep ------------------------------------
        sweep_results = []
        if run_sweep:
            logger.info("research_engine: step 6/7 — parameter sweep")
            sweep_results = self._param_sweep.sweep(primary_candles)
        else:
            logger.info("research_engine: step 6/7 — parameter sweep skipped")

        # ---- 7. Benchmark comparison -------------------------------
        logger.info("research_engine: step 7/7 — benchmark comparison")
        benchmark_results = {}
        for ar in asset_results:
            candles = asset_candles.get(ar.symbol, [])
            if candles:
                benchmark_results[ar.symbol] = self._bench_engine.compare(
                    ar.results, candles
                )

        # ---- Synthesise report -------------------------------------
        report = self._analyzer.analyze(
            asset_results      = asset_results,
            regime_performance = regime_analysis,
            adx_results        = adx_results,
            volatility_results = vol_results,
            breakout_results   = breakout_results,
            sweep_results      = sweep_results,
            benchmark_results  = benchmark_results,
        )

        logger.info(
            "research_engine: complete | edge_confirmed=%s recommended=%s",
            report.edge_confirmed,
            report.recommended_assets,
        )
        return report
