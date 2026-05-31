"""Multi-asset research runner — backtests the strategy on a list of assets.

Answers: 'On which assets does this strategy have a repeatable edge?'

Each asset is backtested independently using the same engine configuration.
Results are ranked by expectancy.  Assets with insufficient trade counts
are flagged and should be excluded from deployment decisions.

No broker code. No API calls. Research only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults, StrategyHealth
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.research.benchmark_engine import BenchmarkEngine, BenchmarkResult
from src.research.market_regime import MarketRegime, MarketRegimeDetector
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.signals.signal_engine import SignalEngine

logger = logging.getLogger(__name__)

SAFEGUARD_MIN_TRADES = 30   # flag assets below this count


@dataclass
class AssetResearchResult:
    """Complete research output for a single asset."""

    symbol:          str
    candle_count:    int
    results:         BacktestResults
    health:          StrategyHealth
    benchmark:       Optional[BenchmarkResult]
    current_regime:  MarketRegime
    note:            str = ""

    @property
    def sufficient(self) -> bool:
        return self.results.total_trades >= SAFEGUARD_MIN_TRADES

    @property
    def recommended(self) -> bool:
        """True when results pass health check AND sample size is adequate."""
        return self.sufficient and self.health.passed

    @property
    def expectancy(self) -> float:
        return self.results.expectancy


class MultiAssetRunner:
    """Runs backtests on multiple assets and returns per-asset research results.

    Parameters
    ----------
    config      : settings dict (backtest, risk sections)
    min_warmup  : minimum bars before first signal
    run_benchmark : include buy-and-hold comparison (adds computation)
    """

    def __init__(
        self,
        config:        dict,
        min_warmup:    int  = 210,
        run_benchmark: bool = True,
    ) -> None:
        self._config       = config
        self._min_warmup   = min_warmup
        self._run_benchmark = run_benchmark
        self._bench_engine = BenchmarkEngine()
        self._regime_det   = MarketRegimeDetector()

    def run(
        self,
        asset_candles: Dict[str, List[Candle]],
    ) -> List[AssetResearchResult]:
        """Backtest each asset and return results sorted by expectancy."""
        results = []

        for symbol, candles in asset_candles.items():
            if not candles:
                logger.warning("multi_asset: no candles for %s — skipping", symbol)
                continue

            logger.info(
                "multi_asset: backtesting %s (%d candles)", symbol, len(candles)
            )
            result = self._run_single(symbol, candles)
            results.append(result)

        results.sort(key=lambda r: r.expectancy, reverse=True)
        self._log_summary(results)
        return results

    def recommended_assets(
        self, results: List[AssetResearchResult]
    ) -> List[str]:
        """Return symbols that pass health check with sufficient sample."""
        return [r.symbol for r in results if r.recommended]

    def avoid_assets(
        self, results: List[AssetResearchResult]
    ) -> List[str]:
        """Return symbols where strategy consistently fails or has no edge."""
        return [
            r.symbol for r in results
            if r.sufficient and not r.health.passed and r.results.expectancy < 0
        ]

    def best_asset(
        self, results: List[AssetResearchResult]
    ) -> Optional[AssetResearchResult]:
        sufficient = [r for r in results if r.sufficient]
        return max(sufficient, key=lambda r: r.expectancy) if sufficient else None

    def worst_asset(
        self, results: List[AssetResearchResult]
    ) -> Optional[AssetResearchResult]:
        sufficient = [r for r in results if r.sufficient]
        return min(sufficient, key=lambda r: r.expectancy) if sufficient else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single(
        self, symbol: str, candles: List[Candle]
    ) -> AssetResearchResult:
        bt_cfg = self._config.get("backtest", {})
        start  = float(bt_cfg.get("starting_balance", 10_000.0))

        engine = BacktestEngine(
            signal_engine = SignalEngine(),
            risk_engine   = RiskEngine.from_config(self._config),
            portfolio     = Portfolio(start),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            min_warmup = self._min_warmup,
            symbol     = symbol,
            timeframe  = candles[0].timeframe if candles else "D1",
        )

        bt     = engine.run(candles)
        health = StrategyHealth.evaluate(
            bt, min_trades=bt_cfg.get("minimum_trades_required", SAFEGUARD_MIN_TRADES)
        )
        bench  = (
            self._bench_engine.compare(bt, candles, start)
            if self._run_benchmark else None
        )
        regime = self._regime_det.classify(candles)
        note   = (
            "" if bt.total_trades >= SAFEGUARD_MIN_TRADES else
            f"INSUFFICIENT SAMPLE: {bt.total_trades} trades"
        )

        return AssetResearchResult(
            symbol         = symbol,
            candle_count   = len(candles),
            results        = bt,
            health         = health,
            benchmark      = bench,
            current_regime = regime,
            note           = note,
        )

    @staticmethod
    def _log_summary(results: List[AssetResearchResult]) -> None:
        recommended = [r.symbol for r in results if r.recommended]
        logger.info(
            "multi_asset_complete: assets=%d recommended=%s",
            len(results),
            recommended or "none",
        )
