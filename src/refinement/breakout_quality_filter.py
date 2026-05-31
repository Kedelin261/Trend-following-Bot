"""Breakout quality filter for Strategy V2 — enforces higher breakout standards.

Research finding: higher breakout thresholds reduce false signals.
V2 requires close > resistance × (1 + 1.0%) instead of 0.25%.

Research capability: compare multiple threshold/volume combinations
to find the optimal balance of trade count vs quality.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults, StrategyHealth
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.risk.risk_engine import RiskEngine
from src.signals.breakout_detector import BreakoutDetector
from src.signals.signal_engine import SignalEngine
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)

SAFEGUARD_MIN_TRADES = 30
RESEARCH_THRESHOLDS  = [0.0100, 0.0125, 0.0150]   # 1.0 %, 1.25 %, 1.5 %


@dataclass
class BreakoutResearchResult:
    """Result of one breakout threshold research backtest."""

    threshold:    float
    results:      BacktestResults
    health:       StrategyHealth
    note:         str = ""

    @property
    def sufficient(self) -> bool:
        return self.results.total_trades >= SAFEGUARD_MIN_TRADES


class BreakoutQualityFilter:
    """Research tool: tests breakout threshold combinations in V2 context.

    Unlike Phase 4.5 BreakoutQualityResearcher, this class builds
    a V2-configured engine (EMA20/50 trend + filters) for each test.

    SAFEGUARD: results below SAFEGUARD_MIN_TRADES are flagged; the
    'best_threshold' method only returns statistically sufficient results.
    """

    def __init__(
        self,
        config:        dict,
        ema_fast:      int   = 20,
        ema_slow:      int   = 50,
        min_warmup:    int   = 210,
    ) -> None:
        self._config    = config
        self._ema_fast  = ema_fast
        self._ema_slow  = ema_slow
        self._min_warmup = min_warmup

    def research(
        self,
        candles:    List[Candle],
        thresholds: List[float] = None,
    ) -> List[BreakoutResearchResult]:
        """Run backtests for each threshold and return ranked results."""
        thresholds = thresholds or RESEARCH_THRESHOLDS
        results    = []

        for thresh in thresholds:
            logger.info("breakout_quality_v2: testing threshold=%.4f", thresh)
            bt     = self._run(candles, thresh)
            health = StrategyHealth.evaluate(bt, min_trades=SAFEGUARD_MIN_TRADES)
            note   = (
                "" if bt.total_trades >= SAFEGUARD_MIN_TRADES else
                f"⚠ INSUFFICIENT SAMPLE: {bt.total_trades} trades"
            )
            results.append(BreakoutResearchResult(thresh, bt, health, note))

        results.sort(key=lambda r: r.results.expectancy, reverse=True)
        return results

    def best_threshold(
        self, results: List[BreakoutResearchResult]
    ) -> Optional[float]:
        """Best expectancy threshold among sufficient results."""
        sufficient = [r for r in results if r.sufficient]
        return max(sufficient, key=lambda r: r.results.expectancy).threshold \
               if sufficient else None

    def _run(self, candles: List[Candle], threshold: float) -> BacktestResults:
        bt_cfg = self._config.get("backtest", {})
        start  = float(bt_cfg.get("starting_balance", 10_000))

        engine = BacktestEngine(
            signal_engine = SignalEngine(
                trend_detector    = TrendDetector(self._ema_fast, self._ema_slow),
                breakout_detector = BreakoutDetector(threshold),
                min_candles       = self._ema_slow + 10,
            ),
            risk_engine   = RiskEngine.from_config(self._config),
            portfolio     = Portfolio(start),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            min_warmup = self._min_warmup,
        )
        return engine.run(candles)
