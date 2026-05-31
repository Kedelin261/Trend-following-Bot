"""Breakout quality research — compares strategy performance across
different breakout confirmation thresholds.

Research methodology: run separate backtests for each configuration and
report results side-by-side.  No automatic adoption of the best result.
The quant team reviews the table and decides whether a threshold change
is warranted by a statistically significant improvement.

No broker code. No API calls. Research only.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults, StrategyHealth
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.risk.risk_engine import RiskEngine
from src.signals.breakout_detector import BreakoutDetector
from src.signals.signal_engine import SignalEngine
from src.signals.support_resistance import SupportResistanceDetector
from src.signals.trend_detector import TrendDetector
from src.signals.volume_confirmation import VolumeConfirmation

logger = logging.getLogger(__name__)

# Research configurations — not production settings
BREAKOUT_THRESHOLDS  = [0.0025, 0.0050, 0.0075, 0.0100]
VOLUME_MULTIPLIERS   = [1.0, 1.25, 1.5, 2.0]

SAFEGUARD_MIN_TRADES = 30   # never recommend a config below this count


@dataclass
class BreakoutResearchConfig:
    """A single breakout threshold / volume multiplier research configuration."""

    breakout_threshold:  float
    description:         str = field(default="")

    def __post_init__(self) -> None:
        if not self.description:
            self.description = f"Breakout ≥ {self.breakout_threshold * 100:.2f}%"


@dataclass
class BreakoutQualityResult:
    """Result of one breakout configuration backtest."""

    config:        BreakoutResearchConfig
    results:       BacktestResults
    health:        StrategyHealth
    note:          str = ""

    @property
    def sufficient(self) -> bool:
        return self.results.total_trades >= SAFEGUARD_MIN_TRADES


class BreakoutQualityResearcher:
    """Runs multiple backtests with varying breakout thresholds.

    SAFEGUARD: results with fewer than SAFEGUARD_MIN_TRADES trades are
    flagged as statistically insufficient and excluded from recommendations.
    """

    def __init__(
        self,
        config:    dict,
        min_warmup: int = 210,
    ) -> None:
        self._config    = config
        self._min_warmup = min_warmup

    def research(
        self,
        candles:    List[Candle],
        thresholds: List[float] = None,
    ) -> List[BreakoutQualityResult]:
        """Run a backtest for each breakout threshold and return ranked results."""
        thresholds = thresholds or BREAKOUT_THRESHOLDS
        results    = []

        for threshold in thresholds:
            cfg = BreakoutResearchConfig(breakout_threshold=threshold)
            logger.info("breakout_research: testing threshold=%.4f", threshold)
            bt = self._run_backtest(candles, threshold)
            health = StrategyHealth.evaluate(bt, min_trades=SAFEGUARD_MIN_TRADES)
            note = "" if bt.total_trades >= SAFEGUARD_MIN_TRADES else (
                f"INSUFFICIENT SAMPLE: {bt.total_trades} trades < {SAFEGUARD_MIN_TRADES} minimum"
            )
            results.append(BreakoutQualityResult(cfg, bt, health, note))

        return sorted(results, key=lambda r: r.results.expectancy, reverse=True)

    def best_threshold(
        self, results: List[BreakoutQualityResult]
    ) -> Optional[BreakoutQualityResult]:
        """Return the threshold with highest expectancy among sufficient results."""
        sufficient = [r for r in results if r.sufficient]
        return max(sufficient, key=lambda r: r.results.expectancy) if sufficient else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_backtest(self, candles: List[Candle], threshold: float) -> BacktestResults:
        bt_cfg = self._config.get("backtest", {})
        engine = BacktestEngine(
            signal_engine = SignalEngine(
                breakout_detector = BreakoutDetector(threshold=threshold),
            ),
            risk_engine   = RiskEngine.from_config(self._config),
            portfolio     = Portfolio(float(bt_cfg.get("starting_balance", 10_000))),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            min_warmup = self._min_warmup,
        )
        return engine.run(candles)
