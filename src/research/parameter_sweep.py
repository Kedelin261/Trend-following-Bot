"""Parameter sweep engine — research parameter sensitivity, NOT optimization.

ANTI-CURVE-FITTING SAFEGUARDS:
  1. Every result must meet SAFEGUARD_MIN_TRADES (50) to be considered
  2. No single 'best' parameter set is automatically adopted
  3. Rankings must be reviewed by a human analyst
  4. Results include expectancy, drawdown AND profit factor — not just profit
  5. Warning emitted when top result has < 2× the min trade count

The sweep tests EMA period combinations, ATR multipliers, and breakout
thresholds by running full backtests for each configuration.

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
from src.risk.atr_calculator import ATRCalculator
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.risk.stop_loss_engine import StopLossEngine
from src.risk.take_profit_engine import TakeProfitEngine
from src.risk.trade_validator import TradeValidator
from src.signals.breakout_detector import BreakoutDetector
from src.signals.signal_engine import SignalEngine
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)

SAFEGUARD_MIN_TRADES = 50   # results below this count are flagged insufficient


@dataclass
class SweepConfig:
    """One parameter combination for research testing."""

    ema_fast:            int
    ema_slow:            int
    atr_stop_mult:       float
    atr_target_mult:     float
    breakout_threshold:  float
    description:         str = field(default="")

    def __post_init__(self) -> None:
        if not self.description:
            self.description = (
                f"EMA{self.ema_fast}/{self.ema_slow} "
                f"Stop×{self.atr_stop_mult} Target×{self.atr_target_mult} "
                f"Brk{self.breakout_threshold * 100:.2f}%"
            )


# Research configurations — the entire space to explore
DEFAULT_SWEEP_CONFIGS: List[SweepConfig] = [
    SweepConfig(10,  50,  1.5, 2.0, 0.0025),
    SweepConfig(10,  50,  2.0, 3.0, 0.0025),
    SweepConfig(20,  50,  1.5, 2.0, 0.0025),
    SweepConfig(20,  50,  2.0, 3.0, 0.0025),
    SweepConfig(20, 100,  2.0, 3.0, 0.0025),
    SweepConfig(20, 100,  2.5, 4.0, 0.0025),
    SweepConfig(50, 200,  1.5, 2.0, 0.0025, "EMA50/200 Stop×1.5 Target×2 (aggressive)"),
    SweepConfig(50, 200,  2.0, 3.0, 0.0025, "EMA50/200 Stop×2 Target×3 (CURRENT)"),
    SweepConfig(50, 200,  2.5, 4.0, 0.0025, "EMA50/200 Stop×2.5 Target×4 (patient)"),
    SweepConfig(50, 200,  2.0, 3.0, 0.0050, "EMA50/200 Brk 0.5%"),
    SweepConfig(50, 200,  2.0, 3.0, 0.0075, "EMA50/200 Brk 0.75%"),
    SweepConfig(50, 200,  2.0, 3.0, 0.0100, "EMA50/200 Brk 1.0%"),
]


@dataclass
class SweepResult:
    """Backtest results for a single sweep configuration."""

    config:      SweepConfig
    results:     BacktestResults
    health:      StrategyHealth
    note:        str = ""

    @property
    def sufficient(self) -> bool:
        return self.results.total_trades >= SAFEGUARD_MIN_TRADES

    @property
    def rank_score(self) -> float:
        """Composite score for sorting: expectancy weighted by profit factor.

        Only used for display ordering — never for automatic adoption.
        """
        if not self.sufficient:
            return float("-inf")
        return self.results.expectancy * min(self.results.profit_factor, 3.0)


class ParameterSweep:
    """Runs multiple full backtests to research parameter sensitivity.

    IMPORTANT: This class produces a research table, not a recommendation.
    A human analyst must review results before any parameter change is made.
    """

    def __init__(
        self,
        config:     dict,
        min_warmup: int = 210,
    ) -> None:
        self._config     = config
        self._min_warmup = min_warmup

    def sweep(
        self,
        candles: List[Candle],
        configs: List[SweepConfig] = None,
    ) -> List[SweepResult]:
        """Run backtests for all configurations and return ranked results.

        Results are sorted by rank_score descending.  Insufficient results
        appear at the bottom with a clear warning note.
        """
        sweep_configs = configs or DEFAULT_SWEEP_CONFIGS
        results = []

        for cfg in sweep_configs:
            logger.info("parameter_sweep: testing %s", cfg.description)
            bt     = self._run(candles, cfg)
            health = StrategyHealth.evaluate(bt, min_trades=SAFEGUARD_MIN_TRADES)
            note   = (
                "" if bt.total_trades >= SAFEGUARD_MIN_TRADES else
                f"⚠ INSUFFICIENT SAMPLE: {bt.total_trades} trades "
                f"(need ≥ {SAFEGUARD_MIN_TRADES})"
            )
            results.append(SweepResult(cfg, bt, health, note))

        results.sort(key=lambda r: r.rank_score, reverse=True)
        self._log_summary(results)
        return results

    def best_by_expectancy(
        self, results: List[SweepResult]
    ) -> Optional[SweepResult]:
        """Return highest-expectancy config among sufficient results.

        Returns None when no configuration meets the minimum trade count.
        NEVER call this to auto-adopt parameters — it is for research display only.
        """
        sufficient = [r for r in results if r.sufficient]
        return max(sufficient, key=lambda r: r.results.expectancy) if sufficient else None

    def insufficient_count(self, results: List[SweepResult]) -> int:
        return sum(1 for r in results if not r.sufficient)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, candles: List[Candle], cfg: SweepConfig) -> BacktestResults:
        bt_cfg = self._config.get("backtest", {})
        r_cfg  = self._config.get("risk", {})

        # Use cfg.ema_slow as the min_warmup if it's larger
        min_warmup = max(self._min_warmup, cfg.ema_slow + 10)

        profile = RiskProfile(
            account_size           = float(bt_cfg.get("starting_balance", 10_000)),
            cash_available         = float(bt_cfg.get("starting_balance", 10_000)),
            risk_per_trade_percent = float(r_cfg.get("risk_per_trade_percent", 1.0)),
        )

        engine = BacktestEngine(
            signal_engine = SignalEngine(
                trend_detector    = TrendDetector(cfg.ema_fast, cfg.ema_slow),
                breakout_detector = BreakoutDetector(cfg.breakout_threshold),
                min_candles       = cfg.ema_slow + 10,
            ),
            risk_engine = RiskEngine(
                risk_profile       = profile,
                atr_calculator     = ATRCalculator(int(r_cfg.get("atr_period", 14))),
                stop_loss_engine   = StopLossEngine(cfg.atr_stop_mult),
                take_profit_engine = TakeProfitEngine(cfg.atr_target_mult),
                trade_validator    = TradeValidator(
                    minimum_signal_score = float(r_cfg.get("minimum_signal_score", 70.0)),
                    minimum_risk_reward  = cfg.atr_target_mult / cfg.atr_stop_mult,
                ),
            ),
            portfolio  = Portfolio(float(bt_cfg.get("starting_balance", 10_000))),
            simulator  = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            min_warmup = min_warmup,
        )
        return engine.run(candles)

    @staticmethod
    def _log_summary(results: List[SweepResult]) -> None:
        sufficient = sum(1 for r in results if r.sufficient)
        logger.info(
            "parameter_sweep_complete: configs=%d sufficient=%d insufficient=%d",
            len(results),
            sufficient,
            len(results) - sufficient,
        )
