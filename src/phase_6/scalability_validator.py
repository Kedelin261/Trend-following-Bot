"""Phase 6.1 — Edge Scalability Validator.

Purpose
-------
Determine whether the three surviving candidate families survive:
  1. History expansion  : 1000 → 3000 → 5000 bars
  2. Asset expansion    : all 8 ETF assets simultaneously

Candidates evaluated:
  MULTI_TIMEFRAME_ALIGNMENT  (Phase 6.0 winner)
  MARKET_LEADERSHIP          (Phase 6.0 additional candidate)
  RELATIVE_STRENGTH          (Phase 6.0 additional candidate)

Benchmark:
  MOMENTUM_ROTATION

Gate criteria (at least one candidate must satisfy ALL):
  PF >= 1.20
  Expectancy > 0
  Trades >= 100

Absolute constraints
--------------------
- No parameter changes
- No strategy logic changes
- No threshold modifications
- No optimization of any kind
- Research and measurement only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.edge_discovery.market_leadership_research import MarketLeadershipStrategy
from src.edge_discovery.relative_strength_research import RelativeStrengthStrategy
from src.edge_discovery.strategy_family import FamilyID
from src.edge_discovery.timeframe_alignment_research import MultiTimeframeAlignmentStrategy
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.risk_engine import RiskEngine
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 6.1 gate thresholds — DO NOT MODIFY
# ---------------------------------------------------------------------------
GATE_MIN_PF         = 1.20
GATE_MIN_EXPECTANCY = 0.0      # strictly > 0
GATE_MIN_TRADES     = 100
BAR_HORIZONS        = [1000, 3000, 5000]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class HorizonResult:
    """Backtest metrics for one candidate at one bar-horizon."""
    family_id:   FamilyID
    family_name: str
    bar_count:   int
    trades:      int
    profit_factor: float
    expectancy:  float
    max_drawdown: float
    win_rate:    float
    robustness:  str

    @property
    def passes_gate(self) -> bool:
        return (
            self.profit_factor >= GATE_MIN_PF
            and self.expectancy > GATE_MIN_EXPECTANCY
            and self.trades >= GATE_MIN_TRADES
        )

    @property
    def pf_display(self) -> str:
        return "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.3f}"


@dataclass
class CandidateScalabilityResult:
    """All three horizon results for one candidate."""
    family_id:   FamilyID
    family_name: str
    horizons:    List[HorizonResult] = field(default_factory=list)

    @property
    def best_horizon(self) -> Optional[HorizonResult]:
        """Horizon with highest trade count that passes gate, else best PF."""
        passing = [h for h in self.horizons if h.passes_gate]
        if passing:
            return max(passing, key=lambda h: h.trades)
        return max(self.horizons, key=lambda h: h.profit_factor) if self.horizons else None

    @property
    def passes_any_horizon(self) -> bool:
        return any(h.passes_gate for h in self.horizons)

    @property
    def passes_5000_bar(self) -> bool:
        h = next((x for x in self.horizons if x.bar_count == 5000), None)
        return h.passes_gate if h else False

    def get_horizon(self, bars: int) -> Optional[HorizonResult]:
        return next((h for h in self.horizons if h.bar_count == bars), None)


@dataclass
class ScalabilityReport:
    """Full Phase 6.1 results across all candidates and horizons."""
    data_source:   str
    candidates:    List[CandidateScalabilityResult]
    benchmark:     Optional[CandidateScalabilityResult]
    gate_passes:   bool
    gate_reason:   str

    @property
    def passing_candidates(self) -> List[CandidateScalabilityResult]:
        return [c for c in self.candidates if c.passes_any_horizon]

    @property
    def best_candidate(self) -> Optional[CandidateScalabilityResult]:
        passing = self.passing_candidates
        if not passing:
            return None
        # Rank by 5000-bar horizon first, then 3000, then 1000
        for bars in [5000, 3000, 1000]:
            at_horizon = [c for c in passing if c.get_horizon(bars) and
                          c.get_horizon(bars).passes_gate]
            if at_horizon:
                return max(at_horizon,
                           key=lambda c: c.get_horizon(bars).trades)
        return passing[0]


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class ScalabilityValidator:
    """Run Phase 6.1 scalability validation.

    Parameters
    ----------
    config        : EDGE_CONFIG dict (identical to Phase 6.0)
    asset_candles : symbol → List[Candle]  (must have >= 5000 bars)
    data_source   : "LIVE_IBKR" or "SYNTHETIC"
    """

    def __init__(
        self,
        config:        dict,
        asset_candles: Dict[str, List[Candle]],
        data_source:   str = "SYNTHETIC",
    ) -> None:
        self._config        = config
        self._asset_candles = asset_candles
        self._data_source   = data_source
        self._stability     = EdgeStabilityAnalyzer()

    def run(self) -> ScalabilityReport:
        """Execute scalability validation for all candidates and benchmark."""
        strategies = self._build_strategies()

        candidate_results: List[CandidateScalabilityResult] = []
        benchmark_result:  Optional[CandidateScalabilityResult] = None

        for strategy, family_id, is_benchmark in strategies:
            logger.info("phase_6_1: evaluating %s ...", strategy.name)
            result = self._evaluate_candidate(strategy, family_id)
            if is_benchmark:
                benchmark_result = result
            else:
                candidate_results.append(result)

        gate_passes, gate_reason = self._evaluate_gate(candidate_results)

        return ScalabilityReport(
            data_source=self._data_source,
            candidates=candidate_results,
            benchmark=benchmark_result,
            gate_passes=gate_passes,
            gate_reason=gate_reason,
        )

    # ------------------------------------------------------------------
    # Strategy factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_strategies() -> List[Tuple[StrategyInterface, FamilyID, bool]]:
        """Return (strategy, family_id, is_benchmark) triples."""
        return [
            (MultiTimeframeAlignmentStrategy(), FamilyID.MULTI_TIMEFRAME_ALIGNMENT, False),
            (MarketLeadershipStrategy(),         FamilyID.MARKET_LEADERSHIP,          False),
            (RelativeStrengthStrategy(),          FamilyID.RELATIVE_STRENGTH,          False),
            (MomentumRotationStrategy(),          FamilyID.MOMENTUM_ROTATION,          True),
        ]

    # ------------------------------------------------------------------
    # Per-candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        strategy:  StrategyInterface,
        family_id: FamilyID,
    ) -> CandidateScalabilityResult:
        """Evaluate one candidate across all bar horizons."""
        result = CandidateScalabilityResult(
            family_id=family_id,
            family_name=strategy.name,
        )

        for bars in BAR_HORIZONS:
            horizon_candles = self._slice_candles(bars)
            # Skip if insufficient candles for this horizon
            if not horizon_candles:
                logger.warning(
                    "phase_6_1: %s skipping %d-bar horizon (insufficient data)",
                    strategy.name, bars,
                )
                continue

            hr = self._run_horizon(strategy, family_id, bars, horizon_candles)
            result.horizons.append(hr)
            logger.info(
                "phase_6_1: %s @ %d bars → trades=%d PF=%s exp=%.2f",
                strategy.name, bars, hr.trades, hr.pf_display, hr.expectancy,
            )

        return result

    def _run_horizon(
        self,
        strategy:       StrategyInterface,
        family_id:      FamilyID,
        bars:           int,
        asset_candles:  Dict[str, List[Candle]],
    ) -> HorizonResult:
        """Run backtest for one strategy at one bar horizon across all assets."""
        all_trades = []
        all_results: List[BacktestResults] = []

        for sym, candles in asset_candles.items():
            engine = self._build_engine(strategy)
            bt     = engine.run(candles)
            all_trades.extend(bt.trades)
            all_results.append(bt)

        total = len(all_trades)
        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if t.is_loss]
        gw     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))
        pf     = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
        exp_   = sum(t.pnl for t in all_trades) / total if total > 0 else 0.0
        wr     = len(wins) / total if total > 0 else 0.0
        dd     = max((bt.max_drawdown for bt in all_results), default=0.0)

        # Robustness via EdgeStabilityAnalyzer
        robustness, _ = self._stability.analyze(
            strategy, asset_candles, lambda s: self._build_engine(s)
        )

        return HorizonResult(
            family_id=family_id,
            family_name=strategy.name,
            bar_count=bars,
            trades=total,
            profit_factor=pf,
            expectancy=exp_,
            max_drawdown=dd,
            win_rate=wr,
            robustness=robustness,
        )

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_gate(
        candidates: List[CandidateScalabilityResult],
    ) -> Tuple[bool, str]:
        """Return (passes, reason) for the Phase 6.1 gate.

        Gate: at least one candidate must satisfy ALL of:
            PF >= 1.20
            Expectancy > 0
            Trades >= 100
        """
        passing = [c for c in candidates if c.passes_any_horizon]
        if not passing:
            return False, (
                f"No candidate met gate criteria "
                f"(PF>={GATE_MIN_PF}, Exp>0, Trades>={GATE_MIN_TRADES}) "
                f"at any bar horizon. STOP — RETURN TO DISCOVERY."
            )

        best  = passing[0]
        bh    = best.best_horizon
        names = ", ".join(c.family_name for c in passing)
        return True, (
            f"{len(passing)} candidate(s) passed: [{names}]. "
            f"Best: {best.family_name} @ {bh.bar_count} bars "
            f"— trades={bh.trades} PF={bh.pf_display} exp={bh.expectancy:.2f}. "
            f"PROCEED TO PHASE 6.2."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slice_candles(self, bars: int) -> Dict[str, List[Candle]]:
        """Return up to `bars` candles per asset (most recent)."""
        result: Dict[str, List[Candle]] = {}
        for sym, candles in self._asset_candles.items():
            if len(candles) >= bars:
                result[sym] = candles[:bars]      # take first N (earliest) for reproducibility
            elif len(candles) > 0:
                result[sym] = candles             # use all available
        return result if result else {}

    def _build_engine(self, strategy: StrategyInterface) -> BacktestEngine:
        """Build a BacktestEngine with Phase 6.0 shared config."""
        bt_cfg = self._config.get("backtest", {})
        start  = float(bt_cfg.get("starting_balance", 10_000.0))

        return BacktestEngine(
            signal_engine=strategy,
            risk_engine=RiskEngine.from_config(self._config),
            portfolio=Portfolio(start),
            simulator=TradeSimulator(
                slippage_percent=float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade=float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            min_warmup=strategy.min_candles,
            symbol="",
            timeframe="D1",
        )
