"""DiscoveryEngine — Phase 6.0 master orchestrator.

Runs all 6 research strategy families (plus MOMENTUM_ROTATION benchmark)
on identical data with identical risk assumptions.

FAIRNESS GUARANTEE: same assets, same config, same risk engine, same
commission, same slippage, same promotion criteria for every family.
Only the entry logic differs.

No promotion occurs in Phase 6.0.  This is discovery only.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.edge_discovery.breakout_continuation_research import BreakoutContinuationStrategy
from src.edge_discovery.discovery_comparator import ComparisonReport, DiscoveryComparator
from src.edge_discovery.discovery_profile import (
    MIN_SAMPLE,
    AssetSummary,
    DiscoveryProfile,
    PROMO_MAX_DD,
    PROMO_MIN_EXP,
    PROMO_MIN_PF,
    PROMO_MIN_TRADES,
)
from src.edge_discovery.market_leadership_research import MarketLeadershipStrategy
from src.edge_discovery.relative_strength_research import RelativeStrengthStrategy
from src.edge_discovery.strategy_family import FAMILY_DESCRIPTIONS, FamilyID
from src.edge_discovery.timeframe_alignment_research import MultiTimeframeAlignmentStrategy
from src.edge_discovery.trend_persistence_research import TrendPersistenceStrategy
from src.edge_discovery.volatility_transition_research import VolatilityTransitionStrategy
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy

logger = logging.getLogger(__name__)


class DiscoveryEngine:
    """Runs all 6 strategy families + benchmark on identical data.

    Parameters
    ----------
    config        : EDGE_CONFIG dict (backtest/risk sections)
    asset_candles : symbol → List[Candle]
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

    def run(self) -> ComparisonReport:
        """Evaluate all families and return a ranked ComparisonReport."""
        strategies = self._build_strategies()
        profiles: List[DiscoveryProfile] = []

        for strategy, family_id in strategies:
            logger.info("discovery_engine: evaluating %s ...", strategy.name)
            profile = self._evaluate_family(strategy, family_id)
            profiles.append(profile)

        comparator = DiscoveryComparator(profiles)
        report     = comparator.compare()
        return report

    # ------------------------------------------------------------------ #
    # Strategy factory                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_strategies() -> List[tuple]:
        """Return (strategy, FamilyID) pairs — 6 research + 1 benchmark."""
        return [
            (TrendPersistenceStrategy(),          FamilyID.TREND_PERSISTENCE),
            (BreakoutContinuationStrategy(),       FamilyID.BREAKOUT_CONTINUATION),
            (RelativeStrengthStrategy(),           FamilyID.RELATIVE_STRENGTH),
            (MarketLeadershipStrategy(),           FamilyID.MARKET_LEADERSHIP),
            (VolatilityTransitionStrategy(),       FamilyID.VOLATILITY_TRANSITION),
            (MultiTimeframeAlignmentStrategy(),    FamilyID.MULTI_TIMEFRAME_ALIGNMENT),
            (MomentumRotationStrategy(),           FamilyID.MOMENTUM_ROTATION),  # benchmark
        ]

    # ------------------------------------------------------------------ #
    # Single-family evaluation                                             #
    # ------------------------------------------------------------------ #

    def _evaluate_family(
        self,
        strategy:  StrategyInterface,
        family_id: FamilyID,
    ) -> DiscoveryProfile:
        """Run one strategy family across all assets and build DiscoveryProfile."""

        # Per-asset backtests
        asset_results: Dict[str, BacktestResults] = {}
        for sym, candles in self._asset_candles.items():
            if candles:
                engine               = self._build_engine(strategy)
                asset_results[sym]   = engine.run(candles)

        # Aggregate trades
        all_trades = [t for bt in asset_results.values() for t in bt.trades]
        total      = len(all_trades)

        # Insufficient sample guard
        if total < MIN_SAMPLE:
            return self._insufficient_profile(strategy, family_id)

        # Core metrics
        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if t.is_loss]
        gw     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))
        pf     = gw / gl if gl > 0 else float("inf")
        exp_   = sum(t.pnl for t in all_trades) / total
        wr     = len(wins) / total if total > 0 else 0.0
        worst_dd = max((bt.max_drawdown for bt in asset_results.values()), default=0.0)

        # Robustness (3-window stability)
        robustness, _ = self._stability.analyze(
            strategy, self._asset_candles, lambda s: self._build_engine(s)
        )

        # Survivability: fraction of assets with positive expectancy
        surviving = 0
        asset_summaries: List[AssetSummary] = []
        for sym, bt in asset_results.items():
            sym_trades = bt.trades
            sym_total  = len(sym_trades)
            sym_wins   = [t for t in sym_trades if t.is_win]
            sym_losses = [t for t in sym_trades if t.is_loss]
            sgw = sum(t.pnl for t in sym_wins)
            sgl = abs(sum(t.pnl for t in sym_losses))
            spf = sgw / sgl if sgl > 0 else (float("inf") if sgw > 0 else 0.0)
            sexp = sum(t.pnl for t in sym_trades) / sym_total if sym_total > 0 else 0.0
            if sexp > 0:
                surviving += 1
            asset_summaries.append(
                AssetSummary(
                    symbol=sym,
                    trades=sym_total,
                    profit_factor=spf,
                    expectancy=sexp,
                )
            )

        n_assets      = max(1, len([s for s in asset_results if asset_results[s].trades]))
        survivability = surviving / n_assets

        # Sort asset summaries for best/worst
        finite_summaries = [a for a in asset_summaries if a.profit_factor != float("inf")]
        inf_summaries    = [a for a in asset_summaries if a.profit_factor == float("inf")]

        sorted_by_pf = sorted(finite_summaries, key=lambda a: a.profit_factor, reverse=True)
        all_sorted   = inf_summaries + sorted_by_pf  # inf first (best)

        best_assets  = all_sorted[:3]
        worst_assets = all_sorted[-3:] if len(all_sorted) >= 3 else all_sorted

        return DiscoveryProfile(
            family_id=family_id,
            family_name=strategy.name,
            description=strategy.description,
            data_source=self._data_source,
            trade_count=total,
            profit_factor=pf,
            expectancy=exp_,
            win_rate=wr,
            max_drawdown=worst_dd,
            robustness=robustness,
            survivability=survivability,
            scalable=(total >= PROMO_MIN_TRADES),
            best_assets=best_assets,
            worst_assets=worst_assets,
            insufficient_sample=False,
        )

    def _insufficient_profile(
        self, strategy: StrategyInterface, family_id: FamilyID
    ) -> DiscoveryProfile:
        """Return a zero-metric profile flagged as INSUFFICIENT SAMPLE."""
        return DiscoveryProfile(
            family_id=family_id,
            family_name=strategy.name,
            description=strategy.description,
            data_source=self._data_source,
            trade_count=0,
            profit_factor=0.0,
            expectancy=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            robustness="UNSTABLE",
            survivability=0.0,
            scalable=False,
            insufficient_sample=True,
        )

    # ------------------------------------------------------------------ #
    # Engine factory                                                       #
    # ------------------------------------------------------------------ #

    def _build_engine(self, strategy: StrategyInterface) -> BacktestEngine:
        """Build a BacktestEngine with shared risk config."""
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
