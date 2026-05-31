"""Edge Engine — runs every strategy on the same data and produces EdgeProfiles.

FAIRNESS GUARANTEE: identical risk engine, position sizer, commission,
slippage, and assets for every strategy.  Only the entry signal differs.
"""

import logging
from typing import Dict, List, Optional

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.edge_lab.edge_profile import EdgeProfile
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer
from src.edge_lab.strategy_interface import StrategyInterface
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

# ---- Shared risk parameters for ALL strategies ----------------------------
DEFAULT_STARTING_BALANCE = 10_000.0
DEFAULT_RISK_PCT         = 1.0
DEFAULT_SLIPPAGE         = 0.05
DEFAULT_COMMISSION       = 1.0


class EdgeEngine:
    """Evaluates a set of strategies on identical data using identical risk.

    Parameters
    ----------
    config  : settings dict (backtest / risk sections)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._stability = EdgeStabilityAnalyzer()

    def evaluate_all(
        self,
        strategies:    List[StrategyInterface],
        asset_candles: Dict[str, List[Candle]],
    ) -> List[EdgeProfile]:
        """Evaluate every strategy and return a list of EdgeProfiles."""
        profiles = []
        for strategy in strategies:
            logger.info("edge_engine: evaluating %s ...", strategy.name)
            profile = self._evaluate_one(strategy, asset_candles)
            profiles.append(profile)
        return profiles

    def _evaluate_one(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
    ) -> EdgeProfile:
        """Run one strategy, compute stability, return EdgeProfile."""
        bt_cfg  = self._config.get("backtest", {})
        r_cfg   = self._config.get("risk", {})
        start   = float(bt_cfg.get("starting_balance", DEFAULT_STARTING_BALANCE))
        risk_pct = float(r_cfg.get("risk_per_trade_percent", DEFAULT_RISK_PCT))

        # Full backtest
        asset_results: Dict = {}
        for sym, candles in asset_candles.items():
            if candles:
                engine = self._build_engine(strategy)
                asset_results[sym] = engine.run(candles)

        profile = EdgeProfile.from_asset_results(
            strategy_name = strategy.name,
            description   = strategy.description,
            asset_results = asset_results,
        )

        # Stability
        rating, windows = self._stability.analyze(
            strategy, asset_candles, lambda s: self._build_engine(s)
        )
        profile.robustness_rating = rating
        profile.window_pf = [w.pf for w in windows]

        logger.info(
            "edge_engine: %s trades=%d pf=%.2f exp=%.2f dd=%.1f%% robustness=%s",
            strategy.name,
            profile.total_trades,
            profile.profit_factor if not __import__("math").isinf(profile.profit_factor) else 99,
            profile.expectancy,
            profile.max_drawdown,
            rating,
        )
        return profile

    def _build_engine(self, strategy: StrategyInterface) -> BacktestEngine:
        """Build a BacktestEngine with the shared risk configuration."""
        bt_cfg = self._config.get("backtest", {})
        r_cfg  = self._config.get("risk", {})
        start  = float(bt_cfg.get("starting_balance", DEFAULT_STARTING_BALANCE))

        risk_profile = RiskProfile(
            account_size           = start,
            cash_available         = start,
            risk_per_trade_percent = float(r_cfg.get("risk_per_trade_percent", DEFAULT_RISK_PCT)),
        )
        risk_engine = RiskEngine.from_config(self._config)

        return BacktestEngine(
            signal_engine = strategy,
            risk_engine   = risk_engine,
            portfolio     = Portfolio(start),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", DEFAULT_SLIPPAGE)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", DEFAULT_COMMISSION)),
            ),
            min_warmup = strategy.min_candles,
            symbol     = "",
            timeframe  = "D1",
        )
