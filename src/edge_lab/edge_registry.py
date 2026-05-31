"""Edge registry — central list of all strategy candidates."""

from src.edge_lab.strategy_interface import StrategyInterface
from src.signals.strategies.breakout_strategy import BreakoutStrategy
from src.signals.strategies.pullback_strategy import PullbackStrategy
from src.signals.strategies.donchian_strategy import DonchianStrategy
from src.signals.strategies.volatility_expansion_strategy import VolatilityExpansionStrategy
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy


def build_all_strategies() -> list:
    """Return one instance of every registered strategy."""
    return [
        BreakoutStrategy(),
        PullbackStrategy(),
        DonchianStrategy(),
        VolatilityExpansionStrategy(),
        MomentumRotationStrategy(),
    ]
