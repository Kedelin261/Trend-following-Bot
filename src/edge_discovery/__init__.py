"""Phase 6.0 Edge Discovery package.

Exports the primary public API for the edge discovery research framework.

Six new strategy families are evaluated (NOT Momentum Rotation variations):
    1. TREND_PERSISTENCE
    2. BREAKOUT_CONTINUATION
    3. RELATIVE_STRENGTH
    4. MARKET_LEADERSHIP
    5. VOLATILITY_TRANSITION
    6. MULTI_TIMEFRAME_ALIGNMENT

MOMENTUM_ROTATION serves as the benchmark (researched/rejected in Phases 5.x).

No promotion decisions are made in Phase 6.0.
No execution, no paper trading, no live trading.
"""

from src.edge_discovery.breakout_continuation_research import BreakoutContinuationStrategy
from src.edge_discovery.discovery_comparator import ComparisonReport, DiscoveryComparator, FamilyRanking
from src.edge_discovery.discovery_engine import DiscoveryEngine
from src.edge_discovery.discovery_profile import AssetSummary, DiscoveryProfile, MIN_SAMPLE
from src.edge_discovery.discovery_report import DiscoveryReport
from src.edge_discovery.edge_score import EdgeScoreCalculator
from src.edge_discovery.market_leadership_research import MarketLeadershipStrategy
from src.edge_discovery.relative_strength_research import RelativeStrengthStrategy
from src.edge_discovery.strategy_family import FAMILY_DESCRIPTIONS, FamilyID
from src.edge_discovery.timeframe_alignment_research import MultiTimeframeAlignmentStrategy
from src.edge_discovery.trend_persistence_research import TrendPersistenceStrategy
from src.edge_discovery.volatility_transition_research import VolatilityTransitionStrategy

__all__ = [
    # Strategy families
    "TrendPersistenceStrategy",
    "BreakoutContinuationStrategy",
    "RelativeStrengthStrategy",
    "MarketLeadershipStrategy",
    "VolatilityTransitionStrategy",
    "MultiTimeframeAlignmentStrategy",
    # Taxonomy
    "FamilyID",
    "FAMILY_DESCRIPTIONS",
    # Profile
    "DiscoveryProfile",
    "AssetSummary",
    "MIN_SAMPLE",
    # Scoring
    "EdgeScoreCalculator",
    # Comparison
    "DiscoveryComparator",
    "ComparisonReport",
    "FamilyRanking",
    # Engine
    "DiscoveryEngine",
    # Report
    "DiscoveryReport",
]
