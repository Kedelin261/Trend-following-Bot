"""DiscoveryComparator — rank strategy families by edge score.

Phase 6.0: Research ranking only.  No promotion decisions.

Responsibilities:
    1. Accept a list of DiscoveryProfiles
    2. Assign edge scores via EdgeScoreCalculator
    3. Rank by edge score descending
    4. Identify the discovery winner (highest score)
    5. Identify promising candidates (promising flag + score > 0)
    6. Build ComparisonReport for printing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.edge_discovery.discovery_profile import DiscoveryProfile
from src.edge_discovery.edge_score import EdgeScoreCalculator
from src.edge_discovery.strategy_family import FamilyID


@dataclass
class FamilyRanking:
    """Single-row ranking entry."""

    rank:          int
    family_id:     FamilyID
    family_name:   str
    edge_score:    float
    profit_factor: float
    expectancy:    float
    max_drawdown:  float
    trade_count:   int
    robustness:    str
    is_promising:  bool
    insufficient:  bool


@dataclass
class ComparisonReport:
    """Full comparison report output from DiscoveryComparator."""

    rankings:          List[FamilyRanking] = field(default_factory=list)
    discovery_winner:  Optional[FamilyRanking] = None
    promising_candidates: List[FamilyRanking] = field(default_factory=list)
    benchmark_ranking: Optional[FamilyRanking] = None  # MOMENTUM_ROTATION


class DiscoveryComparator:
    """Ranks all DiscoveryProfiles and identifies winners and candidates.

    Usage:
        comparator = DiscoveryComparator(profiles)
        report = comparator.compare()
    """

    def __init__(self, profiles: List[DiscoveryProfile]) -> None:
        self._profiles = profiles
        self._scorer   = EdgeScoreCalculator()

    def compare(self) -> ComparisonReport:
        """Score, rank, and analyse all profiles."""
        # Score all profiles
        for profile in self._profiles:
            profile.edge_score = self._scorer.compute(profile)

        # Separate benchmark from research families
        research = [p for p in self._profiles if p.family_id != FamilyID.MOMENTUM_ROTATION]
        benchmark_profiles = [p for p in self._profiles if p.family_id == FamilyID.MOMENTUM_ROTATION]

        # Sort research families by edge score descending
        research_sorted = sorted(research, key=lambda p: p.edge_score, reverse=True)

        rankings: List[FamilyRanking] = []
        for rank, profile in enumerate(research_sorted, start=1):
            rankings.append(self._to_ranking(rank, profile))

        # Benchmark entry (not ranked among research families)
        benchmark_ranking: Optional[FamilyRanking] = None
        if benchmark_profiles:
            benchmark_ranking = self._to_ranking(0, benchmark_profiles[0])

        # Discovery winner: highest-scoring non-benchmark family (score > 0)
        scored_rankings = [r for r in rankings if r.edge_score > 0]
        discovery_winner = scored_rankings[0] if scored_rankings else None

        # Promising candidates: promising flag set, score > 0, not benchmark
        promising = [r for r in rankings if r.is_promising and r.edge_score > 0]

        report = ComparisonReport(
            rankings=rankings,
            discovery_winner=discovery_winner,
            promising_candidates=promising,
            benchmark_ranking=benchmark_ranking,
        )
        return report

    @staticmethod
    def _to_ranking(rank: int, profile: DiscoveryProfile) -> FamilyRanking:
        return FamilyRanking(
            rank=rank,
            family_id=profile.family_id,
            family_name=profile.family_name,
            edge_score=profile.edge_score,
            profit_factor=profile.profit_factor,
            expectancy=profile.expectancy,
            max_drawdown=profile.max_drawdown,
            trade_count=profile.trade_count,
            robustness=profile.robustness,
            is_promising=profile.is_promising,
            insufficient=profile.insufficient_sample,
        )
