"""DiscoveryProfile — answers the 12 discovery questions for each strategy family.

Phase 6.0: Edge Discovery (research only, no promotion decisions).

DATA SOURCE NOTE: Profiles generated from synthetic data must carry
data_source="SYNTHETIC" and must not be used for promotion decisions.
IBKR live data is required for any official research conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.edge_discovery.strategy_family import FamilyID

# Minimum trades required for a statistically meaningful sample.
MIN_SAMPLE = 50

# Promotion thresholds (for reference — no promotion occurs in Phase 6.0).
PROMO_MIN_PF     = 1.50
PROMO_MIN_EXP    = 0.0
PROMO_MAX_DD     = 15.0
PROMO_MIN_TRADES = 500


@dataclass
class AssetSummary:
    """Per-asset performance summary for a strategy family."""

    symbol:        str
    trades:        int
    profit_factor: float
    expectancy:    float

    def __repr__(self) -> str:
        return (
            f"AssetSummary({self.symbol}: "
            f"T={self.trades} PF={self.profit_factor:.2f} "
            f"Exp=${self.expectancy:.2f})"
        )


@dataclass
class DiscoveryProfile:
    """Aggregated discovery profile answering 12 research questions.

    Answers
    -------
    Q1  trade_count       — total trades across all assets
    Q2  profit_factor     — aggregate profit factor
    Q3  expectancy        — average $ expectancy per trade
    Q4  max_drawdown      — worst drawdown % observed
    Q5  robustness        — ROBUST / MARGINAL / UNSTABLE (3-window)
    Q6  survivability     — fraction of assets with positive expectancy
    Q7  scalable          — True when trade_count >= 500
    Q8  best_assets       — top-3 assets by PF
    Q9  worst_assets      — bottom-3 assets by PF
    Q10 best_regime       — most favourable regime (placeholder: NOT_ANALYZED)
    Q11 worst_regime      — least favourable regime (placeholder: NOT_ANALYZED)
    Q12 edge_score        — normalized 0-100 composite score
    """

    family_id:    FamilyID
    family_name:  str
    description:  str
    data_source:  str  # "LIVE_IBKR" or "SYNTHETIC"

    # Q1-Q4 core metrics
    trade_count:   int
    profit_factor: float
    expectancy:    float
    win_rate:      float
    max_drawdown:  float

    # Q5 robustness
    robustness: str  # ROBUST / MARGINAL / UNSTABLE

    # Q6 survivability (fraction of assets with positive expectancy)
    survivability: float

    # Q7 scalability
    scalable: bool  # trade_count >= PROMO_MIN_TRADES

    # Q8-Q9 asset ranking
    best_assets:  List[AssetSummary] = field(default_factory=list)
    worst_assets: List[AssetSummary] = field(default_factory=list)

    # Q10-Q11 regime analysis (Phase 6.0: not analyzed)
    best_regime:  str = "NOT_ANALYZED"
    worst_regime: str = "NOT_ANALYZED"

    # Q12 edge score (computed by EdgeScoreCalculator)
    edge_score: float = 0.0

    # Insufficient sample flag
    insufficient_sample: bool = False

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def pf_str(self) -> str:
        """Formatted profit factor string."""
        if self.profit_factor == float("inf"):
            return "∞"
        return f"{self.profit_factor:.2f}"

    @property
    def promotion_gap(self) -> str:
        """Text summary of distance from each promotion criterion."""
        if self.insufficient_sample:
            return "INSUFFICIENT SAMPLE"
        gaps = []
        if self.profit_factor < PROMO_MIN_PF:
            gaps.append(f"PF needs +{PROMO_MIN_PF - self.profit_factor:.2f}")
        if self.expectancy <= PROMO_MIN_EXP:
            gaps.append("Expectancy ≤ 0")
        if self.max_drawdown > PROMO_MAX_DD:
            gaps.append(f"DD over by {self.max_drawdown - PROMO_MAX_DD:.1f}%")
        if self.trade_count < PROMO_MIN_TRADES:
            gaps.append(f"Trades need +{PROMO_MIN_TRADES - self.trade_count}")
        if self.robustness != "ROBUST":
            gaps.append(f"Robustness={self.robustness}")
        if not gaps:
            return "MEETS ALL CRITERIA"
        return " | ".join(gaps)

    @property
    def is_promising(self) -> bool:
        """Heuristic flag: PF > 1.10 AND expectancy > 0 AND trades >= MIN_SAMPLE."""
        return (
            self.trade_count >= MIN_SAMPLE
            and self.profit_factor >= 1.10
            and self.expectancy > 0.0
        )

    def __repr__(self) -> str:
        return (
            f"DiscoveryProfile({self.family_id.value}: "
            f"T={self.trade_count} PF={self.pf_str} "
            f"Exp=${self.expectancy:.2f} Score={self.edge_score:.1f})"
        )
