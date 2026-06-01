"""EdgeScoreCalculator — normalized 0–100 composite edge score.

Purpose: RANK strategy families for research prioritization.
         NOT a promotion tool.  NOT an optimization target.

Scoring Formula (inputs → normalized components → weighted sum):
    1. Profit Factor       weight=0.30  — primary edge indicator
    2. Expectancy          weight=0.25  — economic significance
    3. Drawdown            weight=0.20  — risk-adjusted quality
    4. Trade Count         weight=0.10  — statistical confidence
    5. Robustness          weight=0.10  — cross-window stability
    6. Consistency         weight=0.05  — win-rate stability

Component normalization:
    PF:        linear 1.0→0, 2.0→100 (capped); below 1.0 = 0
    Exp:       linear 0→0, 50→100 (capped); below 0 = 0
    DD:        inverted linear 0%→100, 30%+→0
    Trades:    log-scaled: 0→0, 500+→100
    Robustness: ROBUST=100, MARGINAL=50, UNSTABLE=0
    Consistency (win_rate): linear 0%→0, 60%+→100

Anti-curve-fitting constraints:
    - Scores are informational only
    - No score threshold triggers promotion
    - INSUFFICIENT_SAMPLE profiles receive score=0
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.edge_discovery.discovery_profile import DiscoveryProfile

# Component weights (must sum to 1.0)
_W_PF          = 0.30
_W_EXP         = 0.25
_W_DD          = 0.20
_W_TRADES      = 0.10
_W_ROBUSTNESS  = 0.10
_W_CONSISTENCY = 0.05

assert abs(_W_PF + _W_EXP + _W_DD + _W_TRADES + _W_ROBUSTNESS + _W_CONSISTENCY - 1.0) < 1e-9, \
    "Edge score weights must sum to 1.0"

# Normalization parameters
_PF_MIN        = 1.0    # PF below this → 0
_PF_MAX        = 2.0    # PF at/above this → 100
_EXP_MAX       = 50.0   # expectancy $ at/above this → 100
_DD_MAX_BAD    = 30.0   # drawdown at/above this → 0
_TRADE_TARGET  = 500.0  # trades at/above this → 100 (log-scaled)
_WR_MAX        = 0.60   # win rate at/above this → 100


class EdgeScoreCalculator:
    """Compute a normalized 0–100 edge score for a DiscoveryProfile.

    Usage:
        calc = EdgeScoreCalculator()
        score = calc.compute(profile)
    """

    def compute(self, profile: "DiscoveryProfile") -> float:
        """Return score in [0, 100].  Returns 0.0 for insufficient samples."""
        from src.edge_discovery.discovery_profile import MIN_SAMPLE

        if profile.insufficient_sample or profile.trade_count < MIN_SAMPLE:
            return 0.0

        pf_score    = self._score_pf(profile.profit_factor)
        exp_score   = self._score_expectancy(profile.expectancy)
        dd_score    = self._score_drawdown(profile.max_drawdown)
        trade_score = self._score_trades(profile.trade_count)
        rob_score   = self._score_robustness(profile.robustness)
        con_score   = self._score_consistency(profile.win_rate)

        raw = (
            pf_score    * _W_PF
            + exp_score   * _W_EXP
            + dd_score    * _W_DD
            + trade_score * _W_TRADES
            + rob_score   * _W_ROBUSTNESS
            + con_score   * _W_CONSISTENCY
        )

        return round(min(100.0, max(0.0, raw)), 2)

    # ------------------------------------------------------------------ #
    # Component scorers (all return 0–100)                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_pf(pf: float) -> float:
        """Profit factor: linear 1.0→0, 2.0→100."""
        if pf == float("inf"):
            return 100.0
        if pf < _PF_MIN:
            return 0.0
        return min(100.0, (pf - _PF_MIN) / (_PF_MAX - _PF_MIN) * 100.0)

    @staticmethod
    def _score_expectancy(exp: float) -> float:
        """Expectancy: linear 0→0, _EXP_MAX→100.  Below 0 = 0."""
        if exp <= 0:
            return 0.0
        return min(100.0, exp / _EXP_MAX * 100.0)

    @staticmethod
    def _score_drawdown(dd: float) -> float:
        """Drawdown: inverted — 0%→100, ≥30%→0."""
        if dd >= _DD_MAX_BAD:
            return 0.0
        return min(100.0, (1.0 - dd / _DD_MAX_BAD) * 100.0)

    @staticmethod
    def _score_trades(trades: int) -> float:
        """Trades: log-scaled.  500+ → 100."""
        if trades <= 0:
            return 0.0
        # log scale: ln(trades) / ln(target) × 100
        score = math.log(trades + 1) / math.log(_TRADE_TARGET + 1) * 100.0
        return min(100.0, score)

    @staticmethod
    def _score_robustness(robustness: str) -> float:
        """Robustness: ROBUST=100, MARGINAL=50, UNSTABLE=0."""
        mapping = {"ROBUST": 100.0, "MARGINAL": 50.0, "UNSTABLE": 0.0}
        return mapping.get(robustness, 0.0)

    @staticmethod
    def _score_consistency(win_rate: float) -> float:
        """Win rate: 0%→0, 60%+→100."""
        return min(100.0, (win_rate / _WR_MAX) * 100.0)
