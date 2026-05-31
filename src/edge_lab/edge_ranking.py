"""Edge ranking — scores and sorts strategy candidates.

Score formula (all components normalised to 0-100):
  40 % Profit Factor
  25 % Expectancy
  15 % Drawdown (inverted)
  10 % Trade Count
  10 % Robustness

No single metric determines the winner.  A strategy with mediocre PF
but strong consistency and high trade count can outscore a high-PF
strategy with few trades and poor robustness.
"""

import math
from typing import List

from src.edge_lab.edge_profile import EdgeProfile

PROMO_MIN_PF      = 1.50
PROMO_MIN_EXP     = 0.0
PROMO_MAX_DD      = 15.0
PROMO_MIN_TRADES  = 100
PROMO_MIN_ASSETS  = 2
PROMO_ROBUSTNESS  = {"ROBUST", "MARGINAL"}


class EdgeRanking:
    """Scores every EdgeProfile and assigns promotion status."""

    def score(self, profile: EdgeProfile) -> float:
        """Return composite edge score in [0, 100]."""
        safe_pf = profile.profit_factor if not math.isinf(profile.profit_factor) else 3.0

        # PF component: 1.0 = 0 pts, 3.0 = 100 pts
        pf_score = max(0.0, min(100.0, (safe_pf - 1.0) / 2.0 * 100.0))

        # Expectancy: $0 = 0 pts, $100/trade = 100 pts
        exp_score = max(0.0, min(100.0, profile.expectancy / 100.0 * 100.0))

        # Drawdown (inverted): 0% = 100 pts, 15%+ = 0 pts
        dd_score = max(0.0, min(100.0, (15.0 - profile.max_drawdown) / 15.0 * 100.0))

        # Trade count: 0 = 0 pts, 200 = 100 pts
        trade_score = max(0.0, min(100.0, profile.total_trades / 200.0 * 100.0))

        # Robustness
        rob_map = {"ROBUST": 100.0, "MARGINAL": 50.0, "UNSTABLE": 0.0, "UNKNOWN": 0.0}
        rob_score = rob_map.get(profile.robustness_rating, 0.0)

        return round(
            pf_score   * 0.40 +
            exp_score  * 0.25 +
            dd_score   * 0.15 +
            trade_score * 0.10 +
            rob_score  * 0.10,
            1,
        )

    def evaluate_promotion(self, profile: EdgeProfile) -> tuple:
        """Return (is_candidate, rejection_reason)."""
        safe_pf = profile.profit_factor if not math.isinf(profile.profit_factor) else 99.0

        if profile.total_trades < PROMO_MIN_TRADES:
            return False, f"trades {profile.total_trades} < {PROMO_MIN_TRADES}"
        if safe_pf < PROMO_MIN_PF:
            return False, f"PF {safe_pf:.2f} < {PROMO_MIN_PF}"
        if profile.expectancy <= PROMO_MIN_EXP:
            return False, f"expectancy ${profile.expectancy:.2f} ≤ $0"
        if profile.max_drawdown >= PROMO_MAX_DD:
            return False, f"drawdown {profile.max_drawdown:.1f}% ≥ {PROMO_MAX_DD}%"
        if profile.assets_passing < PROMO_MIN_ASSETS:
            return False, f"only {profile.assets_passing} assets pass individually"
        if profile.robustness_rating not in PROMO_ROBUSTNESS:
            return False, f"robustness {profile.robustness_rating} is UNSTABLE"
        return True, None

    def rank(self, profiles: List[EdgeProfile]) -> List[EdgeProfile]:
        """Score, mark candidates, and sort descending."""
        ranked = []
        for p in profiles:
            p.edge_score = self.score(p)
            p.promotion_candidate, p.rejection_reason = self.evaluate_promotion(p)
            ranked.append(p)
        return sorted(ranked, key=lambda p: p.edge_score, reverse=True)
