"""Promotion candidate evaluator — Phase 5.1 gate.

Additional criterion vs Phase 4.x: History Tested ≥ 3000 bars.

Reuses PromotionValidator thresholds. No duplicate logic.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from src.edge_lab.edge_profile import EdgeProfile
from src.edge_validation.strategy_survivability import SurvivabilityResult

PROMO_MIN_TRADES   = 100
PROMO_MIN_PF       = 1.50
PROMO_MIN_EXP      = 0.0
PROMO_MAX_DD       = 15.0
PROMO_MIN_ASSETS   = 2
PROMO_MIN_BARS     = 3000     # Phase 5.1 additional requirement
VALID_ROBUSTNESS   = {"ROBUST", "MARGINAL"}


@dataclass
class PromotionCandidateResult:
    strategy_name:      str
    is_candidate:       bool
    trades:             int
    profit_factor:      float
    expectancy:         float
    drawdown:           float
    robustness:         str
    assets_passing:     int
    history_tested:     int
    survivability:      str
    pass_criteria:      List[str]
    fail_criteria:      List[str]

    @property
    def status(self) -> str:
        return "PROMOTION CANDIDATE" if self.is_candidate else "NOT READY"

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


class PromotionCandidateEvaluator:

    def evaluate(
        self,
        profile:       EdgeProfile,
        survivability: SurvivabilityResult,
        history_bars:  int,
    ) -> PromotionCandidateResult:
        safe_pf = profile.profit_factor if not math.isinf(profile.profit_factor) else 99.0

        checks = [
            (profile.total_trades   >= PROMO_MIN_TRADES,
             f"Trades {profile.total_trades} ≥ {PROMO_MIN_TRADES}",
             f"Trades {profile.total_trades} < {PROMO_MIN_TRADES}"),
            (safe_pf >= PROMO_MIN_PF,
             f"PF {safe_pf:.2f} ≥ {PROMO_MIN_PF}",
             f"PF {safe_pf:.2f} < {PROMO_MIN_PF}"),
            (profile.expectancy > PROMO_MIN_EXP,
             f"Expectancy ${profile.expectancy:.2f} > $0",
             f"Expectancy ${profile.expectancy:.2f} ≤ $0"),
            (profile.max_drawdown < PROMO_MAX_DD,
             f"DD {profile.max_drawdown:.1f}% < {PROMO_MAX_DD}%",
             f"DD {profile.max_drawdown:.1f}% ≥ {PROMO_MAX_DD}%"),
            (profile.robustness_rating in VALID_ROBUSTNESS,
             f"Robustness {profile.robustness_rating}",
             f"Robustness {profile.robustness_rating} (must be ROBUST/MARGINAL)"),
            (profile.assets_passing >= PROMO_MIN_ASSETS,
             f"{profile.assets_passing} assets pass individually",
             f"Only {profile.assets_passing} assets pass (need ≥ {PROMO_MIN_ASSETS})"),
            (history_bars >= PROMO_MIN_BARS,
             f"History {history_bars:,} bars ≥ {PROMO_MIN_BARS:,}",
             f"History {history_bars:,} bars < {PROMO_MIN_BARS:,}"),
            (survivability.survives,
             f"Survivability: SURVIVES",
             f"Survivability: FAILS — {'; '.join(survivability.failure_reasons[:2])}"),
        ]

        pass_c = [msg for ok, msg, _ in checks if ok]
        fail_c = [msg for ok, _, msg in checks if not ok]

        return PromotionCandidateResult(
            strategy_name   = profile.strategy_name,
            is_candidate    = all(ok for ok, _, _ in checks),
            trades          = profile.total_trades,
            profit_factor   = profile.profit_factor,
            expectancy      = profile.expectancy,
            drawdown        = profile.max_drawdown,
            robustness      = profile.robustness_rating,
            assets_passing  = profile.assets_passing,
            history_tested  = history_bars,
            survivability   = survivability.verdict,
            pass_criteria   = pass_c,
            fail_criteria   = fail_c,
        )
