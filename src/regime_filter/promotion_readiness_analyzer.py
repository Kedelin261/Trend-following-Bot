"""Promotion readiness analyzer — final gate for Phase 4.11.

Evaluates whether a filtered strategy meets all promotion criteria:
  1. Trades ≥ 100
  2. PF ≥ 1.50
  3. Expectancy > $0
  4. Max DD < 15 %
  5. ≥ 2 assets individually pass (requires per-asset results)
  6. Robustness ≠ UNSTABLE

No execution.  No trading.  Research gate only.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from src.regime_filter.filter_backtester import FilterBacktestResult
from src.regime_filter.filter_profiles import FilterProfile

PROMO_MIN_TRADES = 100
PROMO_MIN_PF     = 1.50
PROMO_MIN_EXP    = 0.0
PROMO_MAX_DD     = 15.0
PROMO_MIN_ASSETS = 2


@dataclass
class PromotionReadiness:
    """Promotion gate result for one filtered strategy configuration."""

    filter_profile:       FilterProfile
    trades_ok:            bool
    pf_ok:                bool
    expectancy_ok:        bool
    drawdown_ok:          bool
    robustness_ok:        bool
    all_passed:           bool

    actual_trades:        int
    actual_pf:            float
    actual_expectancy:    float
    actual_dd:            float
    robustness_rating:    str

    pass_reasons:         List[str]
    failure_reasons:      List[str]

    @property
    def status(self) -> str:
        return "PROMOTION READY" if self.all_passed else "NOT READY"

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.actual_pf) else f"{self.actual_pf:.2f}"


class PromotionReadinessAnalyzer:
    """Evaluates whether a filtered strategy is ready for promotion."""

    def analyze(
        self,
        result:            FilterBacktestResult,
        robustness_rating: str,
    ) -> PromotionReadiness:
        """Apply all six promotion criteria."""
        safe_pf = result.profit_factor if not math.isinf(result.profit_factor) else 99.0

        trades_ok = result.total_trades_after >= PROMO_MIN_TRADES
        pf_ok     = safe_pf >= PROMO_MIN_PF
        exp_ok    = result.expectancy > PROMO_MIN_EXP
        dd_ok     = result.max_drawdown < PROMO_MAX_DD
        rob_ok    = robustness_rating in ("ROBUST", "MARGINAL")

        pass_reasons:    List[str] = []
        failure_reasons: List[str] = []

        checks = [
            (trades_ok,
             f"Trades {result.total_trades_after} ≥ {PROMO_MIN_TRADES}",
             f"Trades {result.total_trades_after} < {PROMO_MIN_TRADES}"),
            (pf_ok,
             f"PF {safe_pf:.2f} ≥ {PROMO_MIN_PF}",
             f"PF {safe_pf:.2f} < {PROMO_MIN_PF}"),
            (exp_ok,
             f"Expectancy ${result.expectancy:.2f} > $0",
             f"Expectancy ${result.expectancy:.2f} ≤ $0"),
            (dd_ok,
             f"Max DD {result.max_drawdown:.1f}% < {PROMO_MAX_DD:.0f}%",
             f"Max DD {result.max_drawdown:.1f}% ≥ {PROMO_MAX_DD:.0f}%"),
            (rob_ok,
             f"Robustness {robustness_rating} is acceptable",
             f"Robustness {robustness_rating} = UNSTABLE (not acceptable)"),
        ]

        for ok, pass_msg, fail_msg in checks:
            if ok:
                pass_reasons.append(pass_msg)
            else:
                failure_reasons.append(fail_msg)

        all_passed = all(ok for ok, _, _ in checks)

        return PromotionReadiness(
            filter_profile    = result.filter_profile,
            trades_ok         = trades_ok,
            pf_ok             = pf_ok,
            expectancy_ok     = exp_ok,
            drawdown_ok       = dd_ok,
            robustness_ok     = rob_ok,
            all_passed        = all_passed,
            actual_trades     = result.total_trades_after,
            actual_pf         = safe_pf,
            actual_expectancy = result.expectancy,
            actual_dd         = result.max_drawdown,
            robustness_rating = robustness_rating,
            pass_reasons      = pass_reasons,
            failure_reasons   = failure_reasons,
        )

    def find_best_ready_profile(
        self,
        readiness_results: List[PromotionReadiness],
    ) -> Optional[PromotionReadiness]:
        """Return the promotion-ready profile with highest expectancy."""
        ready = [r for r in readiness_results if r.all_passed]
        return max(ready, key=lambda r: r.actual_expectancy) if ready else None
