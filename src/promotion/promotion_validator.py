"""Promotion validator — applies the final pass/fail criteria.

PASS only when ALL of:
  Trades ≥ 100
  PF ≥ 1.50
  Expectancy > $0
  Max DD < 15 %
  ≥ 2 assets individually pass

All criteria are hardcoded here.  Only extending historical data or
adding approved asset candidates may change the outcome.
The strategy parameters are NOT to be touched.

No broker code. No API calls.
"""

from dataclasses import dataclass
from typing import List, Optional

PROMOTION_MIN_TRADES = 100
PROMOTION_MIN_PF     = 1.50
PROMOTION_MIN_EXP    = 0.0
PROMOTION_MAX_DD     = 15.0
PROMOTION_MIN_ASSETS = 2


@dataclass
class PromotionValidationResult:
    """Detailed pass/fail result for the promotion gate."""

    trades_ok:      bool
    pf_ok:          bool
    expectancy_ok:  bool
    drawdown_ok:    bool
    assets_ok:      bool
    all_passed:     bool

    actual_trades:   int
    actual_pf:       float
    actual_exp:      float
    actual_dd:       float
    actual_assets:   int

    failure_reasons: List[str]
    pass_reasons:    List[str]

    @property
    def status_label(self) -> str:
        return "PROMOTE_TO_PHASE_5" if self.all_passed else "RETURN_TO_RESEARCH"


class PromotionValidator:
    """Applies the five promotion criteria and returns a structured result."""

    def validate(
        self,
        total_trades:  int,
        profit_factor: float,
        expectancy:    float,
        max_drawdown:  float,
        assets_passing: int,
    ) -> PromotionValidationResult:
        """Evaluate all five promotion criteria.

        Parameters are portfolio-level aggregates (not per-asset).
        """
        import math
        safe_pf = profit_factor if not math.isinf(profit_factor) else 99.0

        trades_ok   = total_trades  >= PROMOTION_MIN_TRADES
        pf_ok       = safe_pf       >= PROMOTION_MIN_PF
        exp_ok      = expectancy    >  PROMOTION_MIN_EXP
        dd_ok       = max_drawdown  <  PROMOTION_MAX_DD
        assets_ok   = assets_passing >= PROMOTION_MIN_ASSETS

        pass_reasons:    List[str] = []
        failure_reasons: List[str] = []

        checks = [
            (trades_ok,
             f"Trade count {total_trades} ≥ {PROMOTION_MIN_TRADES}",
             f"Trade count {total_trades} < {PROMOTION_MIN_TRADES} minimum"),
            (pf_ok,
             f"Profit factor {safe_pf:.2f} ≥ {PROMOTION_MIN_PF}",
             f"Profit factor {safe_pf:.2f} < {PROMOTION_MIN_PF} minimum"),
            (exp_ok,
             f"Expectancy ${expectancy:.2f} > $0",
             f"Expectancy ${expectancy:.2f} ≤ $0"),
            (dd_ok,
             f"Max drawdown {max_drawdown:.1f}% < {PROMOTION_MAX_DD:.0f}%",
             f"Max drawdown {max_drawdown:.1f}% ≥ {PROMOTION_MAX_DD:.0f}% limit"),
            (assets_ok,
             f"{assets_passing} assets individually pass",
             f"Only {assets_passing} asset(s) pass (need ≥ {PROMOTION_MIN_ASSETS})"),
        ]

        for ok, pass_msg, fail_msg in checks:
            if ok:
                pass_reasons.append(pass_msg)
            else:
                failure_reasons.append(fail_msg)

        all_passed = all(ok for ok, _, _ in checks)

        return PromotionValidationResult(
            trades_ok       = trades_ok,
            pf_ok           = pf_ok,
            expectancy_ok   = exp_ok,
            drawdown_ok     = dd_ok,
            assets_ok       = assets_ok,
            all_passed      = all_passed,
            actual_trades   = total_trades,
            actual_pf       = safe_pf,
            actual_exp      = expectancy,
            actual_dd       = max_drawdown,
            actual_assets   = assets_passing,
            failure_reasons = failure_reasons,
            pass_reasons    = pass_reasons,
        )
