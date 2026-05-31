"""Regime filter engine — ALLOW or REJECT individual trades based on filter profile.

Applies a FilterProfile's rules to a LabelledTrade (from Phase 4.10) and
returns a FilterDecision.

Decision priority:
  1. Blocked regimes (blacklist) → REJECT if any match
  2. Allowed regimes (whitelist) → REJECT if none match when whitelist is set
  3. Default → ALLOW

No strategy parameters are changed.  This is a post-trade gate only.
No broker code.  No API calls.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime_filter.filter_profiles import FilterProfile

logger = logging.getLogger(__name__)


class FilterDecision(str, Enum):
    ALLOW  = "ALLOW"
    REJECT = "REJECT"


@dataclass
class FilterEvaluationDetail:
    """Detailed explanation of a filter decision."""

    decision:     FilterDecision
    trade_symbol: str
    reason:       str     # why the decision was made


class RegimeFilterEngine:
    """Evaluates whether a labelled trade passes a FilterProfile.

    Designed for post-hoc research: apply the filter to the trade list
    produced by the existing strategy without re-running the backtest.
    """

    def evaluate(
        self,
        lt:      LabelledTrade,
        profile: FilterProfile,
    ) -> FilterDecision:
        """Return ALLOW or REJECT for one trade under one filter profile."""
        # --- Block rules (blacklist) ------------------------------------
        if lt.market_regime in profile.blocked_market_regimes:
            return FilterDecision.REJECT
        if lt.volatility_regime in profile.blocked_volatility_regimes:
            return FilterDecision.REJECT
        if lt.macro_regime in profile.blocked_macro_regimes:
            return FilterDecision.REJECT
        if lt.drawdown_env in profile.blocked_drawdown_envs:
            return FilterDecision.REJECT
        if lt.trend_regime in profile.blocked_trend_regimes:
            return FilterDecision.REJECT

        # --- Allow rules (whitelist) ------------------------------------
        if profile.allowed_market_regimes:
            if lt.market_regime not in profile.allowed_market_regimes:
                return FilterDecision.REJECT

        return FilterDecision.ALLOW

    def evaluate_with_detail(
        self,
        lt:      LabelledTrade,
        profile: FilterProfile,
    ) -> FilterEvaluationDetail:
        """Return a FilterEvaluationDetail with the reason for the decision."""
        decision = self.evaluate(lt, profile)
        sym = lt.trade.symbol

        if decision == FilterDecision.REJECT:
            # Determine which rule triggered
            if lt.market_regime in profile.blocked_market_regimes:
                reason = f"blocked_market_regime={lt.market_regime.value}"
            elif lt.volatility_regime in profile.blocked_volatility_regimes:
                reason = f"blocked_volatility={lt.volatility_regime.value}"
            elif lt.macro_regime in profile.blocked_macro_regimes:
                reason = f"blocked_macro={lt.macro_regime.value}"
            elif lt.drawdown_env in profile.blocked_drawdown_envs:
                reason = f"blocked_drawdown={lt.drawdown_env.value}"
            elif lt.trend_regime in profile.blocked_trend_regimes:
                reason = f"blocked_trend={lt.trend_regime.value}"
            elif profile.allowed_market_regimes and \
                    lt.market_regime not in profile.allowed_market_regimes:
                reason = (
                    f"market_regime={lt.market_regime.value} "
                    f"not in whitelist={[r.value for r in profile.allowed_market_regimes]}"
                )
            else:
                reason = "unknown rule"
        else:
            reason = "all rules passed"

        return FilterEvaluationDetail(decision=decision, trade_symbol=sym, reason=reason)

    def apply_to_trades(
        self,
        labelled_trades: List[LabelledTrade],
        profile:         FilterProfile,
    ) -> List[LabelledTrade]:
        """Return only the trades that pass the filter."""
        return [lt for lt in labelled_trades if self.evaluate(lt, profile) == FilterDecision.ALLOW]

    def count_by_decision(
        self,
        labelled_trades: List[LabelledTrade],
        profile:         FilterProfile,
    ) -> dict:
        """Return {'ALLOW': N, 'REJECT': N} for the trade list."""
        from collections import Counter
        counts = Counter(self.evaluate(lt, profile).value for lt in labelled_trades)
        return {"ALLOW": counts.get("ALLOW", 0), "REJECT": counts.get("REJECT", 0)}
