"""Trade validation — guards that reject candidates failing risk standards.

All rules are config-driven.  The validator is stateless; results depend
only on the TradeCandidate values passed in.

Rejection rules (evaluated in priority order):
  1. ATR ≤ 0             — no volatility measurement available
  2. Entry == Stop       — zero risk distance (division-by-zero risk)
  3. Signal Score < min  — signal quality too low
  4. Risk/Reward < min   — unfavourable trade geometry
  5. Position Size ≤ 0   — capital too small for even 1 share at this risk

No broker code. No API calls. Pure validation logic.
"""

import logging
from typing import Optional, Tuple

from src.risk.models import TradeCandidate

logger = logging.getLogger(__name__)

_FLOAT_EPSILON = 1e-8


class TradeValidator:
    """Validates a TradeCandidate against configurable risk thresholds.

    Parameters
    ----------
    minimum_signal_score : floor signal quality (default 70.0)
    minimum_risk_reward  : floor R:R ratio (default 1.5)
    """

    def __init__(
        self,
        minimum_signal_score: float = 70.0,
        minimum_risk_reward:  float = 1.5,
    ) -> None:
        self.minimum_signal_score = minimum_signal_score
        self.minimum_risk_reward  = minimum_risk_reward

    def validate_trade(
        self, candidate: TradeCandidate
    ) -> Tuple[bool, Optional[str]]:
        """Apply all validation rules to *candidate*.

        Returns
        -------
        (True, None)           — trade passes all rules
        (False, reason_str)    — trade rejected; reason explains why
        """
        # Rule 1: ATR must be positive
        if candidate.atr <= 0:
            return self._reject(
                candidate,
                "ATR is zero or negative — cannot calculate stop distance",
            )

        # Rule 2: Entry must not equal stop (prevents division-by-zero in sizers)
        if abs(candidate.entry_price - candidate.stop_loss) < _FLOAT_EPSILON:
            return self._reject(
                candidate,
                "Entry price equals stop loss — no meaningful risk distance",
            )

        # Rule 3: Signal quality threshold
        if candidate.signal_score < self.minimum_signal_score:
            return self._reject(
                candidate,
                f"Signal score {candidate.signal_score:.1f} is below "
                f"minimum {self.minimum_signal_score:.1f}",
            )

        # Rule 4: Minimum risk/reward
        if candidate.risk_reward_ratio < self.minimum_risk_reward:
            return self._reject(
                candidate,
                f"Risk/reward {candidate.risk_reward_ratio:.2f} is below "
                f"minimum {self.minimum_risk_reward:.2f}",
            )

        # Rule 5: Viable position size
        if candidate.position_size <= 0:
            return self._reject(
                candidate,
                "Position size is zero — dollar risk too small for this "
                "risk-per-share at current account settings",
            )

        logger.info(
            "trade_validator: APPROVED %s/%s score=%.1f rr=%.2f size=%d",
            candidate.symbol,
            candidate.timeframe,
            candidate.signal_score,
            candidate.risk_reward_ratio,
            candidate.position_size,
        )
        return True, None

    @staticmethod
    def _reject(
        candidate: TradeCandidate, reason: str
    ) -> Tuple[bool, str]:
        logger.info(
            "trade_validator: REJECTED %s/%s — %s",
            candidate.symbol,
            candidate.timeframe,
            reason,
        )
        return False, reason
