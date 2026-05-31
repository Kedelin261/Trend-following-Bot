"""Position sizing based on fixed-dollar risk per trade.

Formula:
    Position Size = Dollar Risk ÷ Risk Per Share

where:
    Dollar Risk     = account_cash × risk_percent / 100
    Risk Per Share  = abs(Entry − Stop)

No broker code. No API calls. Pure arithmetic.
"""

import logging
from typing import Optional

from src.risk.models import RiskProfile

logger = logging.getLogger(__name__)


class PositionSizer:
    """Converts a dollar risk budget into a discrete share count.

    Fractional shares are floored to whole numbers.  Future phases can
    extend this class to support fractional shares or lot-based sizing
    (e.g. Forex mini-lots) without changing callers.
    """

    def calculate_position_size(
        self,
        dollar_risk: float,
        risk_per_share: float,
    ) -> int:
        """Return the number of whole shares to trade.

        Parameters
        ----------
        dollar_risk     : maximum dollar loss permitted (from RiskProfile)
        risk_per_share  : abs(entry − stop); must be > 0

        Returns
        -------
        Position size as an integer (0 when risk_per_share ≤ 0).
        """
        if risk_per_share <= 0:
            logger.warning(
                "position_sizer: risk_per_share=%.5f ≤ 0 — returning 0 shares",
                risk_per_share,
            )
            return 0

        raw = dollar_risk / risk_per_share
        size = max(0, int(raw))  # floor to whole shares

        logger.debug(
            "position_sizer: dollar_risk=%.2f risk_per_share=%.5f raw=%.3f size=%d",
            dollar_risk,
            risk_per_share,
            raw,
            size,
        )
        return size

    def dollar_risk_from_profile(self, profile: RiskProfile) -> float:
        """Extract the permissible dollar risk from a RiskProfile."""
        return profile.dollar_risk_per_trade
