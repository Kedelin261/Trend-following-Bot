"""ATR-based take profit calculation.

LONG  target = Entry + (ATR × multiplier)
SHORT target = Entry − (ATR × multiplier)

No broker code. No API calls. Pure arithmetic.
"""

import logging

from src.signals.models import SignalType

logger = logging.getLogger(__name__)


class TakeProfitEngine:
    """Calculates ATR-based take profit levels.

    Parameters
    ----------
    multiplier : ATR multiple for target distance (default 3.0)
    """

    def __init__(self, multiplier: float = 3.0) -> None:
        if multiplier <= 0:
            raise ValueError(f"Target multiplier must be > 0, got {multiplier}")
        self.multiplier = multiplier

    def calculate_take_profit(
        self,
        signal_type: SignalType,
        entry_price: float,
        atr: float,
    ) -> float:
        """Return the take profit price for an entry.

        Parameters
        ----------
        signal_type : LONG → target above entry; SHORT → target below entry
        entry_price : intended entry price
        atr         : current ATR value

        Returns
        -------
        take_profit price
        """
        target_distance = atr * self.multiplier

        if signal_type == SignalType.LONG:
            target = entry_price + target_distance
        else:
            target = entry_price - target_distance

        logger.debug(
            "take_profit_engine: type=%s entry=%.5f atr=%.5f mult=%.1f target=%.5f",
            signal_type.value,
            entry_price,
            atr,
            self.multiplier,
            target,
        )
        return target
