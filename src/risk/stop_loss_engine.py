"""ATR-based stop loss calculation.

LONG  stop = Entry − (ATR × multiplier)
SHORT stop = Entry + (ATR × multiplier)

No broker code. No API calls. Pure arithmetic.
"""

import logging

from src.signals.models import SignalType

logger = logging.getLogger(__name__)


class StopLossEngine:
    """Calculates ATR-based stop loss levels.

    Parameters
    ----------
    multiplier : ATR multiple for stop distance (default 2.0)
    """

    def __init__(self, multiplier: float = 2.0) -> None:
        if multiplier <= 0:
            raise ValueError(f"Stop multiplier must be > 0, got {multiplier}")
        self.multiplier = multiplier

    def calculate_stop_loss(
        self,
        signal_type: SignalType,
        entry_price: float,
        atr: float,
    ) -> float:
        """Return the stop loss price for an entry.

        Parameters
        ----------
        signal_type : LONG → stop below entry; SHORT → stop above entry
        entry_price : intended entry price
        atr         : current ATR value

        Returns
        -------
        stop_loss price
        """
        stop_distance = atr * self.multiplier

        if signal_type == SignalType.LONG:
            stop = entry_price - stop_distance
        else:
            stop = entry_price + stop_distance

        logger.debug(
            "stop_loss_engine: type=%s entry=%.5f atr=%.5f mult=%.1f stop=%.5f",
            signal_type.value,
            entry_price,
            atr,
            self.multiplier,
            stop,
        )
        return stop
