"""Registry of standard timeframe identifiers used across all providers.

Provider-specific constants (e.g. mt5.TIMEFRAME_H1) live inside the
provider implementation — this module is provider-free.
"""

from typing import List

STANDARD_TIMEFRAMES: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]


class TimeframeRegistry:
    """Validates and enumerates standard timeframe strings."""

    _valid: frozenset = frozenset(STANDARD_TIMEFRAMES)

    @classmethod
    def is_valid(cls, timeframe: str) -> bool:
        """Return True if *timeframe* is a recognised standard identifier."""
        return timeframe in cls._valid

    @classmethod
    def get_all(cls) -> List[str]:
        """Return all standard timeframes in canonical order."""
        return STANDARD_TIMEFRAMES.copy()

    @classmethod
    def validate(cls, timeframe: str) -> str:
        """Return *timeframe* unchanged, raising ValueError if unknown."""
        if not cls.is_valid(timeframe):
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Valid: {STANDARD_TIMEFRAMES}"
            )
        return timeframe
