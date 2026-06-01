"""Asset Exclusion Filter — Phase 5.4.

Gates trade candidates by symbol.  If a symbol appears in the
excluded_assets set, the trade is rejected before entry.

This is the mechanism to validate the Phase 5.3 SCHD finding:
  SCHD: PF=0.95, Exp=-$2.52 (negative expectancy)

CRITICAL CONSTRAINTS:
  - Does NOT modify entry logic
  - Does NOT modify signal generation
  - Does NOT modify risk engine
  - Does NOT modify position sizing
  - Only gates already-approved trade candidates by asset symbol
  - No searching for alternative ETFs
  - Only validates the exact asset found in Phase 5.3

Research only. No execution. No broker code.
"""

import logging
from typing import FrozenSet

logger = logging.getLogger(__name__)


class AssetExclusionFilter:
    """Rejects trade candidates whose symbol is in the exclusion set.

    Parameters
    ----------
    excluded_assets : frozenset of ticker symbols to exclude (e.g. {"SCHD"})
    """

    def __init__(self, excluded_assets: FrozenSet[str]) -> None:
        self._excluded = frozenset(s.upper() for s in excluded_assets)

    @property
    def excluded_assets(self) -> FrozenSet[str]:
        return self._excluded

    def allows(self, symbol: str) -> bool:
        """Return True if the symbol is allowed (not excluded).

        Parameters
        ----------
        symbol : ticker symbol (case-insensitive)
        """
        allowed = symbol.upper() not in self._excluded
        if not allowed:
            logger.debug(
                "asset_exclusion_filter: BLOCKED symbol=%s", symbol
            )
        return allowed

    def __repr__(self) -> str:
        return f"AssetExclusionFilter(excluded={sorted(self._excluded)})"
