"""Registry for canonical trading symbols with broker-suffix alias resolution.

Brokers append suffixes to standard symbol names (EURUSD.r, EURUSDm, etc.).
This registry tries aliases in order and returns the first broker match,
keeping the canonical name stable for the rest of the system.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SymbolRegistry:
    """Maps canonical symbol names to broker-specific variants."""

    def __init__(
        self,
        symbols: List[str],
        aliases: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._symbols: List[str] = list(symbols)
        self._aliases: Dict[str, List[str]] = aliases or {}

    def get_symbols(self) -> List[str]:
        """Return configured canonical symbol list."""
        return self._symbols.copy()

    def get_candidates(self, symbol: str) -> List[str]:
        """Return symbol followed by all configured aliases for broker lookup."""
        seen: set = set()
        candidates: List[str] = []
        for name in [symbol] + self._aliases.get(symbol, []):
            if name not in seen:
                seen.add(name)
                candidates.append(name)
        return candidates

    def resolve(self, symbol: str, available: List[str]) -> Optional[str]:
        """Find the first candidate present in *available* broker symbols.

        Returns the resolved broker symbol, or None if none match.
        Logs resolution result at INFO/WARNING level.
        """
        logger.debug(
            "symbol_resolution_attempt: symbol=%s candidates=%s",
            symbol,
            self.get_candidates(symbol),
        )
        available_set = set(available)
        for candidate in self.get_candidates(symbol):
            if candidate in available_set:
                logger.info(
                    "symbol_resolution_success: %s -> %s", symbol, candidate
                )
                return candidate

        logger.warning(
            "symbol_resolution_failure: '%s' not found in broker symbols. "
            "Hint: Add the broker suffix as an alias in config/settings.yaml "
            "under symbol_aliases.",
            symbol,
        )
        return None

    @classmethod
    def from_config(cls, config: dict) -> "SymbolRegistry":
        """Construct a SymbolRegistry from the loaded settings dict."""
        return cls(
            symbols=config.get("symbols", []),
            aliases=config.get("symbol_aliases", {}),
        )
