"""Factory that creates a MarketDataProvider from the configured provider name.

To add a new provider:
  1. Create src/providers/your_provider.py implementing MarketDataProvider
  2. Add an elif branch here
  3. Set 'provider: your_name' in config/settings.yaml
  No other file needs to change.
"""

import logging
from typing import Dict

from src.data.symbol_registry import SymbolRegistry
from src.providers.market_data_provider import MarketDataProvider

logger = logging.getLogger(__name__)

_IMPLEMENTED = ("mt5", "ibkr")
_PLANNED = ("polygon", "yahoo", "alpaca", "oanda")


class ProviderFactory:
    """Instantiates the correct MarketDataProvider from config."""

    @staticmethod
    def create(config: Dict, symbol_registry: SymbolRegistry) -> MarketDataProvider:
        """Return a MarketDataProvider for the name in config['provider'].

        Raises ValueError for unknown or not-yet-implemented providers.
        """
        name = config.get("provider", "").lower().strip()

        if not name:
            raise ValueError(
                "No provider configured. "
                "Set 'provider: mt5' in config/settings.yaml."
            )

        if name == "mt5":
            from src.providers.mt5_provider import MT5Provider
            logger.info("provider_factory: creating MT5Provider")
            return MT5Provider(config=config, symbol_registry=symbol_registry)

        if name == "ibkr":
            from src.providers.ibkr_provider import IBKRProvider
            logger.info("provider_factory: creating IBKRProvider")
            return IBKRProvider(config=config, symbol_registry=symbol_registry)

        if name in _PLANNED:
            raise ValueError(
                f"Provider '{name}' is planned but not yet implemented. "
                f"Create src/providers/{name}_provider.py implementing "
                f"MarketDataProvider, then register it in ProviderFactory."
            )

        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Implemented: {_IMPLEMENTED}. "
            f"Planned (not yet built): {_PLANNED}."
        )
