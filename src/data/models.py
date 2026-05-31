"""Standard data models shared across all market data providers.

All providers normalize their raw data into Candle and Quote.
No provider-specific types belong here.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Candle:
    """Normalized OHLCV candle from any provider."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    provider: str
    resolved_symbol: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        if self.resolved_symbol is None:
            self.resolved_symbol = self.symbol


@dataclass
class Quote:
    """Latest bid/ask tick from any provider."""

    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    provider: str
