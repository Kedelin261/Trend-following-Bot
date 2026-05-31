"""Asset selector — manages which assets the strategy should trade.

Research findings (Phase 4.5) revealed that the strategy works
differently across asset classes. This module codifies those findings
and prevents deployment on assets with no confirmed edge.

Default configuration based on Phase 4.5 research:
  Recommended : SPY, VOO, DIA
  Excluded    : QQQ, XLE, XLF, IWM (negative or insufficient edge)

No broker code. No API calls. Pure configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


DEFAULT_RECOMMENDED = ["SPY", "VOO", "DIA"]
DEFAULT_EXCLUDED    = ["QQQ", "XLE", "XLF", "IWM"]


@dataclass
class AssetMetadata:
    """Research-backed metadata for a single asset."""

    symbol:       str
    recommended:  bool
    reason:       str          # why recommended / excluded
    expectancy:   Optional[float] = None
    profit_factor: Optional[float] = None
    trade_count:  Optional[int]   = None


class AssetSelector:
    """Manages the tradeable universe based on research findings.

    Parameters
    ----------
    recommended : symbols with confirmed (or research-backed) edge
    excluded    : symbols explicitly excluded from trading
    """

    def __init__(
        self,
        recommended: List[str] = None,
        excluded:    List[str] = None,
        metadata:    Dict[str, AssetMetadata] = None,
    ) -> None:
        self._recommended = list(recommended or DEFAULT_RECOMMENDED)
        self._excluded    = set(excluded    or DEFAULT_EXCLUDED)
        self._metadata    = metadata or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_tradeable(self, symbol: str) -> bool:
        """True when symbol is recommended and not explicitly excluded."""
        return symbol in self._recommended and symbol not in self._excluded

    def filter(self, symbols: List[str]) -> List[str]:
        """Return only the tradeable symbols from *symbols*."""
        return [s for s in symbols if self.is_tradeable(s)]

    @property
    def recommended(self) -> List[str]:
        return list(self._recommended)

    @property
    def excluded(self) -> List[str]:
        return list(self._excluded)

    def get_metadata(self, symbol: str) -> Optional[AssetMetadata]:
        return self._metadata.get(symbol)

    def all_metadata(self) -> Dict[str, AssetMetadata]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_research(cls, research_report) -> "AssetSelector":
        """Build an AssetSelector from Phase 4.5 ResearchReport findings."""
        recommended = research_report.recommended_assets or DEFAULT_RECOMMENDED
        excluded    = research_report.avoid_assets        or DEFAULT_EXCLUDED

        metadata: Dict[str, AssetMetadata] = {}
        for ar in research_report.asset_results:
            metadata[ar.symbol] = AssetMetadata(
                symbol        = ar.symbol,
                recommended   = ar.symbol in recommended,
                reason        = "Passed health check" if ar.recommended else
                                f"expectancy={ar.expectancy:.2f}",
                expectancy    = ar.expectancy,
                profit_factor = ar.results.profit_factor,
                trade_count   = ar.results.total_trades,
            )

        return cls(recommended=recommended, excluded=excluded, metadata=metadata)

    @classmethod
    def default(cls) -> "AssetSelector":
        """Default selector based on Phase 4.5 research conclusions."""
        return cls(
            recommended = DEFAULT_RECOMMENDED,
            excluded    = DEFAULT_EXCLUDED,
            metadata    = {
                "SPY": AssetMetadata("SPY", True,  "Best overall performance in research"),
                "VOO": AssetMetadata("VOO", True,  "Correlated with SPY; consistent results"),
                "DIA": AssetMetadata("DIA", True,  "Positive expectancy in bull markets"),
                "QQQ": AssetMetadata("QQQ", False, "Negative expectancy; excluded"),
                "XLE": AssetMetadata("XLE", False, "High volatility; poor fit"),
                "XLF": AssetMetadata("XLF", False, "Insufficient edge"),
                "IWM": AssetMetadata("IWM", False, "Below-average results"),
            },
        )
