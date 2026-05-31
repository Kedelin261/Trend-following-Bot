"""Asset expansion research — evaluates additional assets for inclusion.

Current V2 universe: SPY, VOO, DIA
Candidates for expansion: XLK, VTI, XLP, XLV, SCHD

Each candidate is assessed independently.  No automatic approval.
An asset is recommended only when it passes all quality thresholds AND
adds meaningful trade count to the portfolio.

SAFEGUARD: assets where PF < 1.50 or expectancy ≤ 0 are never recommended,
regardless of trade count.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults, StrategyHealth
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE

logger = logging.getLogger(__name__)

EXPANSION_CANDIDATES = ["XLK", "VTI", "XLP", "XLV", "SCHD"]
QUALITY_MIN_PF       = 1.50
QUALITY_MIN_EXP      = 0.0
QUALITY_MAX_DD       = 15.0
QUALITY_MIN_TRADES   = 10


@dataclass
class AssetExpansionResult:
    """Research result for a single expansion candidate asset."""

    symbol:       str
    candle_count: int
    trade_count:  int
    expectancy:   float
    profit_factor: float
    max_drawdown: float
    win_rate:     float
    recommended:  bool
    note:         str = ""

    @property
    def meets_quality(self) -> bool:
        return (
            self.profit_factor >= QUALITY_MIN_PF
            and self.expectancy > QUALITY_MIN_EXP
            and self.max_drawdown <= QUALITY_MAX_DD
        )


class AssetExpansionResearcher:
    """Researches expansion candidates against quality and density thresholds."""

    def __init__(
        self,
        config:       dict,
        base_profile: StrategyProfile = None,
        candidates:   List[str]      = None,
    ) -> None:
        self._config     = config
        self._base       = base_profile or V2_PROFILE
        self._candidates = candidates or EXPANSION_CANDIDATES

    def research(
        self,
        asset_candles: Dict[str, List[Candle]],
    ) -> List[AssetExpansionResult]:
        """Run backtest on each expansion candidate. Returns results ranked by expectancy."""
        results = []
        for symbol in self._candidates:
            candles = asset_candles.get(symbol, [])
            if not candles:
                logger.info("asset_expansion: no candles for %s — skipping", symbol)
                continue

            logger.info("asset_expansion: testing %s (%d candles)", symbol, len(candles))
            bt   = self._run(candles, symbol)
            rec  = bt.total_trades >= QUALITY_MIN_TRADES and self._quality_passes(bt)
            note = "" if rec else (
                f"⚠ QUALITY FAIL: pf={bt.profit_factor:.2f} exp={bt.expectancy:.2f}"
                if not self._quality_passes(bt)
                else f"⚠ low sample ({bt.total_trades} trades)"
            )
            results.append(AssetExpansionResult(
                symbol        = symbol,
                candle_count  = len(candles),
                trade_count   = bt.total_trades,
                expectancy    = bt.expectancy,
                profit_factor = bt.profit_factor,
                max_drawdown  = bt.max_drawdown,
                win_rate      = bt.win_rate,
                recommended   = rec,
                note          = note,
            ))

        return sorted(results, key=lambda r: r.expectancy, reverse=True)

    def recommended_additions(
        self, results: List[AssetExpansionResult]
    ) -> List[str]:
        """Return symbols that pass all quality and sample-size checks."""
        return [r.symbol for r in results if r.recommended]

    def _run(self, candles: List[Candle], symbol: str) -> BacktestResults:
        profile = replace(
            self._base,
            name=f"V2_{symbol}",
            description=f"V2 on {symbol}",
        )
        engine = profile.build_backtest_engine(self._config)
        return engine.run(candles)

    @staticmethod
    def _quality_passes(bt: BacktestResults) -> bool:
        return (
            bt.profit_factor >= QUALITY_MIN_PF
            and bt.expectancy > QUALITY_MIN_EXP
            and bt.max_drawdown <= QUALITY_MAX_DD
        )
