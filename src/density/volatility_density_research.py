"""Volatility density research — determines whether MEDIUM_ONLY is too restrictive.

Baseline: MEDIUM_ONLY (V2)
Research: MEDIUM_ONLY, LOW_AND_MEDIUM, MEDIUM_AND_HIGH, NONE

SAFEGUARD: modes that reduce expectancy below 0 or PF below 1.50 are rejected.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass, replace
from typing import List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE
from src.refinement.volatility_trade_filter import VolatilityFilterMode

logger = logging.getLogger(__name__)

QUALITY_MIN_PF     = 1.50
QUALITY_MIN_EXP    = 0.0
QUALITY_MAX_DD     = 15.0
QUALITY_MIN_TRADES = 10

RESEARCH_MODES = [
    VolatilityFilterMode.MEDIUM_ONLY,
    VolatilityFilterMode.LOW_AND_MEDIUM,
    VolatilityFilterMode.MEDIUM_AND_HIGH,
    VolatilityFilterMode.NONE,
]


@dataclass
class VolatilityDensityResult:
    """Backtest result for a single volatility filter mode."""

    mode:         VolatilityFilterMode
    trade_count:  int
    expectancy:   float
    profit_factor: float
    max_drawdown: float
    win_rate:     float
    note:         str = ""

    @property
    def meets_quality(self) -> bool:
        return (
            self.profit_factor >= QUALITY_MIN_PF
            and self.expectancy > QUALITY_MIN_EXP
            and self.max_drawdown <= QUALITY_MAX_DD
        )

    @property
    def density_score(self) -> float:
        return float(self.trade_count) if self.meets_quality else 0.0


class VolatilityDensityResearcher:
    """Tests volatility filter modes to find the best trade-density balance."""

    def __init__(
        self,
        config:       dict,
        base_profile: StrategyProfile = None,
        modes:        List[VolatilityFilterMode] = None,
    ) -> None:
        self._config = config
        self._base   = base_profile or V2_PROFILE
        self._modes  = modes or RESEARCH_MODES

    def research(self, candles: List[Candle]) -> List[VolatilityDensityResult]:
        """Run backtest for each volatility mode. Returns sorted by density_score."""
        results = []
        for mode in self._modes:
            logger.info("vol_density: testing mode=%s", mode.value)
            bt   = self._run(candles, mode)
            note = "" if bt.total_trades >= QUALITY_MIN_TRADES else "⚠ low sample"
            if not self._quality_passes(bt):
                note = f"⚠ QUALITY FAIL: pf={bt.profit_factor:.2f}"
            results.append(VolatilityDensityResult(
                mode          = mode,
                trade_count   = bt.total_trades,
                expectancy    = bt.expectancy,
                profit_factor = bt.profit_factor,
                max_drawdown  = bt.max_drawdown,
                win_rate      = bt.win_rate,
                note          = note,
            ))
        return sorted(results, key=lambda r: r.density_score, reverse=True)

    def best_mode(self, results: List[VolatilityDensityResult]) -> Optional[VolatilityFilterMode]:
        """Return mode with best density_score that still meets quality."""
        passing = [r for r in results if r.meets_quality and r.trade_count >= QUALITY_MIN_TRADES]
        return max(passing, key=lambda r: r.trade_count).mode if passing else None

    def _run(self, candles: List[Candle], mode: VolatilityFilterMode) -> BacktestResults:
        profile = replace(
            self._base,
            name=f"VOL_{mode.value}",
            description=f"V2 with {mode.value} volatility filter",
            volatility_mode=mode,
        )
        return profile.run_backtest(candles, self._config)

    @staticmethod
    def _quality_passes(bt: BacktestResults) -> bool:
        return (
            bt.profit_factor >= QUALITY_MIN_PF
            and bt.expectancy > QUALITY_MIN_EXP
            and bt.max_drawdown <= QUALITY_MAX_DD
        )
