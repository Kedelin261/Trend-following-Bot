"""ADX density research — finds the ADX threshold that maximises trade
frequency while preserving edge quality.

Baseline: ADX ≥ 25 (V2)
Research: 20, 22, 25, 27, 30

SAFEGUARD: any threshold that produces PF < 1.50, Expectancy ≤ 0, or
DD > 15 % is rejected regardless of trade count.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass, replace
from typing import List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE

logger = logging.getLogger(__name__)

ADX_THRESHOLDS       = [20.0, 22.0, 25.0, 27.0, 30.0]
QUALITY_MIN_PF       = 1.50
QUALITY_MIN_EXP      = 0.0
QUALITY_MAX_DD       = 15.0
QUALITY_MIN_TRADES   = 10    # per-run minimum for a result to be informative


@dataclass
class ADXDensityResult:
    """Backtest result for a single ADX threshold."""

    threshold:    float
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
    def informative(self) -> bool:
        return self.trade_count >= QUALITY_MIN_TRADES

    @property
    def density_score(self) -> float:
        """Composite: trade_count × quality bonus (0 when quality fails)."""
        if not self.meets_quality:
            return 0.0
        return float(self.trade_count)


class ADXDensityResearcher:
    """Tests ADX thresholds to find the best trade-frequency / edge trade-off.

    Uses V2_PROFILE as the baseline; only the ADX threshold varies.
    """

    def __init__(
        self,
        config:        dict,
        base_profile:  StrategyProfile = None,
        thresholds:    List[float]    = None,
    ) -> None:
        self._config      = config
        self._base        = base_profile or V2_PROFILE
        self._thresholds  = thresholds or ADX_THRESHOLDS

    def research(self, candles: List[Candle]) -> List[ADXDensityResult]:
        """Run backtest for each ADX threshold. Returns results sorted by density_score."""
        results = []
        for thresh in self._thresholds:
            logger.info("adx_density: testing ADX ≥ %.0f", thresh)
            bt   = self._run(candles, thresh)
            note = "" if bt.total_trades >= QUALITY_MIN_TRADES else "⚠ low sample"
            if not self._quality_passes(bt):
                note = f"⚠ QUALITY FAIL: pf={bt.profit_factor:.2f} exp={bt.expectancy:.2f}"
            results.append(ADXDensityResult(
                threshold     = thresh,
                trade_count   = bt.total_trades,
                expectancy    = bt.expectancy,
                profit_factor = bt.profit_factor,
                max_drawdown  = bt.max_drawdown,
                win_rate      = bt.win_rate,
                note          = note,
            ))
        return sorted(results, key=lambda r: r.density_score, reverse=True)

    def best_threshold(self, results: List[ADXDensityResult]) -> Optional[float]:
        """Return the threshold with the best density_score (quality-gated)."""
        passing = [r for r in results if r.meets_quality and r.informative]
        return max(passing, key=lambda r: r.trade_count).threshold if passing else None

    def _run(self, candles: List[Candle], thresh: float) -> BacktestResults:
        profile = replace(
            self._base,
            name=f"ADX{thresh:.0f}",
            description=f"V2 with ADX ≥ {thresh:.0f}",
            adx_threshold=thresh,
        )
        return profile.run_backtest(candles, self._config)

    @staticmethod
    def _quality_passes(bt: BacktestResults) -> bool:
        return (
            bt.profit_factor >= QUALITY_MIN_PF
            and bt.expectancy > QUALITY_MIN_EXP
            and bt.max_drawdown <= QUALITY_MAX_DD
        )
