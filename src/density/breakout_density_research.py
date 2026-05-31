"""Breakout density research — finds the minimum breakout threshold that
preserves edge while maximising trade frequency.

Baseline: 1.00 % (V2)
Research: 1.00 %, 0.90 %, 0.80 %, 0.75 %, 0.50 %

SAFEGUARD: thresholds that reduce PF below 1.50 or turn expectancy negative
are rejected even if they generate more trades.

No broker code. No API calls. Candle data only.
"""

import logging
from dataclasses import dataclass, replace
from typing import List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile, V2_PROFILE

logger = logging.getLogger(__name__)

BREAKOUT_THRESHOLDS  = [0.0050, 0.0075, 0.0080, 0.0090, 0.0100]
QUALITY_MIN_PF       = 1.50
QUALITY_MIN_EXP      = 0.0
QUALITY_MAX_DD       = 15.0
QUALITY_MIN_TRADES   = 10


@dataclass
class BreakoutDensityResult:
    """Backtest result for a single breakout threshold."""

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
    def density_score(self) -> float:
        return float(self.trade_count) if self.meets_quality else 0.0

    @property
    def pct_str(self) -> str:
        return f"{self.threshold * 100:.2f}%"


class BreakoutDensityResearcher:
    """Tests breakout thresholds to find the minimum that preserves edge."""

    def __init__(
        self,
        config:       dict,
        base_profile: StrategyProfile = None,
        thresholds:   List[float]    = None,
    ) -> None:
        self._config     = config
        self._base       = base_profile or V2_PROFILE
        self._thresholds = thresholds or BREAKOUT_THRESHOLDS

    def research(self, candles: List[Candle]) -> List[BreakoutDensityResult]:
        """Run backtest for each breakout threshold. Sorted by density_score desc."""
        results = []
        for thresh in sorted(self._thresholds, reverse=True):   # high → low
            logger.info("breakout_density: testing threshold=%.4f", thresh)
            bt   = self._run(candles, thresh)
            note = "" if bt.total_trades >= QUALITY_MIN_TRADES else "⚠ low sample"
            if not self._quality_passes(bt):
                note = f"⚠ QUALITY FAIL: pf={bt.profit_factor:.2f} exp={bt.expectancy:.2f}"
            results.append(BreakoutDensityResult(
                threshold     = thresh,
                trade_count   = bt.total_trades,
                expectancy    = bt.expectancy,
                profit_factor = bt.profit_factor,
                max_drawdown  = bt.max_drawdown,
                win_rate      = bt.win_rate,
                note          = note,
            ))
        return sorted(results, key=lambda r: r.density_score, reverse=True)

    def best_threshold(self, results: List[BreakoutDensityResult]) -> Optional[float]:
        """Return lowest quality-preserving threshold (most trades)."""
        passing = [r for r in results if r.meets_quality and r.trade_count >= QUALITY_MIN_TRADES]
        return max(passing, key=lambda r: r.trade_count).threshold if passing else None

    def _run(self, candles: List[Candle], thresh: float) -> BacktestResults:
        profile = replace(
            self._base,
            name=f"BRK{thresh*100:.2f}%",
            description=f"V2 with breakout ≥ {thresh*100:.2f}%",
            breakout_threshold=thresh,
        )
        return profile.run_backtest(candles, self._config)

    @staticmethod
    def _quality_passes(bt: BacktestResults) -> bool:
        return (
            bt.profit_factor >= QUALITY_MIN_PF
            and bt.expectancy > QUALITY_MIN_EXP
            and bt.max_drawdown <= QUALITY_MAX_DD
        )
