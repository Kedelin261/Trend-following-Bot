"""Robustness checker — verifies strategy stability across market periods.

Tests three non-overlapping 60 % windows of the candle history:
  early  : oldest 60 % (bars 0 → 60 %)
  middle : central 60 % (bars 20 % → 80 %)
  recent : newest 60 % (bars 40 % → 100 %)

Ratings:
  ROBUST   : all 3 windows pass quality
  MARGINAL : 2 of 3 windows pass quality
  UNSTABLE : 1 or fewer windows pass quality

A MARGINAL strategy may still be promoted but should be monitored.
An UNSTABLE strategy must return to research.

No broker code. No API calls. Candle data only.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)

QUALITY_MIN_PF   = 1.50
QUALITY_MIN_EXP  = 0.0
QUALITY_MAX_DD   = 15.0
WINDOW_FRACTION  = 0.60   # each window covers 60 % of the dataset


@dataclass
class WindowResult:
    """Strategy performance in one historical window."""

    window_name:  str       # "early", "middle", "recent"
    candle_count: int
    total_trades: int
    pf:           float
    expectancy:   float
    drawdown:     float
    passes:       bool
    note:         str = ""


@dataclass
class RobustnessResult:
    """Aggregated stability assessment across all windows."""

    window_results:  List[WindowResult]
    windows_passing: int        # 0–3
    rating:          str        # ROBUST / MARGINAL / UNSTABLE
    notes:           List[str]

    @property
    def is_acceptable(self) -> bool:
        """True when rating is ROBUST or MARGINAL (promotable states)."""
        return self.rating in ("ROBUST", "MARGINAL")


class RobustnessChecker:
    """Tests strategy stability across early, middle, and recent market periods."""

    def __init__(
        self,
        config:          dict,
        profile:         StrategyProfile = None,
        window_fraction: float = WINDOW_FRACTION,
    ) -> None:
        self._config   = config
        self._profile  = profile or BEST_DENSITY_PROFILE
        self._fraction = window_fraction

    def check(
        self,
        asset_candles: Dict[str, List[Candle]],
    ) -> RobustnessResult:
        """Run the strategy across three historical windows and rate stability."""
        min_len = min(len(c) for c in asset_candles.values()) if asset_candles else 0

        windows = self._split_windows(asset_candles, min_len)
        window_results = [
            self._test_window(name, candles_map)
            for name, candles_map in windows.items()
        ]

        passing = sum(1 for w in window_results if w.passes)

        if passing == 3:
            rating = "ROBUST"
        elif passing == 2:
            rating = "MARGINAL"
        else:
            rating = "UNSTABLE"

        notes = self._generate_notes(window_results, rating)

        logger.info(
            "robustness: windows_passing=%d/%d rating=%s",
            passing, len(window_results), rating,
        )

        return RobustnessResult(
            window_results  = window_results,
            windows_passing = passing,
            rating          = rating,
            notes           = notes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_windows(
        self,
        asset_candles: Dict[str, List[Candle]],
        min_len:       int,
    ) -> Dict[str, Dict[str, List[Candle]]]:
        """Create the three overlapping windows as slices of the candle series."""
        w = int(min_len * self._fraction)
        if w < 1:
            w = min_len

        return {
            "early":  {sym: c[:w]           for sym, c in asset_candles.items()},
            "middle": {sym: c[int(min_len * 0.2):int(min_len * 0.2) + w]
                       for sym, c in asset_candles.items()},
            "recent": {sym: c[min_len - w:] for sym, c in asset_candles.items()},
        }

    def _test_window(
        self,
        name:          str,
        asset_candles: Dict[str, List[Candle]],
    ) -> WindowResult:
        asset_bt = {}
        for sym, candles in asset_candles.items():
            if candles:
                engine      = self._profile.build_backtest_engine(self._config)
                asset_bt[sym] = engine.run(candles)

        total = sum(bt.total_trades for bt in asset_bt.values())
        pf, exp = self._aggregate(asset_bt)
        worst_dd = max((bt.max_drawdown for bt in asset_bt.values()), default=0.0)

        passes = (
            (pf >= QUALITY_MIN_PF or math.isinf(pf))
            and exp > QUALITY_MIN_EXP
            and worst_dd <= QUALITY_MAX_DD
            and total > 0
        )
        note = f"pf={pf:.2f} exp=${exp:.2f} dd={worst_dd:.1f}% trades={total}"

        return WindowResult(
            window_name  = name,
            candle_count = min((len(c) for c in asset_candles.values()), default=0),
            total_trades = total,
            pf           = pf,
            expectancy   = exp,
            drawdown     = worst_dd,
            passes       = passes,
            note         = note,
        )

    @staticmethod
    def _aggregate(asset_results: Dict[str, BacktestResults]):
        all_trades = [t for bt in asset_results.values() for t in bt.trades]
        if not all_trades:
            return 0.0, 0.0
        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if t.is_loss]
        gross_wins   = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        exp = sum(t.pnl for t in all_trades) / len(all_trades)
        return pf, exp

    @staticmethod
    def _generate_notes(windows: List[WindowResult], rating: str) -> List[str]:
        notes = []
        if rating == "ROBUST":
            notes.append("Strategy passes quality in all three historical windows")
        elif rating == "MARGINAL":
            failing = [w.window_name for w in windows if not w.passes]
            notes.append(
                f"Strategy fails quality in {failing} window(s) — "
                "monitor carefully after promotion"
            )
        else:
            failing = [w.window_name for w in windows if not w.passes]
            notes.append(
                f"Strategy unstable — fails quality in {failing} window(s). "
                "Return to research before promoting."
            )
        return notes
