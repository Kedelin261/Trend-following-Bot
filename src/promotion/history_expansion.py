"""History expansion — tests whether longer historical windows produce ≥ 100 trades.

The strategy is NOT changed.  Only the candle window is extended:
3000 → 3500 → 4000 → maximum available.

SAFEGUARD: we accept a window only when it passes the quality floor
(PF ≥ 1.50, Exp > $0, DD < 15 %).  More history that destroys quality
is rejected.

No broker code. No API calls. Candle data only.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)

CANDLE_COUNTS    = [3000, 3500, 4000, 5000]   # 5000 = effectively "max" for most providers
QUALITY_MIN_PF   = 1.50
QUALITY_MIN_EXP  = 0.0
QUALITY_MAX_DD   = 15.0
PROMOTION_MIN_TRADES = 100


@dataclass
class HistoryExpansionResult:
    """Single-window result from a history expansion run."""

    candle_count:     int
    total_trades:     int
    aggregate_pf:     float
    aggregate_exp:    float
    worst_dd:         float
    assets_passing:   int
    meets_threshold:  bool       # total_trades >= PROMOTION_MIN_TRADES
    is_max_available: bool = False
    meets_quality:    bool = True
    note:             str  = ""

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.aggregate_pf) else f"{self.aggregate_pf:.2f}"


class HistoryExpansionResearcher:
    """Runs the locked strategy with increasing historical windows.

    Uses the most recent *candle_count* bars (tail slice) for each window.
    """

    def __init__(
        self,
        config:   dict,
        profile:  StrategyProfile = None,
        counts:   List[int] = None,
    ) -> None:
        self._config  = config
        self._profile = profile or BEST_DENSITY_PROFILE
        self._counts  = counts or CANDLE_COUNTS

    def research(
        self,
        asset_candles: Dict[str, List[Candle]],
    ) -> List[HistoryExpansionResult]:
        """Test each candle window and return all results sorted by candle_count."""
        results: List[HistoryExpansionResult] = []

        max_available = max(len(c) for c in asset_candles.values()) if asset_candles else 0

        # Fixed windows
        for count in self._counts:
            sliced = {
                sym: candles[-count:] if len(candles) >= count else candles
                for sym, candles in asset_candles.items()
            }
            actual = min(len(c) for c in sliced.values()) if sliced else 0
            logger.info("history_expansion: testing %d candles (actual=%d)", count, actual)
            result = self._run_window(sliced, actual, is_max=False)
            results.append(result)

        # Maximum available window (all candles)
        if max_available > max(self._counts, default=0):
            logger.info("history_expansion: testing max=%d candles", max_available)
            result = self._run_window(asset_candles, max_available, is_max=True)
            results.append(result)

        # Remove duplicate counts (if max_available ≤ largest fixed count)
        seen = set()
        unique = []
        for r in results:
            key = r.candle_count
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return sorted(unique, key=lambda r: r.candle_count)

    def best_window(
        self, results: List[HistoryExpansionResult]
    ):
        """Return the largest window that meets promotion threshold and quality."""
        passing = [r for r in results if r.meets_threshold and r.meets_quality]
        return max(passing, key=lambda r: r.candle_count) if passing else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_window(
        self,
        asset_candles: Dict[str, List[Candle]],
        candle_count:  int,
        is_max:        bool,
    ) -> HistoryExpansionResult:
        asset_results = self._run_portfolio(asset_candles)

        total  = sum(bt.total_trades for bt in asset_results.values())
        all_bt = list(asset_results.values())

        pf, exp = self._aggregate_quality(asset_results)
        worst_dd = max((bt.max_drawdown for bt in all_bt), default=0.0)
        assets_ok = sum(
            1 for bt in all_bt
            if bt.profit_factor >= QUALITY_MIN_PF
            and bt.expectancy > QUALITY_MIN_EXP
            and bt.max_drawdown <= QUALITY_MAX_DD
        )

        meets_q = (
            (pf >= QUALITY_MIN_PF or math.isinf(pf))
            and exp > QUALITY_MIN_EXP
            and worst_dd <= QUALITY_MAX_DD
        )
        note = "" if meets_q else (
            f"⚠ QUALITY FAIL: pf={pf:.2f} exp={exp:.2f} dd={worst_dd:.1f}%"
        )

        return HistoryExpansionResult(
            candle_count     = candle_count,
            total_trades     = total,
            aggregate_pf     = pf,
            aggregate_exp    = exp,
            worst_dd         = worst_dd,
            assets_passing   = assets_ok,
            meets_threshold  = total >= PROMOTION_MIN_TRADES,
            is_max_available = is_max,
            meets_quality    = meets_q,
            note             = note,
        )

    def _run_portfolio(
        self, asset_candles: Dict[str, List[Candle]]
    ) -> Dict[str, BacktestResults]:
        results = {}
        for sym, candles in asset_candles.items():
            if candles:
                engine = self._profile.build_backtest_engine(self._config)
                results[sym] = engine.run(candles)
        return results

    @staticmethod
    def _aggregate_quality(
        asset_results: Dict[str, BacktestResults],
    ):
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
