"""Multi-timeframe research — tests combinations of D1, H4, H1, W1.

Each combination runs independent backtests on each included timeframe,
then aggregates results accounting for signal overlap.

Combinations researched:
  D1 only
  H4 only
  H1 only
  W1 only
  D1 + H4
  D1 + H1
  D1 + H4 + H1

SAFEGUARD: combinations are rejected when:
  - Combined quality falls below thresholds (PF, Exp, DD)
  - Signal overlap ≥ 50% (artificially inflated trade count)

No broker code. No API calls. Candle data only.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.timeframe.signal_overlap_analyzer import (
    CombinedOverlapResult,
    SignalOverlapAnalyzer,
)
from src.timeframe.timeframe_backtester import TimeframeBacktestResult, TimeframeBacktester
from src.timeframe.timeframe_profile import MultiTimeframeProfile

logger = logging.getLogger(__name__)

QUALITY_MIN_PF  = 1.50
QUALITY_MIN_EXP = 0.0
QUALITY_MAX_DD  = 15.0


@dataclass
class MultiTimeframeResult:
    """Result of a multi-timeframe combination backtest."""

    combo:                MultiTimeframeProfile
    per_tf_results:       Dict[str, Dict[str, TimeframeBacktestResult]]  # {tf: {sym: result}}
    overlap:              CombinedOverlapResult

    total_raw_trades:     int      # D1 + H4 + H1 trades (with overlap)
    unique_trades:        int      # estimated after deduplication
    combined_pf:          float
    combined_expectancy:  float
    max_drawdown:         float    # worst single-asset DD across all timeframes
    combined_win_rate:    float
    meets_quality:        bool
    note:                 str = ""

    @property
    def label(self) -> str:
        return self.combo.label

    @property
    def is_viable(self) -> bool:
        return self.meets_quality and self.overlap.acceptable


class MultiTimeframeResearcher:
    """Runs multi-timeframe combination backtests and aggregates results."""

    def __init__(
        self,
        config:   dict,
        backtester: TimeframeBacktester,
        overlap_analyzer: SignalOverlapAnalyzer = None,
    ) -> None:
        self._config   = config
        self._bt       = backtester
        self._overlap  = overlap_analyzer or SignalOverlapAnalyzer()

    def research_single(
        self,
        symbol_candles_by_tf: Dict[str, Dict[str, List[Candle]]],
        combo:                MultiTimeframeProfile,
    ) -> MultiTimeframeResult:
        """Run one multi-timeframe combination and return aggregated result.

        Parameters
        ----------
        symbol_candles_by_tf : {symbol: {timeframe: [candles]}}
        combo                : which timeframe combination to test
        """
        logger.info("multi_tf: testing %s", combo.label)

        # Run backtest for each timeframe
        per_tf: Dict[str, Dict[str, TimeframeBacktestResult]] = {}
        for tf_profile in combo.profiles:
            tf_key = tf_profile.timeframe
            per_tf[tf_key] = {}
            for sym, tf_candles in symbol_candles_by_tf.items():
                candles = tf_candles.get(tf_key, [])
                if candles:
                    per_tf[tf_key][sym] = self._bt.backtest(sym, candles, tf_profile)

        # Collect all trades grouped by timeframe (across all assets)
        trades_by_tf: Dict[str, list] = {
            tf: [
                t
                for sym_res in tf_res.values()
                for t in sym_res.results.trades
            ]
            for tf, tf_res in per_tf.items()
        }

        # Calculate overlap
        overlap = self._overlap.calculate_combined_overlap(trades_by_tf)

        # Aggregate quality metrics from all trades
        all_trades = [t for tl in trades_by_tf.values() for t in tl]
        combined_pf, combined_exp, combined_wr = self._aggregate_quality(all_trades)

        worst_dd = max(
            (r.results.max_drawdown
             for tf_res in per_tf.values()
             for r in tf_res.values()),
            default=0.0,
        )

        meets = (
            combined_pf >= QUALITY_MIN_PF
            and combined_exp > QUALITY_MIN_EXP
            and worst_dd <= QUALITY_MAX_DD
        )
        note = "" if meets else (
            f"⚠ QUALITY FAIL: pf={combined_pf:.2f} "
            f"exp={combined_exp:.2f} dd={worst_dd:.1f}%"
        )

        return MultiTimeframeResult(
            combo              = combo,
            per_tf_results     = per_tf,
            overlap            = overlap,
            total_raw_trades   = overlap.total_raw_trades,
            unique_trades      = overlap.estimated_unique_trades,
            combined_pf        = combined_pf,
            combined_expectancy = combined_exp,
            max_drawdown       = worst_dd,
            combined_win_rate  = combined_wr,
            meets_quality      = meets,
            note               = note,
        )

    def research_all(
        self,
        symbol_candles_by_tf: Dict[str, Dict[str, List[Candle]]],
        combos:               List[MultiTimeframeProfile],
    ) -> List[MultiTimeframeResult]:
        """Research all combinations and return sorted by unique_trades."""
        results = [
            self.research_single(symbol_candles_by_tf, combo)
            for combo in combos
        ]
        return sorted(results, key=lambda r: r.unique_trades, reverse=True)

    def best_combo(
        self, results: List[MultiTimeframeResult]
    ) -> Optional[MultiTimeframeResult]:
        """Return the viable combination with most unique trades."""
        viable = [r for r in results if r.is_viable]
        return max(viable, key=lambda r: r.unique_trades) if viable else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_quality(trades) -> tuple:
        """Return (pf, expectancy, win_rate) from a flat trade list."""
        if not trades:
            return 0.0, 0.0, 0.0

        wins   = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]

        gross_wins   = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))

        pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        exp = sum(t.pnl for t in trades) / len(trades)
        wr  = len(wins) / len(trades)

        return pf, exp, wr
