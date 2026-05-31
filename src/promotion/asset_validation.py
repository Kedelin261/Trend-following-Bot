"""Asset validation — evaluates one candidate asset at a time.

Candidates: XLV, SCHD, VTI

A candidate is approved only when adding it:
  1. Increases total portfolio trade count
  2. Does NOT reduce PF below 1.50
  3. Does NOT make expectancy negative
  4. Does NOT increase max DD above 15%

Automatic rejection is better than false approval.

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

CANDIDATE_SYMBOLS = ["XLV", "SCHD", "VTI"]
QUALITY_MIN_PF    = 1.50
QUALITY_MIN_EXP   = 0.0
QUALITY_MAX_DD    = 15.0


@dataclass
class AssetValidationResult:
    """Result of validating one expansion candidate asset."""

    candidate_symbol:      str
    baseline_trades:       int
    with_candidate_trades: int
    trade_increase:        int
    baseline_pf:           float
    with_candidate_pf:     float
    candidate_pf:          float       # candidate alone
    candidate_exp:         float       # candidate alone
    candidate_dd:          float       # candidate alone
    quality_maintained:    bool
    adds_trades:           bool
    approved:              bool
    note:                  str = ""


class AssetValidator:
    """Tests whether adding one candidate asset improves the portfolio.

    The baseline portfolio (SPY, VOO, DIA) is run first, then the
    candidate is added to measure the incremental impact.
    """

    def __init__(
        self,
        config:  dict,
        profile: StrategyProfile = None,
    ) -> None:
        self._config  = config
        self._profile = profile or BEST_DENSITY_PROFILE

    def validate(
        self,
        baseline_candles:   Dict[str, List[Candle]],
        candidate_symbol:   str,
        candidate_candles:  List[Candle],
    ) -> AssetValidationResult:
        """Validate whether the candidate asset should be added."""
        logger.info(
            "asset_validation: testing candidate=%s against %d baseline assets",
            candidate_symbol, len(baseline_candles),
        )

        # Run baseline
        base_bt = self._run_portfolio(baseline_candles)
        base_total = sum(bt.total_trades for bt in base_bt.values())
        base_pf, base_exp = self._aggregate(base_bt)

        # Run candidate alone
        cand_bt  = self._run_one(candidate_symbol, candidate_candles)
        cand_pf  = cand_bt.profit_factor if not math.isinf(cand_bt.profit_factor) else 99.0
        cand_exp = cand_bt.expectancy
        cand_dd  = cand_bt.max_drawdown

        # Run combined
        combined_candles = {**baseline_candles, candidate_symbol: candidate_candles}
        combined_bt      = self._run_portfolio(combined_candles)
        combined_total   = sum(bt.total_trades for bt in combined_bt.values())
        combined_pf, combined_exp = self._aggregate(combined_bt)
        combined_dd      = max(bt.max_drawdown for bt in combined_bt.values())

        adds_trades = combined_total > base_total
        quality_ok  = (
            combined_pf >= QUALITY_MIN_PF
            and combined_exp > QUALITY_MIN_EXP
            and combined_dd <= QUALITY_MAX_DD
        )

        approved = adds_trades and quality_ok
        note = (
            "" if approved else (
                f"✗ adds_trades={adds_trades} quality_ok={quality_ok}"
                + (f" pf={combined_pf:.2f}" if not quality_ok else "")
            )
        )

        logger.info(
            "asset_validation: %s approved=%s trades %d→%d",
            candidate_symbol, approved, base_total, combined_total,
        )

        return AssetValidationResult(
            candidate_symbol      = candidate_symbol,
            baseline_trades       = base_total,
            with_candidate_trades = combined_total,
            trade_increase        = combined_total - base_total,
            baseline_pf           = base_pf,
            with_candidate_pf     = combined_pf,
            candidate_pf          = cand_pf,
            candidate_exp         = cand_exp,
            candidate_dd          = cand_dd,
            quality_maintained    = quality_ok,
            adds_trades           = adds_trades,
            approved              = approved,
            note                  = note,
        )

    def validate_all(
        self,
        baseline_candles:    Dict[str, List[Candle]],
        candidate_candles:   Dict[str, List[Candle]],
    ) -> List[AssetValidationResult]:
        """Validate all candidates and return results sorted by trade_increase."""
        results = []
        for sym, candles in candidate_candles.items():
            if candles and sym not in baseline_candles:
                results.append(self.validate(baseline_candles, sym, candles))
        return sorted(results, key=lambda r: r.trade_increase, reverse=True)

    def approved_candidates(
        self, results: List[AssetValidationResult]
    ) -> List[str]:
        return [r.candidate_symbol for r in results if r.approved]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_one(self, symbol: str, candles: List[Candle]) -> BacktestResults:
        engine = self._profile.build_backtest_engine(self._config)
        return engine.run(candles)

    def _run_portfolio(
        self, asset_candles: Dict[str, List[Candle]]
    ) -> Dict[str, BacktestResults]:
        return {
            sym: self._run_one(sym, c)
            for sym, c in asset_candles.items()
            if c
        }

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
