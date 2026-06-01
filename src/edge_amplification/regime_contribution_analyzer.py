"""Module 2 — Regime Contribution Analysis.

Answers: Which market regimes create edge? Which destroy it?

Classifies each trade's entry bar into a macro regime:
  EXPANSION / RECOVERY / CONTRACTION / CRISIS / UNKNOWN

Uses MacroRegimeDetector.classify_at_timestamp() to determine the regime
active when each trade was entered.  This avoids lookahead — only bars up
to and including the entry timestamp are used.

Reports per-regime: Trades, PF, Expectancy, Profit%, Loss%

Anti-curve-fitting: regimes with < 50 trades are flagged INSUFFICIENT SAMPLE.

No execution code. No broker code. No live trading.
No entry logic modifications.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest.performance_metrics import profit_factor, expectancy
from src.data.models import Candle
from src.regime.macro_regime_detector import MacroRegime, MacroRegimeDetector

MIN_SAMPLE = 50


@dataclass
class RegimeContribution:
    """Metrics for a single macro regime bucket."""
    regime:           str
    trades:           int
    profit_factor:    float
    expectancy:       float
    gross_profit:     float
    gross_loss:       float
    profit_pct:       float   # gross_profit / total_gross_profit × 100
    loss_pct:         float   # gross_loss   / total_gross_loss   × 100
    rank:             int     = 0
    insufficient_sample: bool = False

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


class RegimeContributionAnalyzer:
    """Classifies trades by the macro regime at entry and computes per-regime metrics.

    Parameters
    ----------
    trades       : closed trades from multi-asset backtest
    asset_candles: per-symbol candle history (used for regime classification)
    detector     : optional MacroRegimeDetector (default constructed if None)
    """

    def __init__(
        self,
        trades:        List[BacktestTrade],
        asset_candles: Dict[str, List[Candle]],
        detector:      Optional[MacroRegimeDetector] = None,
    ) -> None:
        self._trades       = trades
        self._candles      = asset_candles
        self._detector     = detector or MacroRegimeDetector()

    def analyze(self) -> List[RegimeContribution]:
        """Return per-regime contributions ranked best → worst by PF."""
        buckets: Dict[str, List[BacktestTrade]] = {r.value: [] for r in MacroRegime}

        for trade in self._trades:
            sym_candles = self._candles.get(trade.symbol, [])
            regime = self._classifier(sym_candles, trade)
            buckets[regime.value].append(trade)

        total_gp = sum(t.pnl for t in self._trades if t.is_win)
        total_gl = abs(sum(t.pnl for t in self._trades if t.is_loss))

        results: List[RegimeContribution] = []
        for regime_name, bucket_trades in buckets.items():
            if not bucket_trades:
                continue

            wins   = [t for t in bucket_trades if t.is_win]
            losses = [t for t in bucket_trades if t.is_loss]
            gp     = sum(t.pnl for t in wins)
            gl     = abs(sum(t.pnl for t in losses))
            n      = len(bucket_trades)

            pf_val  = profit_factor(bucket_trades)
            exp_val = expectancy(bucket_trades)

            pp = (gp / total_gp * 100.0) if total_gp > 0 else 0.0
            lp = (gl / total_gl * 100.0) if total_gl > 0 else 0.0

            results.append(RegimeContribution(
                regime            = regime_name,
                trades            = n,
                profit_factor     = pf_val,
                expectancy        = exp_val,
                gross_profit      = gp,
                gross_loss        = gl,
                profit_pct        = pp,
                loss_pct          = lp,
                insufficient_sample = n < MIN_SAMPLE,
            ))

        def _sort_key(r: RegimeContribution):
            pf = r.profit_factor if not math.isinf(r.profit_factor) else 99.0
            return (not r.insufficient_sample, pf, r.expectancy)

        results.sort(key=_sort_key, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def _classifier(
        self,
        candles: List[Candle],
        trade:   BacktestTrade,
    ) -> MacroRegime:
        """Return the regime active at trade entry (no lookahead)."""
        if not candles:
            return MacroRegime.UNKNOWN
        history = [c for c in candles if c.timestamp <= trade.entry_time]
        if not history:
            return MacroRegime.UNKNOWN
        return self._detector.classify(history)
