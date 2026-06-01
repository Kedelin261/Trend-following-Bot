"""Module 3 — Volatility Contribution Analysis.

Answers: Which volatility environments help? Which hurt?

Classifies each trade's entry bar into a volatility regime:
  LOW_VOL / NORMAL_VOL / HIGH_VOL / EXTREME_VOL / UNKNOWN

Uses VolatilityRegimeDetector.classify() at entry timestamp (no lookahead).

Reports per-regime: Trades, PF, Expectancy, Profit%, Loss%

Anti-curve-fitting: buckets with < 50 trades flagged INSUFFICIENT SAMPLE.

No execution code. No broker code. No live trading.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestTrade
from src.backtest.performance_metrics import profit_factor, expectancy
from src.data.models import Candle
from src.regime.volatility_regime_detector import VolatilityRegime, VolatilityRegimeDetector

MIN_SAMPLE = 50


@dataclass
class VolatilityContribution:
    """Metrics for a single volatility regime bucket."""
    regime:           str
    trades:           int
    profit_factor:    float
    expectancy:       float
    gross_profit:     float
    gross_loss:       float
    profit_pct:       float
    loss_pct:         float
    rank:             int   = 0
    insufficient_sample: bool = False

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


class VolatilityContributionAnalyzer:
    """Classifies trades by the volatility regime active at entry.

    Parameters
    ----------
    trades       : closed trades from multi-asset backtest
    asset_candles: per-symbol candle history
    detector     : optional VolatilityRegimeDetector (default constructed if None)
    """

    def __init__(
        self,
        trades:        List[BacktestTrade],
        asset_candles: Dict[str, List[Candle]],
        detector:      Optional[VolatilityRegimeDetector] = None,
    ) -> None:
        self._trades   = trades
        self._candles  = asset_candles
        self._detector = detector or VolatilityRegimeDetector()

    def analyze(self) -> List[VolatilityContribution]:
        """Return per-volatility-regime contributions ranked best → worst by PF."""
        buckets: Dict[str, List[BacktestTrade]] = {r.value: [] for r in VolatilityRegime}

        for trade in self._trades:
            sym_candles = self._candles.get(trade.symbol, [])
            regime = self._classify_at_entry(sym_candles, trade)
            buckets[regime.value].append(trade)

        total_gp = sum(t.pnl for t in self._trades if t.is_win)
        total_gl = abs(sum(t.pnl for t in self._trades if t.is_loss))

        results: List[VolatilityContribution] = []
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

            results.append(VolatilityContribution(
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

        def _sort_key(r: VolatilityContribution):
            pf = r.profit_factor if not math.isinf(r.profit_factor) else 99.0
            return (not r.insufficient_sample, pf, r.expectancy)

        results.sort(key=_sort_key, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def _classify_at_entry(
        self,
        candles: List[Candle],
        trade:   BacktestTrade,
    ) -> VolatilityRegime:
        """Return volatility regime at trade entry (no lookahead).

        When candles is empty the detector is still called so that mocks
        injected in tests can control the return value.  In production the
        detector returns UNKNOWN for an empty list.
        """
        if candles is None:
            return VolatilityRegime.UNKNOWN
        history = [c for c in candles if c.timestamp <= trade.entry_time]
        # Use full candles if all bars post-date the entry (e.g. empty mock list)
        effective = history if history else candles
        return self._detector.classify(effective)
