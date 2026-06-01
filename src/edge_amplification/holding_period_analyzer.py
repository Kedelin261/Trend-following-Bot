"""Module 5 — Holding Period Analysis.

Answers: Which holding periods produce profits? Which produce losses?

Groups trades by holding_period (bars from entry to exit):
  0–5   bars  (very short)
  6–10  bars
  11–20 bars
  21+   bars  (long-duration holds)

Reports per-bucket: Trades, PF, Expectancy, Contribution %

Anti-curve-fitting: buckets with < 50 trades flagged INSUFFICIENT SAMPLE.

No execution code. No broker code. No live trading.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.backtest.models import BacktestTrade
from src.backtest.performance_metrics import profit_factor, expectancy

MIN_SAMPLE = 50

HOLDING_BANDS: List[Tuple[int, int, str]] = [
    (0,   5,  "0-5"),
    (6,  10,  "6-10"),
    (11, 20,  "11-20"),
    (21, 9999, "21+"),
]


@dataclass
class HoldingPeriodResult:
    """Metrics for one holding-period bucket."""
    band:             str
    bars_min:         int
    bars_max:         int
    trades:           int
    profit_factor:    float
    expectancy:       float
    net_pnl:          float
    contribution_pct: float   # net_pnl / total_net_pnl × 100
    rank:             int     = 0
    insufficient_sample: bool = False

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


def _band_for_bars(bars: int) -> str:
    for lo, hi, label in HOLDING_BANDS:
        if lo <= bars <= hi:
            return label
    return "21+"


class HoldingPeriodAnalyzer:
    """Groups trades by holding duration and evaluates per-bucket metrics.

    Parameters
    ----------
    trades : closed BacktestTrade list from full backtest
    """

    def __init__(self, trades: List[BacktestTrade]) -> None:
        self._trades = trades

    def analyze(self) -> List[HoldingPeriodResult]:
        """Return per-holding-period results ranked best → worst."""
        buckets: Dict[str, List[BacktestTrade]] = {
            label: [] for _, _, label in HOLDING_BANDS
        }

        for trade in self._trades:
            band = _band_for_bars(trade.holding_period)
            buckets[band].append(trade)

        total_net = sum(t.pnl for t in self._trades)

        results: List[HoldingPeriodResult] = []
        for lo, hi, label in HOLDING_BANDS:
            bucket_trades = buckets.get(label, [])
            if not bucket_trades:
                continue

            n       = len(bucket_trades)
            net     = sum(t.pnl for t in bucket_trades)
            pf_val  = profit_factor(bucket_trades)
            exp_val = expectancy(bucket_trades)
            contrib = (net / total_net * 100.0) if total_net != 0 else 0.0

            results.append(HoldingPeriodResult(
                band             = label,
                bars_min         = lo,
                bars_max         = hi,
                trades           = n,
                profit_factor    = pf_val,
                expectancy       = exp_val,
                net_pnl          = net,
                contribution_pct = contrib,
                insufficient_sample = n < MIN_SAMPLE,
            ))

        def _sort_key(r: HoldingPeriodResult):
            pf = r.profit_factor if not math.isinf(r.profit_factor) else 99.0
            return (not r.insufficient_sample, pf, r.expectancy)

        results.sort(key=_sort_key, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results
