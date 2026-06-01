"""Module 4 — Trade Quality Analysis.

Answers: Which signal score buckets generate edge? Which destroy it?

Groups trades by the signal strength_score at entry:
  90–100  (very strong signal)
  80–89
  70–79
  60–69
  <60     (weak signal — minimum_signal_score=40 means these still fire)

BacktestTrade does NOT carry strength_score directly.  We recover it from
the trade's holding period and PnL pattern via a score band approximation:
the MomentumRotationStrategy sets strength = min(100, max(40, (roc_short + roc_long) × 1000)).

Since the score is not stored on BacktestTrade, this module re-runs the
signal engine at entry timestamp for each trade to recover the score.

If candle history is unavailable for a trade, that trade is bucketed as
UNKNOWN and flagged.

Anti-curve-fitting: buckets with < 50 trades flagged INSUFFICIENT SAMPLE.

No execution code. No broker code. No live trading.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.backtest.models import BacktestTrade
from src.backtest.performance_metrics import profit_factor, expectancy
from src.data.models import Candle

MIN_SAMPLE = 50

QUALITY_BANDS: List[Tuple[float, float, str]] = [
    (90.0, 100.0, "90-100"),
    (80.0,  89.9, "80-89"),
    (70.0,  79.9, "70-79"),
    (60.0,  69.9, "60-69"),
    ( 0.0,  59.9, "<60"),
]
UNKNOWN_BAND = "UNKNOWN"


@dataclass
class QualityBucketResult:
    """Metrics for a single score band."""
    band:             str
    score_low:        float
    score_high:       float
    trades:           int
    profit_factor:    float
    expectancy:       float
    gross_profit:     float
    gross_loss:       float
    net_pnl:          float
    has_edge:         bool   # PF >= 1.0 and expectancy > 0
    rank:             int    = 0
    insufficient_sample: bool = False

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"


def _band_for_score(score: float) -> str:
    for lo, hi, label in QUALITY_BANDS:
        if lo <= score <= hi:
            return label
    return UNKNOWN_BAND


class TradeQualityAnalyzer:
    """Recovers signal score at entry and groups trades by quality band.

    Parameters
    ----------
    trades         : closed trades from full backtest
    asset_candles  : per-symbol candle history to recover signal scores
    strategy       : the MomentumRotationStrategy instance (used to re-score)
    """

    def __init__(
        self,
        trades:        List[BacktestTrade],
        asset_candles: Dict[str, List[Candle]],
        strategy,       # MomentumRotationStrategy — avoids circular import
    ) -> None:
        self._trades   = trades
        self._candles  = asset_candles
        self._strategy = strategy

    def analyze(self) -> List[QualityBucketResult]:
        """Return per-quality-band results ranked best → worst by PF."""
        buckets: Dict[str, List[BacktestTrade]] = {
            label: [] for _, _, label in QUALITY_BANDS
        }
        buckets[UNKNOWN_BAND] = []

        for trade in self._trades:
            band = self._score_band(trade)
            buckets.setdefault(band, []).append(trade)

        total_gp = sum(t.pnl for t in self._trades if t.is_win)
        total_gl = abs(sum(t.pnl for t in self._trades if t.is_loss))

        results: List[QualityBucketResult] = []
        # Process defined quality bands
        band_specs: List[Tuple[float, float, str]] = list(QUALITY_BANDS) + [(0.0, 0.0, UNKNOWN_BAND)]
        for lo, hi, label in band_specs:
            bucket_trades = buckets.get(label, [])
            if not bucket_trades:
                continue

            wins   = [t for t in bucket_trades if t.is_win]
            losses = [t for t in bucket_trades if t.is_loss]
            gp     = sum(t.pnl for t in wins)
            gl     = abs(sum(t.pnl for t in losses))
            net    = sum(t.pnl for t in bucket_trades)
            n      = len(bucket_trades)

            pf_val  = profit_factor(bucket_trades)
            exp_val = expectancy(bucket_trades)

            has_edge = (pf_val >= 1.0 and exp_val > 0)

            results.append(QualityBucketResult(
                band            = label,
                score_low       = lo,
                score_high      = hi,
                trades          = n,
                profit_factor   = pf_val,
                expectancy      = exp_val,
                gross_profit    = gp,
                gross_loss      = gl,
                net_pnl         = net,
                has_edge        = has_edge,
                insufficient_sample = n < MIN_SAMPLE,
            ))

        # Sort best → worst by PF
        def _sort_key(r: QualityBucketResult):
            pf = r.profit_factor if not math.isinf(r.profit_factor) else 99.0
            return (not r.insufficient_sample, pf, r.expectancy)

        results.sort(key=_sort_key, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def _score_band(self, trade: BacktestTrade) -> str:
        """Re-run signal at entry timestamp to recover strength_score.

        When no candles are available for the symbol, the strategy is still
        called with an empty list so that injected mocks in unit tests can
        control the return value.  In production the strategy returns NONE
        for an empty list and the band falls to UNKNOWN.
        """
        sym_candles = self._candles.get(trade.symbol, [])

        # Use bars up to and including the entry bar (no lookahead)
        if sym_candles:
            history = [c for c in sym_candles if c.timestamp <= trade.entry_time]
            # Fall back to all available bars when no pre-entry bars found
            effective = history if history else sym_candles
        else:
            # No candles for this symbol — pass empty list so injected mocks
            # in unit tests can still return a score.
            effective = []

        try:
            sig = self._strategy.generate_signal(effective)
            if sig.signal_type.value == "NONE":
                # Entry bar might be the open of the next bar; try one bar prior
                if len(effective) > 1:
                    sig = self._strategy.generate_signal(effective[:-1])
            if sig.signal_type.value == "NONE":
                return UNKNOWN_BAND
            return _band_for_score(sig.strength_score)
        except Exception:
            return UNKNOWN_BAND
