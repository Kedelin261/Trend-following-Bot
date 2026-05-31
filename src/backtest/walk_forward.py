"""Walk-forward analysis — prevents curve fitting by separating in-sample
and out-of-sample backtests.

Default split: 70 % in-sample (training), 30 % out-of-sample (test).

The out-of-sample window has *warmup* candles prepended from the end of the
training period so the signal engine has enough history at the test start.

No broker code. No API calls. Pure data splitting.
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from src.data.models import Candle

if TYPE_CHECKING:
    from src.backtest.backtest_engine import BacktestEngine
    from src.backtest.models import BacktestResults

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Describes one train/test division of a candle series."""

    train_candles: List[Candle]
    test_candles:  List[Candle]   # includes warmup prefix from training window
    train_count:   int            # actual training candles
    test_count:    int            # actual new test candles (excl. warmup prefix)
    warmup_count:  int            # overlap bars prepended to test window


@dataclass
class WalkForwardResult:
    """Combined in-sample and out-of-sample backtest results.

    ``consistency_ratio`` measures how closely OOS performance tracks
    in-sample: > 0.5 = reasonably consistent; < 0.2 = likely overfit.
    """

    in_sample:         "BacktestResults"
    out_of_sample:     "BacktestResults"
    split:             WalkForwardSplit
    consistency_ratio: float   # oos_profit_factor / is_profit_factor


class WalkForwardAnalyzer:
    """Manages train/test splitting and coordinated backtest execution.

    Parameters
    ----------
    train_pct : fraction of data used for in-sample period (default 0.70)
    warmup    : minimum bars required before the signal engine can fire
                (must match BacktestEngine.min_warmup, default 210)
    """

    def __init__(
        self,
        train_pct: float = 0.70,
        warmup:    int   = 210,
    ) -> None:
        if not 0.1 <= train_pct <= 0.9:
            raise ValueError(
                f"train_pct must be between 0.1 and 0.9, got {train_pct}"
            )
        self.train_pct = train_pct
        self.warmup    = warmup

    def split(self, candles: List[Candle]) -> WalkForwardSplit:
        """Divide candles into training and test windows.

        The test window always begins with *warmup* bars from the end of the
        training window to ensure the signal engine has sufficient history.
        """
        n         = len(candles)
        split_idx = int(n * self.train_pct)

        train = candles[:split_idx]

        warmup_start = max(0, split_idx - self.warmup)
        test         = candles[warmup_start:]

        actual_test = n - split_idx

        logger.info(
            "walk_forward_split: total=%d split=%d train=%d test_new=%d warmup_prefix=%d",
            n,
            split_idx,
            split_idx,
            actual_test,
            split_idx - warmup_start,
        )

        return WalkForwardSplit(
            train_candles = train,
            test_candles  = test,
            train_count   = split_idx,
            test_count    = actual_test,
            warmup_count  = split_idx - warmup_start,
        )

    def analyze(
        self,
        engine:  "BacktestEngine",
        candles: List[Candle],
    ) -> WalkForwardResult:
        """Run in-sample and out-of-sample backtests and return combined results."""
        split = self.split(candles)

        logger.info(
            "walk_forward: running in-sample backtest (%d candles)...",
            len(split.train_candles),
        )
        in_sample = engine.run(split.train_candles)

        logger.info(
            "walk_forward: running out-of-sample backtest (%d candles)...",
            len(split.test_candles),
        )
        out_of_sample = engine.run(split.test_candles)

        # Consistency ratio — measures performance degradation OOS
        is_pf  = in_sample.profit_factor
        oos_pf = out_of_sample.profit_factor

        if is_pf > 0 and not math.isinf(is_pf):
            consistency = oos_pf / is_pf
        else:
            consistency = 0.0

        logger.info(
            "walk_forward_complete: IS pf=%.2f | OOS pf=%.2f | consistency=%.2f",
            is_pf,
            oos_pf,
            consistency,
        )

        return WalkForwardResult(
            in_sample         = in_sample,
            out_of_sample     = out_of_sample,
            split             = split,
            consistency_ratio = consistency,
        )
