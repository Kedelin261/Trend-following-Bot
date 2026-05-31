"""Filter backtester — runs strategy, labels trades, applies filter profiles.

Workflow:
  1. Run the locked strategy backtest on all assets
  2. Label every closed trade with 5 regime dimensions (Phase 4.10)
  3. Apply a FilterProfile via RegimeFilterEngine
  4. Compute performance metrics on the filtered trade set

Two modes:
  a) full_backtest=True  — runs Phase 4.10 labelling pipeline
  b) from_labelled       — accepts pre-labelled trades (fast, avoids repeat backtest)

No broker code.  No API calls.  Candle data only.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest import performance_metrics as pm
from src.data.models import Candle
from src.regime.drawdown_environment_detector import DrawdownEnvironmentDetector
from src.regime.macro_regime_detector import MacroRegimeDetector
from src.regime.market_regime_classifier import MarketRegimeClassifier
from src.regime.regime_performance_analyzer import LabelledTrade
from src.regime.trend_regime_detector import TrendRegimeDetector
from src.regime.volatility_regime_detector import VolatilityRegimeDetector
from src.regime_filter.filter_profiles import FilterProfile, NO_FILTER
from src.regime_filter.regime_filter_engine import RegimeFilterEngine
from src.refinement.strategy_v2 import StrategyProfile
from src.timeframe.timeframe_profile import BEST_DENSITY_PROFILE

logger = logging.getLogger(__name__)


@dataclass
class FilterBacktestResult:
    """Performance of the strategy after applying one FilterProfile."""

    filter_profile:       FilterProfile
    total_trades_before:  int
    total_trades_after:   int
    trades_removed:       int
    win_rate:             float
    profit_factor:        float
    expectancy:           float
    max_drawdown:         float
    net_pnl:              float
    note:                 str = ""

    @property
    def retention_pct(self) -> float:
        return (self.total_trades_after / self.total_trades_before * 100
                if self.total_trades_before > 0 else 0.0)

    @property
    def pf_str(self) -> str:
        return "∞" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"

    @property
    def meets_quality(self) -> bool:
        return (
            (self.profit_factor >= 1.50 or math.isinf(self.profit_factor))
            and self.expectancy > 0
            and self.max_drawdown < 15.0
        )


class FilterBacktester:
    """Runs the strategy, labels trades, and evaluates filter profiles.

    Parameters
    ----------
    config           : settings dict (backtest, risk sections)
    strategy_profile : strategy to use (default = BEST_DENSITY_PROFILE)
    """

    def __init__(
        self,
        config:           dict,
        strategy_profile: StrategyProfile = None,
    ) -> None:
        self._config   = config
        self._profile  = strategy_profile or BEST_DENSITY_PROFILE
        self._engine   = RegimeFilterEngine()

        # Regime detectors (Phase 4.10)
        self._market_clf = MarketRegimeClassifier()
        self._trend_det  = TrendRegimeDetector()
        self._vol_det    = VolatilityRegimeDetector()
        self._dd_det     = DrawdownEnvironmentDetector()
        self._macro_det  = MacroRegimeDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest_with_filter(
        self,
        asset_candles:    Dict[str, List[Candle]],
        filter_profile:   FilterProfile,
    ) -> FilterBacktestResult:
        """Full pipeline: run backtest → label → filter → metrics."""
        labelled = self._run_and_label(asset_candles)
        return self.apply_filter(labelled, filter_profile)

    def backtest_all_profiles(
        self,
        asset_candles: Dict[str, List[Candle]],
        profiles:      List[FilterProfile],
    ) -> List[FilterBacktestResult]:
        """Run once, then apply every profile to the same labelled trade list.

        Significantly faster than calling backtest_with_filter() per profile.
        """
        labelled = self._run_and_label(asset_candles)
        return [self.apply_filter(labelled, p) for p in profiles]

    def apply_filter(
        self,
        labelled_trades: List[LabelledTrade],
        profile:         FilterProfile,
    ) -> FilterBacktestResult:
        """Apply *profile* to a pre-labelled trade list and compute metrics."""
        passing    = self._engine.apply_to_trades(labelled_trades, profile)
        all_trades = [lt.trade for lt in labelled_trades]
        filt_trades = [lt.trade for lt in passing]

        pf   = self._pf(filt_trades)
        exp  = pm.expectancy(filt_trades)
        wr   = pm.win_rate(filt_trades)
        dd   = self._max_dd(filt_trades, all_trades)
        pnl  = sum(t.pnl for t in filt_trades)

        note = (
            "" if filt_trades else
            "⚠ All trades removed — filter too restrictive"
        )

        logger.info(
            "filter_backtest: profile=%s trades=%d→%d pf=%.2f exp=%.2f",
            profile.name, len(labelled_trades), len(passing),
            pf if not math.isinf(pf) else 99.0, exp,
        )

        return FilterBacktestResult(
            filter_profile      = profile,
            total_trades_before = len(labelled_trades),
            total_trades_after  = len(passing),
            trades_removed      = len(labelled_trades) - len(passing),
            win_rate            = wr,
            profit_factor       = pf,
            expectancy          = exp,
            max_drawdown        = dd,
            net_pnl             = pnl,
            note                = note,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_and_label(
        self, asset_candles: Dict[str, List[Candle]]
    ) -> List[LabelledTrade]:
        labelled: List[LabelledTrade] = []
        for sym, candles in asset_candles.items():
            if not candles:
                continue
            bt = self._profile.build_backtest_engine(self._config).run(candles)
            for trade in bt.trades:
                ts = trade.entry_time
                labelled.append(LabelledTrade(
                    trade             = trade,
                    market_regime     = self._market_clf.classify_at_timestamp(candles, ts),
                    trend_regime      = self._trend_det.classify_at_timestamp(candles, ts),
                    volatility_regime = self._vol_det.classify_at_timestamp(candles, ts),
                    drawdown_env      = self._dd_det.classify_at_timestamp(candles, ts),
                    macro_regime      = self._macro_det.classify_at_timestamp(candles, ts),
                ))
        return labelled

    @staticmethod
    def _pf(trades) -> float:
        if not trades:
            return 0.0
        wins   = [t for t in trades if t.is_win]
        losses = [t for t in trades if t.is_loss]
        gross_wins   = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        return gross_wins / gross_losses if gross_losses > 0 else float("inf")

    @staticmethod
    def _max_dd(filtered_trades, all_trades) -> float:
        """Approximate max DD using filtered trades' PnL sequence."""
        if not filtered_trades:
            return 0.0
        equity = 10_000.0
        peak   = equity
        max_dd = 0.0
        for t in filtered_trades:
            equity += t.pnl
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        return max_dd
