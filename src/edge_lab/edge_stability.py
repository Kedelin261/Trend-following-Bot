"""Edge stability — tests robustness across early / middle / recent windows."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.backtest.models import BacktestResults
from src.data.models import Candle
from src.edge_lab.strategy_interface import StrategyInterface

WINDOW_FRACTION = 0.60
QUALITY_MIN_PF  = 1.20   # lower bar for stability — we want positive direction
QUALITY_MIN_EXP = 0.0


@dataclass
class StabilityWindow:
    name:    str
    trades:  int
    pf:      float
    exp:     float
    passes:  bool


class EdgeStabilityAnalyzer:
    """Tests one strategy across three historical windows."""

    def analyze(
        self,
        strategy:      StrategyInterface,
        asset_candles: Dict[str, List[Candle]],
        build_engine_fn,  # callable: (strategy) -> BacktestEngine
    ) -> tuple:
        """Returns (rating, windows: List[StabilityWindow])."""
        min_len = min(len(c) for c in asset_candles.values()) if asset_candles else 0
        w = max(1, int(min_len * WINDOW_FRACTION))

        windows_def = {
            "early":  {s: c[:w]            for s, c in asset_candles.items()},
            "middle": {s: c[int(min_len * 0.2):int(min_len * 0.2) + w]
                       for s, c in asset_candles.items()},
            "recent": {s: c[min_len - w:]  for s, c in asset_candles.items()},
        }

        window_results: List[StabilityWindow] = []
        for win_name, win_candles in windows_def.items():
            trades_all, pf_val, exp_val = self._run(strategy, win_candles, build_engine_fn)
            passes = (pf_val >= QUALITY_MIN_PF and exp_val > QUALITY_MIN_EXP and trades_all > 0)
            window_results.append(StabilityWindow(win_name, trades_all, pf_val, exp_val, passes))

        passing = sum(1 for w in window_results if w.passes)
        rating  = "ROBUST" if passing == 3 else ("MARGINAL" if passing == 2 else "UNSTABLE")

        return rating, window_results

    @staticmethod
    def _run(
        strategy:       StrategyInterface,
        asset_candles:  Dict[str, List[Candle]],
        build_engine_fn,
    ) -> tuple:
        all_trades = []
        for sym, candles in asset_candles.items():
            if candles:
                engine = build_engine_fn(strategy)
                bt = engine.run(candles)
                all_trades.extend(bt.trades)

        total = len(all_trades)
        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if t.is_loss]
        gw = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses))
        pf  = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
        exp = sum(t.pnl for t in all_trades) / total if total > 0 else 0.0
        return total, pf, exp
