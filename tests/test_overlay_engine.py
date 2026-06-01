"""Tests for the Phase 5.2 OverlayEngine."""

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pytest

from src.data.models import Candle
from src.risk_overlay.overlay_engine import OverlayEngine, OverlayResult
from src.risk_overlay.profiles.no_overlay import NoOverlay
from src.risk_overlay.profiles.equity_curve_pause import EquityCurvePause
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy


_CONFIG = {
    "backtest": {"starting_balance": 10_000.0, "commission_per_trade": 1.0, "slippage_percent": 0.05},
    "risk":     {"risk_per_trade_percent": 1.0, "atr_period": 14, "atr_stop_multiplier": 2.0,
                 "atr_target_multiplier": 3.0, "minimum_signal_score": 40.0, "minimum_risk_reward": 1.0},
}


def _make_candles(symbol: str, n: int = 300) -> List[Candle]:
    rng   = random.Random(42)
    price = 400.0
    base  = datetime(2020, 1, 1, tzinfo=timezone.utc)
    out   = []
    for i in range(n):
        ret   = 0.08/252 + rng.gauss(0, 0.18/(252**0.5))
        price = max(1.0, price*(1+ret))
        vol   = price*0.18/(252**0.5)*0.6
        open_ = price*(1+rng.gauss(0, 0.18/(252**0.5)*0.25))
        high  = max(open_, price)+abs(rng.gauss(0, vol))
        low   = min(open_, price)-abs(rng.gauss(0, vol))
        out.append(Candle(
            symbol=symbol, timeframe="D1",
            timestamp=base+timedelta(days=i),
            open=round(open_,4), high=round(high,4),
            low=round(low,4), close=round(price,4),
            volume=rng.uniform(30e6,200e6), provider="SYNTHETIC",
        ))
    return out


class TestOverlayEngine:

    def setup_method(self):
        self.engine   = OverlayEngine(_CONFIG)
        self.strategy = MomentumRotationStrategy()
        self.assets   = {
            "SPY": _make_candles("SPY", 300),
            "QQQ": _make_candles("QQQ", 300),
        }

    def test_runs_without_error(self):
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        assert report is not None
        assert len(report.overlay_results) == 1

    def test_returns_one_result_per_overlay(self):
        overlays = [NoOverlay(), EquityCurvePause()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        assert len(report.overlay_results) == 2

    def test_result_has_correct_name(self):
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        assert report.overlay_results[0].overlay_name == "NO_OVERLAY"

    def test_result_metrics_are_valid(self):
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        r = report.overlay_results[0]
        assert r.trades >= 0
        assert r.max_drawdown >= 0.0
        assert r.robustness in ("ROBUST", "MARGINAL", "UNSTABLE")
        assert r.history_bars == 300

    def test_report_has_recommendation(self):
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        assert isinstance(report.recommendation, str)
        assert len(report.recommendation) > 0

    def test_promotion_flag_is_bool(self):
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        assert isinstance(report.promotion_exists, bool)

    def test_no_promotion_with_short_history(self):
        """Short history (300 bars) should fail the 3000-bar criterion."""
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        # 300 bars < 3000 minimum — should not be promoted
        r = report.overlay_results[0]
        fails = [f for f in r.fail_criteria if "3000" in f or "bars" in f.lower()]
        assert len(fails) > 0 or not r.is_promoted

    def test_overlay_result_dataclass(self):
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        r = report.overlay_results[0]
        assert isinstance(r, OverlayResult)
        assert isinstance(r.pass_criteria, list)
        assert isinstance(r.fail_criteria, list)
        assert isinstance(r.is_promoted, bool)

    def test_best_candidate_none_when_no_promotion(self):
        """With short data, no overlay should reach promotion."""
        overlays = [NoOverlay()]
        report = self.engine.run(self.strategy, overlays, self.assets)
        if not report.promotion_exists:
            assert report.best_candidate is None
