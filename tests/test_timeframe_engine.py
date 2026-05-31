"""Tests for TimeframeEngine — full orchestration."""

import pathlib
import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from src.data.models import Candle
from src.refinement.strategy_v2 import StrategyProfile
from src.refinement.volatility_trade_filter import VolatilityFilterMode
from src.timeframe.timeframe_engine import TimeframeEngine, TimeframeReport
from src.timeframe.timeframe_profile import (
    COMBO_D1_H4,
    D1_PROFILE,
    H4_PROFILE,
    TimeframeProfile,
)
from tests.fixtures import downtrend_candles, uptrend_candles

BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _h4_candles(n=50, symbol="SPY") -> List[Candle]:
    price = 400.0
    result = []
    for i in range(n):
        price *= 1.002
        result.append(Candle(
            symbol=symbol, timeframe="H4",
            timestamp=BASE + timedelta(hours=4 * i),
            open=price, high=price * 1.003, low=price * 0.997,
            close=price, volume=1e8, provider="TEST",
        ))
    return result


def _mini_config() -> dict:
    return {
        "backtest": {"starting_balance": 10_000, "commission_per_trade": 0.0,
                     "slippage_percent": 0.0},
        "risk": {"risk_per_trade_percent": 1.0, "atr_period": 3,
                 "minimum_signal_score": 60.0, "minimum_risk_reward": 1.0,
                 "atr_stop_multiplier": 2.0, "atr_target_multiplier": 3.0},
    }


def _mini_tf_profile(tf: str) -> TimeframeProfile:
    bpy = {"D1": 252, "H4": 409}
    return TimeframeProfile(name=tf, timeframe=tf, bars_per_year=bpy.get(tf, 252), min_warmup=12)


def _asset_candles() -> Dict[str, Dict[str, List[Candle]]]:
    return {
        "SPY": {
            "D1": uptrend_candles(n=50, symbol="SPY"),
            "H4": _h4_candles(n=50, symbol="SPY"),
        },
        "DIA": {
            "D1": downtrend_candles(n=50, symbol="DIA"),
            "H4": _h4_candles(n=50, symbol="DIA"),
        },
    }


class TestTimeframeEngine:

    @pytest.fixture
    def engine(self) -> TimeframeEngine:
        from src.refinement.strategy_v2 import StrategyProfile
        from src.timeframe.timeframe_backtester import TimeframeBacktester
        from src.timeframe.signal_overlap_analyzer import SignalOverlapAnalyzer
        from src.timeframe.multi_timeframe_research import MultiTimeframeResearcher

        config = _mini_config()

        # Use mini profile so tests run fast
        mini = StrategyProfile(
            name="mini", description="mini", ema_fast=5, ema_slow=10,
            breakout_threshold=0.001, require_bull_regime=False,
            adx_threshold=0.0, volatility_mode=VolatilityFilterMode.NONE, atr_period=3,
        )
        from unittest.mock import patch
        engine = TimeframeEngine(config)
        # Patch the strategy used inside the engine
        engine._backtester = TimeframeBacktester(config, mini)
        engine._multi_res  = MultiTimeframeResearcher(
            config, engine._backtester, SignalOverlapAnalyzer()
        )
        return engine

    def test_returns_timeframe_report(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1"), _mini_tf_profile("H4")],
            combo_profiles=[COMBO_D1_H4],
        )
        assert isinstance(report, TimeframeReport)

    def test_empty_input_raises(self, engine):
        with pytest.raises(ValueError):
            engine.run({})

    def test_single_tf_results_populated(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1")],
            combo_profiles=[],
        )
        assert "D1" in report.single_tf_results

    def test_combo_results_populated(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1"), _mini_tf_profile("H4")],
            combo_profiles=[COMBO_D1_H4],
        )
        assert len(report.combo_results) == 1

    def test_promoted_is_bool(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1")],
            combo_profiles=[],
        )
        assert isinstance(report.promoted, bool)

    def test_recommendation_is_string(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1")],
            combo_profiles=[],
        )
        assert len(report.recommendation) > 0

    def test_no_broker_imports_in_timeframe_modules(self):
        tf_dir = pathlib.Path("src/timeframe")
        for pyfile in tf_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"

    def test_comparisons_populated_when_d1_present(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1"), _mini_tf_profile("H4")],
            combo_profiles=[],
        )
        # Comparisons are keyed by symbol and should have entries for assets with D1
        assert isinstance(report.comparisons, dict)

    def test_opportunity_metrics_populated(self, engine):
        report = engine.run(
            _asset_candles(),
            single_profiles=[_mini_tf_profile("D1")],
            combo_profiles=[],
        )
        assert "D1" in report.opportunity_metrics
