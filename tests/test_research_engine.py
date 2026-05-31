"""Tests for ResearchEngine — full orchestration."""

import pathlib
import pytest
from typing import Dict, List

from src.data.models import Candle
from src.research.research_engine import ResearchEngine
from src.research.strategy_analyzer import ResearchReport
from tests.fixtures import downtrend_candles, flat_candles, uptrend_candles


@pytest.fixture
def config() -> dict:
    return {
        "backtest": {
            "starting_balance": 10_000,
            "commission_per_trade": 0.0,
            "slippage_percent": 0.0,
            "minimum_trades_required": 3,
            "min_warmup": 12,
        },
        "risk": {
            "risk_per_trade_percent": 1.0,
            "atr_period": 3,
            "atr_stop_multiplier": 2.0,
            "atr_target_multiplier": 3.0,
            "minimum_signal_score": 60.0,
            "minimum_risk_reward": 1.0,
        },
    }


@pytest.fixture
def engine(config) -> ResearchEngine:
    return ResearchEngine(config, min_warmup=12)


def _asset_candles() -> Dict[str, List[Candle]]:
    return {
        "SPY": uptrend_candles(n=50, symbol="SPY"),
        "QQQ": downtrend_candles(n=50, symbol="QQQ"),
    }


class TestResearchEngineRun:

    def test_returns_research_report(self, engine):
        report = engine.run(
            _asset_candles(),
            run_sweep=False, run_breakout=False,
        )
        assert isinstance(report, ResearchReport)

    def test_empty_input_raises(self, engine):
        with pytest.raises(ValueError):
            engine.run({})

    def test_asset_results_populated(self, engine):
        report = engine.run(_asset_candles(), run_sweep=False, run_breakout=False)
        assert len(report.asset_results) == 2

    def test_edge_confirmed_is_bool(self, engine):
        report = engine.run(_asset_candles(), run_sweep=False, run_breakout=False)
        assert isinstance(report.edge_confirmed, bool)

    def test_warnings_is_list(self, engine):
        report = engine.run(_asset_candles(), run_sweep=False, run_breakout=False)
        assert isinstance(report.warnings, list)

    def test_benchmark_results_populated(self, engine):
        report = engine.run(_asset_candles(), run_sweep=False, run_breakout=False)
        assert isinstance(report.benchmark_results, dict)

    def test_single_asset_runs(self, engine):
        report = engine.run(
            {"SPY": uptrend_candles(n=50, symbol="SPY")},
            run_sweep=False, run_breakout=False,
        )
        assert len(report.asset_results) == 1

    def test_skip_sweep_when_flag_false(self, engine):
        report = engine.run(
            _asset_candles(), run_sweep=False, run_breakout=False
        )
        assert report.sweep_results == []

    def test_skip_breakout_when_flag_false(self, engine):
        report = engine.run(
            _asset_candles(), run_sweep=False, run_breakout=False
        )
        assert report.breakout_results == []

    def test_sweep_populated_when_enabled(self, engine):
        report = engine.run(
            _asset_candles(), run_sweep=True, run_breakout=False
        )
        assert len(report.sweep_results) > 0

    def test_no_broker_imports_in_research_modules(self):
        research_dir = pathlib.Path("src/research")
        for pyfile in research_dir.glob("*.py"):
            text = pyfile.read_text()
            assert "ib_insync"   not in text, f"{pyfile} imports ib_insync"
            assert "MetaTrader"  not in text, f"{pyfile} imports MetaTrader"

    def test_expected_metrics_nonnegative_drawdown(self, engine):
        report = engine.run(_asset_candles(), run_sweep=False, run_breakout=False)
        assert report.expected_max_drawdown >= 0.0
