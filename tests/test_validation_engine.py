"""Tests for validation_engine."""
import pytest
from src.edge_validation.validation_engine import ValidationEngine, ValidationReport
from src.signals.strategies.donchian_strategy import DonchianStrategy
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from tests.fixtures import uptrend_candles, downtrend_candles

CONFIG = {"backtest":{"starting_balance":10_000,"commission_per_trade":0.0,"slippage_percent":0.0},
          "risk":{"risk_per_trade_percent":1.0,"atr_period":3,"minimum_signal_score":50.0,
                  "minimum_risk_reward":1.0,"atr_stop_multiplier":2.0,"atr_target_multiplier":3.0}}

def _candles():
    return {"SPY": uptrend_candles(n=80, symbol="SPY"), "DIA": downtrend_candles(n=80, symbol="DIA")}

def test_returns_validation_report():
    engine = ValidationEngine(CONFIG)
    report = engine.run([DonchianStrategy(period=5)], _candles())
    assert isinstance(report, ValidationReport)

def test_results_count_matches_strategies():
    engine = ValidationEngine(CONFIG)
    report = engine.run([DonchianStrategy(period=5)], _candles())
    assert len(report.strategy_results) == 1

def test_recommendation_is_string():
    engine = ValidationEngine(CONFIG)
    report = engine.run([DonchianStrategy(period=5)], _candles())
    assert isinstance(report.recommendation, str)
    assert len(report.recommendation) > 0

def test_all_validation_components_present():
    engine = ValidationEngine(CONFIG)
    report = engine.run([DonchianStrategy(period=5)], _candles())
    r = report.strategy_results[0]
    assert r.history_slices is not None
    assert r.asset_expansion is not None
    assert r.scalability is not None
    assert r.robustness is not None
    assert r.survivability is not None
    assert r.promotion is not None
