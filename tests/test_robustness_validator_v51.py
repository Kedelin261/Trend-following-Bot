"""Tests for robustness_validator (Phase 5.1)."""
from src.edge_validation.robustness_validator import RobustnessValidator
from src.signals.strategies.donchian_strategy import DonchianStrategy
from tests.fixtures import uptrend_candles

CONFIG = {"backtest":{"starting_balance":10_000,"commission_per_trade":0.0,"slippage_percent":0.0},
          "risk":{"risk_per_trade_percent":1.0,"atr_period":3,"minimum_signal_score":50.0,
                  "minimum_risk_reward":1.0,"atr_stop_multiplier":2.0,"atr_target_multiplier":3.0}}

def test_returns_result():
    v = RobustnessValidator(CONFIG)
    result = v.validate(DonchianStrategy(period=5), {"SPY": uptrend_candles(n=100)})
    assert result.rating in ("ROBUST","MARGINAL","UNSTABLE")

def test_three_windows():
    v = RobustnessValidator(CONFIG)
    result = v.validate(DonchianStrategy(period=5), {"SPY": uptrend_candles(n=100)})
    assert len(result.windows) == 3

def test_passes_property():
    v = RobustnessValidator(CONFIG)
    result = v.validate(DonchianStrategy(period=5), {"SPY": uptrend_candles(n=100)})
    assert isinstance(result.passes, bool)
