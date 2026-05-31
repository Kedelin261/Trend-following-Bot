"""Tests for EdgeEngine."""
import pytest
from src.edge_lab.edge_engine import EdgeEngine
from src.edge_lab.edge_profile import EdgeProfile
from src.signals.strategies.donchian_strategy import DonchianStrategy
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from tests.fixtures import uptrend_candles, downtrend_candles

@pytest.fixture
def config():
    return {"backtest":{"starting_balance":10_000,"commission_per_trade":0.0,"slippage_percent":0.0},
            "risk":{"risk_per_trade_percent":1.0,"atr_period":3,"minimum_signal_score":50.0,
                    "minimum_risk_reward":1.0,"atr_stop_multiplier":2.0,"atr_target_multiplier":3.0}}

@pytest.fixture
def engine(config): return EdgeEngine(config)

def _asset_candles():
    return {"SPY": uptrend_candles(n=60, symbol="SPY"), "DIA": downtrend_candles(n=60, symbol="DIA")}

class TestEdgeEngine:
    def test_evaluate_all_returns_profiles(self, engine):
        strats = [DonchianStrategy(period=5), MomentumRotationStrategy(roc_short=5, roc_long=10, ema_fast=5, ema_slow=10)]
        profiles = engine.evaluate_all(strats, _asset_candles())
        assert len(profiles) == 2
    def test_each_profile_has_correct_name(self, engine):
        strats = [DonchianStrategy(period=5)]
        profiles = engine.evaluate_all(strats, _asset_candles())
        assert profiles[0].strategy_name == "DONCHIAN_20"
    def test_robustness_rating_is_valid(self, engine):
        strats = [DonchianStrategy(period=5)]
        profiles = engine.evaluate_all(strats, _asset_candles())
        assert profiles[0].robustness_rating in ("ROBUST","MARGINAL","UNSTABLE","UNKNOWN")
    def test_profile_fields_populated(self, engine):
        strats = [DonchianStrategy(period=5)]
        profiles = engine.evaluate_all(strats, _asset_candles())
        p = profiles[0]
        assert p.total_trades >= 0
        assert 0.0 <= p.win_rate <= 1.0
        assert p.max_drawdown >= 0.0
