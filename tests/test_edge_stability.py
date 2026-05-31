"""Tests for EdgeStabilityAnalyzer."""
import pytest
from src.edge_lab.edge_stability import EdgeStabilityAnalyzer
from src.signals.strategies.donchian_strategy import DonchianStrategy
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.risk.risk_engine import RiskEngine
from tests.fixtures import uptrend_candles

def _build(strat):
    config = {"backtest":{"starting_balance":10_000},"risk":{"risk_per_trade_percent":1.0,"atr_period":3,"atr_stop_multiplier":2.0,"atr_target_multiplier":3.0,"minimum_signal_score":50.0,"minimum_risk_reward":1.0}}
    risk = RiskEngine.from_config(config)
    return BacktestEngine(signal_engine=strat, risk_engine=risk,
                          portfolio=Portfolio(10_000),
                          simulator=TradeSimulator(0.0, 0.0),
                          min_warmup=strat.min_candles)

class TestEdgeStabilityAnalyzer:
    def test_returns_rating_and_windows(self):
        strat = DonchianStrategy(period=5)
        candles = {"SPY": uptrend_candles(n=100)}
        analyzer = EdgeStabilityAnalyzer()
        rating, windows = analyzer.analyze(strat, candles, _build)
        assert rating in ("ROBUST","MARGINAL","UNSTABLE")
        assert len(windows) == 3
    def test_window_names(self):
        strat = DonchianStrategy(period=5)
        candles = {"SPY": uptrend_candles(n=100)}
        _, windows = EdgeStabilityAnalyzer().analyze(strat, candles, _build)
        assert {w.name for w in windows} == {"early","middle","recent"}
