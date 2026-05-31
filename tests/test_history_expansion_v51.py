"""Tests for history_expansion (Phase 5.1)."""
import pytest
from src.edge_validation.history_expansion import HistoryExpansionResearcher
from src.signals.strategies.donchian_strategy import DonchianStrategy
from tests.fixtures import uptrend_candles

CONFIG = {"backtest":{"starting_balance":10_000,"commission_per_trade":0.0,"slippage_percent":0.0},
          "risk":{"risk_per_trade_percent":1.0,"atr_period":3,"minimum_signal_score":50.0,
                  "minimum_risk_reward":1.0,"atr_stop_multiplier":2.0,"atr_target_multiplier":3.0}}

def test_returns_slices():
    r = HistoryExpansionResearcher(CONFIG, counts=[50, 80])
    candles = {"SPY": uptrend_candles(n=100)}
    slices = r.research(DonchianStrategy(period=5), candles)
    assert len(slices) >= 2

def test_slices_capped_at_available():
    r = HistoryExpansionResearcher(CONFIG, counts=[200])
    candles = {"SPY": uptrend_candles(n=80)}
    slices = r.research(DonchianStrategy(period=5), candles)
    assert slices[0].actual_bars <= 80

def test_profile_populated():
    r = HistoryExpansionResearcher(CONFIG, counts=[50])
    candles = {"SPY": uptrend_candles(n=60)}
    slices = r.research(DonchianStrategy(period=5), candles)
    assert slices[0].profile is not None
    assert slices[0].trades >= 0
