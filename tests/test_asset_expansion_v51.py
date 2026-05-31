"""Tests for asset_expansion (Phase 5.1)."""
import pytest
from src.edge_validation.asset_expansion import AssetExpansionResearcher
from src.signals.strategies.donchian_strategy import DonchianStrategy
from tests.fixtures import uptrend_candles, downtrend_candles

CONFIG = {"backtest":{"starting_balance":10_000,"commission_per_trade":0.0,"slippage_percent":0.0},
          "risk":{"risk_per_trade_percent":1.0,"atr_period":3,"minimum_signal_score":50.0,
                  "minimum_risk_reward":1.0,"atr_stop_multiplier":2.0,"atr_target_multiplier":3.0}}

def test_per_asset_results():
    r = AssetExpansionResearcher(CONFIG)
    candles = {"SPY": uptrend_candles(n=60, symbol="SPY"), "DIA": downtrend_candles(n=60, symbol="DIA")}
    result = r.research(DonchianStrategy(period=5), candles)
    assert len(result.asset_contributions) == 2
    assert result.strategy_name == "DONCHIAN_20"

def test_portfolio_profile_populated():
    r = AssetExpansionResearcher(CONFIG)
    candles = {"SPY": uptrend_candles(n=60)}
    result = r.research(DonchianStrategy(period=5), candles)
    assert result.portfolio_profile is not None

def test_approved_assets_method():
    r = AssetExpansionResearcher(CONFIG)
    candles = {"SPY": uptrend_candles(n=60)}
    result = r.research(DonchianStrategy(period=5), candles)
    assert isinstance(result.approved_assets, list)
