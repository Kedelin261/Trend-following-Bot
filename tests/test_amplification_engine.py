"""Tests for AmplificationResearchEngine — Phase 5.3."""

import pytest
import math
from datetime import datetime, timedelta, timezone
from src.data.models import Candle
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from src.edge_amplification.amplification_research_engine import (
    AmplificationResearchEngine, BaselineMetrics, AmplificationCandidate,
    PROMO_MIN_TRADES,
)

EDGE_CONFIG = {
    "backtest": {
        "starting_balance":     10_000.0,
        "commission_per_trade":  1.0,
        "slippage_percent":      0.05,
    },
    "risk": {
        "risk_per_trade_percent": 1.0,
        "atr_period":             14,
        "atr_stop_multiplier":    2.0,
        "atr_target_multiplier":  3.0,
        "minimum_signal_score":   40.0,
        "minimum_risk_reward":    1.0,
    },
}


def _candles(symbol: str, n: int = 300, seed: int = 42) -> list:
    """Build minimal synthetic candles for unit tests."""
    import random
    rng = random.Random(seed)
    price = 200.0
    base = datetime(2020, 1, 2, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        ret = 0.0003 + rng.gauss(0, 0.015)
        price = max(1.0, price * (1 + ret))
        r = price * 0.015
        open_ = price * (1 + rng.gauss(0, 0.003))
        high = max(open_, price) + abs(rng.gauss(0, r * 0.6))
        low = min(open_, price) - abs(rng.gauss(0, r * 0.6))
        out.append(Candle(
            symbol=symbol, timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=round(open_, 4), high=round(high, 4),
            low=round(low, 4), close=round(price, 4),
            volume=rng.uniform(1e6, 10e6),
            provider="TEST",
        ))
    return out


def test_engine_produces_result():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles, data_source="TEST")
    assert result is not None
    assert result.baseline is not None


def test_baseline_fields_populated():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    b = result.baseline
    assert isinstance(b.trades, int)
    assert b.trades >= 0
    assert b.profit_factor >= 0
    assert 0.0 <= b.win_rate <= 1.0
    assert b.max_drawdown >= 0.0


def test_asset_contributions_populated():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300), "QQQ": _candles("QQQ", 300, seed=7)}
    result = engine.run(strategy, asset_candles)
    assert len(result.asset_contributions) <= 2  # only assets with trades


def test_regime_contributions_present():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert isinstance(result.regime_contributions, list)


def test_vol_contributions_present():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert isinstance(result.vol_contributions, list)


def test_quality_buckets_present():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert isinstance(result.quality_buckets, list)


def test_holding_period_present():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert isinstance(result.holding_periods, list)


def test_profit_concentration_present():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert result.profit_concentration is not None


def test_loss_concentration_present():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert result.loss_concentration is not None


def test_recommendation_valid_string():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert result.recommendation in ("PROCEED TO PHASE 5.4", "RETURN TO STRATEGY RESEARCH")


def test_data_source_propagated():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles, data_source="LIVE (IBKR)")
    assert result.data_source == "LIVE (IBKR)"


def test_history_bars_set():
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    asset_candles = {"SPY": _candles("SPY", 300)}
    result = engine.run(strategy, asset_candles)
    assert result.history_bars == 300


def test_baseline_metrics_with_no_trades():
    """Degenerate: no candles → no trades."""
    strategy = MomentumRotationStrategy()
    engine = AmplificationResearchEngine(EDGE_CONFIG)
    result = engine.run(strategy, {})
    assert result.baseline.trades == 0
    assert result.baseline.profit_factor == 0.0


def test_pf_str_finite():
    b = BaselineMetrics(10, 1.25, 5.0, 0.6, 12.0, 500.0, 400.0)
    assert b.pf_str == "1.25"


def test_pf_str_inf():
    b = BaselineMetrics(5, float("inf"), 10.0, 1.0, 5.0, 200.0, 0.0)
    assert b.pf_str == "∞"
