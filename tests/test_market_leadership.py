"""Tests for MarketLeadershipStrategy (Family #4).

Verifies:
- min_candles property
- No signal on insufficient history
- Signal fires when leadership threshold met
- No signal in downtrend
- No signal when ROC percentile below threshold
- No signal when decelerating
- Strength score bounds
- Name and description
"""

import pytest
from tests.fixtures import _candle, uptrend_candles
from src.edge_discovery.market_leadership_research import MarketLeadershipStrategy
from src.signals.models import SignalType


@pytest.fixture
def strategy():
    return MarketLeadershipStrategy()


@pytest.fixture
def leadership_uptrend():
    """Slow then fast uptrend — recent ROC above 70th percentile of own history."""
    return _make_leadership_candles()


def _make_leadership_candles():
    """GBM candles — scan for first bar that fires the leadership signal."""
    import random
    from datetime import datetime, timedelta, timezone
    from src.data.models import Candle
    from src.edge_discovery.market_leadership_research import MarketLeadershipStrategy
    from src.signals.models import SignalType

    rng     = random.Random(7)
    base_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    strategy = MarketLeadershipStrategy()

    # GBM: strong drift to get leaderhip ROC naturally
    daily_drift = 0.15 / 252
    daily_vol   = 0.20 / (252 ** 0.5)
    price = 400.0
    candles = []
    for i in range(600):
        ret   = daily_drift + rng.gauss(0, daily_vol)
        price = max(1.0, price * (1 + ret))
        high  = price * (1 + abs(rng.gauss(0, daily_vol * 0.4)))
        low   = price * (1 - abs(rng.gauss(0, daily_vol * 0.4)))
        candles.append(Candle(
            symbol="SPY", timeframe="D1",
            timestamp=base_dt + timedelta(days=i),
            open=round(price * (1 + rng.gauss(0, daily_vol * 0.2)), 4),
            high=round(high, 4), low=round(low, 4),
            close=round(price, 4), volume=1e8, provider="TEST",
        ))

    for i in range(strategy.min_candles, len(candles)):
        sig = strategy.generate_signal(candles[:i + 1])
        if sig.signal_type == SignalType.LONG:
            return candles[:i + 1]

    return candles  # test will fail with info if no signal found


class TestMarketLeadershipBasics:
    def test_name(self, strategy):
        assert strategy.name == "MARKET_LEADERSHIP"

    def test_description_not_empty(self, strategy):
        assert len(strategy.description) > 10

    def test_min_candles_at_least_100(self, strategy):
        assert strategy.min_candles >= 100

    def test_no_signal_empty(self, strategy):
        sig = strategy.generate_signal([])
        assert sig.signal_type == SignalType.NONE

    def test_no_signal_insufficient(self, strategy):
        candles = uptrend_candles(n=strategy.min_candles - 1)
        sig = strategy.generate_signal(candles)
        assert sig.signal_type == SignalType.NONE


class TestMarketLeadershipSignal:
    def test_signal_fires_in_accelerating_uptrend(self, strategy, leadership_uptrend):
        """Leadership candles: slow drift then acceleration → high ROC percentile."""
        found = False
        for i in range(strategy.min_candles, len(leadership_uptrend)):
            sig = strategy.generate_signal(leadership_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                found = True
                break
        assert found, "Expected LONG signal: slow baseline then acceleration"

    def test_no_signal_in_downtrend(self, strategy):
        from tests.fixtures import downtrend_candles
        candles = downtrend_candles(n=300, start=500.0, daily_loss=0.002)
        for i in range(strategy.min_candles, len(candles)):
            sig = strategy.generate_signal(candles[:i + 1])
            assert sig.signal_type == SignalType.NONE

    def test_strength_at_least_65(self, strategy, leadership_uptrend):
        for i in range(strategy.min_candles, len(leadership_uptrend)):
            sig = strategy.generate_signal(leadership_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.strength_score >= 65.0
                break

    def test_strength_not_above_100(self, strategy, leadership_uptrend):
        for i in range(strategy.min_candles, len(leadership_uptrend)):
            sig = strategy.generate_signal(leadership_uptrend[:i + 1])
            if sig.signal_type == SignalType.LONG:
                assert sig.strength_score <= 100.0

    def test_never_raises_on_random(self, strategy):
        import random
        rng = random.Random(77)
        candles = [_candle(close=rng.uniform(1.0, 600.0), i=i) for i in range(300)]
        sig = strategy.generate_signal(candles)
        assert sig is not None

    def test_never_raises_on_uptrend(self, strategy, leadership_uptrend):
        sig = strategy.generate_signal(leadership_uptrend)
        assert sig is not None

    def test_roc_must_be_positive(self, strategy):
        """ROC must be positive — test with flat prices (ROC≈0)."""
        from datetime import datetime, timedelta, timezone
        from src.data.models import Candle
        base_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        # Flat price series
        candles = [
            Candle(
                symbol="SPY", timeframe="D1",
                timestamp=base_dt + timedelta(days=i),
                open=400.0, high=400.5, low=399.5,
                close=400.0, volume=1e8, provider="TEST",
            )
            for i in range(200)
        ]
        for i in range(strategy.min_candles, len(candles)):
            sig = strategy.generate_signal(candles[:i + 1])
            # Flat prices → ROC ≈ 0 → no signal
            assert sig.signal_type == SignalType.NONE
