"""Strategy V2 — refined strategy built from Phase 4.5 research findings.

Research-driven refinements over Strategy V1 (current):
  1. EMA 20/50 trend detector  (vs 50/200 in V1)
  2. Market regime filter       (BULL only: EMA20 > EMA50 > EMA200)
  3. ADX ≥ 25 trend-strength gate
  4. MEDIUM volatility filter   (blocks low and high volatility)
  5. Breakout threshold 1.00 %  (vs 0.25 % in V1)

The StrategyProfile dataclass captures all parameters so future phases
can evaluate multiple strategy versions simultaneously without rewriting
any downstream code.

No broker code. No API calls. No execution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.models import BacktestResults
from src.backtest.portfolio import Portfolio
from src.backtest.trade_simulator import TradeSimulator
from src.data.models import Candle
from src.refinement.adx_trade_filter import ADXTradeFilter
from src.refinement.market_regime_filter import MarketRegimeFilter
from src.refinement.volatility_trade_filter import VolatilityFilterMode, VolatilityTradeFilter
from src.risk.atr_calculator import ATRCalculator
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.risk.stop_loss_engine import StopLossEngine
from src.risk.take_profit_engine import TakeProfitEngine
from src.risk.trade_validator import TradeValidator
from src.signals.breakout_detector import BreakoutDetector
from src.signals.models import Signal, SignalType, TrendDirection
from src.signals.signal_engine import SignalEngine
from src.signals.trend_detector import TrendDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Profile
# ---------------------------------------------------------------------------

@dataclass
class StrategyProfile:
    """Immutable configuration for a complete strategy version.

    Build a BacktestEngine from this profile via build_backtest_engine().
    New strategy hypotheses → new StrategyProfile → no code rewrites needed.
    """

    name:                str
    description:         str

    # Signal engine
    ema_fast:            int        # trend detector fast EMA
    ema_slow:            int        # trend detector slow EMA
    breakout_threshold:  float      # minimum breakout clearance

    # Risk engine
    atr_period:          int   = 14
    atr_stop_mult:       float = 2.0
    atr_target_mult:     float = 3.0
    min_signal_score:    float = 70.0
    min_risk_reward:     float = 1.5

    # V2 filters (all False/None = disabled → compatible with V1)
    require_bull_regime: bool  = False
    adx_threshold:       float = 0.0       # 0 = no ADX filter
    volatility_mode:     VolatilityFilterMode = VolatilityFilterMode.NONE
    require_volume:      bool  = False

    @property
    def min_warmup(self) -> int:
        """Minimum candle history needed before first signal."""
        warmup = self.ema_slow + 10
        if self.require_bull_regime:
            warmup = max(warmup, 200 + 10)  # regime filter uses EMA200
        if self.adx_threshold > 0:
            warmup = max(warmup, 2 * 14 + 1)  # ADX(14) needs 29 bars
        return warmup

    def build_backtest_engine(self, config: dict) -> BacktestEngine:
        """Instantiate a fully-configured BacktestEngine from this profile."""
        bt_cfg  = config.get("backtest", {})
        r_cfg   = config.get("risk", {})
        start   = float(bt_cfg.get("starting_balance", 10_000.0))

        # Signal engine with profile EMA periods and breakout threshold
        base_signal = SignalEngine(
            trend_detector    = TrendDetector(self.ema_fast, self.ema_slow),
            breakout_detector = BreakoutDetector(self.breakout_threshold),
            min_candles       = self.ema_slow + 10,
        )

        # Optional filters
        regime_filter = MarketRegimeFilter() if self.require_bull_regime else None
        adx_filter    = ADXTradeFilter(self.adx_threshold) \
                        if self.adx_threshold > 0 else None
        vol_filter    = VolatilityTradeFilter(mode=self.volatility_mode) \
                        if self.volatility_mode != VolatilityFilterMode.NONE else None

        composite = CompositeSignalEngine(
            base_engine    = base_signal,
            regime_filter  = regime_filter,
            adx_filter     = adx_filter,
            vol_filter     = vol_filter,
        )

        # Risk engine
        profile = RiskProfile(
            account_size           = start,
            cash_available         = start,
            risk_per_trade_percent = float(r_cfg.get("risk_per_trade_percent", 1.0)),
        )
        risk_engine = RiskEngine(
            risk_profile       = profile,
            atr_calculator     = ATRCalculator(period=self.atr_period),
            stop_loss_engine   = StopLossEngine(multiplier=self.atr_stop_mult),
            take_profit_engine = TakeProfitEngine(multiplier=self.atr_target_mult),
            trade_validator    = TradeValidator(
                minimum_signal_score = self.min_signal_score,
                minimum_risk_reward  = self.min_risk_reward,
            ),
        )

        return BacktestEngine(
            signal_engine = composite,
            risk_engine   = risk_engine,
            portfolio     = Portfolio(start),
            simulator     = TradeSimulator(
                slippage_percent     = float(bt_cfg.get("slippage_percent", 0.05)),
                commission_per_trade = float(bt_cfg.get("commission_per_trade", 1.0)),
            ),
            min_warmup = self.min_warmup,
        )

    def run_backtest(
        self,
        candles: List[Candle],
        config:  dict,
    ) -> BacktestResults:
        """Convenience: build engine and run backtest in one call."""
        engine = self.build_backtest_engine(config)
        return engine.run(candles)


# ---------------------------------------------------------------------------
# Composite Signal Engine — filter wrapper
# ---------------------------------------------------------------------------

def _none_signal(candles: List[Candle], reason: str) -> Signal:
    """Return a NONE signal with the filter's rejection reason."""
    from datetime import timezone as _tz
    if candles:
        sym, tf, ts, px = (
            candles[-1].symbol, candles[-1].timeframe,
            candles[-1].timestamp, candles[-1].close,
        )
    else:
        sym, tf, ts, px = "UNKNOWN", "UNKNOWN", datetime.now(tz=_tz.utc), 0.0

    return Signal(
        symbol            = sym,
        timeframe         = tf,
        timestamp         = ts,
        signal_type       = SignalType.NONE,
        trend_direction   = TrendDirection.NEUTRAL,
        strength_score    = 0.0,
        breakout_detected = False,
        volume_confirmed  = None,
        support_level     = None,
        resistance_level  = None,
        close_price       = px,
        reasoning         = [reason],
    )


class CompositeSignalEngine:
    """Wraps a base SignalEngine with pre-trade filter gates.

    Filters are applied in order before the base engine runs:
    1. Market regime filter (is BULL?)
    2. ADX filter (trend strong enough?)
    3. Volatility filter (ATR% in target range?)

    If any filter fails, NONE is returned immediately.
    The base engine is only called when all filters pass.
    """

    def __init__(
        self,
        base_engine:    SignalEngine,
        regime_filter:  Optional[MarketRegimeFilter]   = None,
        adx_filter:     Optional[ADXTradeFilter]       = None,
        vol_filter:     Optional[VolatilityTradeFilter] = None,
    ) -> None:
        self._base    = base_engine
        self._regime  = regime_filter
        self._adx     = adx_filter
        self._vol     = vol_filter

    @property
    def min_candles(self) -> int:
        return self._base.min_candles

    def generate_signal(self, candles: List[Candle]) -> Signal:
        """Run filters then delegate to the base signal engine."""
        if not candles:
            return _none_signal(candles if candles else [], "No candles")

        if self._regime and not self._regime.is_bull_regime(candles):
            return _none_signal(candles, "Regime filter: not BULL (EMA20>EMA50>EMA200)")

        if self._adx and not self._adx.passes(candles):
            adx_val = self._adx.current_adx(candles)
            adx_str = f"{adx_val:.1f}" if adx_val is not None else "N/A"
            return _none_signal(
                candles,
                f"ADX filter: {adx_str} < threshold {self._adx.threshold}",
            )

        if self._vol and not self._vol.passes(candles):
            cat = self._vol.current_category(candles)
            return _none_signal(
                candles,
                f"Volatility filter: {cat.value} not in allowed categories",
            )

        return self._base.generate_signal(candles)


# ---------------------------------------------------------------------------
# Pre-defined strategy profiles
# ---------------------------------------------------------------------------

V1_PROFILE = StrategyProfile(
    name                = "Trend V1",
    description         = "Original strategy: EMA50/200, 0.25% breakout, no filters",
    ema_fast            = 50,
    ema_slow            = 200,
    breakout_threshold  = 0.0025,
    atr_period          = 14,
    atr_stop_mult       = 2.0,
    atr_target_mult     = 3.0,
    min_signal_score    = 70.0,
    min_risk_reward     = 1.5,
    require_bull_regime = False,
    adx_threshold       = 0.0,
    volatility_mode     = VolatilityFilterMode.NONE,
)

V2_PROFILE = StrategyProfile(
    name                = "Trend V2",
    description         = (
        "Refined strategy: EMA20/50, BULL regime filter, ADX≥25, "
        "MEDIUM volatility, 1.00% breakout threshold"
    ),
    ema_fast            = 20,
    ema_slow            = 50,
    breakout_threshold  = 0.0100,   # 1.00%
    atr_period          = 14,
    atr_stop_mult       = 2.0,
    atr_target_mult     = 3.0,
    min_signal_score    = 70.0,
    min_risk_reward     = 1.5,
    require_bull_regime = True,
    adx_threshold       = 25.0,
    volatility_mode     = VolatilityFilterMode.MEDIUM_ONLY,
)
