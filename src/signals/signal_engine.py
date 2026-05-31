"""Signal Engine — master orchestrator for Phase 2.

Consumes a list of Candle objects (from any provider) and returns
a fully-populated Signal object.

Dependency flow
---------------
Candles → TrendDetector
        → SupportResistanceDetector
        → BreakoutDetector
        → VolumeConfirmation
        → SignalScorer
        → Signal

Signal rules
------------
LONG  : BULLISH trend  AND bullish breakout  AND (volume confirmed OR unknown)
SHORT : BEARISH trend  AND bearish breakout  AND (volume confirmed OR unknown)
NONE  : all other cases

No execution code. No orders. No positions. No broker imports.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.data.models import Candle
from src.signals.breakout_detector import BreakoutDetector
from src.signals.models import Signal, SignalType, TrendDirection
from src.signals.signal_scorer import SignalScorer
from src.signals.support_resistance import SupportResistanceDetector
from src.signals.trend_detector import TrendDetector
from src.signals.volume_confirmation import VolumeConfirmation

logger = logging.getLogger(__name__)


class SignalEngine:
    """Orchestrates all signal-analysis components into a single Signal.

    Parameters
    ----------
    trend_detector     : custom TrendDetector instance (or default)
    sr_detector        : custom SupportResistanceDetector (or default)
    breakout_detector  : custom BreakoutDetector (or default)
    volume_confirmation: custom VolumeConfirmation (or default)
    scorer             : custom SignalScorer (or default)
    min_candles        : minimum candle count required to produce a signal;
                         below this threshold NONE is returned immediately
    """

    def __init__(
        self,
        trend_detector: Optional[TrendDetector] = None,
        sr_detector: Optional[SupportResistanceDetector] = None,
        breakout_detector: Optional[BreakoutDetector] = None,
        volume_confirmation: Optional[VolumeConfirmation] = None,
        scorer: Optional[SignalScorer] = None,
        min_candles: int = 210,
    ) -> None:
        self.trend_detector      = trend_detector      or TrendDetector()
        self.sr_detector         = sr_detector         or SupportResistanceDetector()
        self.breakout_detector   = breakout_detector   or BreakoutDetector()
        self.volume_confirmation = volume_confirmation or VolumeConfirmation()
        self.scorer              = scorer              or SignalScorer()
        self.min_candles         = min_candles

    def generate_signal(self, candles: List[Candle]) -> Signal:
        """Analyse *candles* and return a Signal.

        Always returns a valid Signal (never raises).  When data is
        insufficient the signal_type will be NONE with a descriptive
        reasoning entry.
        """
        if not candles:
            return self._empty_signal("UNKNOWN", "UNKNOWN")

        latest = candles[-1]

        if len(candles) < self.min_candles:
            logger.warning(
                "signal_engine: only %d candles for %s/%s — need %d (returning NONE)",
                len(candles),
                latest.symbol,
                latest.timeframe,
                self.min_candles,
            )
            return self._insufficient_signal(latest)

        closes = [c.close for c in candles]

        # ------------------------------------------------------------------ #
        # Step 1 — Trend                                                       #
        # ------------------------------------------------------------------ #
        trend     = self.trend_detector.get_trend_direction(candles)
        strength  = self.trend_detector.get_trend_strength(candles)
        ema50     = self.trend_detector.calculate_ema(closes, self.trend_detector.fast_period)
        ema200    = self.trend_detector.calculate_ema(closes, self.trend_detector.slow_period)

        # ------------------------------------------------------------------ #
        # Step 2 — Support / Resistance                                        #
        # ------------------------------------------------------------------ #
        support_levels    = self.sr_detector.get_support_levels(candles)
        resistance_levels = self.sr_detector.get_resistance_levels(candles)

        # Nearest levels relative to current price
        nearest_support    = self.sr_detector.nearest_support(candles, latest.close)
        nearest_resistance = self.sr_detector.nearest_resistance(candles, latest.close)

        # ------------------------------------------------------------------ #
        # Step 3 — Breakout                                                    #
        # ------------------------------------------------------------------ #
        bull_breakout, breached_resistance = self.breakout_detector.detect_bullish_breakout(
            latest.close, resistance_levels
        )
        bear_breakout, broken_support = self.breakout_detector.detect_bearish_breakout(
            latest.close, support_levels
        )

        breakout_detected = bull_breakout or bear_breakout

        # ------------------------------------------------------------------ #
        # Step 4 — Volume                                                      #
        # ------------------------------------------------------------------ #
        volume_confirmed = self.volume_confirmation.is_volume_confirmed(candles)
        vol_ok = volume_confirmed is True or volume_confirmed is None

        # ------------------------------------------------------------------ #
        # Step 5 — Signal type                                                 #
        # ------------------------------------------------------------------ #
        reasoning: List[str] = []

        if trend == TrendDirection.BULLISH and bull_breakout and vol_ok:
            signal_type = SignalType.LONG
            reasoning.append("Price above EMA50")
            reasoning.append("EMA50 above EMA200")
            reasoning.append(
                f"Resistance breakout: close {latest.close:.4f} > "
                f"resistance {breached_resistance:.4f}"
            )
            if volume_confirmed is True:
                reasoning.append("Volume above 20-period average")
            else:
                reasoning.append("Volume data unavailable (Forex / broker)")

        elif trend == TrendDirection.BEARISH and bear_breakout and vol_ok:
            signal_type = SignalType.SHORT
            reasoning.append("Price below EMA50")
            reasoning.append("EMA50 below EMA200")
            reasoning.append(
                f"Support breakdown: close {latest.close:.4f} < "
                f"support {broken_support:.4f}"
            )
            if volume_confirmed is True:
                reasoning.append("Volume above 20-period average")
            else:
                reasoning.append("Volume data unavailable (Forex / broker)")

        else:
            signal_type = SignalType.NONE
            if trend == TrendDirection.NEUTRAL:
                reasoning.append("No clear trend — EMA alignment ambiguous")
            elif trend == TrendDirection.BULLISH and not bull_breakout:
                reasoning.append("Bullish trend present but no resistance breakout")
            elif trend == TrendDirection.BEARISH and not bear_breakout:
                reasoning.append("Bearish trend present but no support breakdown")
            if not vol_ok:
                reasoning.append("Volume below average — breakout not confirmed")

        # ------------------------------------------------------------------ #
        # Step 6 — Score                                                       #
        # ------------------------------------------------------------------ #
        score = self.scorer.score_signal(
            trend_direction   = trend,
            trend_strength    = strength,
            breakout_detected = breakout_detected,
            volume_confirmed  = volume_confirmed,
            signal_type       = signal_type,
        )

        logger.info(
            "signal_engine: %s/%s signal=%s trend=%s score=%.1f "
            "breakout=%s volume=%s",
            latest.symbol,
            latest.timeframe,
            signal_type.value,
            trend.value,
            score,
            breakout_detected,
            volume_confirmed,
        )

        return Signal(
            symbol             = latest.symbol,
            timeframe          = latest.timeframe,
            timestamp          = latest.timestamp,
            signal_type        = signal_type,
            trend_direction    = trend,
            strength_score     = score,
            breakout_detected  = breakout_detected,
            volume_confirmed   = volume_confirmed,
            support_level      = nearest_support,
            resistance_level   = (
                breached_resistance if bull_breakout
                else nearest_resistance
            ),
            close_price        = latest.close,
            reasoning          = reasoning,
            ema50              = ema50,
            ema200             = ema200,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_signal(symbol: str, timeframe: str) -> Signal:
        return Signal(
            symbol            = symbol,
            timeframe         = timeframe,
            timestamp         = datetime.now(tz=timezone.utc),
            signal_type       = SignalType.NONE,
            trend_direction   = TrendDirection.NEUTRAL,
            strength_score    = 0.0,
            breakout_detected = False,
            volume_confirmed  = None,
            support_level     = None,
            resistance_level  = None,
            close_price       = 0.0,
            reasoning         = ["No candle data provided"],
        )

    @staticmethod
    def _insufficient_signal(latest: Candle) -> Signal:
        return Signal(
            symbol            = latest.symbol,
            timeframe         = latest.timeframe,
            timestamp         = latest.timestamp,
            signal_type       = SignalType.NONE,
            trend_direction   = TrendDirection.NEUTRAL,
            strength_score    = 0.0,
            breakout_detected = False,
            volume_confirmed  = None,
            support_level     = None,
            resistance_level  = None,
            close_price       = latest.close,
            reasoning         = ["Insufficient candle history for signal analysis"],
        )
