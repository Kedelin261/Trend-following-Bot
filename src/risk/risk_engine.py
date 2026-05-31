"""Risk Engine — Phase 3 master orchestrator.

Consumes a Signal (from Phase 2) and historical Candles, then produces a
fully-evaluated TradeCandidate.  The engine is broker-agnostic: it never
imports IBKR, MT5, or any market data provider.

Dependency flow:
    Signal + Candles
            ↓
        RiskEngine
            ↓
       TradeCandidate

No execution code. No orders. No positions.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.data.models import Candle
from src.risk.atr_calculator import ATRCalculator
from src.risk.models import RiskProfile, TradeCandidate
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss_engine import StopLossEngine
from src.risk.take_profit_engine import TakeProfitEngine
from src.risk.trade_validator import TradeValidator
from src.signals.models import Signal, SignalType

logger = logging.getLogger(__name__)


class RiskEngine:
    """Orchestrates ATR calculation, stop/target placement, sizing, and validation.

    Parameters
    ----------
    risk_profile          : Account parameters (injectable — changes without code edits)
    atr_calculator        : ATRCalculator instance (or default ATR 14)
    stop_loss_engine      : StopLossEngine instance (or default ×2)
    take_profit_engine    : TakeProfitEngine instance (or default ×3)
    position_sizer        : PositionSizer instance (or default)
    trade_validator       : TradeValidator instance (or default)
    """

    def __init__(
        self,
        risk_profile:       RiskProfile,
        atr_calculator:     Optional[ATRCalculator]  = None,
        stop_loss_engine:   Optional[StopLossEngine]  = None,
        take_profit_engine: Optional[TakeProfitEngine] = None,
        position_sizer:     Optional[PositionSizer]   = None,
        trade_validator:    Optional[TradeValidator]  = None,
    ) -> None:
        self._profile   = risk_profile
        self._atr       = atr_calculator    or ATRCalculator()
        self._stop      = stop_loss_engine  or StopLossEngine()
        self._target    = take_profit_engine or TakeProfitEngine()
        self._sizer     = position_sizer    or PositionSizer()
        self._validator = trade_validator   or TradeValidator()

    @classmethod
    def from_config(cls, config: dict) -> "RiskEngine":
        """Construct a fully-configured RiskEngine from settings.yaml."""
        r = config.get("risk", {})
        profile = RiskProfile.from_config(config)
        return cls(
            risk_profile       = profile,
            atr_calculator     = ATRCalculator(period=int(r.get("atr_period", 14))),
            stop_loss_engine   = StopLossEngine(
                multiplier=float(r.get("atr_stop_multiplier", 2.0))
            ),
            take_profit_engine = TakeProfitEngine(
                multiplier=float(r.get("atr_target_multiplier", 3.0))
            ),
            trade_validator    = TradeValidator(
                minimum_signal_score=float(r.get("minimum_signal_score", 70.0)),
                minimum_risk_reward =float(r.get("minimum_risk_reward", 1.5)),
            ),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, signal: Signal, candles: List[Candle]) -> TradeCandidate:
        """Evaluate a signal against the risk profile and return a TradeCandidate.

        Always returns a valid TradeCandidate — never raises.  When a trade
        cannot be approved the candidate has approved=False and a descriptive
        rejection_reason.
        """
        if signal.signal_type == SignalType.NONE:
            return self._build_rejected(
                signal,
                atr=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                rr=0.0,
                position_size=0,
                dollar_risk=0.0,
                reason="No actionable signal (signal_type=NONE)",
            )

        # ------------------------------------------------------------------ #
        # Step 1 — ATR                                                         #
        # ------------------------------------------------------------------ #
        atr = self._atr.calculate_atr(candles)
        if atr is None:
            return self._build_rejected(
                signal,
                atr=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                rr=0.0,
                position_size=0,
                dollar_risk=0.0,
                reason=(
                    f"Insufficient candle history — need ≥ {self._atr.period + 1} "
                    f"candles for ATR({self._atr.period}), got {len(candles)}"
                ),
            )

        # ------------------------------------------------------------------ #
        # Step 2 — Prices                                                      #
        # ------------------------------------------------------------------ #
        entry = signal.close_price

        stop   = self._stop.calculate_stop_loss(signal.signal_type, entry, atr)
        target = self._target.calculate_take_profit(signal.signal_type, entry, atr)

        risk_per_share   = abs(entry - stop)
        reward_per_share = abs(target - entry)
        rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

        # ------------------------------------------------------------------ #
        # Step 3 — Position size                                               #
        # ------------------------------------------------------------------ #
        dollar_risk   = self._sizer.dollar_risk_from_profile(self._profile)
        position_size = self._sizer.calculate_position_size(dollar_risk, risk_per_share)

        # ------------------------------------------------------------------ #
        # Step 4 — Build candidate                                             #
        # ------------------------------------------------------------------ #
        candidate = TradeCandidate(
            symbol            = signal.symbol,
            timeframe         = signal.timeframe,
            signal_type       = signal.signal_type,
            entry_price       = round(entry,          5),
            atr               = round(atr,            5),
            stop_loss         = round(stop,           5),
            take_profit       = round(target,         5),
            risk_per_share    = round(risk_per_share, 5),
            reward_per_share  = round(reward_per_share, 5),
            risk_reward_ratio = round(rr,             3),
            position_size     = position_size,
            dollar_risk       = round(dollar_risk,    2),
            signal_score      = signal.strength_score,
            approved          = False,     # set by validator below
            rejection_reason  = None,
            timestamp         = signal.timestamp,
        )

        # ------------------------------------------------------------------ #
        # Step 5 — Validate                                                    #
        # ------------------------------------------------------------------ #
        approved, reason = self._validator.validate_trade(candidate)
        candidate.approved          = approved
        candidate.rejection_reason  = reason

        logger.info(
            "risk_engine: %s/%s signal=%s score=%.1f atr=%.4f "
            "stop=%.4f target=%.4f rr=%.2f size=%d approved=%s",
            signal.symbol,
            signal.timeframe,
            signal.signal_type.value,
            signal.strength_score,
            atr,
            stop,
            target,
            rr,
            position_size,
            approved,
        )
        return candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rejected(
        signal: Signal,
        *,
        atr: float,
        stop_loss: float,
        take_profit: float,
        risk_per_share: float,
        reward_per_share: float,
        rr: float,
        position_size: int,
        dollar_risk: float,
        reason: str,
    ) -> TradeCandidate:
        logger.info(
            "risk_engine: REJECTED %s/%s — %s", signal.symbol, signal.timeframe, reason
        )
        return TradeCandidate(
            symbol            = signal.symbol,
            timeframe         = signal.timeframe,
            signal_type       = signal.signal_type,
            entry_price       = signal.close_price,
            atr               = atr,
            stop_loss         = stop_loss,
            take_profit       = take_profit,
            risk_per_share    = risk_per_share,
            reward_per_share  = reward_per_share,
            risk_reward_ratio = rr,
            position_size     = position_size,
            dollar_risk       = dollar_risk,
            signal_score      = signal.strength_score,
            approved          = False,
            rejection_reason  = reason,
            timestamp         = signal.timestamp,
        )
