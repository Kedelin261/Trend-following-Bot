"""Phase 5.2 — Momentum Rotation Risk Overlay Research.

Provides risk overlays that intercept trade entry decisions without
modifying entry signals, the backtester, the risk engine, or position
sizing baseline.

Overlays implemented:
    NO_OVERLAY              — baseline reference
    EQUITY_CURVE_PAUSE      — pause after 10% equity drawdown from peak
    BEAR_MARKET_PAUSE       — pause during CONTRACTION/CRISIS macro regime
    MONTHLY_LOSS_LOCKOUT    — stop entries if monthly loss exceeds threshold
    CONSECUTIVE_LOSS_COOLDOWN — pause after 5 consecutive losing trades
    VOLATILITY_RISK_SCALING — reduce size in HIGH/EXTREME volatility
    COMBINED_OVERLAY        — max 2 overlays combined

Research only. No execution. No broker code. No live trading.
"""

from src.risk_overlay.base_overlay import RiskOverlay
from src.risk_overlay.overlay_backtester import OverlayBacktestEngine
from src.risk_overlay.profiles.no_overlay import NoOverlay
from src.risk_overlay.profiles.equity_curve_pause import EquityCurvePause
from src.risk_overlay.profiles.bear_market_pause import BearMarketPause
from src.risk_overlay.profiles.monthly_loss_lockout import MonthlyLossLockout
from src.risk_overlay.profiles.consecutive_loss_cooldown import ConsecutiveLossCooldown
from src.risk_overlay.profiles.volatility_risk_scaling import VolatilityRiskScaling
from src.risk_overlay.profiles.combined_overlay import CombinedOverlay

__all__ = [
    "RiskOverlay",
    "OverlayBacktestEngine",
    "NoOverlay",
    "EquityCurvePause",
    "BearMarketPause",
    "MonthlyLossLockout",
    "ConsecutiveLossCooldown",
    "VolatilityRiskScaling",
    "CombinedOverlay",
]
