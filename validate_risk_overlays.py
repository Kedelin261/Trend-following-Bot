#!/usr/bin/env python3
"""Phase 5.2 — Momentum Rotation Risk Overlay Validation Script.

Tests 7 risk overlay variants on the MOMENTUM_ROTATION strategy:
  1. NO_OVERLAY              — baseline reference
  2. EQUITY_CURVE_PAUSE      — pause after 10% equity drawdown from peak
  3. BEAR_MARKET_PAUSE       — pause during CONTRACTION/CRISIS regimes
  4. MONTHLY_LOSS_LOCKOUT    — lockout rest of month after 5% monthly loss
  5. CONSECUTIVE_LOSS_COOLDOWN — 10-bar pause after 5 consecutive losses
  6. VOLATILITY_RISK_SCALING — reduce size in HIGH/EXTREME volatility
  7. COMBINED_OVERLAY        — EQUITY_CURVE_PAUSE + VOLATILITY_RISK_SCALING

Data source (priority order):
  1. LIVE — IBKR historical data via configured provider (settings.yaml)
  2. SYNTHETIC — deterministic fallback when IBKR is unavailable
             (same generator as Phase 5.1 for reproducibility)

Fairness guarantee (identical for every overlay):
  - Same Momentum Rotation signals
  - Same 8 assets  (SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD)
  - Same candles   (live IBKR D1 bars, or synthetic fallback)
  - Same backtester / risk engine / slippage / commissions
  - Same position sizing baseline
  - Same history window
  - Only the overlay logic differs

Promotion criteria (Phase 5.2):
  Trades >= 500
  PF    >= 1.50
  Exp   >  $0
  MaxDD <  15%
  Robustness = ROBUST
  History >= 3000 bars

Research only. No execution. No broker code. No live trading.
No orders placed. Read-only throughout.
"""

import math
import random
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.data.models import Candle
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from src.risk_overlay.overlay_engine import OverlayEngine, OverlayResult, OverlayResearchReport
from src.risk_overlay.profiles.no_overlay import NoOverlay
from src.risk_overlay.profiles.equity_curve_pause import EquityCurvePause
from src.risk_overlay.profiles.bear_market_pause import BearMarketPause
from src.risk_overlay.profiles.monthly_loss_lockout import MonthlyLossLockout
from src.risk_overlay.profiles.consecutive_loss_cooldown import ConsecutiveLossCooldown
from src.risk_overlay.profiles.volatility_risk_scaling import VolatilityRiskScaling
from src.risk_overlay.profiles.combined_overlay import CombinedOverlay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — identical to Phase 5.1 for fairness
# ---------------------------------------------------------------------------

ASSETS        = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
MAX_BARS      = 5000
REQUEST_DELAY = 1.5   # seconds between IBKR candle requests (rate limiting)

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
        # Matches Phase 5.1 edge validation thresholds (not settings.yaml defaults)
        "minimum_signal_score":   40.0,
        "minimum_risk_reward":    1.0,
    },
}

# ---------------------------------------------------------------------------
# Synthetic data — identical generator to Phase 5.1 for reproducibility
# ---------------------------------------------------------------------------

_START_PRICES = {
    "SPY": 400.0, "VOO": 380.0, "QQQ": 350.0, "DIA": 330.0,
    "IWM": 220.0, "VTI": 210.0, "XLV": 130.0, "SCHD": 75.0,
}
_SEEDS = {
    "SPY": 42, "VOO": 99, "QQQ": 7, "DIA": 13,
    "IWM": 21, "VTI": 55, "XLV": 63, "SCHD": 88,
}


def _synthetic(symbol: str, n: int = 5000) -> List[Candle]:
    """Generate deterministic synthetic OHLCV data — same seed/params as Phase 5.1."""
    rng   = random.Random(_SEEDS.get(symbol, 42))
    price = _START_PRICES.get(symbol, 100.0)
    base  = datetime(2016, 1, 4, tzinfo=timezone.utc)
    out   = []
    daily_drift = 0.08 / 252
    daily_vol   = 0.18 / (252 ** 0.5)

    for i in range(n):
        ret   = daily_drift + rng.gauss(0, daily_vol)
        price = max(1.0, price * (1 + ret))
        rng_r = price * daily_vol * 0.6
        open_ = price * (1 + rng.gauss(0, daily_vol * 0.25))
        high  = max(open_, price) + abs(rng.gauss(0, rng_r))
        low   = min(open_, price) - abs(rng.gauss(0, rng_r))
        out.append(Candle(
            symbol=symbol, timeframe="D1",
            timestamp=base + timedelta(days=i),
            open=round(open_, 4), high=round(high, 4),
            low=round(low, 4),   close=round(price, 4),
            volume=rng.uniform(30e6, 200e6),
            provider="SYNTHETIC",
        ))
    return out


# ---------------------------------------------------------------------------
# Live data — IBKR via existing provider architecture (same as Phase 5.1)
# ---------------------------------------------------------------------------

def _try_live(assets: List[str], n: int) -> Dict[str, List[Candle]]:
    """Attempt to fetch live D1 bars from the configured provider (IBKR).

    Returns a dict of {symbol: candles} on success.
    Returns {} on any failure — caller falls back to synthetic.

    Uses the identical provider path as validate_edge_scalability.py.
    No orders are placed. Read-only throughout.
    """
    try:
        from src.config.settings_loader import load_settings
        from src.data.symbol_registry import SymbolRegistry
        from src.providers.provider_factory import ProviderFactory

        config   = load_settings()
        registry = SymbolRegistry.from_config(config)
        provider = ProviderFactory.create(config, registry)

        if not provider.connect():
            logger.warning("validate_risk_overlays: provider connect failed — using synthetic")
            return {}

        result: Dict[str, List[Candle]] = {}
        for sym in assets:
            candles = provider.get_candles(sym, "D1", n)
            if candles:
                result[sym] = candles
                logger.info("  %s: %d live bars fetched", sym, len(candles))
            else:
                logger.warning("  %s: no candles returned", sym)
            time.sleep(REQUEST_DELAY)

        provider.disconnect()
        return result

    except Exception as exc:
        logger.warning("validate_risk_overlays: live fetch failed (%s) — using synthetic", exc)
        return {}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler("logs/risk_overlay_validation.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def _print_report(
    report:      OverlayResearchReport,
    data_source: str,
    n_bars:      int,
) -> None:
    sep = "=" * 66

    print(f"\n{sep}")
    print("  RISK OVERLAY VALIDATION REPORT  —  Phase 5.2")
    print(f"  Strategy:    MOMENTUM_ROTATION")
    print(f"  Assets:      {', '.join(ASSETS)}")
    print(f"  Data source: {data_source}")
    print(f"  History:     {n_bars:,} D1 bars per asset")
    print(sep)

    for r in report.overlay_results:
        print(f"\n  {'─'*62}")
        print(f"  {r.overlay_name}")
        print()
        print(f"  Trades:      {r.trades}")
        print(f"  PF:          {_pf(r.profit_factor)}")
        print(f"  Expectancy:  ${r.expectancy:+.2f}/trade")
        print(f"  Max DD:      {r.max_drawdown:.1f}%")
        print(f"  Robustness:  {r.robustness}")
        print()

        print(f"  Promotion gate:")
        for c in r.pass_criteria:
            print(f"    ✓ {c}")
        for c in r.fail_criteria:
            print(f"    ✗ {c}")
        print(f"\n  STATUS: {r.status}")

    # Rankings
    print(f"\n{sep}")
    print("  FINAL RANKINGS")
    print(sep)
    print()

    sorted_results = sorted(
        report.overlay_results,
        key=lambda r: (
            r.is_promoted,
            r.expectancy if r.profit_factor >= 1.0 else -999,
            -r.max_drawdown,
        ),
        reverse=True,
    )

    for rank, r in enumerate(sorted_results, 1):
        promoted_tag = " ← CANDIDATE" if r.is_promoted else ""
        print(
            f"  {rank}. {r.overlay_name:<30}"
            f"  PF={_pf(r.profit_factor):<6}"
            f"  Exp=${r.expectancy:+.2f}"
            f"  DD={r.max_drawdown:.1f}%"
            f"  {r.robustness}"
            f"{promoted_tag}"
        )

    print(f"\n{sep}")
    print("  PROMOTION CANDIDATE")
    print(sep)

    if report.best_candidate:
        c = report.best_candidate
        print(f"\n  YES — {c.overlay_name}")
        print(f"\n  Trades:      {c.trades}")
        print(f"  PF:          {_pf(c.profit_factor)}")
        print(f"  Expectancy:  ${c.expectancy:+.2f}/trade")
        print(f"  Max DD:      {c.max_drawdown:.1f}%")
        print(f"  Robustness:  {c.robustness}")
        print(f"  History:     {c.history_bars:,} bars")
    else:
        print(f"\n  NO PROMOTION CANDIDATE")
        if report.overlay_results:
            best  = max(report.overlay_results, key=lambda r: (r.expectancy, -r.max_drawdown))
            fails = " | ".join(best.fail_criteria[:3])
            print(f"\n  Best: {best.overlay_name}")
            print(f"  Failing: {fails}")

    print(f"\n{sep}")
    print("  RECOMMENDATION")
    print(sep)

    if report.promotion_exists:
        best = report.best_candidate
        print(f"\n  PROMOTE TO PHASE 5.3")
        print(f"\n  Overlay: {best.overlay_name}")
        print(
            f"  This overlay reduces Max DD to {best.max_drawdown:.1f}% "
            f"(below 15% limit) while maintaining:"
        )
        print(f"    Trades={best.trades} (sample size adequate)")
        print(f"    PF={_pf(best.profit_factor)} (≥1.50 threshold)")
        print(f"    Expectancy=${best.expectancy:+.2f}/trade (positive edge)")
        print(f"    Robustness={best.robustness}")
    else:
        print(f"\n  RETURN TO EDGE RESEARCH")
        print()
        print("  No risk overlay was able to simultaneously achieve:")
        print("    Trades ≥ 500  AND  PF ≥ 1.50  AND  DD < 15%")
        print("    AND  Robustness = ROBUST  AND  History ≥ 3000 bars")
        print()
        if data_source.startswith("SYNTHETIC"):
            print("  NOTE: These results used synthetic fallback data.")
            print("  Re-run with live IBKR data for the official validation.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _setup_logging()

    print("\nPhase 5.2 — Momentum Rotation Risk Overlay Validation")
    print("=" * 66)

    # ------------------------------------------------------------------
    # Step 1: Attempt live IBKR data (same pattern as Phase 5.1)
    # ------------------------------------------------------------------
    print(f"\nRequesting {MAX_BARS} D1 bars for {len(ASSETS)} assets...")
    print(f"Assets: {', '.join(ASSETS)}")
    asset_candles = _try_live(ASSETS, MAX_BARS)

    if asset_candles:
        # Live data obtained
        source = "LIVE (IBKR)"
        n_live = len(next(iter(asset_candles.values())))
        # Fill any missing symbols with synthetic at the same bar count
        for sym in ASSETS:
            if sym not in asset_candles:
                print(f"  {sym}: live unavailable — using synthetic ({n_live} bars)")
                asset_candles[sym] = _synthetic(sym, n_live)
            else:
                actual = len(asset_candles[sym])
                first  = asset_candles[sym][0].timestamp.strftime("%Y-%m-%d")
                last   = asset_candles[sym][-1].timestamp.strftime("%Y-%m-%d")
                print(f"  {sym}: {actual} bars  {first} → {last}")
        n_bars = n_live
    else:
        # Fallback to synthetic
        source = "SYNTHETIC (deterministic, seed per asset) — IBKR unavailable"
        print("  Live unavailable — generating synthetic data for all assets")
        asset_candles = {sym: _synthetic(sym, MAX_BARS) for sym in ASSETS}
        for sym, c in asset_candles.items():
            print(f"  {sym}: {len(c)} bars  "
                  f"{c[0].timestamp.date()} → {c[-1].timestamp.date()}")
        n_bars = MAX_BARS

    print(f"\nData source: {source}")
    if source.startswith("SYNTHETIC"):
        print("WARNING: Synthetic data results are NOT official.")
        print("         Run with TWS open on port 7497 for official IBKR validation.")
    print(f"Config: $10k · 1% risk · ATR×2 stop · ATR×3 target · $1 commission · 0.05% slip")

    # ------------------------------------------------------------------
    # Step 2: Build overlays and run
    # ------------------------------------------------------------------
    strategy = MomentumRotationStrategy()

    overlays = [
        NoOverlay(),
        EquityCurvePause(),
        BearMarketPause(),
        MonthlyLossLockout(
            starting_balance=EDGE_CONFIG["backtest"]["starting_balance"],
        ),
        ConsecutiveLossCooldown(),
        VolatilityRiskScaling(),
        CombinedOverlay(),
    ]

    print(f"\nOverlays: {', '.join(o.name for o in overlays)}")
    print(f"Evaluating {len(overlays)} overlays × {len(asset_candles)} assets ...")
    print("(Each overlay: full backtest + robustness windows per asset)\n")

    engine = OverlayEngine(EDGE_CONFIG)
    report = engine.run(strategy, overlays, asset_candles)

    # ------------------------------------------------------------------
    # Step 3: Print report
    # ------------------------------------------------------------------
    _print_report(report, source, n_bars)

    # Log location
    print(f"Full log written to: logs/risk_overlay_validation.log\n")

    return 0 if report.promotion_exists else 1


if __name__ == "__main__":
    sys.exit(main())
