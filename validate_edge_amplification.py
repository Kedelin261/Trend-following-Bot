#!/usr/bin/env python3
"""Phase 5.3 — Edge Amplification Validation Script.

Answers 10 research questions about MOMENTUM_ROTATION:

  Q1.  Which assets generate most profits?
  Q2.  Which assets generate most losses?
  Q3.  Which market regimes create edge?
  Q4.  Which market regimes destroy edge?
  Q5.  Which volatility environments help?
  Q6.  Which volatility environments hurt?
  Q7.  Which holding periods produce profits?
  Q8.  Which holding periods produce losses?
  Q9.  Which trade quality scores perform best?
  Q10. Can weak segments be removed while maintaining Trades >= 500?

Data source (priority order):
  1. LIVE  — IBKR historical data via configured provider (settings.yaml)
  2. SYNTHETIC — deterministic fallback when IBKR unavailable

Research only. No execution. No broker code. No live trading.
No entry logic modifications. No signal-engine changes.
"""

import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from src.data.models import Candle
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from src.edge_amplification.amplification_research_engine import AmplificationResearchEngine
from src.edge_amplification.amplification_report import AmplificationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — identical to Phase 5.1 / 5.2 for strict comparability
# ---------------------------------------------------------------------------

ASSETS        = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
MAX_BARS      = 5000
REQUEST_DELAY = 1.5

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
# Synthetic data — identical generator to Phase 5.1 / 5.2
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
    """Generate deterministic synthetic OHLCV data — same seed/params as Phase 5.1/5.2."""
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
# Live data — IBKR via existing provider architecture
# ---------------------------------------------------------------------------

def _try_live(assets: List[str], n: int) -> Dict[str, List[Candle]]:
    """Attempt to fetch live D1 bars from the configured provider (IBKR).

    Returns {symbol: candles} on success.
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
            logger.warning("validate_edge_amplification: provider connect failed — using synthetic")
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
        logger.warning("validate_edge_amplification: live fetch failed (%s) — using synthetic", exc)
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
            logging.FileHandler("logs/edge_amplification.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _setup_logging()

    print("\nPhase 5.3 — Edge Amplification Research")
    print("=" * 66)
    print("Strategy: MOMENTUM_ROTATION")
    print(f"Assets:   {', '.join(ASSETS)}")
    print(f"Config:   $10k · 1% risk · ATR×2 stop · ATR×3 target · "
          f"$1 commission · 0.05% slip")
    print()

    # ── Step 1: Data acquisition ──────────────────────────────────────────
    print(f"Requesting {MAX_BARS} D1 bars for {len(ASSETS)} assets...")
    asset_candles = _try_live(ASSETS, MAX_BARS)

    if asset_candles:
        source = "LIVE (IBKR)"
        n_live = len(next(iter(asset_candles.values())))
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

    # ── Step 2: Run amplification research ────────────────────────────────
    strategy = MomentumRotationStrategy()
    engine   = AmplificationResearchEngine(EDGE_CONFIG)

    print(f"\nRunning amplification research across {len(asset_candles)} assets...")
    print("  Module 1: Asset contribution analysis")
    print("  Module 2: Regime contribution analysis")
    print("  Module 3: Volatility contribution analysis")
    print("  Module 4: Trade quality analysis")
    print("  Module 5: Holding period analysis")
    print("  Module 6: Profit concentration analysis")
    print("  Module 7: Loss concentration analysis")
    print("  Deriving amplification candidates...")
    print()

    result = engine.run(strategy, asset_candles, data_source=source)

    # ── Step 3: Print full validation report ─────────────────────────────
    report = AmplificationReport(result)
    report.print()

    print(f"Full log written to: logs/edge_amplification.log\n")

    return 0 if result.recommendation == "PROCEED TO PHASE 5.4" else 1


if __name__ == "__main__":
    sys.exit(main())
