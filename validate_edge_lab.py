#!/usr/bin/env python3
"""Phase 5.0 — Entry Edge Research Lab Validation Script.

Runs all five strategy candidates on an identical dataset and produces
a ranked comparison report with promotion recommendation.

Data source (in order of preference):
  1. Live IBKR connection  (if available)
  2. Synthetic deterministic candles  (fallback — always produces output)

Assets : SPY, QQQ, VOO  (same for every strategy)
History: 750 D1 candles per asset  (~3 years)

Every strategy uses:
  • Same risk engine      (ATR×2 stop, ATR×3 target, 1% risk)
  • Same position sizing
  • Same commission ($1) and slippage (0.05%)
  • Same backtester

Only the ENTRY LOGIC differs.
"""

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from src.data.models import Candle
from src.edge_lab.edge_engine import EdgeEngine
from src.edge_lab.edge_ranking import EdgeRanking
from src.edge_lab.edge_report import generate_report
from src.edge_lab.edge_registry import build_all_strategies


# ---------------------------------------------------------------------------
# Deterministic synthetic candle generator
# ---------------------------------------------------------------------------

def _synthetic_candles(
    symbol:        str,
    n:             int   = 750,
    start_price:   float = 400.0,
    annual_drift:  float = 0.10,
    annual_vol:    float = 0.16,
    seed:          int   = 42,
) -> List[Candle]:
    """Generate realistic D1 candles using a seeded geometric Brownian motion.

    Parameters produce ~10% annual return with ~16% volatility — similar
    to SPY over the 2010-2023 period.
    """
    rng          = random.Random(seed)
    daily_drift  = annual_drift / 252
    daily_vol    = annual_vol   / (252 ** 0.5)
    price        = start_price
    base_dt      = datetime(2020, 1, 2, tzinfo=timezone.utc)
    candles      = []

    for i in range(n):
        ret       = daily_drift + rng.gauss(0, daily_vol)
        price     = max(1.0, price * (1 + ret))

        # Intraday range: ~70% of ATR driven by daily vol
        intra_vol  = price * daily_vol * 0.7
        open_      = price * (1 + rng.gauss(0, daily_vol * 0.3))
        high       = max(open_, price) + abs(rng.gauss(0, intra_vol))
        low        = min(open_, price) - abs(rng.gauss(0, intra_vol))
        volume     = rng.uniform(40e6, 180e6)

        candles.append(Candle(
            symbol    = symbol,
            timeframe = "D1",
            timestamp = base_dt + timedelta(days=i),
            open      = round(open_, 4),
            high      = round(high, 4),
            low       = round(low, 4),
            close     = round(price, 4),
            volume    = volume,
            provider  = "SYNTHETIC",
        ))
    return candles


def _try_live_download(assets: List[str], candles_per_asset: int) -> Dict[str, List[Candle]]:
    """Attempt live download; return {} on any failure."""
    try:
        from src.config.settings_loader import load_settings
        from src.data.symbol_registry import SymbolRegistry
        from src.providers.provider_factory import ProviderFactory

        config   = load_settings()
        registry = SymbolRegistry.from_config(config)
        provider = ProviderFactory.create(config, registry)

        if not provider.connect():
            return {}

        result = {}
        for sym in assets:
            candles = provider.get_candles(sym, "D1", candles_per_asset)
            if candles:
                result[sym] = candles
        provider.disconnect()
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Shared engine config
# ---------------------------------------------------------------------------

EDGE_CONFIG = {
    "backtest": {
        "starting_balance":    10_000.0,
        "commission_per_trade": 1.0,
        "slippage_percent":     0.05,
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

ASSETS           = ["SPY", "QQQ", "VOO"]
CANDLES_PER_ASSET = 750


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nPhase 5.0 — Entry Edge Research Lab")
    print("=" * 64)

    # -- Data ---------------------------------------------------------------
    print(f"\nAttempting live data download ({', '.join(ASSETS)})...", end=" ", flush=True)
    asset_candles = _try_live_download(ASSETS, CANDLES_PER_ASSET)

    if asset_candles:
        source = "LIVE (IBKR)"
        first  = next(iter(asset_candles.values()))
        print(f"OK  ({len(first)} bars per asset)")
    else:
        print("unavailable — using synthetic data")
        source = "SYNTHETIC (deterministic, seed=42)"
        seeds  = {"SPY": 42, "QQQ": 7, "VOO": 99}
        asset_candles = {
            sym: _synthetic_candles(
                sym,
                n=CANDLES_PER_ASSET,
                start_price={"SPY": 400.0, "QQQ": 350.0, "VOO": 380.0}[sym],
                seed=seeds[sym],
            )
            for sym in ASSETS
        }

    print(f"Data source : {source}")
    print(f"Assets      : {', '.join(asset_candles)}")
    print(f"Candles/asset: {next(len(c) for c in asset_candles.values())}")
    print(f"Config      : $10k account · 1% risk/trade · ATR×2 stop · ATR×3 target")
    print(f"Commission  : $1/side  Slippage: 0.05%")

    # -- Build strategies and engine ----------------------------------------
    strategies = build_all_strategies()
    engine     = EdgeEngine(EDGE_CONFIG)
    ranker     = EdgeRanking()

    print(f"\nRunning {len(strategies)} strategies × {len(asset_candles)} assets...")
    print("(each strategy uses identical risk engine, sizing, and data)\n")

    # -- Evaluate -----------------------------------------------------------
    profiles = engine.evaluate_all(strategies, asset_candles)
    ranked   = ranker.rank(profiles)

    # -- Report -------------------------------------------------------------
    report = generate_report(ranked, benchmark_name="BREAKOUT_V1")
    print(report)

    # Exit code: 0 if a promotion candidate exists
    return 0 if any(p.promotion_candidate for p in ranked) else 1


if __name__ == "__main__":
    sys.exit(main())
