"""Phase 6.0 — Edge Discovery Validation

Discovers 6 NEW strategy families (NOT Momentum Rotation variations):
    1. TREND_PERSISTENCE
    2. BREAKOUT_CONTINUATION
    3. RELATIVE_STRENGTH
    4. MARKET_LEADERSHIP
    5. VOLATILITY_TRANSITION
    6. MULTI_TIMEFRAME_ALIGNMENT

IBKR-FIRST DATA RULE:
    Attempts live IBKR historical download first.
    Falls back to deterministic synthetic candles ONLY when IBKR unavailable.
    No promotion decisions may be made from synthetic data results.

Absolute constraints:
    - No execution, no paper trading, no live trading
    - No ML, no parameter sweeps, no optimization
    - No curve fitting

Exit 0: discovery winner found OR at least one promising candidate
Exit 1: all families insufficient sample / no edge detected
"""

import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from src.data.models import Candle
from src.edge_discovery.discovery_engine import DiscoveryEngine
from src.edge_discovery.discovery_report import DiscoveryReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSETS: List[str] = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
CANDLES_PER_ASSET: int = 1000
REQUEST_DELAY: float = 1.5  # seconds between live asset requests

# Deterministic seeds — one per asset, consistent across all phases
_SEEDS: Dict[str, int] = {
    "SPY":  42,
    "VOO":  99,
    "DIA":  17,
    "QQQ":   7,
    "IWM":  33,
    "VTI":  55,
    "XLV":  71,
    "SCHD": 88,
}

# Start prices matching each ETF's approximate price level
_START_PRICES: Dict[str, float] = {
    "SPY":  400.0,
    "VOO":  370.0,
    "DIA":  330.0,
    "QQQ":  350.0,
    "IWM":  185.0,
    "VTI":  210.0,
    "XLV":  130.0,
    "SCHD": 75.0,
}

# ---------------------------------------------------------------------------
# Shared engine config — identical to prior phases (no changes allowed)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Synthetic candle generator — deterministic, seeded per symbol
# ---------------------------------------------------------------------------

def _synthetic_candles(
    symbol:       str,
    n:            int   = 1000,
    start_price:  float = 400.0,
    annual_drift: float = 0.10,
    annual_vol:   float = 0.16,
    seed:         int   = 42,
) -> List[Candle]:
    """Generate realistic D1 candles using seeded geometric Brownian motion.

    Parameters produce ~10% annual return with ~16% volatility — similar
    to SPY over the 2010-2023 period.  Identical parameters across all phases.
    """
    rng         = random.Random(seed)
    daily_drift = annual_drift / 252
    daily_vol   = annual_vol / (252 ** 0.5)
    price       = start_price
    base_dt     = datetime(2020, 1, 2, tzinfo=timezone.utc)
    candles: List[Candle] = []

    for i in range(n):
        ret   = daily_drift + rng.gauss(0, daily_vol)
        price = max(1.0, price * (1 + ret))

        intra_vol = price * daily_vol * 0.7
        open_     = price * (1 + rng.gauss(0, daily_vol * 0.3))
        high      = max(open_, price) + abs(rng.gauss(0, intra_vol))
        low       = min(open_, price) - abs(rng.gauss(0, intra_vol))
        volume    = rng.uniform(40e6, 180e6)

        candles.append(Candle(
            symbol    = symbol,
            timeframe = "D1",
            timestamp = base_dt + timedelta(days=i),
            open      = round(open_, 4),
            high      = round(high,  4),
            low       = round(low,   4),
            close     = round(price, 4),
            volume    = volume,
            provider  = "SYNTHETIC",
        ))

    return candles

# ---------------------------------------------------------------------------
# IBKR-first live download
# ---------------------------------------------------------------------------

def _try_live_download(
    assets:           List[str],
    candles_per_asset: int,
) -> Dict[str, List[Candle]]:
    """Attempt live IBKR download; return {} on any failure."""
    try:
        from src.config.settings_loader import load_settings
        from src.data.symbol_registry import SymbolRegistry
        from src.providers.provider_factory import ProviderFactory

        config   = load_settings()
        registry = SymbolRegistry.from_config(config)
        provider = ProviderFactory.create(config, registry)

        if not provider.connect():
            return {}

        result: Dict[str, List[Candle]] = {}
        for sym in assets:
            candles = provider.get_candles(sym, "D1", candles_per_asset)
            if candles:
                result[sym] = candles
            time.sleep(REQUEST_DELAY)

        provider.disconnect()
        return result

    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nPhase 6.0 — Edge Discovery")
    print("=" * 54)

    # -- Data ---------------------------------------------------------------
    print(
        f"\nAttempting live data download ({', '.join(ASSETS)})...",
        end=" ",
        flush=True,
    )
    asset_candles = _try_live_download(ASSETS, CANDLES_PER_ASSET)

    if asset_candles:
        data_source = "LIVE (IBKR)"
        first       = next(iter(asset_candles.values()))
        print(f"OK  ({len(first)} bars per asset)")
    else:
        print("unavailable — using synthetic data")
        data_source = "SYNTHETIC"
        asset_candles = {
            sym: _synthetic_candles(
                symbol      = sym,
                n           = CANDLES_PER_ASSET,
                start_price = _START_PRICES[sym],
                seed        = _SEEDS[sym],
            )
            for sym in ASSETS
        }

    print(f"Data source  : {data_source}")
    print(f"Assets       : {', '.join(asset_candles.keys())}")
    first_sym = next(iter(asset_candles.values()))
    print(f"Bars/asset   : {len(first_sym)}")
    print()

    # -- Discovery engine ---------------------------------------------------
    print("Running discovery engine (6 families + benchmark)...")
    print()

    engine = DiscoveryEngine(
        config        = EDGE_CONFIG,
        asset_candles = asset_candles,
        data_source   = data_source,
    )
    report = engine.run()

    # -- Report -------------------------------------------------------------
    discovery_report = DiscoveryReport(
        comparison  = report,
        data_source = data_source,
        asset_names = list(asset_candles.keys()),
    )
    discovery_report.print()

    # -- Exit code ----------------------------------------------------------
    has_winner     = report.discovery_winner is not None and report.discovery_winner.edge_score > 0
    has_promising  = len(report.promising_candidates) > 0

    if has_winner or has_promising:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
