"""Phase 6.1 — Edge Scalability Validation

IBKR-FIRST DATA RULE:
    Attempts live IBKR historical download first.
    Falls back to deterministic synthetic candles ONLY when unavailable.
    Synthetic fallback clearly labelled. No promotion allowed from synthetic.

Candidates:
    MULTI_TIMEFRAME_ALIGNMENT  (Phase 6.0 winner)
    MARKET_LEADERSHIP
    RELATIVE_STRENGTH

Benchmark:
    MOMENTUM_ROTATION

Bar horizons:
    1000 / 3000 / 5000 bars

Gate criteria (at least one candidate must satisfy ALL):
    PF >= 1.20
    Expectancy > 0
    Trades >= 100

Exit 0: gate passes
Exit 1: gate fails — stop pipeline
"""

import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from src.data.models import Candle
from src.phase_6.scalability_validator import ScalabilityValidator
from src.phase_6.phase_6_1_report import Phase61Report

# ---------------------------------------------------------------------------
# Constants — identical to Phase 6.0 (no changes)
# ---------------------------------------------------------------------------

ASSETS: List[str] = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
CANDLES_PER_ASSET: int = 5000    # Phase 6.1 requires up to 5000 bars
REQUEST_DELAY:     float = 1.5

_SEEDS: Dict[str, int] = {
    "SPY":  42,  "VOO":  99,  "DIA":  17,  "QQQ":   7,
    "IWM":  33,  "VTI":  55,  "XLV":  71,  "SCHD":  88,
}
_START_PRICES: Dict[str, float] = {
    "SPY":  400.0,  "VOO":  370.0,  "DIA":  330.0,  "QQQ":  350.0,
    "IWM":  185.0,  "VTI":  210.0,  "XLV":  130.0,  "SCHD":   75.0,
}

EDGE_CONFIG = {
    "backtest": {
        "starting_balance":      10_000.0,
        "commission_per_trade":   1.0,
        "slippage_percent":       0.05,
    },
    "risk": {
        "risk_per_trade_percent":  1.0,
        "atr_period":              14,
        "atr_stop_multiplier":     2.0,
        "atr_target_multiplier":   3.0,
        "minimum_signal_score":    40.0,
        "minimum_risk_reward":     1.0,
    },
}

# ---------------------------------------------------------------------------
# Synthetic candle generator — identical to Phase 6.0
# ---------------------------------------------------------------------------

def _synthetic_candles(
    symbol:       str,
    n:            int   = 5000,
    start_price:  float = 400.0,
    annual_drift: float = 0.10,
    annual_vol:   float = 0.16,
    seed:         int   = 42,
) -> List[Candle]:
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
        candles.append(Candle(
            symbol    = symbol,
            timeframe = "D1",
            timestamp = base_dt + timedelta(days=i),
            open      = round(open_, 4),
            high      = round(high,  4),
            low       = round(low,   4),
            close     = round(price, 4),
            volume    = rng.uniform(40e6, 180e6),
            provider  = "SYNTHETIC",
        ))
    return candles

# ---------------------------------------------------------------------------
# IBKR-first live download
# ---------------------------------------------------------------------------

def _try_live_download(
    assets: List[str],
    n:      int,
) -> Dict[str, List[Candle]]:
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
            candles = provider.get_candles(sym, "D1", n)
            if candles:
                result[sym] = candles
            time.sleep(REQUEST_DELAY)

        provider.disconnect()
        if len(result) < len(assets) // 2:
            return {}
        return result
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def _acquire_data() -> tuple:
    print("Attempting IBKR live data download...")
    live = _try_live_download(ASSETS, CANDLES_PER_ASSET)
    if live:
        print(f"  IBKR live data: {len(live)} assets")
        return live, "LIVE_IBKR"

    print("  IBKR unavailable — using deterministic synthetic data")
    synthetic = {
        sym: _synthetic_candles(
            symbol=sym, n=CANDLES_PER_ASSET,
            start_price=_START_PRICES[sym], seed=_SEEDS[sym],
        )
        for sym in ASSETS
    }
    print(f"  Synthetic: {len(synthetic)} assets × {CANDLES_PER_ASSET} bars")
    return synthetic, "SYNTHETIC"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 70)
    print("  PHASE 6.1 — EDGE SCALABILITY VALIDATION")
    print("=" * 70)
    print()

    asset_candles, data_source = _acquire_data()

    print()
    print("Running scalability validation across 3 bar horizons...")
    print("  Candidates: MULTI_TIMEFRAME_ALIGNMENT, MARKET_LEADERSHIP, RELATIVE_STRENGTH")
    print("  Benchmark:  MOMENTUM_ROTATION")
    print("  Horizons:   1000 / 3000 / 5000 bars")
    print("  Assets:     8 ETFs")
    print()

    validator = ScalabilityValidator(
        config=EDGE_CONFIG,
        asset_candles=asset_candles,
        data_source=data_source,
    )
    report = validator.run()

    # Print formatted report
    Phase61Report(report).print()

    if report.gate_passes:
        print()
        print("PHASE 6.1: GATE PASSED")
        print("Proceeding to Phase 6.2.")
        return 0
    else:
        print()
        print("PHASE 6.1: GATE FAILED")
        print("STOP. Return to Discovery.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
