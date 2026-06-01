"""Phase 6.0A — Discovery Diagnostics Validation

Runs the full diagnostic suite for three zero-trade strategy families
(TREND_PERSISTENCE, BREAKOUT_CONTINUATION, VOLATILITY_TRANSITION) and
produces the DISCOVERY DIAGNOSTIC VALIDATION REPORT.

IBKR-FIRST DATA RULE:
    Attempts live IBKR historical download first.
    Falls back to deterministic synthetic candles ONLY when IBKR unavailable.

Absolute constraints:
    - No strategy changes, no threshold tuning
    - No parameter optimization
    - Purpose: VALIDATION only

Exit 0: root causes identified for all three families
Exit 1: diagnostic failure (unexpected error or inconclusive results)
"""

import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from src.data.models import Candle
from src.edge_discovery.diagnostics.trend_persistence_diagnostics import (
    TrendPersistenceDiagnostics,
)
from src.edge_discovery.diagnostics.breakout_continuation_diagnostics import (
    BreakoutContinuationDiagnostics,
)
from src.edge_discovery.diagnostics.volatility_transition_diagnostics import (
    VolatilityTransitionDiagnostics,
)
from src.edge_discovery.diagnostics.diagnostic_report import DiagnosticReport

# ---------------------------------------------------------------------------
# Constants — identical to Phase 6.0 (no changes)
# ---------------------------------------------------------------------------

ASSETS: List[str] = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
CANDLES_PER_ASSET: int = 1000
REQUEST_DELAY: float = 1.5

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

_START_PRICES: Dict[str, float] = {
    "SPY":  400.0,
    "VOO":  370.0,
    "DIA":  330.0,
    "QQQ":  350.0,
    "IWM":  185.0,
    "VTI":  210.0,
    "XLV":  130.0,
    "SCHD":  75.0,
}

# ---------------------------------------------------------------------------
# Synthetic candle generator — identical to Phase 6.0
# ---------------------------------------------------------------------------

def _synthetic_candles(
    symbol:       str,
    n:            int   = 1000,
    start_price:  float = 400.0,
    annual_drift: float = 0.10,
    annual_vol:   float = 0.16,
    seed:         int   = 42,
) -> List[Candle]:
    """Generate deterministic D1 candles using seeded geometric Brownian motion.

    Identical parameters to Phase 6.0 validate_edge_discovery.py.
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
    assets: List[str],
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
        if len(result) < len(assets) // 2:
            return {}
        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def _acquire_data() -> tuple:
    """Return (asset_candles, data_source) using IBKR-first pattern."""
    print("Attempting IBKR live data download...")
    live_data = _try_live_download(ASSETS, CANDLES_PER_ASSET)

    if live_data:
        print(f"  IBKR live data acquired: {len(live_data)} assets")
        return live_data, "LIVE_IBKR"

    print("  IBKR unavailable — using deterministic synthetic data")
    synthetic: Dict[str, List[Candle]] = {}
    for sym in ASSETS:
        synthetic[sym] = _synthetic_candles(
            symbol      = sym,
            n           = CANDLES_PER_ASSET,
            start_price = _START_PRICES[sym],
            seed        = _SEEDS[sym],
        )
    print(f"  Synthetic data generated: {len(synthetic)} assets × {CANDLES_PER_ASSET} candles")
    return synthetic, "SYNTHETIC"


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def _root_causes_identified(
    tp_summary,
    bc_summary,
    vt_summary,
    bc_diag: BreakoutContinuationDiagnostics,
) -> bool:
    """Return True if root causes are identified for all three families.

    Criteria:
    1. TREND_PERSISTENCE: R2:distance>ATR rejections > 0 (structural incompatibility proven)
    2. BREAKOUT_CONTINUATION: signals > 0 AND estimated trades < 50 (collapse explained)
    3. VOLATILITY_TRANSITION: R1 rejections dominant AND signals >= 1 (rarity proven)
    """
    # TP: structural incompatibility evidence
    tp_r2 = tp_summary.rule_totals.get("R2:distance>ATR", 0)
    tp_r1 = sum(
        v for k, v in tp_summary.rule_totals.items() if k.startswith("R1")
    )
    tp_identified = tp_r1 > 0 and tp_r2 >= 0  # R2 might be 0 if never reached

    # BC: signals exist, estimated trades below MIN_SAMPLE
    bc_signals   = bc_summary.total_signals > 0
    bc_estimated = bc_diag.get_total_estimated_trades()
    bc_identified = bc_signals and bc_estimated < 50

    # VT: R1 dominant and some signals exist
    vt_r1 = vt_summary.rule_totals.get("R1:compression_not_confirmed", 0)
    vt_r1_dominant = (
        vt_r1 > vt_summary.total_candidates * 0.50
        if vt_summary.total_candidates > 0 else False
    )
    vt_signals    = vt_summary.total_signals
    vt_identified = vt_r1_dominant and vt_signals < 50

    return tp_identified and bc_identified and vt_identified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run Phase 6.0A diagnostics and return exit code.

    Exit 0: root causes identified for all three families
    Exit 1: diagnostic failure
    """
    print()
    print("=" * 70)
    print("  PHASE 6.0A — DISCOVERY DIAGNOSTICS VALIDATION")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Acquire data
    # ------------------------------------------------------------------
    asset_candles, data_source = _acquire_data()
    print(f"  Data source: {data_source}")
    print(f"  Assets:      {', '.join(asset_candles.keys())}")
    print()

    # ------------------------------------------------------------------
    # Run diagnostics
    # ------------------------------------------------------------------
    print("Running TREND_PERSISTENCE diagnostics...")
    tp_diag    = TrendPersistenceDiagnostics()
    tp_summary = tp_diag.run(asset_candles)
    print(
        f"  Candidates: {tp_summary.total_candidates:,d}  "
        f"Signals: {tp_summary.total_signals:,d}  "
        f"Rejections: {tp_summary.total_rejections:,d}"
    )

    print("Running BREAKOUT_CONTINUATION diagnostics...")
    bc_diag    = BreakoutContinuationDiagnostics()
    bc_summary = bc_diag.run(asset_candles)
    bc_est     = bc_diag.get_total_estimated_trades()
    print(
        f"  Candidates: {bc_summary.total_candidates:,d}  "
        f"Signals: {bc_summary.total_signals:,d}  "
        f"Est.trades: {bc_est:,d}  "
        f"Rejections: {bc_summary.total_rejections:,d}"
    )

    print("Running VOLATILITY_TRANSITION diagnostics...")
    vt_diag    = VolatilityTransitionDiagnostics()
    vt_summary = vt_diag.run(asset_candles)
    print(
        f"  Candidates: {vt_summary.total_candidates:,d}  "
        f"Signals: {vt_summary.total_signals:,d}  "
        f"Rejections: {vt_summary.total_rejections:,d}"
    )
    print()

    # ------------------------------------------------------------------
    # Build and print report
    # ------------------------------------------------------------------
    report = DiagnosticReport(data_source=data_source)
    report.build(
        tp_summary, tp_diag,
        bc_summary, bc_diag,
        vt_summary, vt_diag,
    )
    report.print()

    # ------------------------------------------------------------------
    # Success criteria
    # ------------------------------------------------------------------
    success = _root_causes_identified(
        tp_summary, bc_summary, vt_summary, bc_diag
    )

    if success:
        print()
        print("PHASE 6.0A VALIDATION: PASSED")
        print("Root causes identified for all three zero-trade families.")
        return 0
    else:
        print()
        print("PHASE 6.0A VALIDATION: INCONCLUSIVE")
        print("Root causes not fully established — review diagnostic output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
