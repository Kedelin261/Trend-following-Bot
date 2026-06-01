#!/usr/bin/env python3
"""Phase 5.4 — Edge Amplification Validation Script.

Validates the Phase 5.3 edge-destroyer findings by running 5 scenarios
against the MOMENTUM_ROTATION strategy:

  Scenario 1: BASELINE           — no changes
  Scenario 2: REMOVE_SCHD        — exclude SCHD (Phase 5.3: PF=0.95, Exp=-$2.52)
  Scenario 3: REMOVE_QUALITY_60_69 — reject signal score 60-69 (PF=0.91, NO EDGE)
  Scenario 4: REMOVE_HIGH_VOL    — reject HIGH_VOL trades (PF=0.93, Exp=-$3.11)
  Scenario 5: COMBINED_FILTERS   — all three filters applied

FAIRNESS GUARANTEE (identical across all scenarios):
  - Same 8 assets (SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD)
  - Same candles (live IBKR D1 bars, or deterministic synthetic fallback)
  - Same RiskEngine.from_config(EDGE_CONFIG)
  - Same TradeSimulator (same slippage, same commission)
  - Same Portfolio (same starting balance)
  - Same warmup (strategy.min_candles = 55)
  - Same EdgeStabilityAnalyzer (3 windows, 60% of history)
  - Only the filter gate differs

DATA SOURCE RULE:
  1. LIVE_IBKR — preferred, uses configured provider (settings.yaml)
  2. SYNTHETIC — fallback ONLY when IBKR unavailable
  If synthetic: report is returned but promotion decisions require IBKR.

ANTI-CURVE-FITTING:
  - No threshold tuning
  - No quality score optimization
  - No volatility threshold optimization
  - Only validates Phase 5.3 findings: SCHD, score 60-69, HIGH_VOL

PROMOTION CRITERIA (same as Phase 5.2):
  Trades >= 500, PF >= 1.50, Exp > $0, Max DD < 15%, Robustness = ROBUST

Research only. No execution. No broker code. No live trading.
No entry logic modifications. No signal engine changes.
"""

import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.data.models import Candle
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from src.amplification_validation.filter_profiles import ALL_SCENARIOS
from src.amplification_validation.amplification_validator import AmplificationValidator
from src.amplification_validation.amplification_comparator import AmplificationComparator
from src.amplification_validation.amplification_report import AmplificationValidationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — identical to Phase 5.1 / 5.2 / 5.3 for strict comparability
# ---------------------------------------------------------------------------

ASSETS        = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
MAX_BARS      = 5000
REQUEST_DELAY = 1.5   # seconds between IBKR candle requests (rate limiting)

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
        # Matches Phase 5.1–5.3 thresholds (not settings.yaml defaults)
        "minimum_signal_score":    40.0,
        "minimum_risk_reward":     1.0,
    },
}

# ---------------------------------------------------------------------------
# Synthetic data — identical generator to Phase 5.1 / 5.2 / 5.3
# ---------------------------------------------------------------------------

_START_PRICES = {
    "SPY": 400.0, "VOO": 380.0, "QQQ": 350.0, "DIA": 330.0,
    "IWM": 220.0, "VTI": 210.0, "XLV": 130.0, "SCHD":  75.0,
}
_SEEDS = {
    "SPY": 42, "VOO": 99, "QQQ": 7, "DIA": 13,
    "IWM": 21, "VTI": 55, "XLV": 63, "SCHD": 88,
}


def _synthetic(symbol: str, n: int = 5000) -> List[Candle]:
    """Generate deterministic synthetic OHLCV data — same seed/params as Phase 5.1–5.3."""
    rng         = random.Random(_SEEDS.get(symbol, 42))
    price       = _START_PRICES.get(symbol, 100.0)
    base        = datetime(2016, 1, 4, tzinfo=timezone.utc)
    out         = []
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

    No orders placed. Read-only throughout.
    """
    try:
        from src.config.settings_loader import load_settings
        from src.data.symbol_registry import SymbolRegistry
        from src.providers.provider_factory import ProviderFactory

        config   = load_settings()
        registry = SymbolRegistry.from_config(config)
        provider = ProviderFactory.create(config, registry)

        if not provider.connect():
            logger.warning(
                "validate_amplification_filters: provider connect failed — using synthetic"
            )
            return {}

        result: Dict[str, List[Candle]] = {}
        try:
            for sym in assets:
                candles = provider.get_candles(sym, "D1", n)
                if candles:
                    result[sym] = candles
                    logger.info("  %s: %d live bars fetched", sym, len(candles))
                else:
                    logger.warning("  %s: no data returned", sym)
                time.sleep(REQUEST_DELAY)
        finally:
            provider.disconnect()

        if len(result) < len(assets):
            logger.warning(
                "validate_amplification_filters: only %d/%d assets fetched",
                len(result), len(assets),
            )
            if not result:
                return {}

        return result

    except Exception as exc:
        logger.error(
            "validate_amplification_filters: live data error — %s", exc
        )
        return {}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure logging: file handler + stdout handler."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "amplification_validation.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    # Stdout handler for the root logger — show WARNING+ on console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.WARNING)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logger.info("validate_amplification_filters: logging started → %s", log_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run Phase 5.4 validation.  Returns exit code: 0=proceed, 1=return."""
    _setup_logging()

    print()
    print("Phase 5.4 — Edge Amplification Validation")
    print("=" * 66)
    print("Strategy: MOMENTUM_ROTATION")
    print(f"Assets:   {', '.join(ASSETS)}")
    print(
        "Config:   $10k · 1% risk · ATR×2 stop · ATR×3 target "
        "· $1 commission · 0.05% slip"
    )
    print()

    # ------------------------------------------------------------------
    # Step 1: Data acquisition — IBKR first, synthetic fallback
    # ------------------------------------------------------------------
    print(f"Requesting {MAX_BARS} D1 bars for {len(ASSETS)} assets...")

    asset_candles = _try_live(ASSETS, MAX_BARS)
    data_source: str

    if asset_candles:
        data_source = "LIVE_IBKR"
        print(f"  Data source: LIVE IBKR")
        for sym, c in asset_candles.items():
            first = c[0].timestamp.strftime("%Y-%m-%d") if c else "?"
            last  = c[-1].timestamp.strftime("%Y-%m-%d") if c else "?"
            print(f"  {sym}: {len(c)} bars  {first} → {last}")
    else:
        data_source = "SYNTHETIC"
        print("  Live unavailable — generating synthetic data for all assets")
        asset_candles = {}
        for sym in ASSETS:
            bars = _synthetic(sym, MAX_BARS)
            asset_candles[sym] = bars
            first = bars[0].timestamp.strftime("%Y-%m-%d")
            last  = bars[-1].timestamp.strftime("%Y-%m-%d")
            print(f"  {sym}: {len(bars)} bars  {first} → {last}")

    print()
    print(f"Data source: {data_source}", end="")
    if data_source == "SYNTHETIC":
        print(" (deterministic, seed per asset) — IBKR unavailable")
        print("WARNING: Synthetic data results are NOT official.")
        print("         Run with TWS open on port 7497 for official IBKR validation.")
    else:
        print()
    print()

    # ------------------------------------------------------------------
    # Step 2: Run all 5 scenarios
    # ------------------------------------------------------------------
    strategy  = MomentumRotationStrategy()
    validator = AmplificationValidator(EDGE_CONFIG, data_source=data_source)

    print("Running 5 validation scenarios...")
    print()

    scenario_results = []
    for profile in ALL_SCENARIOS:
        print(f"  Running scenario: {profile.name} ...")
        result = validator.run_scenario(strategy, profile, asset_candles)
        scenario_results.append(result)
        print(
            f"    → Trades={result.trades}  PF={result.pf_str}  "
            f"Exp=${result.expectancy:+.2f}  DD={result.max_drawdown:.1f}%  "
            f"Rob={result.robustness}"
        )

    print()

    # ------------------------------------------------------------------
    # Step 3: Compare & rank
    # ------------------------------------------------------------------
    comparator = AmplificationComparator(scenario_results)
    comparison = comparator.compare()

    # ------------------------------------------------------------------
    # Step 4: Print full validation report
    # ------------------------------------------------------------------
    report = AmplificationValidationReport(comparison)
    report.print()

    print(f"Full log written to: logs/amplification_validation.log")
    print()

    # ------------------------------------------------------------------
    # Step 5: Determine exit code
    # ------------------------------------------------------------------
    if comparison.recommendation == "PROCEED TO PHASE 5.5":
        return 0
    else:
        # Did Phase 5.3 findings hold?  If so, exit 0 (research validated,
        # but promotion criteria not yet met — needs more investigation).
        # If no findings held at all, exit 1.
        if comparison.phase53_findings_held:
            # Findings held but no promotion candidate — exit 0 to signal
            # the research was valid, while report makes recommendation clear
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
