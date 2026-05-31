#!/usr/bin/env python3
"""Phase 5.1 — Edge Scalability Validation Script.

Tests VOLATILITY_EXPANSION and MOMENTUM_ROTATION across:
  • 8 assets  (SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD)
  • 5 history windows  (750 → 1500 → 3000 → 4000 → 5000 bars)
  • 3 robustness windows  (early / middle / recent)

Promotion requires: Trades ≥ 100, PF ≥ 1.50, Exp > 0,
                    DD < 15%, Robustness ≠ UNSTABLE,
                    Assets Passing ≥ 2, History Tested ≥ 3000 bars.
"""

import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from src.data.models import Candle
from src.edge_validation.validation_engine import ValidationEngine
from src.signals.strategies.volatility_expansion_strategy import VolatilityExpansionStrategy
from src.signals.strategies.momentum_rotation_strategy import MomentumRotationStrategy


ASSETS         = ["SPY", "VOO", "DIA", "QQQ", "IWM", "VTI", "XLV", "SCHD"]
MAX_BARS       = 5000
REQUEST_DELAY  = 1.5

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
# Synthetic data (same generator as Phase 5.0 for reproducibility)
# ---------------------------------------------------------------------------
_START_PRICES = {
    "SPY": 400.0, "VOO": 380.0, "QQQ": 350.0, "DIA": 330.0,
    "IWM": 220.0, "VTI": 210.0, "XLV": 130.0, "SCHD": 75.0,
}
_SEEDS = {"SPY": 42, "VOO": 99, "QQQ": 7, "DIA": 13,
          "IWM": 21, "VTI": 55, "XLV": 63, "SCHD": 88}


def _synthetic(symbol: str, n: int = 5000) -> List[Candle]:
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


def _try_live(assets: List[str], n: int) -> Dict[str, List[Candle]]:
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
            candles = provider.get_candles(sym, "D1", n)
            if candles:
                result[sym] = candles
            time.sleep(REQUEST_DELAY)
        provider.disconnect()
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _pf(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def print_report(report) -> None:
    sep = "=" * 66

    print(f"\n{sep}")
    print("  EDGE SCALABILITY VALIDATION REPORT  —  Phase 5.1")
    print(sep)

    for r in report.strategy_results:
        p    = r.promotion
        rob  = r.robustness
        scal = r.scalability
        surv = r.survivability

        print(f"\n  {'─'*62}")
        print(f"  {r.name}")
        print(f"  {r.strategy.description}")
        print()
        print(f"  Trades:        {p.trades}")
        print(f"  PF:            {p.pf_str}")
        print(f"  Expectancy:    ${p.expectancy:+.2f}/trade")
        print(f"  Max Drawdown:  {p.drawdown:.1f}%")
        print(f"  Robustness:    {p.robustness}")
        print(f"  Survivability: {surv.verdict}")
        print(f"  Assets OK:     {p.assets_passing}")
        print(f"  History Bars:  {p.history_tested:,}")
        print()

        # History stability
        print(f"  HISTORY STABILITY:")
        for s in r.history_slices:
            safe = s.pf if not math.isinf(s.pf) else 99.0
            tag  = ""
            bars_label = f"{s.actual_bars:,} bars"
            print(f"    {bars_label:<16} Trades={s.trades:<5} PF={_pf(s.pf):<6} "
                  f"Exp=${s.expectancy:+.2f}  DD={s.drawdown:.1f}%  "
                  f"Rob={s.robustness}")

        # Robustness windows
        print(f"\n  ROBUSTNESS WINDOWS ({rob.rating}):")
        for w in rob.windows:
            ok = "✓" if w.passes else "✗"
            print(f"    {w.name:<8}  trades={w.trades:<4} PF={_pf(w.pf):<6} "
                  f"Exp=${w.exp:+.2f}  {ok}")

        # Asset contribution
        print(f"\n  ASSET CONTRIBUTION:")
        for ac in r.asset_expansion.asset_contributions:
            add = "✓ adds value" if ac.adds_value else "✗ degrades"
            print(f"    {ac.symbol:<6}  trades={ac.trades:<4} PF={ac.pf_str:<6} "
                  f"Exp=${ac.expectancy:+.2f}  {add}")

        # Scalability
        print(f"\n  SCALABILITY: {'SCALABLE' if scal.is_scalable else 'NOT SCALABLE'}")
        print(f"    {scal.summary}")

        # Promotion gate
        print(f"\n  PROMOTION GATE:")
        for c in p.pass_criteria:
            print(f"    ✓ {c}")
        for c in p.fail_criteria:
            print(f"    ✗ {c}")
        print(f"\n  STATUS: {p.status}")

    # Rankings and recommendation
    print(f"\n{sep}")
    print("  FINAL PROMOTION CANDIDATE")
    print(sep)

    if report.best_candidate:
        c = report.best_candidate
        print(f"\n  Name:           {c.strategy_name}")
        print(f"  Trades:         {c.trades}")
        print(f"  PF:             {c.pf_str}")
        print(f"  Expectancy:     ${c.expectancy:+.2f}/trade")
        print(f"  Drawdown:       {c.drawdown:.1f}%")
        print(f"  Robustness:     {c.robustness}")
        print(f"  Assets Passing: {c.assets_passing}")
        print(f"  History Tested: {c.history_tested:,} bars")
        print(f"  Survivability:  {c.survivability}")
    else:
        # Show best near-miss
        all_p = [r.promotion for r in report.strategy_results]
        best  = max(all_p, key=lambda p: (p.trades, p.expectancy))
        print(f"\n  No candidate met all criteria.")
        print(f"  Best: {best.strategy_name}  "
              f"({best.trades} trades, PF={best.pf_str}, Exp=${best.expectancy:.2f})")
        print(f"  Failing: {' | '.join(best.fail_criteria)}")

    print(f"\n{sep}")
    print("  RECOMMENDATION")
    print(sep)
    print(f"\n  {report.recommendation}")

    if report.notes:
        print("\n  Research Notes:")
        for n in report.notes:
            print(f"    → {n}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nPhase 5.1 — Edge Scalability Validation")
    print("=" * 66)

    # Data
    print(f"\nRequesting {MAX_BARS} D1 bars for {len(ASSETS)} assets...")
    print(f"Assets: {', '.join(ASSETS)}")
    asset_candles = _try_live(ASSETS, MAX_BARS)

    if asset_candles:
        source   = "LIVE (IBKR)"
        n_live   = len(next(iter(asset_candles.values())))
        # Fill any missing with synthetic
        for sym in ASSETS:
            if sym not in asset_candles:
                print(f"  {sym}: live unavailable — using synthetic")
                asset_candles[sym] = _synthetic(sym, n_live)
            else:
                actual = len(asset_candles[sym])
                first  = asset_candles[sym][0].timestamp.strftime("%Y-%m-%d")
                last   = asset_candles[sym][-1].timestamp.strftime("%Y-%m-%d")
                print(f"  {sym}: {actual} bars  {first} → {last}")
    else:
        source = "SYNTHETIC (deterministic, seed per asset)"
        print("  Live unavailable — generating synthetic data for all assets")
        asset_candles = {sym: _synthetic(sym, MAX_BARS) for sym in ASSETS}
        for sym, c in asset_candles.items():
            print(f"  {sym}: {len(c)} bars  {c[0].timestamp.date()} → {c[-1].timestamp.date()}")

    print(f"\nData source: {source}")
    print(f"Config: $10k account · 1% risk · ATR×2 stop · ATR×3 target · $1 commission")

    # Strategies (identified as top candidates in Phase 5.0)
    strategies = [
        VolatilityExpansionStrategy(),
        MomentumRotationStrategy(),
    ]

    print(f"\nValidating {len(strategies)} candidates × {len(asset_candles)} assets")
    print("This runs history expansion, asset expansion, robustness, and survivability...\n")

    engine = ValidationEngine(EDGE_CONFIG)
    report = engine.run(strategies, asset_candles)

    print_report(report)

    return 0 if report.best_candidate else 1


if __name__ == "__main__":
    sys.exit(main())
