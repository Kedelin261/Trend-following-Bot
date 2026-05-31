#!/usr/bin/env python3
"""Phase 3 Risk Engine Validation Script.

Workflow:
  1. Connect to configured provider
  2. Fetch SPY D1 candles
  3. Generate signal (Phase 2 Signal Engine)
  4. Evaluate risk (Phase 3 Risk Engine)
  5. Print full TradeCandidate

No orders placed. No trades executed. Read-only throughout.

Usage:
    python validate_risk_engine.py
    python validate_risk_engine.py --symbol QQQ --candles 300
    python validate_risk_engine.py --account 25000 --risk 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings_loader import load_settings
from src.data.symbol_registry import SymbolRegistry
from src.providers.provider_factory import ProviderFactory
from src.risk.models import RiskProfile
from src.risk.risk_engine import RiskEngine
from src.signals.signal_engine import SignalEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 risk engine validation (read-only)")
    p.add_argument("--symbol", default="SPY",
                   help="Symbol to evaluate (default: SPY)")
    p.add_argument("--timeframe", default="D1",
                   help="Candle timeframe (default: D1)")
    p.add_argument("--candles", type=int, default=300,
                   help="Candles to fetch (default: 300)")
    p.add_argument("--account", type=float, default=None,
                   help="Override account size ($)")
    p.add_argument("--risk", type=float, default=None,
                   help="Override risk per trade (%%, e.g. 0.5)")
    return p.parse_args()


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/risk_validation.log", encoding="utf-8"),
        ],
    )


def print_trade_candidate(candidate) -> None:
    sep = "=" * 54

    if candidate.approved:
        print(f"\n{sep}")
        print("  TRADE CANDIDATE — APPROVED")
        print(sep)
        print(f"  Symbol:             {candidate.symbol}")
        print(f"  Signal:             {candidate.signal_type.value}")
        print(f"  Timeframe:          {candidate.timeframe}")
        print()
        print(f"  Entry:              {candidate.entry_price:.4f}")
        print(f"  ATR:                {candidate.atr:.4f}")
        print()
        print(f"  Stop Loss:          {candidate.stop_loss:.4f}")
        print(f"  Take Profit:        {candidate.take_profit:.4f}")
        print()
        print(f"  Risk / Share:       {candidate.risk_per_share:.4f}")
        print(f"  Reward / Share:     {candidate.reward_per_share:.4f}")
        print(f"  R : R:              {candidate.risk_reward_ratio:.2f}")
        print()
        print(f"  Account:            ${candidate.dollar_risk / (candidate.dollar_risk / candidate.dollar_risk * 0.01):.0f}"
              if False else
              f"  Dollar Risk:        ${candidate.dollar_risk:.2f}")
        print(f"  Position Size:      {candidate.position_size:,} shares")
        print(f"  Position Value:     ${candidate.total_position_value:,.2f}")
        print(f"  Expected Profit:    ${candidate.expected_profit:,.2f}")
        print(f"  Expected Loss:      ${candidate.expected_loss:,.2f}")
        print()
        print(f"  Signal Score:       {candidate.signal_score:.1f} / 100")
        print(f"  Approved:           YES")
        print(sep)
    else:
        print(f"\n{sep}")
        print("  TRADE REJECTED")
        print(sep)
        print(f"  Symbol:             {candidate.symbol}")
        print(f"  Signal:             {candidate.signal_type.value}")
        print(f"  Signal Score:       {candidate.signal_score:.1f}")
        print(f"  ATR:                {candidate.atr:.4f}")
        if candidate.risk_reward_ratio > 0:
            print(f"  R : R:              {candidate.risk_reward_ratio:.2f}")
        print()
        print(f"  Reason:             {candidate.rejection_reason}")
        print(sep)


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        config = load_settings()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Allow CLI overrides of risk profile
    if args.account is not None:
        config.setdefault("risk", {})["account_size"] = args.account
        config["risk"]["cash_available"] = args.account
    if args.risk is not None:
        config.setdefault("risk", {})["risk_per_trade_percent"] = args.risk

    symbol_registry = SymbolRegistry.from_config(config)
    risk_profile    = RiskProfile.from_config(config)
    signal_engine   = SignalEngine()
    risk_engine     = RiskEngine.from_config(config)

    provider_name = config.get("provider", "?").upper()

    try:
        provider = ProviderFactory.create(config, symbol_registry)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nConnecting to {provider_name} provider...")
    if not provider.connect():
        print("ERROR: Could not connect to provider.", file=sys.stderr)
        print("  - For IBKR: ensure TWS is running on port 7497", file=sys.stderr)
        return 1
    print("  OK\n")

    print(f"Risk Profile:")
    print(f"  Account:      ${risk_profile.account_size:,.0f}")
    print(f"  Risk/trade:   {risk_profile.risk_per_trade_percent}%  "
          f"(${risk_profile.dollar_risk_per_trade:.2f} max loss)")
    print(f"  ATR period:   {risk_engine._atr.period}")
    print(f"  Stop mult:    ×{risk_engine._stop.multiplier}")
    print(f"  Target mult:  ×{risk_engine._target.multiplier}")
    print()

    try:
        print(f"Fetching {args.symbol} {args.timeframe} ({args.candles} bars)...")
        candles = provider.get_candles(args.symbol, args.timeframe, args.candles)
        print(f"  Received: {len(candles)} candles")

        if not candles:
            print(f"\nERROR: No data returned for {args.symbol}/{args.timeframe}.")
            print("  Check provider connection and market data subscriptions.")
            return 1

        print(f"\nRunning Signal Engine...")
        signal = signal_engine.generate_signal(candles)
        print(f"  Signal:     {signal.signal_type.value}")
        print(f"  Trend:      {signal.trend_direction.value}")
        print(f"  Score:      {signal.strength_score:.1f}")
        if signal.ema50:
            print(f"  EMA50:      {signal.ema50:.4f}")
        if signal.ema200:
            print(f"  EMA200:     {signal.ema200:.4f}")

        print(f"\nRunning Risk Engine...")
        candidate = risk_engine.evaluate(signal, candles)

        print_trade_candidate(candidate)

    finally:
        provider.disconnect()
        print("\n  Disconnected cleanly.")

    print()
    print("No trades placed. No orders sent. Read-only mode.")
    print(f"Full log: logs/risk_validation.log")
    print()
    return 0 if candidate.approved else 1


if __name__ == "__main__":
    sys.exit(main())
