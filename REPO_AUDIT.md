# Repository Audit Report

Generated: 2026-05-31

## Summary

| Metric | Count |
|--------|-------|
| Total files (excl. venv/git) | 261 |
| Python source files | 233 |
| Test files | 94 |
| Validation scripts | 13 |
| Source packages | 16 |
| Tests passing | 1,217 |
| Tests skipped (integration, need live broker) | 8 |

## Source Package Inventory

| Package | Purpose |
|---------|---------|
| `src/providers/` | Broker-agnostic market data layer (IBKR, MT5) |
| `src/data/` | Candle models, SQLite store, symbol/timeframe registries |
| `src/signals/` | Signal engine + 5 strategy implementations |
| `src/signals/strategies/` | Breakout, Pullback, Donchian, VolExp, Momentum |
| `src/risk/` | ATR calculator, stop/target engines, position sizer |
| `src/backtest/` | BacktestEngine, Portfolio, TradeSimulator, metrics |
| `src/research/` | Regime, ADX, volatility, trend quality, benchmark |
| `src/refinement/` | Strategy V2, composite signal engine, filter profiles |
| `src/density/` | Signal density optimization across assets/timeframes |
| `src/timeframe/` | Multi-timeframe research and overlap analysis |
| `src/promotion/` | Promotion gates, history expansion, robustness checks |
| `src/regime/` | 5-state market classifier, trade labelling |
| `src/regime_filter/` | Regime-aware entry filter research |
| `src/edge_lab/` | 5-strategy comparison framework |
| `src/edge_validation/` | Scalability and survivability validation |
| `src/config/` | Settings loader |

## Generated Artifacts (gitignored)

| Artifact | Location | Notes |
|----------|----------|-------|
| SQLite database | `data/market_data.db` | 20 KB — runtime artifact |
| Validation logs | `logs/*.log` | Runtime artifacts |
| Virtual environment | `venv/` | Not committed |

## Security Observations

- **No secrets found in committed code**
- `config/settings.yaml` references env var names only (e.g., `MT5_PASSWORD`) — no actual values
- `.env.example` contains only empty placeholders — safe to commit
- All broker credentials flow through environment variables
- IBKR connection uses read-only mode (`readonly: true`)
- No API keys, tokens, or credentials in any Python file, YAML, or log

## Technical Debt Observations

| Item | Severity | Notes |
|------|----------|-------|
| Phase 5.0/5.1 found recency bias in strategies | Research finding, not debt | Next step: regime-filtered re-test |
| `venv/` directory 60+ MB | None | Gitignored |
| `data/market_data.db` accumulates over runs | Low | Gitignored; can be regenerated |
| `logs/*.log` accumulate over runs | Low | Gitignored |
| 8 integration tests require live broker | Expected | Skipped in CI with `-m "not integration"` |
| `CLAUDE.md` not present | Low | Project uses README for documentation |

## Current Project Phase

**Phase 5.1 Complete** — Edge Scalability Validation

The system has completed full strategy research through Phase 5.1. Both top candidates (VOLATILITY_EXPANSION and MOMENTUM_ROTATION) showed recency bias under 20-year full-history validation. The recommended next step is Phase 5.2: regime-filtered strategy re-test using the Phase 4.11 regime filter infrastructure.

## Validation Script Inventory

| Script | Phase | Purpose |
|--------|-------|---------|
| `validate_ibkr.py` | 1.5 | IBKR connectivity |
| `validate_signals.py` | 2 | Signal engine |
| `validate_risk_engine.py` | 3 | Risk engine |
| `validate_backtest.py` | 4 | Backtesting |
| `validate_research.py` | 4.5 | Strategy research |
| `validate_refinement.py` | 4.6 | Strategy V2 |
| `validate_density.py` | 4.7 | Signal density |
| `validate_timeframes.py` | 4.8 | Timeframe expansion |
| `validate_promotion.py` | 4.9 | Promotion validation |
| `validate_regimes.py` | 4.10 | Regime stability |
| `validate_regime_filters.py` | 4.11 | Regime filters |
| `validate_edge_lab.py` | 5.0 | Edge Lab comparison |
| `validate_edge_scalability.py` | 5.1 | Scalability |
