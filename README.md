# Trend Following Bot

An **institutional-grade, research-first systematic trading framework** built entirely in Python. The system progresses through a rigorous research pipeline before any live execution is considered — signal quality, edge confirmation, robustness validation, and regime analysis are all mandatory gates.

> **Status: Research Complete through Phase 5.1 — No live trading enabled**

---

## Architecture Overview

```
Market Data Layer (IBKR / MT5 / Polygon)
        ↓
Signal Engine (EMA + Breakout + Volume)
        ↓
Risk Engine (ATR-based stop/target/sizing)
        ↓
Backtesting Engine (bar-by-bar, anti-lookahead)
        ↓
Research Framework (regime, density, timeframe)
        ↓
Edge Lab (multi-strategy comparison)
        ↓
Validation Pipeline (promotion gates)
```

---

## Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Broker-agnostic market data layer | ✅ |
| 1.5 | IBKR connectivity validation | ✅ |
| 2 | Signal engine (EMA + breakout) | ✅ |
| 3 | ATR risk engine | ✅ |
| 4 | Institutional backtesting engine | ✅ |
| 4.5 | Strategy research & edge discovery | ✅ |
| 4.6 | Strategy V2 refinement | ✅ |
| 4.7 | Signal density optimization | ✅ |
| 4.8 | Timeframe expansion research | ✅ |
| 4.9 | Promotion validation | ✅ |
| 4.10 | Regime stability research | ✅ |
| 4.11 | Regime-aware entry filter research | ✅ |
| 5.0 | Edge Lab — 5-strategy comparison | ✅ |
| 5.1 | Edge scalability validation | ✅ |

---

## Project Structure

```
src/
├── providers/          # Market data: IBKR, MT5, factory pattern
├── data/               # Candle models, SQLite store, symbol registry
├── signals/            # Signal engine + 5 strategy implementations
│   └── strategies/     # Breakout, Pullback, Donchian, VolExp, Momentum
├── risk/               # ATR calculator, position sizer, trade validator
├── backtest/           # Bar-by-bar simulator, portfolio, metrics
├── research/           # Market regime, ADX, volatility, benchmark
├── refinement/         # Strategy V2, composite signal engine
├── density/            # Signal density optimization
├── timeframe/          # Multi-timeframe research
├── promotion/          # Promotion gates, history expansion
├── regime/             # 5-state classifier, performance analyzer
├── regime_filter/      # Regime-aware entry filters
├── edge_lab/           # Multi-strategy comparison framework
└── edge_validation/    # Scalability and survivability validation
```

---

## Quick Start

### Requirements

- Python 3.9+
- Interactive Brokers TWS (paper account) — optional for live data
- MetaTrader 5 terminal — optional (Windows only)

### Setup

```bash
git clone https://github.com/your-username/Trend-following-Bot.git
cd Trend-following-Bot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in broker credentials if needed
```

### Run Tests

```bash
pytest tests/ -m "not integration" -q
# Expected: 1217 passed
```

---

## Validation Commands

Each script runs a specific research phase and outputs a report. No orders are placed.

```bash
# Broker connectivity
python validate_ibkr.py

# Signal engine
python validate_signals.py

# Risk engine
python validate_risk_engine.py

# Backtesting
python validate_backtest.py

# Strategy research
python validate_research.py

# Strategy V2 refinement
python validate_refinement.py

# Signal density
python validate_density.py

# Timeframe expansion
python validate_timeframes.py

# Promotion validation
python validate_promotion.py

# Regime stability
python validate_regimes.py

# Regime-aware filters
python validate_regime_filters.py

# 5-strategy Edge Lab (Phase 5.0)
python validate_edge_lab.py

# Scalability validation (Phase 5.1)
python validate_edge_scalability.py
```

---

## Key Research Findings

### Phase 5.0 — Edge Lab Results (750 bars, 3 assets)

| Rank | Strategy | PF | Exp/trade | DD | Robustness |
|------|----------|----|-----------|----|------------|
| #1 | VOLATILITY_EXPANSION | 2.47 | +$51.65 | 2.8% | ROBUST |
| #2 | MOMENTUM_ROTATION | 1.60 | +$26.78 | 3.4% | ROBUST |
| #3 | BREAKOUT_V1 | 1.79 | +$33.72 | 3.9% | MARGINAL |
| #4 | DONCHIAN_20 | 1.55 | +$25.04 | 4.9% | UNSTABLE |
| #5 | PULLBACK_CONTINUATION | 0.93 | −$3.85 | 4.1% | UNSTABLE |

### Phase 5.1 — Scalability over 20 years, 8 assets

| Strategy | Trades | PF | DD | Status |
|----------|--------|----|----|--------|
| VOLATILITY_EXPANSION | 330 | 1.18 | 12.0% | Not ready — recent-period bias detected |
| MOMENTUM_ROTATION | 1,075 | 1.19 | 18.7% | Not ready — DD exceeds 15% at full history |

**Conclusion:** Both candidates showed recency bias. Full-history validation reveals the edge requires further refinement before promotion to live research.

---

## Safety Notice

⚠️ **This is a research-only system.**

- No live trading execution exists anywhere in the codebase
- No order placement functions are present
- No broker account mutations occur
- All backtesting uses historical data only
- The system is designed to **research and validate** strategies, not execute them

---

## Broker Configuration

### IBKR (Interactive Brokers)

1. Open Trader Workstation (paper trading account)
2. Go to **File → Global Configuration → API → Settings**
3. Enable "Enable ActiveX and Socket Clients"
4. Set port to `7497` (paper) or `7496` (live)
5. Copy `.env.example` to `.env` and configure credentials

### MetaTrader 5 (optional, Windows only)

1. Open MetaTrader 5
2. Enable Algo Trading: **Tools → Options → Expert Advisors**
3. Set credentials in `.env` (or leave blank to use logged-in session)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Market Data | ib-insync (IBKR), MetaTrader5 |
| Data Storage | SQLite (via stdlib sqlite3) |
| Testing | pytest (1217 tests) |
| Config | PyYAML + python-dotenv |

---

## License

This project is for educational and research purposes only. Not financial advice. Trading involves significant risk of loss.
