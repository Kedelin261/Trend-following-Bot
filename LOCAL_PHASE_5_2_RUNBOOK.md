# LOCAL PHASE 5.2 VALIDATION RUNBOOK
## Momentum Rotation — Risk Overlay Research

**Phase:** 5.2  
**Strategy:** MOMENTUM_ROTATION  
**Overlays:** NO_OVERLAY, EQUITY_CURVE_PAUSE, BEAR_MARKET_PAUSE, MONTHLY_LOSS_LOCKOUT, CONSECUTIVE_LOSS_COOLDOWN, VOLATILITY_RISK_SCALING, COMBINED_OVERLAY  
**Data Source:** Live IBKR D1 historical bars (official validation only)  
**Last Updated:** 2026-05-31  

> **IMPORTANT:** Sandbox results used synthetic data and are NOT official.  
> This runbook executes on your local machine with TWS running to produce official results.

---

## PRE-FLIGHT CHECKLIST

Work through every item before running any validation command.

---

### 1. Open Trader Workstation (TWS)

Launch TWS on your Mac and log in to your **paper account**.

Verify you see the main TWS window with market data active.

---

### 2. Enable API Access in TWS

```
File → Global Configuration → API → Settings
```

Confirm all of the following are set:

| Setting | Required Value |
|---|---|
| Enable ActiveX and Socket Clients | ✓ CHECKED |
| Socket Port | 7497 |
| Master API client ID | 1 (or your preferred ID) |
| Read-Only API | ✓ CHECKED (for safety) |
| Allow connections from localhost only | ✓ CHECKED |
| Trusted IPs | 127.0.0.1 (add if required) |

Click **Apply** then **OK**.

---

### 3. Confirm Paper Account Is Connected

In the bottom status bar of TWS, verify:

```
Paper Trading | Connected | [account number]
```

If it shows **Disconnected** — log out and log back in.

---

## CONNECTIVITY TEST

### Step 1 — TCP Port Check

Open Terminal and run:

```bash
nc -vz 127.0.0.1 7497
```

**Expected output:**
```
Connection to 127.0.0.1 port 7497 [tcp] succeeded!
```

**If this fails:**
- TWS is not running, or
- API is not enabled (revisit Pre-flight Step 2), or
- Wrong port — confirm you are using paper (7497) not live (7496)

---

## REPOSITORY SETUP

### Step 2 — Pull Latest Phase 5.2 Code

```bash
cd /path/to/Trend-following-Bot

git pull origin genspark_ai_developer
```

> Replace `/path/to/Trend-following-Bot` with your actual local path.

Verify Phase 5.2 files are present:

```bash
ls src/risk_overlay/
ls src/risk_overlay/profiles/
ls validate_risk_overlays.py
```

Expected output from `ls src/risk_overlay/`:
```
__init__.py   base_overlay.py   overlay_backtester.py   overlay_engine.py   profiles/
```

Expected output from `ls src/risk_overlay/profiles/`:
```
__init__.py           bear_market_pause.py        combined_overlay.py
consecutive_loss_cooldown.py   equity_curve_pause.py   monthly_loss_lockout.py
no_overlay.py         volatility_risk_scaling.py
```

---

### Step 3 — Activate Virtual Environment

```bash
source venv/bin/activate
```

Verify Python version (must be 3.10+):

```bash
python --version
```

---

### Step 4 — Run Pytest Baseline Check

```bash
python -m pytest tests/ -q --tb=short
```

**Expected output:**
```
1304 passed, 25 skipped, 0 failed
```

If any tests fail, do NOT proceed. Fix failures before running validation.

---

## IBKR CONNECTIVITY VALIDATION

### Step 5 — Run IBKR Validator

```bash
python validate_ibkr.py
```

**Expected output:**
```
==================================================================
  VALIDATION PASSED
==================================================================
  Provider:              IBKR (ib-insync)
  Connection:            OK
  Account summary:       OK
  EUR.USD H1 candles:    <N> bars
  SPY D1 candles:        <N> bars
  Errors:                0
  Orders placed:         0
==================================================================
```

**If you see `VALIDATION FAILED`:**

| Symptom | Fix |
|---|---|
| `Connection: FAILED` | TWS not running / API not enabled / wrong port |
| `Account summary: EMPTY` | Wait 10s and retry — TWS still initializing |
| `SPY D1 candles: 0 bars` | Market data subscription may be inactive |

Do NOT proceed to Phase 5.2 validation until `validate_ibkr.py` shows **PASSED**.

---

## PHASE 5.2 VALIDATION

### Step 6 — Run Risk Overlay Validation

```bash
python validate_risk_overlays.py
```

**What the script does:**

1. Connects to TWS via `config/settings.yaml` (host: 127.0.0.1, port: 7497, readonly: true)
2. Fetches **5,000 D1 bars** for each of 8 assets: `SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD`
3. Applies a 1.5-second delay between requests (IBKR rate limiting)
4. Runs **7 overlay variants** through the MOMENTUM_ROTATION strategy backtester
5. Evaluates promotion criteria against live data results
6. Prints the full validation report
7. Writes full log to `logs/risk_overlay_validation.log`

**Expected console header (with live IBKR data):**
```
Phase 5.2 — Momentum Rotation Risk Overlay Validation
==================================================================

Requesting 5000 D1 bars for 8 assets...
Assets: SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD
  SPY:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  VOO:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  DIA:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  QQQ:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  IWM:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  VTI:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  XLV:  5000 bars  YYYY-MM-DD → YYYY-MM-DD
  SCHD: 5000 bars  YYYY-MM-DD → YYYY-MM-DD

Data source: LIVE (IBKR)
Config: $10k · 1% risk · ATR×2 stop · ATR×3 target · $1 commission · 0.05% slip
```

> **CRITICAL:** The line `Data source: LIVE (IBKR)` confirms official data is being used.  
> If you see `Data source: SYNTHETIC` — TWS is not connected. Stop and fix the connection.

**Expected report structure:**
```
==================================================================
  RISK OVERLAY VALIDATION REPORT  —  Phase 5.2
  Strategy:    MOMENTUM_ROTATION
  Assets:      SPY, VOO, DIA, QQQ, IWM, VTI, XLV, SCHD
  Data source: LIVE (IBKR)
  History:     5000 D1 bars per asset
==================================================================

  ──────────────────────────────────────────────────────────────
  NO_OVERLAY
  ...
  STATUS: [PASS / FAIL]

  ──────────────────────────────────────────────────────────────
  EQUITY_CURVE_PAUSE
  ...

  [... 5 more overlays ...]

==================================================================
  FINAL RANKINGS
==================================================================

==================================================================
  PROMOTION CANDIDATE
==================================================================
  YES — <overlay_name>   ← if any overlay passes all 5 criteria
  NO PROMOTION CANDIDATE ← if no overlay passes

==================================================================
  RECOMMENDATION
==================================================================
  PROMOTE TO PHASE 5.3  ← if candidate exists
  RETURN TO EDGE RESEARCH ← if no candidate
```

---

## PROMOTION CRITERIA

An overlay passes Phase 5.2 promotion if ALL five criteria are met simultaneously:

| Criterion | Threshold | Notes |
|---|---|---|
| Trades | ≥ 500 | Adequate sample size across 8 assets |
| Profit Factor | ≥ 1.50 | Edge quality preserved through overlay |
| Expectancy | > $0.00/trade | Positive expected value per trade |
| Max Drawdown | < 15.0% | Risk reduction vs baseline 18.7% |
| Robustness | = ROBUST | Edge stable across early/middle/recent windows |
| History | ≥ 3,000 bars | Sufficient market history |

> The Phase 5.1 baseline (NO_OVERLAY with IBKR data) was: Trades≈1075, PF≈1.19, MaxDD≈18.7%.  
> Phase 5.2 overlays must reduce MaxDD below 15% while lifting PF to ≥1.50.

---

## OUTPUT LOCATION

| Output | Location |
|---|---|
| Console validation report | Terminal stdout |
| Full structured log | `logs/risk_overlay_validation.log` |
| IBKR connectivity log | `logs/bot.log` (from settings.yaml) |

View the log after the run:

```bash
cat logs/risk_overlay_validation.log
```

Or tail it during the run (in a second terminal):

```bash
tail -f logs/risk_overlay_validation.log
```

---

## TIMING ESTIMATE

| Phase | Estimated Time |
|---|---|
| IBKR data fetch (8 assets × 1.5s delay) | ~12 seconds |
| 7 overlays × 8 assets backtests | ~90–120 seconds |
| Total wall-clock time | ~2–3 minutes |

---

## TROUBLESHOOTING

### Synthetic Fallback Warning

If you see:
```
WARNING: Synthetic data results are NOT official.
         Run with TWS open on port 7497 for official IBKR validation.
```

This means `_try_live()` failed. Debug steps:

```bash
# 1. Check TWS is running and port is open
nc -vz 127.0.0.1 7497

# 2. Re-run IBKR validator
python validate_ibkr.py

# 3. Check for connection errors in logs
grep -i "error\|fail\|warn" logs/bot.log | tail -20
```

### ib-insync Not Installed

```bash
pip install ib-insync
```

### Client ID Conflict (another script already connected)

Edit `config/settings.yaml`:
```yaml
ibkr:
  client_id: 2   # change from 1 to any unused ID
```

Or pass via command line if using validate_ibkr.py directly:
```bash
python validate_ibkr.py --client-id 2
```

### Market Data Subscription Errors

If individual symbols return 0 bars, the paper account may lack subscriptions.  
Contact IBKR support or use IB Gateway with a funded paper account that includes market data.

---

## QUICK REFERENCE — COMPLETE COMMAND SEQUENCE

```bash
# 0. Start TWS, log in to paper account, enable API on port 7497

# 1. Navigate to project
cd /path/to/Trend-following-Bot

# 2. Pull latest Phase 5.2 code
git pull origin genspark_ai_developer

# 3. Activate environment
source venv/bin/activate

# 4. Verify TCP connectivity
nc -vz 127.0.0.1 7497

# 5. Run IBKR connectivity check
python validate_ibkr.py

# 6. Run Phase 5.2 risk overlay validation (official, IBKR data)
python validate_risk_overlays.py

# 7. Review full log
cat logs/risk_overlay_validation.log
```

---

## PHASE GATE DECISION

| Result | Action |
|---|---|
| `PROMOTE TO PHASE 5.3` | Record the promoted overlay name and metrics. Begin Phase 5.3 live paper-trading integration. |
| `RETURN TO EDGE RESEARCH` | Review which criteria failed (most likely PF < 1.50 or Trades < 500). Consider parameter tuning or alternative overlay designs. |

---

*Generated by Genspark AI Developer — Phase 5.2 Local Validation Preparation*  
*Sandbox commit: see git log for actual hash*
