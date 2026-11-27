# BharatAlpha Unattended Long-Only Forward Paper-Trading Runbook (Windows — Lenovo Yoga 7i Gen 10)

> **HARD SAFETY INVARIANT**
> - `LIVE_TRADING_ENABLED = False`
> - `READ_ONLY_MODE = True`
> - **Zero real-money order placement, modification, cancellation, or broker execution routing.**
> - Automated forward paper-trading evidence **never** authorizes real-money trading (`real_money_authorized: false` always).

---

## 1. Standing Long-Only Permission Scope

During the unattended forward paper-trading soak (`BHARATALPHA-FORWARD-SOAK-AUTO-001`), the runner operates under a strict **standing long-only** paper execution scope:

| Action | Permitted? | Governed Behavior |
| :--- | :---: | :--- |
| **BUY** (open LONG paper position) | **YES** *(Stage C/D only)* | Allowed between `09:30 IST` and `14:45 IST` ONLY after all 8 Stage C preconditions pass and all 5 tiers of `MandatoryExecutionGate` + `DeterministicRiskEngine` approve. |
| **SELL** (close or reduce existing LONG paper position) | **YES** | Always permitted for stop-loss, take-profit, trailing exit, `REDUCING` wind-down (`15:05 IST`), `HALTED -> EXIT_ONLY`, or mandatory `15:15 IST` EOD square-off. |
| **SHORT / SHORT SELL** (open SHORT position) | **STRICTLY PROHIBITED** | Blocked fail-closed at Human Authority (`ALLOW_LONG`), Execution Gate, and Risk Engine. |
| **Live Broker Order Placement** | **STRICTLY PROHIBITED** | Broker adapters (`ZerodhaKiteReadOnlyAdapter`, `AngelOneSmartApiReadOnlyAdapter`) are read-only WebSocket market-data consumers only. |

---

## 2. Automated 5-Stage NSE Session Lifecycle (`Asia/Kolkata`)

Every unattended NSE session follows a deterministic 5-stage state machine governed in **`Asia/Kolkata` (IST)**:

| Stage | IST Window | Effective Authority | Description |
| :--- | :--- | :--- | :--- |
| **Stage A — Preflight** | `08:45 – 09:14 IST` | `NO_NEW_ENTRIES` | Validates NSE trading day, credentials, token map, disk space, lock file, read-only flags, and audit initialization (`--preflight-only`). Aborts fail-closed on any failure. |
| **Stage B — Market Open Observation** | `09:15 – 09:29 IST` | `NO_NEW_ENTRIES` | Streams genuine read-only broker ticks, builds causal 1s/1m/5m bars, and observes opening-range liquidity without taking new entries. |
| **Stage C — Automatic `ALLOW_LONG` Gate** | `09:30 IST` | `NO_NEW_ENTRIES` → `ALLOW_LONG` | Automatically promotes authority to `ALLOW_LONG` **ONLY IF all 8 preconditions below pass**. If any precondition fails, remains fail-closed in `NO_NEW_ENTRIES`. |
| **Stage D — Autonomous Long-Only Execution** | `09:30 – 15:14 IST` | `ALLOW_LONG` (`09:30–14:45`) → `REDUCING` (`15:05`) | Evaluates closed causal 1m bars via 5-tier stack. New LONG entries stop at `14:45 IST`; wind-down (`REDUCING`) begins at `15:05 IST`; `2.5%` daily loss or `3.0%` MTM drawdown transitions immediately to `HALTED` (`EXIT_ONLY`). |
| **Stage E — Mandatory Square-Off & EOD Audit** | `15:15 – 15:35 IST` | `EXIT_ONLY` → `SHUTDOWN` | Force-closes all open paper positions at `15:15 IST`, archives partitioned Parquet ticks + `provenance.json`, verifies SHA-256 authority/risk/ledger audit chains (`15:35 IST`), writes `live_paper_state.json`, and runs `scripts/evaluate_forward_paper_soak.py`. |

### Stage C — 8 Mandatory Preconditions for Automatic `ALLOW_LONG` at `09:30 IST`

1. **Paper-Only & Read-Only Invariants**: `LIVE_TRADING_ENABLED == False` and `READ_ONLY_MODE == True` verified.
2. **Valid NSE Trading Day**: Current date in `Asia/Kolkata` is a weekday and NOT in `NSE_TRADING_HOLIDAYS`.
3. **Live Read-Only Feed Health**: Broker WebSocket connected and fresh genuine ticks received with no stale-feed timeout.
4. **Zero Synthetic Contamination**: `provider != "synthetic"`, `source_mode != "SYNTHETIC_TEST"`, and complete `--token-map` for all configured symbols.
5. **Cryptographic Audit Chains Clean**: Authority gate, `DeterministicRiskEngine`, and `IntradayPaperPortfolio` ledger chains initialized and verified.
6. **Risk Engine `ACTIVE`**: Zero margin call deficit (`MARGIN_CALL_DEFICIT`), zero daily/MTM drawdown breach, and initial cash/equity reconciled.
7. **Host & Lock Integrity**: Single-instance `--lock-file` acquired, sufficient free disk space available, and session output directory writable.
8. **No Human Override Active**: Standing long-only policy active with no `STOP_ALL`, `PAUSE`, or `EXIT_ONLY` override registered.

---

## 3. Timezone & NSE Holiday Calendar Rules (Zero Melbourne DST Drift)

- **Source of Truth**: All market hours and holiday checks use Python's `ZoneInfo("Asia/Kolkata")` (`IST_TZ`) and `is_nse_trading_day()` in `trader/config.py`.
- **Never Hard-Code Melbourne Offsets**:
  - NSE opens at `09:15 IST` (`UTC+05:30`, no DST).
  - Melbourne switches between **AEDT (`UTC+11:00`, Oct–Apr)** where `08:50 IST = 14:20 AEDT` and **AEST (`UTC+10:00`, Apr–Oct)** where `08:50 IST = 13:20 AEST`.
  - `scripts/windows/register_bharatalpha_task.ps1` registers **two weekday triggers** (`13:20` and `14:20` Melbourne local time) calling `run_bharatalpha_soak.ps1 -RequirePreOpenWindow`.
  - On each trigger, Python checks `datetime.now(ZoneInfo("Asia/Kolkata"))` and `is_nse_trading_day()`:
    - Whichever trigger lands inside `08:30 – 09:25 IST` on an NSE trading day runs the session.
    - The off-season trigger detects it is outside the IST pre-open window and exits cleanly (`exit 0`).
    - Weekends and official NSE holidays exit cleanly (`exit 0`).
- **Windows `tzdata` Requirement**: Ensure Python has IANA timezone data installed on Windows:
  ```powershell
  python -m pip install --upgrade tzdata
  ```

---

## 4. Secure Credential Setup on Windows (DPAPI / Credential Manager)

**Strict Security Rules**:
- NEVER commit credentials to Git.
- NEVER put API keys or tokens in source code, PowerShell scripts, Task Scheduler arguments, or logs.

### Required Environment Variables (Read-Only Market Data Only)

- **Zerodha KiteTicker (`--provider zerodha`)**:
  - `KITE_API_KEY`
  - `KITE_ACCESS_TOKEN` *(Note: Zerodha access tokens expire daily around 06:00 IST and must be refreshed before `08:45 IST`.)*
- **Angel One SmartAPI WebSocket v2 (`--provider angel_one`)**:
  - `ANGEL_API_KEY`
  - `ANGEL_CLIENT_CODE`
  - `ANGEL_AUTH_TOKEN`
  - `ANGEL_FEED_TOKEN`

### Option A (Recommended): Windows DPAPI Encrypted Secrets File (`%LOCALAPPDATA%\BharatAlpha\secrets.xml`)

DPAPI encrypts secrets using your Windows user login key — only your Windows account on this Lenovo laptop can decrypt them. Run once (or when refreshing daily tokens) in PowerShell:

```powershell
$secretsDir = Join-Path $env:LOCALAPPDATA "BharatAlpha"
New-Item -ItemType Directory -Path $secretsDir -Force | Out-Null

# Example for Zerodha (prompts securely without echoing to screen/history):
$secrets = @{
    KITE_API_KEY      = Read-Host "Enter KITE_API_KEY" -AsSecureString
    KITE_ACCESS_TOKEN = Read-Host "Enter KITE_ACCESS_TOKEN" -AsSecureString
}
$secrets | Export-Clixml -Path (Join-Path $secretsDir "secrets.xml")
Write-Host "Encrypted credentials saved to $secretsDir\secrets.xml"
```

*(For Angel One, store `ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_AUTH_TOKEN`, and `ANGEL_FEED_TOKEN` as `-AsSecureString` keys in the same hashtable.)*

### Option B: User-Level Windows Environment Variables

```powershell
[Environment]::SetEnvironmentVariable("KITE_API_KEY", "<YOUR_KEY>", "User")
[Environment]::SetEnvironmentVariable("KITE_ACCESS_TOKEN", "<YOUR_DAILY_TOKEN>", "User")
```

`scripts/windows/run_bharatalpha_soak.ps1` automatically loads `%LOCALAPPDATA%\BharatAlpha\secrets.xml` (if present) into process-only memory without ever echoing values.

---

## 5. Lenovo Yoga 7i Gen 10 Power, Sleep Prevention & Windows Task Scheduler

### 5.1 Sleep Prevention (Reversible & Non-Destructive)

- `scripts/windows/run_bharatalpha_soak.ps1` calls Win32 `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` while the session is active, keeping the CPU and network awake while still allowing the OLED display to dim/turn off normally.
- In its `finally` block, it **always** restores `SetThreadExecutionState(ES_CONTINUOUS)` when the session completes or exits.
- **Recommended AC Power Settings on Lenovo Yoga 7i Gen 10**:
  - Keep the laptop connected to USB-C AC power during NSE sessions (`08:45–15:40 IST`).
  - Enable Wake Timers on AC power (reversible via `powercfg`):
    ```powershell
    # Allow wake timers when plugged in (AC = 1)
    powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
    powercfg /SETACTIVE SCHEME_CURRENT
    ```
  - Optional: To revert wake timers later, run `powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_SLEEP RTCWAKE 0 && powercfg /SETACTIVE SCHEME_CURRENT`.
  - In Windows Settings → System → Power & battery → Lid close action (plugged in): set to **Do nothing** if you plan to run with the lid closed.

### 5.2 Registering the Scheduled Task

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\register_bharatalpha_task.ps1 `
    -Provider zerodha `
    -Symbols RELIANCE,TCS,ITC `
    -TokenMapPath .\config\token_map.json `
    -Profile ORB_BREAKOUT `
    -Capital 50000
```

To unregister later:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\register_bharatalpha_task.ps1 -Unregister
```

### 5.3 Failure Handling & Restart Checkpoint Recovery

| Failure Mode | Automated Protection & Recovery |
| :--- | :--- |
| **Duplicate Process Launch** | Blocked at TWO layers: Windows Task Scheduler `MultipleInstances IgnoreNew` + Python `--lock-file` single-instance lock. |
| **WebSocket Disconnect / Brief Internet Drop** | Read-only broker adapter reconnects with bounded backoff; `CausalTickAggregator` drops duplicate/out-of-order ticks and never fabricates missing candles. Prolonged feed outage keeps entries fail-closed. |
| **Unexpected Crash / Power Loss** | Task Scheduler retries (`RestartCount 2`, 5-minute interval). Runner resumes safely from session checkpoint / lock cleanup, never double-books trades, and flags any incomplete session so the evaluator excludes it from valid forward counts. |
| **Low Disk Space** | Stage A preflight verifies free disk space before launch; aborting fail-closed prevents corrupt Parquet/JSON archives. |
| **Daily Loss (`2.5%`) or MTM Drawdown (`3.0%`)** | `DeterministicRiskEngine` transitions to `HALTED` (`EXIT_ONLY`): blocks all new entries while allowing exits and `15:15 IST` square-off. |

---

## 6. Copy-Paste Operational Commands (PowerShell)

Run all commands from the repository root (e.g. `C:\dev\indian-stock-ai-agent`).

### 6.1 Preflight Check (`--preflight-only`)

```powershell
python scripts\run_live_paper_session.py `
    --provider zerodha `
    --symbols RELIANCE TCS ITC `
    --token-map config\token_map.json `
    --profile ORB_BREAKOUT `
    --capital 50000 `
    --output-dir data\forward_soak\preflight_check `
    --lock-file data\forward_soak\bharatalpha_soak.lock `
    --preflight-only
```

### 6.2 Dry-Run Validation (`--dry-run`)

```powershell
python scripts\run_live_paper_session.py `
    --provider zerodha `
    --symbols RELIANCE TCS ITC `
    --token-map config\token_map.json `
    --profile ORB_BREAKOUT `
    --capital 50000 `
    --output-dir data\forward_soak\dry_run `
    --dry-run
```

### 6.3 First Autonomous Long-Only Session (`--autonomous-long-only`)

Recommended via the Windows wrapper (enables reversible sleep guard, DPAPI secret load, preflight abort, and post-session evaluator):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\run_bharatalpha_soak.ps1 `
    -Provider zerodha `
    -Symbols RELIANCE,TCS,ITC `
    -TokenMapPath .\config\token_map.json `
    -Profile ORB_BREAKOUT `
    -Capital 50000 `
    -SoakRoot .\data\forward_soak
```

Or directly via Python:

```powershell
python scripts\run_live_paper_session.py `
    --provider zerodha `
    --symbols RELIANCE TCS ITC `
    --token-map config\token_map.json `
    --profile ORB_BREAKOUT `
    --capital 50000 `
    --output-dir data\forward_soak\2026-12-01 `
    --lock-file data\forward_soak\bharatalpha_soak.lock `
    --autonomous-long-only
```

### 6.4 Post-Session Forward Evidence Evaluator

```powershell
python scripts\evaluate_forward_paper_soak.py `
    --soak-dir data\forward_soak `
    --output-dir data\forward_soak
```

Outputs generated in `data\forward_soak\`:
- `session_index.json`
- `forward_evaluation_report.json`
- `forward_evaluation_report.md`

---

## 7. Seven-Gate Forward Qualification Protocol (Verbatim)

All 7 gates from `BHARATALPHA_FORWARD_PAPER_PROTOCOL.md` are enforced automatically by `scripts/evaluate_forward_paper_soak.py`:

| # | Gate Name | Exact Qualification Criterion |
| :-: | :--- | :--- |
| **1** | **Data Provenance** | 100% genuine read-only broker ticks (`source_mode != "SYNTHETIC_TEST"`, `provider != "synthetic"`), zero fabricated timestamps, zero lookahead, `live_trading_enabled == False`, `read_only_mode == True`. |
| **2** | **Minimum Forward Sample** | `>= 30` valid complete NSE trading sessions AND `>= 60` completed round-trip trades across `>= 3` distinct symbols. *(Incomplete, interrupted, non-NSE-holiday, audit-failed, or synthetic-contaminated sessions are explicitly flagged and EXCLUDED from valid counts.)* |
| **3** | **Net-of-Friction Expectancy** | `total_net_pnl_inr > 0.0` AND `profit_factor >= 1.25` after full NSE MIS statutory costs + slippage. |
| **4** | **Drawdown & Risk Compliance** | Zero breaches of 15% per-stock / 80% gross portfolio caps, zero `margin_call_triggered` / `MARGIN_CALL_DEFICIT` events, and `max_cumulative_drawdown_pct <= 6.0`. |
| **5** | **Cryptographic Audit Integrity** | 100% of valid sessions have `authority_audit_verified == True`, `risk_audit_verified == True`, `ledger_integrity_verified == True`, and `0` open positions at EOD (`abs(final_cash_inr - final_equity_inr) < 0.05` or `open_positions == 0`). |
| **6** | **Sealed Holdout Governance** | `data/lab/sealed_holdout_registry.json` exists and is valid JSON (strictly read-only; never modified by forward evaluation). |
| **7** | **Real-Money Boundary** | Always sets `"real_money_authorized": False` and `"qualification_status": "QUALIFIED_FOR_FURTHER_REVIEW"` ONLY if Gates 1–6 all pass, otherwise `"NOT_QUALIFIED"`. Never authorizes real-money trading. |
