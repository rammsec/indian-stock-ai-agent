# BharatAlpha — Python ↔ JavaScript Parity Report (`BHARATALPHA-AUTO-QA-REMEDIATION-001`)

**Engines Compared**:
- **Python**: `trader/authority/gate.py`, `trader/authority/instructions.py`, `trader/timing/monitoring_session.py`, `trader/risk/engine.py`, `trader/config.py`
- **JavaScript**: `docs/candle-agent/engine.js` (+ `docs/candle-agent/index.html` UI controls)

**Final Parity Score (Post-Remediation)**: **17 / 20 EXACT MATCH**, **3 / 20 DOCUMENTED ARCHITECTURAL DIFFERENCES**, **0 / 20 DEFECT DIVERGENCES**.

---

## 20-Scenario Parity Matrix

| # | Scenario | Stage B Status | Post-Fix Status | Python Behavior | JavaScript Behavior |
|---:|---|---|---|---|---|
| 1 | Missing Human instruction | MATCH | **MATCH** | Fail-closed (`HUMAN_INSTRUCTION_MISSING`), `RISK_EXIT` allowed | `humanPermission = null` (UI default `UNSET`), blocks entries, `RISK_EXIT` allowed |
| 2 | Human `ALLOW_LONG` | MATCH | **MATCH** | Long entries & exits allowed; short entries denied | Identical |
| 3 | Human `ALLOW_SHORT` | MATCH | **MATCH** | Short entries & exits allowed; long entries denied | Identical |
| 4 | Human `ALLOW_BOTH` | MATCH | **MATCH** | Long/short entries & exits allowed | Identical |
| 5 | Human `PAUSE` | MATCH | **MATCH** | Suspends entries & discretionary exits immediately; `RISK_EXIT` allowed | Identical (`call='REJECT'`, `gateStatus='REJECTED'`, `RISK_EXIT` allowed) |
| 6 | Human `EXIT_ONLY` | MATCH | **MATCH** | Blocks new entries; allows exits & bypasses min hold bars | Identical |
| 7 | Human `STOP_ALL` | MATCH | **MATCH** | Blocks entries; immediately flattens open position (incl. bar 0) | Identical (`EMERGENCY_HUMAN_STOP_ALL` incl. `barIdx === 0`) |
| 8 | Central direction changes | MATCH | **MATCH** | Revalidated at bar open; opposing/neutral blocks pending entry | Identical (`OPEN_GATE_BLOCKED:CENTRAL_DIRECTION_CHANGED`) |
| 9 | `AUTO` direction | MATCH | **MATCH** | 4-vote causal regime (`mtf5m`, `supertrend`, `vwap`, `orb`); yields to explicit Central/Human | Identical 4-vote output across all 5 probed regime combinations |
| 10 | Pending orders | MATCH | **MATCH** | Queued at bar `t-1` close, filled at bar `t` open, gap-stop/gap-target cancel | Identical |
| 11 | Mid-session authority changes | MATCH | **MATCH** | Applied prior to Step A (`<= ts_open`); revalidates queued orders | Identical |
| 12 | Risk `HALTED` (2.5% daily / 3% MTM) | MATCH | **MATCH** | Enters `EXIT_ONLY`, preserves existing positions (no auto-liquidation) | Identical |
| 13 | Stop-loss execution (LONG & SHORT) | DIVERGENCE (`PAR-02`) | **MATCH (FIXED)** | Strict conservative stop-first across OHLC bar | Fixed (`engine.js:978-1012`): `GAP_STOP_LOSS` → `INTRABAR_STOP_LOSS` checked before target |
| 14 | Target execution (LONG & SHORT) | DIVERGENCE (`PAR-02`) | **MATCH (FIXED)** | Target fills only if SL not breached on same bar | Fixed (`engine.js:978-1012`): matches Python on gap-target-then-SL reversals |
| 15 | 14:45 entry cutoff | MATCH | **MATCH** | `14:44` allowed; `14:45` rejected | Identical (`OPEN_GATE_BLOCKED:PAST_1445_OR_REDUCING`) |
| 16 | 15:15 square-off | MATCH | **MATCH** | Mandatory `RISK_EXIT` succeeds under `PAUSE`, `EXIT_ONLY`, `STOP_ALL`, `HALTED`, `MISSING` | Identical (`MANDATORY_1515_SQUAREOFF` under all 5 states) |
| 17 | Shutdown | ARCH DIFF | **DOCUMENTED ARCH DIFF** | `LivePaperTradingRunner` has `finally:` `SESSION_SHUTDOWN_SQUARE_OFF` | Browser simulator is step-by-step replay (`reset()` / `STOP_ALL`, no daemon lifecycle) |
| 18 | Duplicate & delayed instructions | ARCH DIFF | **DOCUMENTED ARCH DIFF** | Structured `InstructionRegistrationResult` + SHA-256 `instruction_audit_trail` | Synchronous browser UI dropdown (`setHumanPermission(str)` without distributed IDs) |
| 19 | Insufficient capital & risk sizing | DIVERGENCE (`PAR-01`) | **MATCH (FIXED)** | 1.0% risk/trade (`cap=100k, px=100, sl=90 -> qty=100`), fee-inclusive cash cap | Fixed (`engine.js:811` `eq * 0.01`): identical `qty=100` and `qty=0` under insufficient capital |
| 20 | Multiple simultaneous symbols | ARCH DIFF | **DOCUMENTED ARCH DIFF** | Multi-symbol portfolio + `CashReservationLedger` + 80% gross cap | Single-symbol browser replay simulator (`this.symbol`, single `this.position`) |

---

## Automated Parity Tests

Both parity regressions (`PAR-01` and `PAR-02`) plus the 5 core REMED-01 scenarios are permanently enforced in:
- `tests/intraday/test_remed01_authority_risk_capital.py::test_remed01_python_js_five_scenario_parity_matrix`
- `tests/intraday/test_autonomous_qa_remediation.py::test_par01_python_js_1pct_risk_per_trade_sizing_parity`
- `tests/intraday/test_autonomous_qa_remediation.py::test_par02_python_js_strict_conservative_stop_first_on_reversal_bars`
