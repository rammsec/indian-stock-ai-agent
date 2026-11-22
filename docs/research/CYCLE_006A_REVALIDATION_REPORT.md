# BharatAlpha — CYCLE-006A Historical Research Revalidation Report

- **Cycle ID**: `CYCLE-006A`
- **Repository Baseline**: `main` at `61382c8fae7f381bb1fdd52b4b75420560bc9ac7`
- **Machine-Readable Artifact**: [`data/lab/cycle_006a_revalidation_report.json`](../../data/lab/cycle_006a_revalidation_report.json)
- **Classification**: **EXPLORATORY_RESEARCH** (revalidation on previously exposed historical dates; zero new formal qualification or holdout consumption)

---

## A. Executive Summary & Final Assessment

**Final Assessment: `PARTIAL_PASS`** — All completed deterministic strategy backtests across `CYCLE-001` (8/8 deterministic strategies), `CYCLE-002` (9/9 entries), and `CYCLE-003` (8/8 entries across 30m Track A & 60m Track B) reproduced **exact trade counts and net returns (`0.00%` delta)**, all 7 qualification gates and `CYCLE-005` integrity protections remain fail-closed, and all **278/278** regression tests pass. It is classified `PARTIAL_PASS` rather than `PASS` because:
1. Two `CYCLE-001` AI-agent replays (`AGENT-CENTRAL-COUNCIL`, `AGENT-AUTONOMOUS-TRADER`) were **not completed** (`AGENT-CENTRAL-COUNCIL` hit the 600s timeout; `AGENT-AUTONOMOUS-TRADER` was not run per bounded Option A).
2. `data/lab/sealed_holdout_registry.json` has `48/48` dates `EXPOSED` (`0` `AVAILABLE`), and `1h` provenance remains `PARTIALLY_VERIFIED`, so **no genuine unseen predictive qualification** can be granted.
3. The historical `5m` dataset is not committed in `main` and had to be re-fetched from Yahoo Finance (`yfinance`) into `/tmp/cycle_006a_iso`.

---

## B. Historical Revalidation Comparison (`CYCLE-001` to `CYCLE-003`)

All runs executed in an isolated temporary directory (`/tmp/cycle_006a_iso`) using unmodified `ChiefResearchOrchestrator` code (`run_live_pytest=False` inside the orchestrator; full suite run independently in Stage 6).

### 1. `CYCLE-001` (5m Intraday Baseline — 56 Eligible Sessions, 3 Excluded: `2026-08-25 -> 2026-11-20`)

| Candidate | Status | Orig Trades | Rerun Trades | Orig Net % | Rerun Net % | Rerun Gross % | Delta Net % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `STRAT-002-ORB-BREAKOUT` | `COMPLETED` | 25 | 25 | -2.20% | -2.20% | -0.52% | 0.00% |
| `STRAT-003-EMA-CROSS` | `COMPLETED` | 101 | 101 | -10.27% | -10.27% | -3.11% | 0.00% |
| `STRAT-001-VWAP-V2` | `COMPLETED` | 104 | 104 | -18.69% | -18.69% | -11.21% | 0.00% |
| `STRAT-005-RSI-MACD` | `COMPLETED` | 230 | 230 | -29.25% | -29.25% | -11.96% | 0.00% |
| `STRAT-001-VWAP-PULLBACK` | `COMPLETED` | 229 | 229 | -30.44% | -30.44% | -14.95% | 0.00% |
| `STRAT-006-VOLUME-BREAKOUT` | `COMPLETED` | 253 | 253 | -32.30% | -32.30% | -12.81% | 0.00% |
| `STRAT-004-SUPERTREND` | `COMPLETED` | 375 | 375 | -43.74% | -43.74% | -18.59% | 0.00% |
| `STRAT-007-MULTI-INDICATOR` | `COMPLETED` | 522 | 522 | -60.56% | -60.56% | -24.49% | 0.00% |
| `AGENT-CENTRAL-COUNCIL` | `NOT_COMPLETED` (600s timeout) | 433 | — | -55.19% | — | — | — |
| `AGENT-AUTONOMOUS-TRADER` | `NOT_COMPLETED` (not run) | 547 | — | -69.38% | — | — | — |

*Historical-vs-current-code note*: Original `research_cycle_report.json` did not populate `gross_return_pct` (`None`) and had minor fee-breakdown differences (`₹7.44`–`₹176.11` / `±0.04` profit factor) before the `CYCLE-002` fee-schema update.

### 2. `CYCLE-002` (5m vs 15m/30m Discovery — 56 Eligible Sessions: `2026-08-25 -> 2026-11-20`)

| Candidate | Timeframe | Orig Trades | Rerun Trades | Orig Gross % | Rerun Gross % | Orig Net % | Rerun Net % | Delta (All Metrics) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BENCHMARK-CASH-NO-TRADE` *(Benchmark)* | 30m | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | `0` |
| `STRAT-011-30M-BREAKOUT-VOL` | 30m | 26 | 26 | +0.40% | +0.40% | -1.59% | -1.59% | `0` |
| `STRAT-002-ORB-BREAKOUT` | 5m | 25 | 25 | -0.52% | -0.52% | -2.20% | -2.20% | `0` |
| `STRAT-010-30M-EMA-TREND` | 30m | 16 | 16 | -2.78% | -2.78% | -4.02% | -4.02% | `0` |
| `STRAT-003-EMA-CROSS` | 5m | 101 | 101 | -3.11% | -3.11% | -10.27% | -10.27% | `0` |
| `STRAT-012-LOW-FREQ-TREND` | 30m | 32 | 32 | -12.12% | -12.12% | -14.18% | -14.18% | `0` |
| `STRAT-009-15M-VWAP-TREND` | 15m | 157 | 157 | -5.05% | -5.05% | -16.27% | -16.27% | `0` |
| `STRAT-008-15M-RS-LEADER` | 15m | 105 | 105 | -10.84% | -10.84% | -18.70% | -18.70% | `0` |
| `STRAT-001-VWAP-PULLBACK` | 5m | 229 | 229 | -14.95% | -14.95% | -30.44% | -30.44% | `0` |

### 3. `CYCLE-003` (Track A 30m: 56 Sessions; Track B 60m: 240 Sessions, `2025-11-21 -> 2026-11-20`)

| Candidate | Track | Orig Trades | Rerun Trades | Orig Gross % | Rerun Gross % | Orig Net % | Rerun Net % | Delta (All Metrics) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BENCHMARK-CASH-NO-TRADE` *(Benchmark)* | 60m | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | `0` |
| `BENCHMARK-CASH-NO-TRADE` *(Benchmark)* | 30m | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | `0` |
| `STRAT-011-30M-BREAKOUT-VOL` | 30m | 26 | 26 | +0.40% | +0.40% | -1.59% | -1.59% | `0` |
| `STRAT-014-30M-VWAP-SQUEEZE` | 30m | 8 | 8 | -0.99% | -0.99% | -1.61% | -1.61% | `0` |
| `STRAT-013-30M-CONC-BREAKOUT` | 30m | 9 | 9 | -1.18% | -1.18% | -1.86% | -1.86% | `0` |
| `STRAT-017-60M-ORB-FILTERED` | 60m | 109 | 109 | +3.19% | +3.19% | -4.65% | -4.65% | `0` |
| `STRAT-015-60M-TREND-BREAKOUT` | 60m | 84 | 84 | +0.26% | +0.26% | -5.76% | -5.76% | `0` |
| `STRAT-016-60M-VWAP-MOMENTUM` | 60m | 282 | 282 | -7.57% | -7.57% | -28.13% | -28.13% | `0` |

---

## C. Data Provenance, Holdout & Qualification Integrity Audit

### Input Datasets (`Asia/Kolkata`)
- **Committed `1h` Parquet (`data/intraday/*_1h.parquet`, `2025-11-21 -> 2026-11-20`)**:
  - `RELIANCE_1h.parquet`: `1,703` rows (`f8931a85587e923a77214e344e0dde35c8db06db57a810c47ad939a48123dcfe`)
  - `TCS_1h.parquet`: `1,696` rows (`542d5c0b8408957568f9c1354c5500f9a93eba45a15b2e70b9156fa8087dbbf4`)
  - `BIRLACABLE_1h.parquet`: `1,709` rows (`42914a6089a9270cfea784fe6d2084ccb00bd7841e396901f6ce9b72cebb4b26`)
  - `ITC_1h.parquet`: `1,704` rows (`2a5890a1e85ef7502f45e552dce3effdf340176cf33ba49433b4afa9dadcd9be`)
  - `TATAELXSI_1h.parquet`: `1,691` rows (`e028a9f72039613e5a178a89b5e1f06ab22b3847d42b1037fae1fe6e49e3e875`)
- **Fetched `5m` Parquet (`yfinance` -> `/tmp/cycle_006a_iso/data/intraday/*_NSE_5m.parquet`, `2026-08-25 -> 2026-11-20`)**:
  - `RELIANCE_NSE_5m.parquet`: `4,325` rows (`41b6fc7638c305b89018e9a7e46f70460ff176fa6a357958e3a53f1832e3365e`)
  - `TCS_NSE_5m.parquet`: `4,315` rows (`1bc1fcd036a0251309928908d0ee5590d9f0c0b6bf9bb1330a07ca47c03d0a6c`)
  - `BIRLACABLE_NSE_5m.parquet`: `4,354` rows (`bc54d18b370e88df4ad2232470fb4b979994aae563b7214f3059019878da0744`)
  - `ITC_NSE_5m.parquet`: `4,326` rows (`2bf641bd37627f345d90c4c2b68d1d70be77e2f7db6bea22990dfe96927529a0`)
  - `TATAELXSI_NSE_5m.parquet`: `4,310` rows (`4e1ee21e66c3f784c7458672991cee451a55908d86faf9f18e1dd13719fead44`)

### Holdout & `CYCLE-004` / `CYCLE-005` / 7-Gate Audit
- **Holdout registry**: Real [`sealed_holdout_registry.json`](../../data/lab/sealed_holdout_registry.json) remains unchanged (`48/48` `EXPOSED`, `0` `AVAILABLE`, `0` `SEALED`).
- **`CYCLE-004` & `CYCLE-005` (`cycle_005_report.json` preserved, `cycle_005_report_v2.json` active)**:
  - `CYCLE-005` v2 records `evaluation_type = EXPLORATORY_TRAIN_VAL_ONLY`, `provenance_status = PARTIALLY_VERIFIED`, `qualification_allowed = False`, `unseen_return_pct = null`, `unseen_evaluation_status = NOT_EVALUATED`, and `Gate 2 = INSUFFICIENT_EVIDENCE` (no `0.0%` coercion).
- **7-Gate Qualification (`evaluate_seven_gates`, `trader/research/qualification.py`)**:
  - Gate 1 (`Data & Holdout Integrity`): `ResearchIntegrityGuard` + `prov_ok`/`hold_ok` — fail-closed (`BLOCKED_INTEGRITY` / `INVALIDATED`).
  - Gate 2 (`Net Profitability`): checks `net_ret > 0`, `net_pnl > 0`, and `unseen_ret > 0`; marks `INSUFFICIENT_EVIDENCE` when `unseen_metrics` is `NOT_EVALUATED` or `None`.
  - Gate 3 (`Benchmark Comparison`): `alpha_bh > 0` and `net_ret > 0`.
  - Gate 4 (`Risk Controls`): `MaxDD <= 10%`, `PF > 1.20`, `Sharpe > 1.0`, `Expectancy > 0`.
  - Gate 5 (`Statistical Reliability`): `sessions >= 60`, `trades >= 30`.
  - Gate 6 (`Execution Robustness`): `10 bps` slippage net return `> 0%`.
  - Gate 7 (`Independent Approval`): `StatisticalValidationAgent == PASS` and `StrategyChallengerRedTeamAgent == APPROVED`.

---

## D. Regression & Immutability Verification

- **Targeted PR #10/#11/#12 audit suite** (60s budget): **21/21 passed in 1.99s** (`0` failures, `0` skips).
- **Full regression suite** (`.venv/bin/pytest tests/intraday/ tests/phase6/ -v --tb=short`, 10m budget): **278/278 passed in 339.10s (0:05:39)** (`0` failed, `0` skipped).
- **SHA-256 Immutability**: All **18 protected files** (all historical cycle reports, `sealed_holdout_registry.json`, `provenance_report.json`, `lab_index.json`, and all 5 `*_1h.parquet` files) verified byte-for-byte unchanged (`ALL_18_PROTECTED_FILES_SHA256_MATCH`).
- **Working tree**: Transient test side effects cleaned; only the two new `CYCLE-006A` report files exist as untracked outputs.

---

## E. Outstanding Risks & Recommendation

1. **Performance bottleneck (`df.iterrows()` in `trader/data/loader.py:139, 186`)**: Causes single 5m strategy replays to take ~170–350s and the full test suite to take ~5m39s.
2. **Zero `AVAILABLE` holdout dates**: All `48` tracked dates in `sealed_holdout_registry.json` are `EXPOSED`, and `1h` provenance is `PARTIALLY_VERIFIED`.
3. **Readiness recommendation**: The deterministic historical research foundation (`CYCLE-001`–`CYCLE-003` + `CYCLE-005` integrity controls) is **reproducible and technically sound** for exploratory research. Before launching `CYCLE-006` integrated paper trading, archive fresh post-`2026-11-20` sessions if unseen qualification is desired, and consider vectorizing `trader/data/loader.py` to remove the 5m replay bottleneck.
