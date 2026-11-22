# CYCLE-006C — Real-Data Validation & Input Integrity Report

- **Task ID**: `CYCLE-006C-REAL-DATA-20261122`
- **Cycle ID**: `CYCLE-006C`
- **Session Date**: `2026-11-19` (as-of cutoff `2026-11-19T09:15:00+05:30`)
- **Primary Classification**: **`PASS_INFRASTRUCTURE_WITH_ZERO_TRADES`**
- **Qualification Allowed**: `False`
- **Unseen Evaluation Status**: `NOT_EVALUATED`
- **Generated At (IST)**: `2026-11-22T13:18:43.307361+05:30`

> **Scope Notice**: This report validates paper-trading infrastructure execution on genuine archived market and intelligence data only. It makes **no** strategy profitability claim, **no** formal qualification claim, consumes **zero** holdout dates, and executes **zero** live broker or paid API calls.

## 1. Root Cause of CYCLE-006B Input Misalignment

phase5_intelligence.json was refreshed on 2026-11-20T17:29:48+05:30, overwriting generated_at and per-stock observation_date to 2026-11-20 while latest_predictions.json and docs/data/regime.json remained at observation_date=2026-11-18 (pre-market for 2026-11-19). CYCLE-006B correctly failed closed to prevent lookahead/misalignment.

| Committed File | `generated_at` | `observation_date` |
|---|---|---|
| `data/predictions/latest_predictions.json` | `2026-11-18T21:04:26.529445+00:00` | `['2026-11-18']` |
| `docs/data/regime.json` | *(none)* | `2026-11-18` |
| `data/predictions/phase5_intelligence.json` | `2026-11-20T17:29:48.155332+05:30` | `['2026-11-20']` |

## 2. Genuine Historical As-Of Snapshot Reconstruction

- **Status**: `RECONSTRUCTED_FROM_ARCHIVES` (`ARCHIVED_PRE_CUTOFF_RECONSTRUCTED`)
- **Target Observation Date**: `2026-11-18`
- **Prediction Generated At (IST)**: `2026-11-19T02:34:26.591210+05:30`
- **Regime Availability (IST)**: `2026-11-19T02:34:26.591210+05:30`
- **Intelligence Generated At (IST)**: `2026-11-19T02:34:53.810677+05:30`
- **Max Verified Event Collection (IST)**: `2026-11-19T02:34:53.810677+05:30`
- **Verified Pre-Cutoff Events**: `81` / `280`
- **Per-Symbol Verified Events**: `{'RELIANCE': 19, 'TCS': 31, 'BIRLACABLE': 9, 'ITC': 15, 'TATAELXSI': 7}`
- **Excluded Events Breakdown**: `{'missing_collection_timestamp': 156, 'missing_publication_timestamp': 0, 'published_after_cutoff': 4, 'collected_after_cutoff': 39, 'outside_target_universe': 0}`

## 3. Real-Data Paper-Trading Execution Summary (`CYCLE-006B` Engine)

- **Run Status**: `COMPLETED`
- **Classification**: `PASS_INFRASTRUCTURE_WITH_ZERO_TRADES`
- **Market Regime**: `Weak Bull`
- **Initial Capital**: `₹50,000.00`
- **Ending Cash / Equity**: `₹50,000.00` / `₹50,000.00`
- **Net P&L**: `₹0.00` (`0.0000%`)
- **Total Transaction Friction**: `₹0.00` (brokerage `₹0.00`, statutory `₹0.00`, slippage `₹0.00`)
- **Risk Decisions Evaluated**: `5`
- **Orders Created / Filled**: `0` / `0`
- **Zero-Trade Explanation**: All 5 genuine pre-market predictions for `2026-11-19` had `prob_next_day_up` between `0.4618` and `0.5049` (below the `0.55` buy threshold) and `expected_next_day_return_pct` between `-0.4444%` and `+0.0226%` (below the `0.25%` threshold). The engine legitimately stood aside with zero orders.
- **Ledger Events**: `2` | **SHA-256 Chain Valid**: `True` | **Equity Reconciles**: `True` (`diff=0.0`)
- **Synthetic Candles Used**: `0`
- **External API Calls**: `0` | **Broker Connections**: `0` | **Paid API Cost**: `$0.00`

## 4. Timeframe Provenance Classification (`1h`, `5m`, `15m`)

| Timeframe | Committed Repo Classification | Isolated Yahoo Finance Cache Classification | Notes |
|---|---|---|---|
| **1h** | `PARTIALLY_VERIFIED` | `PARTIALLY_VERIFIED` | Committed `data/intraday/*_1h.parquet`, fingerprint `e508f71047493e5db1aaaee3ceb16d4db0eb2199a41a842743b077b7bb986394` |
| **5m** | `UNAVAILABLE` | `PARTIALLY_VERIFIED` | Not committed in repo; isolated Yahoo Finance cache is non-exchange-authoritative with minor missing bars on `2026-11-19` |
| **15m** | `UNAVAILABLE` | `PARTIALLY_VERIFIED` | Not committed in repo; deterministic completed-bar aggregation from 5m (no interpolation) |

## 5. Governance & Safety Attestation

- `qualification_allowed`: `false`
- `unseen_evaluation_status`: `NOT_EVALUATED`
- `holdout_registry_unchanged`: `true`
- `live_trading_enabled`: `false`
- `cycle_007_implemented`: `false`
