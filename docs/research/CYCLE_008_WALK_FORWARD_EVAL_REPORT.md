# CYCLE-008 — Historical Out-of-Sample Evaluation & Strategy Validation Report

- **Cycle ID**: `CYCLE-008`
- **Phase**: `HISTORICAL_OUT_OF_SAMPLE_EVALUATION_AND_STRATEGY_VALIDATION`
- **Overall Classification**: **`INSUFFICIENT_EVIDENCE`**
- **Confidence Calibration Status**: `UNCALIBRATED`
- **Qualification Allowed**: `false`
- **Unseen Evaluation Status**: `NOT_EVALUATED`
- **Live Trading Enabled**: `false`
- **Code Commit SHA**: `719a191e2448d1e2fbba93e21892511060dff911`
- **Completed At (IST)**: `2026-11-22T15:53:23.088373+05:30`
- **Output Report SHA-256**: `642a8505899ca6a7a8c6ea12c4dd2e5103f96dcfcff5809d461758c509ad86eb`

> **Governance & Evidence Notice**:
> Classification is **`INSUFFICIENT_EVIDENCE`**.
> Reasons: `STAGE1_HISTORICAL_ARCHIVE_ONLY_3_OBSERVATION_DATES`, `ZERO_AVAILABLE_SEALED_HOLDOUT_DATES_48_EXPOSED`, `MARKET_DATA_1H_PROVENANCE_PARTIALLY_VERIFIED_YAHOO_FINANCE`, `COMMITTED_15M_AND_5M_CANDLES_UNAVAILABLE`, `DIRECTIONAL_CONFIDENCE_UNCALIBRATED`.
> `UNCALIBRATED` — Heuristic directional confidence score; not a calibrated probability of a profitable trade.

## 1. Phase 1 — Historical Data Readiness & Provenance Audit

- **1h Archive Provenance**: `PARTIALLY_VERIFIED` (`verified=False`, Yahoo Finance — never upgraded to `VERIFIED`)
- **Common 1h Trading Days**: `245` (`2025-11-21` .. `2026-11-20`)
- **Committed 15m Archive**: `UNAVAILABLE` (never fabricated/substituted)
- **Committed 5m Archive**: `UNAVAILABLE` (never fabricated/substituted)
- **Stage 1 Archives**: `prediction_log`=`3` dates (`['2026-11-16', '2026-11-17', '2026-11-18']`), `forward_predictions`=`3` dates, `market_regimes`=`3` dates, `structured_events`=`280` rows (`collection_timestamp` completeness: `44.29%` — `124/280`)
- **Holdout Registry (`sealed_holdout_registry.json`)**: `48 EXPOSED`, `0 AVAILABLE`, `0 SEALED` (SHA-256: `5fc9013346b5e4209576b98f7f035f8179f0f7c10173acf289badd3faa60e352`)
- **Date Partitioning**: `196` non-holdout trading days (`2025-11-21` .. `2026-09-08`) vs `48` exposed holdout trading days (`2026-09-09` .. `2026-11-20`, strictly forbidden for unseen claims).

### Per-Symbol 1h Archive Integrity

| Symbol | Rows | Days | Date Range | Duplicates | NaNs | OHLC Violations | Timezone | >15% Jumps |
|---|---:|---:|---|---:|---:|---:|---|---:|
| `RELIANCE` | 1703 | 245 | `2025-11-21`..`2026-11-20` | 0 | 0 | 0 | `Asia/Kolkata` | 0 |
| `TCS` | 1696 | 246 | `2025-11-21`..`2026-11-20` | 0 | 0 | 0 | `Asia/Kolkata` | 0 |
| `BIRLACABLE` | 1709 | 246 | `2025-11-21`..`2026-11-20` | 0 | 0 | 0 | `Asia/Kolkata` | 0 |
| `ITC` | 1704 | 245 | `2025-11-21`..`2026-11-20` | 0 | 0 | 0 | `Asia/Kolkata` | 0 |
| `TATAELXSI` | 1691 | 245 | `2025-11-21`..`2026-11-20` | 0 | 0 | 0 | `Asia/Kolkata` | 0 |

## 2. Phase 2 — Chronological Walk-Forward Splits & Data Leakage Audit

| Split | Days | Date Min | Date Max |
|---|---:|---|---|
| `train` | 117 | `2025-11-21` | `2026-05-18` |
| `validation` | 39 | `2026-05-19` | `2026-07-14` |
| `dev_test_oos` | 40 | `2026-07-15` | `2026-09-08` |

- **Leakage Audit Passed**: `true`
- **Exposed Holdout Dates in Walk-Forward**: `0`
- **Lookahead Bar Violations**: `0` (checked `3445` symbol-sessions with `max_feature_ts < as_of_ist`)
- **Parameter Tuning on `dev_test_oos`**: `0`
- **Synthetic Candles / Fabricated Timeframes**: `0` / `0`

## 3. Phase 3 — Non-Holdout Walk-Forward Strategy Comparison (`dev_test_oos`)

| Strategy | Trades | Win Rate | Dir Acc | Net Return % | Net P&L (₹) | Friction (₹) | Profit Factor | Max DD % | Sharpe | Sortino | Avg MAE % | Avg MFE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `independent_candle_timing_agent` | 9 | 0.3333 | 0.6364 | -1.9135% | -956.76 | 219.79 | 0.1944 | 1.9556% | -5.2228 | -5.1459 | -1.5612% | +1.4027% |
| `simple_technical_baseline` | 10 | 0.3000 | 0.5000 | -1.7749% | -887.44 | 321.09 | 0.2286 | 2.0149% | -3.8702 | -3.9857 | -1.2018% | +0.5521% |
| `no_trade_baseline` | 0 | N/A | N/A | +0.0000% | +0.00 | 0.00 | N/A | 0.0000% | N/A | N/A | N/A | N/A |

### Full Non-Holdout Walk-Forward Summary (196 Trading Days)

| Strategy | Trades | Win Rate | Dir Acc | Net Return % | Net P&L (₹) | Friction (₹) | Profit Factor | Max DD % | Sharpe | Sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `independent_candle_timing_agent` | 41 | 0.3415 | 0.3846 | -5.2810% | -2640.49 | 929.53 | 0.3053 | 5.5615% | -3.0726 | -3.2600 |
| `simple_technical_baseline` | 37 | 0.3243 | 0.4324 | -4.0329% | -2016.42 | 1175.29 | 0.5037 | 4.3362% | -1.7982 | -2.2202 |
| `no_trade_baseline` | 0 | N/A | N/A | +0.0000% | +0.00 | 0.00 | N/A | 0.0000% | N/A | N/A |

## 4. Archived Historical Diagnostics (`EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY`)

> These runs cover only the 3 committed archive sessions (`2026-11-17`, `2026-11-18`, `2026-11-19`) inside the exposed holdout window. Verdict: **`INSUFFICIENT_EVIDENCE`**.

| Strategy | Scope | Verdict | Completed / Requested | Trades | Dir Acc | Net P&L (₹) | Friction (₹) |
|---|---|---|---:|---:|---:|---:|---:|
| `central_directional_agent_archived_diagnostic` | `EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY` | **`INSUFFICIENT_EVIDENCE`** | 2/3 | 0 | 0.0000 | +0.00 | 0.00 |
| `combined_cycle007_pipeline_archived_diagnostic` | `EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY` | **`INSUFFICIENT_EVIDENCE`** | 2/3 | 0 | N/A | +0.00 | 0.00 |

## 5. Safety, Isolation, Holdout Integrity & Execution Provenance

- `external_api_calls`: `0` | `broker_connections`: `0` | `paid_api_cost`: `0.0`
- `synthetic_candles_used`: `0` | `visual_ai_enabled`: `false`
- `holdout_registry_unchanged`: `true` (`before=5fc9013346b5e4209576b98f7f035f8179f0f7c10173acf289badd3faa60e352`, `after=5fc9013346b5e4209576b98f7f035f8179f0f7c10173acf289badd3faa60e352`)
- `implementation_commit_sha`: `719a191e2448d1e2fbba93e21892511060dff911`
- `working_tree_clean_at_execution`: `true`
- `dirty_paths_at_execution`: `[]` (clean)
- `report_commit_distinction`: implementation_commit_sha identifies the committed source revision executed to produce this report; the subsequent commit recording updated JSON/Markdown report files is necessarily a descendant commit.
