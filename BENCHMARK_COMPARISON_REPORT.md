# BharatAlpha Benchmark Comparison Report

**Generated**: `2026-11-19` | **Module**: `backend/backtesting/benchmark_eval.py` | **Output**: `data/backtests/benchmark_comparison.json`

## 1. Executive Summary
Under strict look-ahead exclusion, daily `(symbol, observation_date)` deduplication, and minimum sample-size gating (`>= 30` matured observations and `>= 10` unique observation dates), **all three live forward horizons (`1D`, `5D`, `10D`) are currently `INSUFFICIENT_DATA`**.

Key audit corrections applied relative to `docs/data/forward_validation.json` and `docs/data/phase5_intelligence.json`:
1. **`NIFTY_50` Benchmark Lookup Fixed**: `data/processed/indices_daily.parquet` stores Nifty 50 as `NIFTY_50` (495 rows), whereas `forward_validation.py` queried `"NIFTY 50"`, causing `nifty_50_buy_hold` to report `UNAVAILABLE` and all `benchmark_return_pct` values to be `null`.
2. **Look-Ahead Timestamp Leak Excluded**: Snapshots with `observation_date = 2026-11-17` had `prediction_timestamp = 2026-11-18T17:32:47+05:30` (post-close on Nov 18). Requiring `entry_date > max(observation_date, prediction_timestamp[:10])` properly classifies Nov 17 `POST_MARKET` snapshots as `PENDING` (awaiting Nov 19 open) rather than evaluating them on Nov 18 prices.
3. **Same-Day Session Double-Counting Eliminated**: `65` raw snapshots collapse to `15` unique `(symbol, observation_date)` daily canonical snapshots (`5` pilot symbols × `3` dates: `2026-11-16`..`2026-11-18`).

## 2. Live Forward Horizon Summary (`data/backtests/benchmark_comparison.json`)

| Horizon | Canonical Daily Snapshots | Matured Observations | Pending Observations | Unique Dates | Status | Gating Reason |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1D** | 15 | **5** (`2026-11-16` -> `2026-11-17`) | 10 | **1** | `INSUFFICIENT_DATA` | `Matured observations=5 (min 30), unique_dates=1 (min 10).` |
| **5D** | 15 | **0** | 15 | **0** | `INSUFFICIENT_DATA` | `Matured observations=0 (min 30), unique_dates=0 (min 10).` |
| **10D** | 15 | **0** | 15 | **0** | `INSUFFICIENT_DATA` | `Matured observations=0 (min 30), unique_dates=0 (min 10).` |

## 3. Strategy & Benchmark Comparison (All Horizons Gated)

| Strategy | 1D Status | 5D Status | 10D Status | Cost Assumption | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `bharat_alpha` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | 0.347% CNC | Metrics gated to `null` until `n >= 30` & `dates >= 10` |
| `naive_momentum` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | 0.347% CNC | Horizon-matched `benchmark_prob_{h}_up` |
| `equal_weight_universe` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | 0.347% CNC | Identical `[entry_date, exit_date]` windows |
| `nifty_50_buy_hold` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` | 0.347% CNC | `NIFTY_50` matched in `indices_daily.parquet` |
| `cash_no_trade` | `OK` (`0.0%`) | `OK` (`0.0%`) | `OK` (`0.0%`) | 0.000% | Risk-free 0% cash baseline |

## 4. Historical Walk-Forward Reference (`docs/data/backtest_summary.json`)
For historical context only (separate 15-symbol universe, `2025-11-14` to `2026-11-17`, 6 purged/embargoed folds, `3,735` OOS rows):
- `ml_ensemble`: `72` active stock-days (`19` executed trades), net return `-6.77%`, Sharpe `-2.448`, ROC-AUC `0.477`.
- `naive_momentum`: `302` active stock-days (`58` executed trades), net return `+10.48%`, Sharpe `+1.016`, ROC-AUC `0.5104`.
- `nifty_50_buy_hold`: `-1.9%` (`close`-to-`close`, 0 cost); `equal_weight_universe`: `+0.11%` (`close`-to-`close`, 0 cost).

## 5. Next Steps for Statistical Readiness
- Continue daily prospective snapshot collection until at least **10+ unique trading dates** and **30+ matured `(symbol, observation_date)` observations** accumulate for `1D`, `5D`, and `10D` before drawing any performance conclusions.
