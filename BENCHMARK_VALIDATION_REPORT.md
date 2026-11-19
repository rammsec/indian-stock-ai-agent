# Benchmark & Chronology Validation Report (Workstream 2)

**Branch:** `remediation/chronology-benchmark`  
**Engine:** `backend/backtesting/chronology_benchmark.py`  
**Test Suite:** `tests/test_remediation_chronology_benchmark.py`  
**Freeze Compliance:** Frozen prediction logic (`backend/prediction/`, `backend/features/`, `backend/news_intel/features.py`, `combiner.py`, `pipeline.py`, `forward_validation.py`) and original `data/` files were **not modified**.

---

## Issue 1: Look-Ahead Chronology Violations (`entry_open_ts <= prediction_timestamp`)

- **Original finding:** Forward outcome evaluation allowed trades to enter at `09:15 IST` on `entry_date` even when `prediction_timestamp` occurred after `entry_date 09:15 IST` (e.g., `observation_date=2026-11-16`, `prediction_timestamp=2026-11-17T10:25:29.884689+05:30`, `entry_date=2026-11-17`, lead time `-70.50` minutes).
- **Root cause:** `backend/news_intel/forward_validation.py` (line 368) resolved entry strictly using `grp["observation_date"] > obs_dt`, ignoring `prediction_timestamp` and NSE session timing. Additionally, `backend/backtesting/benchmark_eval.py` (lines 228–237) used date-only string truncation `cutoff_dt = max(obs_dt, pred_dt)` with `>`, which prevented same-day `PRE_MARKET` execution while lacking an explicit `entry_open_ts > prediction_timestamp` timestamp invariant.
- **Affected files & records:**
  - `backend/news_intel/forward_validation.py` (line 368)
  - `data/predictions/forward_outcomes.parquet`: **45 of 55** `MATURED` rows (`81.8%`) violate `entry_open_ts > prediction_timestamp` (e.g., `fwd_BIRLACABLE_2026-11-16_20261117T102529`, `1D`, `-70.4981` min).
- **Remediation:** Implemented `resolve_eligible_entry_session()` and `enforce_prediction_before_entry()` in `backend/backtesting/chronology_benchmark.py` enforcing `entry_open_ts > prediction_timestamp` (where `entry_open_ts` is `09:15:00+05:30` on the eligible NSE trading day).
- **Tests:** `test_prediction_timestamp_strictly_before_entry_open` in `tests/test_remediation_chronology_benchmark.py`.
- **Evidence of successful execution:** On `data/predictions/forward_predictions.parquet` (65 raw rows -> 15 canonical daily snapshots, 45 horizon evaluations), `chronology_valid == True` for `45/45` (`100%`) rows; all `1D` entries shift to `2026-11-17` (5), `2026-11-19` (5), and `2026-11-20` (5).
- **Remaining limitations:** Legacy `data/predictions/forward_outcomes.parquet` remains unmodified on disk per freeze rules; downstream consumers must read via `chronology_benchmark.py` rather than the legacy `forward_outcomes.parquet`.

---

## Issue 2: Session-Unaware Execution (`PRE_MARKET`, `INTRADAY`, `POST_MARKET`) and NSE Holiday/Weekend Handling

- **Original finding:** Session classification (`PRE_MARKET` `< 09:15 IST`, `INTRADAY` `09:15–15:30 IST`, `POST_MARKET` `> 15:30 IST`) and NSE trading holidays/weekends were not enforced when scheduling entry and horizon exit dates.
- **Root cause:** `backend/backtesting/benchmark_eval.py` (lines 229–237, 271) indexed directly into symbol bar positions (`grp.index[after_mask][0]` and `entry_idx + (hold_n - 1)`) without consulting `NSE_TRADING_HOLIDAYS` or distinguishing `PRE_MARKET` (`< 09:15 IST`) from `INTRADAY`/`POST_MARKET`.
- **Affected files & records:**
  - `backend/backtesting/benchmark_eval.py` (lines 228–275)
  - `data/predictions/forward_predictions.parquet` (`PRE_MARKET`: 10 rows, `INTRADAY`: 30 rows, `POST_MARKET`: 25 rows).
- **Remediation:** Implemented `classify_nse_session()`, `is_nse_trading_day_cal()`, `next_nse_trading_day()`, `add_nse_trading_days()`, and `resolve_eligible_entry_session()` in `backend/backtesting/chronology_benchmark.py` using `NSE_TRADING_HOLIDAYS` from `backend/config.py`:
  - `PRE_MARKET` on trading day `T` -> `T 09:15 IST`
  - `INTRADAY` or `POST_MARKET` on trading day `T` -> `T_next 09:15 IST`
  - Holidays/weekends -> `T_next 09:15 IST`
- **Tests:** `test_pre_market_intraday_and_post_market_sessions` and `test_nse_holidays_and_weekends_roll_forward` in `tests/test_remediation_chronology_benchmark.py`.
- **Evidence of successful execution:** Both session boundary tests (`09:14:59`, `09:15:00`, `15:30:00`, `15:30:01`, UTC `03:00:00Z`) and NSE holiday/weekend tests (`2026-10-02` Gandhi Jayanti + weekend rolling to `2026-10-05`, 5D calendar advance skipping holidays) passed.
- **Remaining limitations:** `NSE_TRADING_HOLIDAYS` in `backend/config.py` covers 2025–2026; future calendar years require updating `NSE_TRADING_HOLIDAYS`.

---

## Issue 3: Benchmark Symbol Mismatch (`NIFTY 50` vs `NIFTY_50`)

- **Original finding:** `NIFTY 50` benchmark returns were completely missing (`None`) in `forward_outcomes.parquet` (`0/195` populated).
- **Root cause:** `backend/news_intel/forward_validation.py` (lines 344, 1137) filtered `indices_df["index_code"].str.upper() == "NIFTY 50"`, whereas `data/processed/indices_daily.parquet` stores `"NIFTY_50"` (495 rows, 0 rows for `"NIFTY 50"`).
- **Affected files & records:**
  - `backend/news_intel/forward_validation.py` (lines 344, 1137)
  - `data/predictions/forward_outcomes.parquet`: **195 of 195** rows have `benchmark_return_pct = None`.
  - `data/processed/indices_daily.parquet`: **495** `NIFTY_50` rows ignored by legacy query.
- **Remediation:** Implemented centralised `normalize_benchmark_symbol()` in `backend/backtesting/chronology_benchmark.py` mapping `"NIFTY 50"`, `"NIFTY_50"`, `"NIFTY50"`, `"NIFTY-50"`, `"^NSEI"` -> `"NIFTY_50"`, and used it in `build_chronology_aligned_outcomes()`.
- **Tests:** `test_benchmark_symbol_normalisation` in `tests/test_remediation_chronology_benchmark.py`.
- **Evidence of successful execution:** On `data/processed/indices_daily.parquet`, `chronology_benchmark.py` matched `5/5` matured `1D` observations with valid `NIFTY_50` returns (`valid_nifty50_comparisons=5`).
- **Remaining limitations:** Legacy `forward_outcomes.parquet` retains `None` for `benchmark_return_pct` because original `data/` files are frozen.

---

## Issue 4: Missing/Invalid Price & Date Handling and Premature Metric Reporting

- **Original finding:** Missing equity bars between `entry_date` and `exit_date` could silently shift exit indices or remain `PENDING` indefinitely; non-positive/NaN prices were labeled `"INVALID"` in `benchmark_eval.py`; `cash_no_trade` reported `status="OK"` and `0.0%` return even when all other strategies were `INSUFFICIENT_DATA`; and metrics were not gated on having a valid `NIFTY_50` benchmark comparison.
- **Root cause:** `backend/backtesting/benchmark_eval.py` (lines 271–284, 550–562) used positional bar indexing instead of trading-calendar dates and lacked a benchmark-validity gate for `cash_no_trade` and strategy metrics.
- **Affected files & records:**
  - `backend/backtesting/benchmark_eval.py` (lines 271–284, 550–562)
  - `data/backtests/benchmark_comparison.json` (`cash_no_trade` reported `status="OK"` with `matured_observations=5` < `min_observations_required=30`).
- **Remediation:** In `backend/backtesting/chronology_benchmark.py`:
  - Trading-calendar exit date resolution (`add_nse_trading_days`) separates `PENDING` (`ref_date < exit_d`) from `UNAVAILABLE` (past `exit_d` with missing bar, missing/non-positive price, or invalid timestamp/date).
  - Independent `1D`, `5D`, `10D` evaluation across `bharat_alpha`, `naive_momentum`, `equal_weight`, `NIFTY_50`, and `cash_no_trade` gates **all** strategies (including `cash_no_trade`) to `INSUFFICIENT_DATA` / `UNAVAILABLE` with `None` metrics whenever `matured_observations < min_samples`, `unique_observation_dates < min_unique_dates`, or `valid_nifty50_comparisons` is insufficient.
- **Tests:** `test_missing_and_invalid_price_data_marked_unavailable` and `test_1d_5d_10d_outcome_maturity_and_benchmark_gating` in `tests/test_remediation_chronology_benchmark.py`.
- **Evidence of successful execution:** Running `run_chronology_benchmark()` on the live `data/` snapshot yields `status="INSUFFICIENT_DATA"` across all 3 horizons (`1D`: 5 matured / 10 pending / 1 unique date; `5D`: 0 matured / 15 pending; `10D`: 0 matured / 15 pending) with all strategy metrics gated to `None`.
- **Remaining limitations:** Live prospective sample size (`n=5` matured on `1D`, `n=0` on `5D`/`10D`) remains below the `min_samples=30, min_unique_dates=10` statistical threshold until additional NSE sessions elapse.

---

## Execution Summary

Command run in `/home/antigravity/worktrees/rem-chronology-benchmark`:
```bash
/home/antigravity/.venv/bin/pytest tests/test_remediation_chronology_benchmark.py -v
```
Result: **6 passed in 1.45s** (100% pass rate).
