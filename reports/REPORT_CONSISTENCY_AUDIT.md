# BharatAlpha AI — Report Consistency & Data Integrity Audit

**Branch:** `feature/data-integrity-audit`  
**Base Commit SHA (`main`):** `8f328b8cdb8f7a1e508c1c47a648849945873013`  
**Audit Date:** 2026-11-18  
**Scope:** Read-only root-cause investigation of reporting discrepancies between `docs/data/forward_validation.json` (Report A) and `docs/data/phase5_intelligence.json` (Report B) without modifying production prediction logic, thresholds, schedules, or historical snapshots.

---

## 1. Executive Summary

The two published monitoring reports (`docs/data/forward_validation.json` and `docs/data/phase5_intelligence.json`) are **not** measuring the same underlying dataset, model, universe, or outcome definition:

| Dimension | Report A (`forward_validation.json`) | Report B (`phase5_intelligence.json` → `monitoring`) |
| :--- | :--- | :--- |
| **Reported `model_version`** | `5.3.0-pilot` (hardcoded subsystem label) | `5.0.0-pilot` (hardcoded subsystem label; underlying log is `ml_ensemble 3.2.0` + `3.0.0`) |
| **Underlying Store** | `data/predictions/forward_predictions.parquet` (55 rows) + `forward_outcomes.parquet` (165 rows) | `data/predictions/prediction_log.parquet` (75 rows) |
| **Stock Universe** | 5 pilot stocks (`RELIANCE`, `TCS`, `TATAELXSI`, `ITC`, `BIRLACABLE`) | 15 quant universe stocks |
| **Deduplication** | Filtered to `is_duplicate_run == False` (`20` canonical snapshots; `35` duplicate runs) | **None** — every run's `prediction_id` includes `run_tag`, so retries/re-runs are counted independently |
| **1D Outcome Definition** | `open[t+1] → close[t+1]` (`evaluate_forward_outcomes`) | `close[t] → close[t+1]` (`DuckDBStore.evaluate_realized_outcomes`) |
| **Reported Evaluated / Matured** | `10` matured (out of `60` canonical outcome rows across 1D/5D/10D) | `60` evaluated outcomes (`eval_df.tail(50)` used for metrics) |
| **1D Accuracy / ROC-AUC** | `40.0%` / `0.6667` (on 10 rows; 5 prospective + 5 look-ahead contaminated) | `60.0%` / `0.7452` (computed on `tail(50)` of 60 un-deduped rows) |

**Authoritative Source:** Neither report should be merged with the other, and neither should have its numbers forced to match. The **canonical forward stores (`forward_predictions.parquet` + `forward_outcomes.parquet`)** are the authoritative source for prospective Phase 5 validation—**only after** timing-based reclassification (separating genuinely prospective observations from look-ahead-contaminated `INTRADAY` and post-close `POST_MARKET` snapshots). At the current baseline (`8f328b8`), only **5** 1D observations are genuinely prospective and matured (insufficient for statistical inference against `MIN_FORWARD_SAMPLES = 10`).

---

## 2. Source of the 60 Evaluated Outcomes (Report B: `phase5_intelligence.json`)

1. **Code Path:** `backend/news_intel/monitoring.py::build_monitoring_snapshot` (L113) calls `_compute_horizon_metrics` (L21), which reads `data/predictions/prediction_log.parquet` directly without deduplication.
2. **Row Generation:** `scripts/run_phase3_backtest.py` (~L205) writes 15 rows per run via `DuckDBStore.upsert_predictions` (`backend/storage/duckdb_store.py:338`), keyed only on `prediction_id`. In `backend/prediction/ensemble_model.py:707`, `prediction_id` is constructed as `{sym}_{obs_date}_{ensemble.name}_{ensemble.version}_{run_tag}`. Every re-run therefore creates a new row.
3. **Exact Composition of the 75 Rows:**
   - **2026-11-16 (60 evaluated rows = 4 runs × 15 stocks):**
     - `05:42:38 UTC` (`ml_ensemble 3.0.0`, 15 rows)
     - `06:47:59 UTC` (`ml_ensemble 3.2.0`, 15 rows)
     - `07:03:07 UTC` (`ml_ensemble 3.2.0`, 15 rows)
     - `13:34:55 UTC` (`ml_ensemble 3.2.0`, 15 rows)
     - All three `3.2.0` runs have identical `prob_next_day_up` per symbol (the exact same forecast counted 3 times).
   - **2026-11-17 (15 pending rows):** generated `21:30:07 UTC` (`ml_ensemble 3.2.0`), not yet evaluated.
4. **Silent `tail(50)` Truncation (`monitoring.py:50`):**
   - `evaluated_outcomes_count` reports `len(eval_df) = 60`, while rolling accuracy (`60.0%`), ROC-AUC (`0.7452`), Brier (`0.2436`), and Rank IC (`-0.0671`) are computed on `recent = eval_df.tail(50)`.
   - The `tail(50)` slice silently drops 10 rows (`BHARTIARTL` ×4, `BIRLACABLE` ×4, `HDFCBANK` ×2—including pilot stock `BIRLACABLE`) and mixes `3.2.0` (38 rows) with `3.0.0` (12 rows).
   - When deduplicated to the 15 unique `(symbol, observation_date)` pairs on `2026-11-16`, 1D accuracy is **73.33%** across the 15-stock universe and **80.0%** on the 5 pilot stocks.
5. **Outcome Definition:** `DuckDBStore.evaluate_realized_outcomes` (`backend/storage/duckdb_store.py:355-432`) computes `close[t] → close[t+1]` for 1D, whereas forward validation uses `open[t+1] → close[t+H]`.

---

## 3. Source of the 10 Matured Outcomes (Report A: `forward_validation.json`)

1. **Code Path:** `backend/news_intel/forward_validation.py::build_forward_validation_payload` (L1248) reads `forward_predictions.parquet` (55 rows, 5 pilot stocks) and filters to `is_duplicate_run == False` (`20` canonical rows), set by `deduplicate_forward_predictions` (L202) keyed on `(symbol, observation_date, nse_session)`.
2. **Outcome Rows:** `forward_outcomes.parquet` contains 165 rows total (55 predictions × 3 horizons), of which 60 belong to the 20 canonical predictions:
   - **1D:** `10 MATURED` + `10 PENDING`
   - **5D:** `20 PENDING`
   - **10D:** `20 PENDING`
   - `matured_outcomes_total = 10` sums across all horizons.
3. **Inspection of the 10 Matured Canonical 1D Rows (all `entry_date = exit_date = 2026-11-17`):**
   - **5 × `obs = 2026-11-16`, `POST_MARKET`** (`2026-11-16T19:23:27 IST`, `run_id = NaN`): **genuinely prospective** (predicted before `2026-11-17T09:15:00+05:30` entry open).
   - **5 × `obs = 2026-11-16`, `INTRADAY`** (`2026-11-17T10:25:29 IST`, `run_id = NaN`): **look-ahead contaminated (`INVALID`) & duplicate target** — generated 70 minutes *after* the `2026-11-17 09:15 IST` entry open, targeting the exact same 1D session with near-identical probabilities (e.g., `RELIANCE` `0.5183` vs `0.5232`) and identical realized returns.
   - Because `deduplicate_forward_predictions` keys on `(symbol, observation_date, nse_session)`, `POST_MARKET` and `INTRADAY` for `obs=2026-11-16` were both kept as "canonical", doubling the 1D matured count from **5** independent prospective observations to **10**.
4. **Inspection of the 10 Pending Canonical 1D Rows (`obs = 2026-11-17`, both targeting `2026-11-18`):**
   - **5 × `PRE_MARKET`** (`2026-11-18T03:00:20 IST`, GH run `44180088619`): **genuinely prospective (`PENDING`)**.
   - **5 × `POST_MARKET`** (`2026-11-18T17:32:47 IST`, GH run `44300078913`): generated **after** the `2026-11-18 15:30 IST` close (`equities_daily` max date is `2026-11-17`), so the 1D outcome (`2026-11-18` open→close) had already occurred (`HISTORICAL` for 1D, `INVALID` look-ahead for 5D/10D).

---

## 4. Delayed Cron Runs, Session Mislabelling & Staleness

1. **Inverted Session Labels via Wall-Clock Time:**
   - `.github/workflows/daily_nse_research.yml` (L60-87) selects the pre-market branch when `github.event.schedule == '30 3 * * 1-5'` (03:30 UTC / 09:00 IST) and the post-market branch otherwise (`00 13 * * 1-5`, 13:00 UTC / 18:30 IST).
   - `resolve_nse_session` (`backend/news_intel/forward_validation.py:68`) ignores workflow branch and classifies session solely from wall-clock IST time (`<09:15` → `PRE_MARKET`, `09:15–15:30` → `INTRADAY`, `>15:30` → `POST_MARKET`).
   - **GH Run `44180088619`** (created `2026-11-17T21:27:57Z`, commit `efa8f3c`): post-market cron delayed ~8.5h; ran the full post-market pipeline (`equities_daily`, `prediction_log`, `latest_predictions`) at `2026-11-18T03:00:20 IST` and was mislabelled `PRE_MARKET`.
   - **GH Run `44300078913`** (created `2026-11-18T12:02:05Z`, commit `8f328b8`): pre-market cron delayed ~8.5h; ran the pre-market Phase 5 branch at `2026-11-18T17:32:47 IST`, reused stale `latest_predictions.json` (generated `14.54h` earlier at `2026-11-18T03:00:07 IST`, `obs=2026-11-17`), and was mislabelled `POST_MARKET`.
2. **Staleness & Timestamp Ordering:**
   - `docs/data/` and `data/predictions/` JSONs are in sync (`forward_validation.json` and `phase5_intelligence.json` were both written in run `44300078913` at `17:32:47`–`17:33:01 IST`).
   - `phase5_intelligence.json` sets `stocks[*].observation_date = "2026-11-18"` (wall-clock `now`) while the underlying quant inputs and snapshots have `observation_date = "2026-11-17"`.
   - `phase5_intelligence.json` sets `generated_at` / `latest_prediction_at` (`17:32:47.787090+05:30`) at pipeline start, while 60 timeline events carry `collection_timestamp` ~13.8s later (`17:33:01 IST`), making it appear as if news collected *after* prediction generation was used.

---

## 5. Additional Integrity Findings (Model Versions, News, Benchmarks)

1. **Model Version Labels Are Hardcoded Subsystem Tags:**
   - `5.0.0-pilot` is hardcoded in `backend/news_intel/pipeline.py:177` and `backend/models.py:766`.
   - `5.3.0-pilot` is hardcoded in `backend/news_intel/forward_validation.py:121, 504` and `backend/models.py:801, 878, 999`.
   - The actual quant ensemble is `ml_ensemble 3.2.0` (`backend/prediction/ensemble_model.py:71`), with legacy `3.0.0` rows in `prediction_log.parquet`.
2. **Benchmark Key Mismatch (`NIFTY_50` vs `"NIFTY 50"`):**
   - `indices_daily.parquet` stores `index_code = "NIFTY_50"` (494 rows), whereas `forward_validation.py:344` queries `"NIFTY 50"`. Consequently, `benchmark_return_pct` is `None` on all 45 `MATURED` rows, and `nifty_50_buy_hold` reports `status = "UNAVAILABLE"` with no explanatory `reason` field.
   - `paper_portfolio.benchmarks` also retains an unlabelled `naive_momentum` 1D alias alongside `naive_momentum_1d/5d/10d`.
3. **News Quarantine Leak & Duplicates:**
   - `19` `(ticker, url)` pairs (16 rows in `structured_events.parquet`) appear in **both** `structured_events.parquet` and `quarantined_events.parquet` (including `ITC` `066b50de4daebafe` / `dd90edd5827f6680` and `TATAELXSI` `0f6f95d76eeab2bf`).
   - `156` historical backfill rows lack `collection_timestamp` (`validation_status = NaN`).
   - `29` unflagged duplicate `(ticker, source_url)` groups exist where the same article is stored under multiple `event_category` values without `is_duplicate_report = True`.

---

## 6. Authoritative Source & Recommended Corrections

### Authoritative Source
- **Prospective Validation:** `data/predictions/forward_predictions.parquet` + `data/predictions/forward_outcomes.parquet`, filtered by the read-only timing classifier (`backend/audit/classification.py`). Under strict prospective rules (`prediction_timestamp < entry_open`, deduplicated per `(symbol, target_entry_date, horizon)`), the current 1D dataset has **5** matured prospective observations (accuracy `40.0%`, `n < 10`, statistically insufficient) and **5** pending prospective observations.
- **Legacy Quant Log:** `prediction_log.parquet` is a **non-authoritative** legacy quant-only log (`close[t]→close[t+1]`) and must be labelled as such rather than compared against `forward_validation.json`.

### Recommended Engineering Corrections (Post-Freeze / Audit-Tracked)
1. **Derive `nse_session` from workflow branch / `github.event.schedule`**, not wall-clock time, so delayed crons cannot invert `PRE_MARKET` and `POST_MARKET` labels.
2. **Enforce prediction-timestamp invariant (`prediction_timestamp < entry_open`)** at snapshot ingestion or quarantine post-open snapshots as `INVALID` / `HISTORICAL` rather than mixing them into prospective metrics.
3. **Deduplicate forward predictions by `(symbol, target_entry_date, horizon)`** so `POST_MARKET` and `INTRADAY` snapshots targeting the same trading session are never counted as independent observations.
4. **Fix `monitoring.py`:** deduplicate `prediction_log` by `(symbol, observation_date)`, remove or explicitly disclose `eval_df.tail(50)` truncation, filter by a single `model_version`, and clearly label the metric as `Legacy Quant Log (close->close)`.
5. **Unify version registry** so every report states both the quant model version (`ml_ensemble 3.2.0`) and the Phase 5 pipeline/validation schema version.
6. **Fix Nifty 50 lookup key** in `forward_validation.py` to accept `NIFTY_50` / `"NIFTY 50"` and include explicit `reason` fields for `UNAVAILABLE` benchmarks.
7. **Block quarantined `(ticker, url)` pairs** from `structured_events.parquet` and `phase5_intelligence.json` timelines, and set `latest_prediction_at` *after* news collection completes.
