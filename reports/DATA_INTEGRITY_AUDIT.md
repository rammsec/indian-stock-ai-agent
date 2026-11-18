# BharatAlpha AI — Data Integrity & Classification Audit Report

**Branch:** `feature/data-integrity-audit`  
**Base Commit SHA (`main`):** `8f328b8cdb8f7a1e508c1c47a648849945873013`  
**Audit Engine:** `backend/audit/data_integrity.py` & `backend/audit/classification.py`  
**Machine-Readable Report:** `reports/DATA_QUALITY_REPORT.json`  
**Overall Audit Status:** `FAIL` (`0 CRITICAL`, `4 HIGH`, `13 MEDIUM`, `6 LOW`, `40 INFO/PASSED`)

---

## 1. Audit Engine Architecture (`Step 3`)

The deterministic, read-only audit engine in `backend/audit/data_integrity.py` inspects `1,197` records across 7 datasets without mutating any production Parquet or JSON files:

| Dataset | Records Inspected | Passed Checks | Failed Checks | Dataset Status |
| :--- | ---: | ---: | ---: | :--- |
| `forward_predictions` | 55 | 11 | 2 | `FAIL` |
| `prediction_log` | 75 | 4 | 2 | `PASS_WITH_WARNINGS` |
| `forward_outcomes` | 165 | 11 | 1 | `PASS_WITH_WARNINGS` |
| `news_events` | 226 | 10 | 6 | `FAIL` |
| `benchmark` | 494 | 4 | 4 | `FAIL` |
| `paper_portfolio` | 179 | 10 | 2 | `FAIL` |
| `published_reports` | 3 | 1 | 6 | `PASS_WITH_WARNINGS` |
| **Total** | **1,197** | **51** | **23** | **`FAIL`** |

Every check returns a structured `CheckResult` containing `check_id`, `category`, `severity` (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`), `passed`, `affected_records`, `record_identifiers`, `human_explanation`, and `recommended_remediation`.

---

## 2. Complete Findings by Severity

### 2.1 High-Severity Findings (`4`)
1. **`PRED_POST_EVENT_TIMING` (`prediction_integrity`, `HIGH`, affected=`10`):**
   - 10 canonical (`is_duplicate_run=False`) snapshots were generated at or after their target entry session open (`09:15 IST` on the next NSE trading day):
     - 5 `INTRADAY` snapshots for `obs=2026-11-16` generated at `2026-11-17T10:25:29.884689+05:30` (>= `2026-11-17T09:15:00+05:30`).
     - 5 `POST_MARKET` snapshots for `obs=2026-11-17` generated at `2026-11-18T17:32:47.787090+05:30` (>= `2026-11-18T09:15:00+05:30` and after the `15:30 IST` close).
2. **`NEWS_QUARANTINE_LEAK` (`news_integrity`, `HIGH`, affected=`16`):**
   - 16 rows in `structured_events.parquet` (19 `(ticker, url)` pairs across `structured_events` and `quarantined_events`, including `ITC|dd90edd5827f6680`, `TCS|2356b1896be1de85`, `TATAELXSI|0f6f95d76eeab2bf`) match `(symbol, url)` entries in `quarantined_events.parquet` (`irrelevant_entity_contamination` / `generic_price_headline`) and also surface in `phase5_intelligence.json` stock timelines.
3. **`BENCH_KEY_MISMATCH` (`benchmark_integrity`, `HIGH`, affected=`494`):**
   - `data/processed/indices_daily.parquet` stores `index_code = "NIFTY_50"` (494 rows), whereas `backend/news_intel/forward_validation.py:344, 1126` queries `"NIFTY 50"` (with a space). This causes `benchmark_return_pct` to be `None` on all 45 matured outcome rows and `nifty_50_buy_hold` to report `status = "UNAVAILABLE"`.
4. **`PORT_LOOKAHEAD_CONTAMINATED_ENTRY` (`portfolio_integrity`, `HIGH`, affected=`1`):**
   - The single executed 10D paper trade (`fwd_BIRLACABLE_2026-11-16_20261117T102529`) entered at the `2026-11-17 09:15 IST` open (`₹300.25`) using a snapshot generated 70 minutes *after* that open (`2026-11-17T10:25:29.884689+05:30`).

### 2.2 Medium-Severity Findings (`13`)
1. **`PRED_DUPLICATE_TARGET_SESSION` (`affected=10`):** Canonical snapshots across different `nse_session` labels target the same `(symbol, entry_date)` (`2026-11-17` and `2026-11-18` each have 2 canonical snapshots per pilot stock).
2. **`LOG_DUPLICATE_SYMBOL_DATE` (`affected=45`):** `prediction_log.parquet` has 75 rows for only 30 unique `(symbol, observation_date)` pairs (45 duplicates from 4 runs on `2026-11-16`).
3. **`LOG_MIXED_MODEL_VERSIONS` (`affected=2`):** `prediction_log.parquet` mixes `ml_ensemble 3.2.0` (60 rows) and `3.0.0` (15 rows) in a single rolling metric.
4. **`OUT_BENCHMARK_MISSING` (`affected=45`):** All 45 `MATURED` rows in `forward_outcomes.parquet` have `benchmark_return_pct = None` due to `BENCH_KEY_MISMATCH`.
5. **`NEWS_MISSING_COLL_TS` (`affected=156`):** 156 historical RSS backfill rows in `structured_events.parquet` have `collection_timestamp = NaN`.
6. **`NEWS_ENTITY_MISATTRIBUTION` (`affected=1`):** Event `0f6f95d76eeab2bf` (`TATAELXSI`) is a `company_news` item that mentions `TCS` without any `TATAELXSI` alias.
7. **`NEWS_LATE_EFFECTIVE_DATE` (`affected=18`):** 18 events collected after the latest snapshot timestamp carry `effective_trading_date <= 2026-11-17` because `compute_pit_news_features()` filters only on `publication_timestamp <= as_of_ts`, not `collection_timestamp`.
8. **`PORT_DUPLICATE_SIGNAL_COUNTING` (`affected=10`):** All 20 canonical snapshots (2 per stock per date) are passed into `run_paper_research_portfolio()`, inflating `signals_received = 20` instead of `10` unique stock-date signals.
9. **`REP_MODEL_VERSION_MISMATCH` (`affected=2`):** `forward_validation.json` reports `5.3.0-pilot` while `phase5_intelligence.json` reports `5.0.0-pilot`.
10. **`REP_OUTCOME_POPULATION_MISMATCH` (`affected=2`):** `phase5_intelligence.monitoring.evaluated_outcomes_count = 60` vs `forward_validation.matured_outcomes_total = 10`.
11. **`REP_MONITORING_TRUNCATION` (`affected=10`):** `monitoring.py:50` computes metrics on `eval_df.tail(50)` while reporting `evaluated_outcomes_count = 60`.
12. **`REP_OBS_DATE_MISMATCH` (`affected=5`):** `phase5_intelligence.json` stamps `stocks[*].observation_date = "2026-11-18"` while underlying quant inputs and snapshots have `observation_date = "2026-11-17"`.
13. **`REP_SESSION_WORKFLOW_CONSISTENCY` (`affected=5`):** GH Run `44300078913` (delayed pre-market cron at `2026-11-18T17:32:47+05:30`) was labelled `POST_MARKET` by wall-clock time while `equities_daily` max date remained `2026-11-17`.

### 2.3 Low-Severity Findings (`6`)
1. **`NEWS_UNFLAGGED_DUP_URL` (`affected=29`):** 29 `(ticker, source_url)` groups appear across multiple `event_category` values without `is_duplicate_report = True`.
2. **`NEWS_COLL_AFTER_PRED_WARN` (`affected=60`):** 60 timeline events in `phase5_intelligence.json` have `collection_timestamp` ~13.8s after `latest_prediction_at` because `now_iso` is captured at the start of `run_phase5_intelligence_pipeline()`.
3. **`BENCH_NON_TRADING_DAYS` (`affected=5`):** 5 Nifty 50 rows fall on weekend/holiday dates (e.g., Union Budget Saturday `2025-02-01` and Muhurat trading `2025-10-21`).
4. **`BENCH_HORIZON_KEYS_AND_ALIAS` (`affected=1`):** `paper_portfolio.benchmarks` includes an unlabelled `naive_momentum` 1D alias alongside `naive_momentum_1d/5d/10d`.
5. **`BENCH_UNAVAILABLE_NO_REASON` (`affected=5`):** `UNAVAILABLE` and `INSUFFICIENT_DATA` benchmark entries lack an explanatory `reason` field.
6. **`REP_LATEST_PRED_STALE` (`affected=1`):** `latest_predictions.json` was generated `14.54h` before `phase5_intelligence.json` during the delayed pre-market run.

---

## 3. Dataset Classification Rules & Results (`Step 4`)

### 3.1 Forward Predictions & Outcomes (`backend/audit/classification.py::classify_forward_predictions`)
For each `(prediction, horizon)` pair, let `entry_open` be `09:15 IST` on the first NSE trading session after `observation_date`, and `exit_close` be `15:30 IST` on the $H$-th NSE trading session from `entry_date`:
- **`DUPLICATE`:** `is_duplicate_run == True`, **or** a later prospective snapshot targeting the same `(symbol, target_entry_date, horizon)` as an earlier prospective snapshot.
- **`INVALID`:** Missing/unparseable timestamp or `entry_open <= prediction_timestamp < exit_close` (look-ahead contamination).
- **`HISTORICAL`:** `prediction_timestamp >= exit_close` (recorded after the target outcome window had already closed).
- **`PROSPECTIVE`:** `prediction_timestamp < entry_open` and first snapshot for `(symbol, target_entry_date, horizon)`. Its final `reporting_class` is `MATURED` or `PENDING` from `forward_outcomes.parquet`.

| Horizon | Total Stored | Unique Canonical (`~is_duplicate_run`) | `PROSPECTIVE` | `MATURED` (Prospective) | `PENDING` (Prospective) | `HISTORICAL` | `INVALID` (Look-Ahead) | `DUPLICATE` | Classification Conflicts | Prospective 1D Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **1D** | 55 | 20 | **10** | **5** | **5** | **5** | **5** | **35** | 10 | `40.0%` (`n=5 < 10`, insufficient) |
| **5D** | 55 | 20 | **10** | **0** | **10** | **0** | **10** | **35** | 10 | `None` (`n=0`) |
| **10D** | 55 | 20 | **10** | **0** | **10** | **0** | **10** | **35** | 10 | `None` (`n=0`) |

### 3.2 News Events (`backend/audit/classification.py::classify_news_events`)
- **Rule:** A news article published before a prediction (`publication_timestamp <= earliest_snapshot_ts`) but collected after that prediction (`collection_timestamp > earliest_snapshot_ts` or `collection_timestamp` is null) is classified as **`HISTORICAL`**, never `PROSPECTIVE` for earlier predictions, and flagged with `has_collection_leakage_conflict = True`.
- **Results on `structured_events.parquet` (`226` rows):**
  - `PROSPECTIVE`: **11** (collected prior to/at initial snapshot without quarantine conflict)
  - `HISTORICAL`: **164** (156 null `collection_timestamp` backfills + 8 unique articles published before the initial `2026-11-16T19:23:27+05:30` snapshot but collected on `2026-11-17` or `2026-11-18`)
  - `DUPLICATE`: **35** (`is_duplicate_report == True` or repeat `(ticker, source_url)`)
  - `INVALID`: **16** (quarantined `(ticker, source_url)` leaks)
  - `collection_leakage_conflicts`: **63** (rows where `publication_timestamp` alone would misclassify a late-collected or quarantined article as available at initial prediction time).
