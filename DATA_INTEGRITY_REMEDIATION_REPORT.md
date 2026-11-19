# Data Integrity Remediation Report (Workstream 1)

**Programme:** BharatAlpha AI Remediation Programme  
**Workstream:** 1 — Data Integrity (`remediation/data-integrity`)  
**Remediation Version:** `rem-1.0.0` | **Schema Version:** `1.0.0`  
**Prospective Cutoff:** `2026-11-17`  
**Freeze Boundary Compliance:** Zero modifications to frozen model weights, `features.py`, `combiner.py`, `pipeline.py`, `forward_validation.py`, or original `data/predictions/*.parquet`, `data/processed/*.parquet`, and `docs/data/*.json` files. All remediated datasets are materialized non-destructively in `data/shadow/v1_remediated/`.

---

## Issue 1: Missing Formal Prediction & Event Identity Contract and Provenance Schema

- **Original finding:** Prediction snapshots and structured events lacked a formal, deterministic identity contract and uniform provenance metadata across datasets (`forward_predictions.parquet`, `prediction_log.parquet`, `structured_events.parquet`).
- **Root cause:** Identity keys (`prediction_id`, `canonical_prediction_id`) were constructed ad hoc with mixed timestamp/run suffixes (`fwd_BIRLACABLE_2026-11-16_20261117T102529`, `fwd_BIRLACABLE_2026-11-16_44081380157`, `fwd_2026-11-16_BIRLACABLE`), and provenance columns (`schema_version`, `remediation_version`, `source_dataset`, `validation_status`, `rejection_reason`) were absent or inconsistently populated.
- **Affected files & records:**
  - `data/predictions/forward_predictions.parquet` (65 rows)
  - `data/predictions/prediction_log.parquet` (90 rows)
  - `data/predictions/structured_events.parquet` (254 rows)
- **Remediation:**
  - Implemented `backend/audit/identity_contract.py` with deterministic key generators:
    - `canonical_prediction_id(symbol, observation_date, nse_session)` -> `cpid_{YYYY-MM-DD}_{SYMBOL}_{NSE_SESSION}`
    - `daily_canonical_key(symbol, target_entry_date)` -> `dck_{YYYY-MM-DD}_{SYMBOL}`
  - Implemented `enforce_provenance()` and `check_provenance_completeness()` enforcing all 9 mandatory provenance fields: `schema_version`, `remediation_version="rem-1.0.0"`, `source_dataset`, `model_version`, `feature_set_version`, `collection_timestamp`, `prediction_timestamp`, `validation_status`, `rejection_reason`.
- **Tests:**
  - `test_deterministic_identity_keys_and_validation`
  - `test_provenance_enforcement_and_completeness_check`
- **Evidence of successful execution:**
  - Both tests pass in `tests/test_remediation_data_integrity.py`.
  - `check_provenance_completeness()` returns `is_complete=True` and `missing_columns=[]` across all three shadow datasets in `data/shadow/v1_remediated/`.
- **Remaining limitations:**
  - Upstream writers (`forward_validation.py`, `pipeline.py`) remain frozen and continue writing legacy IDs; downstream consumers must read from `data/shadow/v1_remediated/` or pass records through `validate_prediction_records()` / `validate_event_records()`.

---

## Issue 2: `forward_predictions.parquet` Duplicate Runs and `prediction_timestamp` vs `entry_date` Look-Ahead Violations

- **Original finding:** `forward_predictions.parquet` (65 rows) contained 35 duplicate-run snapshots on `2026-11-16` and 15 canonical snapshots with look-ahead violations (`prediction_timestamp > entry_date 09:15:00+05:30`: `2026-11-17T10:25:29+05:30` INTRADAY `+70.48` min, `2026-11-18T17:32:47+05:30` POST_MARKET `+497.8` min, `2026-11-19T17:28:47+05:30` POST_MARKET `+493.79` min).
- **Root cause:** `record_forward_prediction_snapshots` anchored `observation_date` to stale `latest_predictions.json` without checking `prediction_timestamp` against NSE market open (`09:15:00+05:30`), and multiple intraday/post-market runs created duplicate snapshots for the same `(symbol, observation_date, nse_session)`.
- **Affected files & records:**
  - `data/predictions/forward_predictions.parquet` (65 rows: 35 duplicate, 15 canonical look-ahead, 10 valid prospective, 5 valid historical)
  - `data/predictions/forward_outcomes.parquet` (195 rows)
- **Remediation:**
  - Implemented `validate_prediction_records()` in `backend/audit/identity_contract.py` to sort deterministically by `(prediction_timestamp_ist, prediction_id)`, retain the earliest canonical record per `canonical_prediction_id`, classify duplicates as `DUPLICATE` (`rejection_reason="DUPLICATE_CANONICAL_PREDICTION_ID"`), and classify look-ahead violations as `INVALID_LOOKAHEAD` (`rejection_reason="LOOKAHEAD_VIOLATION_DELTA_{delta}_MIN"`).
  - Materialized `data/shadow/v1_remediated/forward_predictions_remediated.parquet` (65 rows preserved with explicit classifications).
- **Tests:**
  - `test_duplicate_prediction_and_lookahead_gate`
  - `test_shadow_layer_non_mutation_guarantee`
- **Evidence of successful execution:**
  - Shadow output classification counts on live data: `DUPLICATE=35`, `INVALID_LOOKAHEAD=15`, `PROSPECTIVE=10`, `HISTORICAL=5` (total 65 rows).
  - SHA-256 hashes and `mtime_ns` of `data/predictions/forward_predictions.parquet` and `forward_outcomes.parquet` verified unchanged.
- **Remaining limitations:**
  - Upstream `forward_validation.py` is frozen; any new snapshots generated by it must be gated via `validate_prediction_records()` before evaluation.

---

## Issue 3: `prediction_log.parquet` Duplicate `(symbol, observation_date)` Contamination

- **Original finding:** `prediction_log.parquet` (90 rows across 15 symbols) contained 45 duplicate `(symbol, observation_date)` rows on `2026-11-16` (4 runs × 15 symbols = 60 rows) with UTC (`+00:00`) timestamps and no `feature_set_version` column.
- **Root cause:** Repeated pipeline executions on `2026-11-16` appended rows with distinct `prediction_id`s and `generated_at` timestamps without enforcing a `(symbol, observation_date)` uniqueness gate.
- **Affected files & records:**
  - `data/predictions/prediction_log.parquet` (90 rows -> 45 canonical + 45 duplicates)
- **Remediation:**
  - Configured `validate_prediction_records(..., dedupe_by_symbol_obs_date=True, default_feature_set_version="237+17")` to normalize `generated_at` UTC timestamps to IST `prediction_timestamp`, enforce provenance, mark 45 duplicate rows as `DUPLICATE` (`rejection_reason="DUPLICATE_SYMBOL_OBSERVATION_DATE"`), and materialize both:
    - `data/shadow/v1_remediated/prediction_log_remediated.parquet` (90 rows with classifications)
    - `data/shadow/v1_remediated/prediction_log_deduplicated.parquet` (45 unique `(symbol, observation_date)` rows)
- **Tests:**
  - `test_prediction_log_live_deduplication_and_provenance`
  - `test_shadow_layer_non_mutation_guarantee`
- **Evidence of successful execution:**
  - Live `prediction_log.parquet` classification counts: `DUPLICATE=45`, `PROSPECTIVE=30`, `HISTORICAL=15`.
  - `prediction_log_deduplicated.parquet` contains 45 rows with `0` duplicate `(symbol, observation_date)` pairs.
- **Remaining limitations:**
  - Historical `model_version` values in `prediction_log.parquet` (`3.0.0` and `3.2.0`) are preserved as-is for lineage accuracy rather than overwritten.

---

## Issue 4: `structured_events.parquet` Missing `collection_timestamp` and Duplicate Reports

- **Original finding:** `structured_events.parquet` (254 rows) contained 156 rows with `NULL` `collection_timestamp` (pre-baseline archive `2026-10-18` to `2026-11-16`) and 7 `is_duplicate_report=True` rows (2 with non-null `collection_timestamp`, 5 with `NULL` `collection_timestamp`).
- **Root cause:** Pre-baseline archived news events were ingested without `collection_timestamp` provenance and without a validation gate separating genuinely forward-collected events from retroactive archives.
- **Affected files & records:**
  - `data/predictions/structured_events.parquet` (254 rows: 156 missing `collection_timestamp`, 2 non-null duplicates, 96 valid prospective)
- **Remediation:**
  - Implemented `validate_event_records()` in `backend/audit/identity_contract.py` to quarantine any event missing `collection_timestamp` (`QUARANTINED`, `rejection_reason="MISSING_COLLECTION_TIMESTAMP"`), flag `publication_timestamp > collection_timestamp` as `INVALID_LOOKAHEAD` (`PUBLICATION_AFTER_COLLECTION`), and flag duplicate reports as `DUPLICATE` (`DUPLICATE_EVENT_REPORT`).
  - Materialized `data/shadow/v1_remediated/structured_events_remediated.parquet`.
- **Tests:**
  - `test_event_validation_missing_collection_ts_and_duplicates`
- **Evidence of successful execution:**
  - Live `structured_events.parquet` classification counts: `QUARANTINED=156`, `PROSPECTIVE=96`, `DUPLICATE=2` (total 254 rows).
- **Remaining limitations:**
  - The 156 archived rows cannot be retroactively assigned genuine `collection_timestamp` values and remain permanently `QUARANTINED` from prospective evaluation.

---

## Issue 5: `5.0.0-pilot` vs `5.3.0-pilot` Metadata Version Drift

- **Original finding:** `data/predictions/phase5_intelligence.json` and `docs/data/phase5_intelligence.json` reported `model_version="5.0.0-pilot"`, whereas `forward_validation.json` and `forward_predictions.parquet` reported `model_version="5.3.0-pilot"` (`feature_set_version="237+17"`).
- **Root cause:** Independent JSON writers emitted inconsistent hardcoded/stale `model_version` strings across Phase 5 artifacts.
- **Affected files & records:**
  - `data/predictions/phase5_intelligence.json` (`5.0.0-pilot`)
  - `docs/data/phase5_intelligence.json` (`5.0.0-pilot`)
  - `data/predictions/forward_validation.json` (`5.3.0-pilot`)
  - `docs/data/forward_validation.json` (`5.3.0-pilot`)
  - `data/predictions/forward_predictions.parquet` (`5.3.0-pilot`)
- **Remediation:**
  - Implemented `resolve_metadata_version_drift()` in `backend/audit/identity_contract.py` to detect raw drift (`["5.0.0-pilot", "5.3.0-pilot"]`) and emit a canonical shadow manifest (`data/shadow/v1_remediated/metadata_manifest.json`) resolving all 4 JSON targets to `canonical_model_version="5.3.0-pilot"` and `canonical_feature_set_version="237+17"` while preserving raw drift lineage.
- **Tests:**
  - `test_model_version_drift_detection_and_resolution`
- **Evidence of successful execution:**
  - `resolve_metadata_version_drift()` detects `drift_detected_in_raw=True`, `raw_unique_versions=["5.0.0-pilot", "5.3.0-pilot"]`, and outputs `drift_resolved=True` with all `shadow_resolved_versions` set to `"5.3.0-pilot"`.
- **Remaining limitations:**
  - Original JSON files under `data/predictions/` and `docs/data/` remain untouched per freeze boundary; dashboard/API consumers should read `data/shadow/v1_remediated/metadata_manifest.json` or apply canonical version resolution at read time.

---

## Test Execution Summary

Command:
```bash
/home/antigravity/.venv/bin/pytest tests/test_remediation_data_integrity.py -v
```
Result: **7 passed in 0.59s** (all 7 tests in `tests/test_remediation_data_integrity.py` passed, and all 7 existing audit tests in `tests/test_data_integrity_audit.py` also continue to pass: **14 passed in 0.71s**).
