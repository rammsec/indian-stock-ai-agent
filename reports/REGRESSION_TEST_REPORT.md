# BharatAlpha AI — Regression Test Report

**Branch:** `feature/data-integrity-audit`  
**Base Commit SHA (`main`):** `8f328b8cdb8f7a1e508c1c47a648849945873013`  
**Audit Date:** 2026-11-18  
**Test Suite Files:** `tests/test_data_integrity_audit.py` (16 tests) & `tests/test_data_integrity_engine.py` (12 tests), plus the existing 90 repository tests (`118` total).

---

## 1. Execution Commands & Results Summary

| Command Executed | Total Tests | Passed | Failed | Skipped | Duration |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `.venv/bin/pytest tests/test_data_integrity_audit.py tests/test_data_integrity_engine.py -v` | 28 | **28** | 0 | 0 | `2.53s` |
| `.venv/bin/pytest -q` (Complete repository test suite) | 118 | **118** | 0 | 0 | `39.15s` |

---

## 2. Coverage Across the 13 Required Integrity Test Areas

| Required Test Area | Test Functions Covering Area | Valid & Invalid / Boundary & Real Repo Regression Scenarios Verified |
| :--- | :--- | :--- |
| **1. Prediction identity & deduplication** | `test_multiple_predictions_same_stock_and_session`, `test_legitimate_retries_vs_independent_runs`, `test_classification_duplicate_target_session`, `test_legacy_prediction_log_classification_and_checks` | Verifies same-session retry deduplication, cross-session duplicate target detection (`POST_MARKET` + `PRE_MARKET` targeting `2026-11-17`), and `prediction_log.parquet` `(symbol, observation_date)` deduplication (`75` stored vs `30` unique). |
| **2. Prediction timestamp integrity** | `test_missing_and_invalid_timestamps`, `test_classification_boundary_exactly_at_entry_open_and_holiday` | Verifies missing/corrupt ISO timestamps fail `PRED_INVALID_TIMESTAMP`, and tests the exact `09:14:59+05:30` (`PROSPECTIVE`) vs `09:15:00+05:30` (`INVALID` look-ahead) boundary. |
| **3. Historical vs. prospective classification** | `test_classification_post_event_and_counts`, `test_historical_news_vs_prospective_separation` | Reproduces real `2026-11-16` `POST_MARKET` (`MATURED`), `2026-11-16` `INTRADAY` (`INVALID`), `2026-11-17` `PRE_MARKET` (`PENDING`), and `2026-11-18 17:32:47 IST` `POST_MARKET` (`HISTORICAL` for 1D, `INVALID` for 5D). |
| **4. News collection-time leakage** | `test_news_collection_time_leakage_classification`, `test_quarantine_leak_and_entity_misattribution_detected` | Reproduces real event `c378197f3aa255ca` (`BIRLACABLE`, published `2026-11-16T04:12:00+05:30`, collected `2026-11-17T10:19:13+05:30` after the `2026-11-16T19:23:27+05:30` snapshot → classified `HISTORICAL` with `has_collection_leakage_conflict=True`), plus `ITC` quarantine leaks and `TATAELXSI`/`TCS` (`0f6f95d76eeab2bf`) misattribution. |
| **5. Outcome maturity** | `test_outcomes_not_matured`, `test_finalised_outcome_immutability_and_invalid_prices` | Verifies premature maturity detection when `< H` future NSE sessions exist in `equities_daily`, and verifies immutability of `MATURED` rows against a reference snapshot (`OUT_MODIFIED_AFTER_FINALISATION`). |
| **6. Trading-session calculations** | `test_nse_holidays_and_weekends`, `test_classification_boundary_exactly_at_entry_open_and_holiday` | Verifies weekend and `NSE_TRADING_HOLIDAYS` (`2026-11-24`) handling so an observation on `2026-11-23` targets `2026-11-25` as its next entry session. |
| **7. Benchmark horizon alignment** | `test_missing_benchmark_data`, `test_benchmark_key_mismatch_and_valid_match` | Reproduces the real `NIFTY_50` vs `"NIFTY 50"` key mismatch (`BENCH_KEY_MISMATCH`) and verifies passing behaviour when `"NIFTY 50"` is present. |
| **8. Portfolio cash reconciliation** | `test_portfolio_cash_and_exposure_consistency`, `test_portfolio_integrity_valid_and_broken_reconciliation` | Verifies `cash + invested == starting + realized` (`PORT_CASH_INVESTED_RECONCILIATION`) and `cash + invested + unrealized == ending_capital` (`PORT_EQUITY_RECONCILIATION`) on both valid `10D` (`₹9,98,018.32`) and deliberately broken (`₹8,00,000` cash) fixtures. |
| **9. Position sizing & exposure** | `test_portfolio_cash_and_exposure_consistency`, `test_portfolio_integrity_valid_and_broken_reconciliation` | Verifies `10%` single-position limit (`PORT_POSITION_ALLOCATION_LIMIT`) and `80%` max portfolio exposure limit (`PORT_EXPOSURE_LIMIT`) fail when breached (`15%` weight / `85%` peak exposure). |
| **10. Entry & exit sequencing** | `test_lookahead_bias_prevention`, `test_portfolio_integrity_valid_and_broken_reconciliation` | Reproduces the real `fwd_BIRLACABLE_2026-11-16_20261117T102529` look-ahead entry (`PORT_LOOKAHEAD_CONTAMINATED_ENTRY`) and verifies `entry_date == next_trading_session`. |
| **11. Duplicate execution prevention** | `test_portfolio_integrity_valid_and_broken_reconciliation` | Verifies `PORT_DUPLICATE_ENTRIES` (`CRITICAL`) fails when two `ENTRY_EXECUTED` rows share `(strategy_horizon, symbol, entry_date)`. |
| **12. Missing & invalid market prices** | `test_missing_entry_and_exit_prices`, `test_finalised_outcome_immutability_and_invalid_prices`, `test_probability_out_of_range_detected` | Verifies `OUT_INVALID_PRICES` (`CRITICAL`) on negative/missing entry or exit prices and `PRED_PROB_RANGE` (`CRITICAL`) on out-of-range probabilities (`1.25`). |
| **13. Model-version & report consistency** | `test_reporting_source_of_truth_envelope`, `test_published_reports_consistency_checks`, `test_full_repo_data_integrity_report_schema` | Verifies detection of `5.3.0-pilot` vs `5.0.0-pilot` mismatch (`REP_MODEL_VERSION_MISMATCH`), `60` vs `10` outcome mismatch (`REP_OUTCOME_POPULATION_MISMATCH`), `tail(50)` truncation (`REP_MONITORING_TRUNCATION`), `observation_date` mismatch (`REP_OBS_DATE_MISMATCH`), and stale `latest_predictions.json` (`REP_LATEST_PRED_STALE`). |

---

## 3. Known Limitations Observed During Testing
- Running the full existing repository test suite (`pytest -q`) executes live NSE bulk/block deal collector tests that touch `data/raw/nse_bulk_block/bulk_latest.csv` in the working directory; this file was immediately restored via `git checkout -- data/raw/nse_bulk_block/bulk_latest.csv` so zero production data files are modified on the branch.
