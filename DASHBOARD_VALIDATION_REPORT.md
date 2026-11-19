# Dashboard Validation & Reconciliation Report — Workstream 5

**Branch / Worktree:** `remediation/dashboard-reporting` (`/home/antigravity/worktrees/rem-dashboard-reporting`)  
**Reconciliation Engine:** [`backend/audit/dashboard_reconciler.py`](backend/audit/dashboard_reconciler.py)  
**Freeze & Deployment Boundary:** Frozen prediction/model logic untouched; `docs/index.html` untouched (no GitHub Pages deployment performed).

---

## Executive Summary

Automated read-only reconciliation of `docs/data/*.json` (`forward_validation.json`, `phase5_intelligence.json`, `phase5_experiments.json`, `backtest_summary.json`, `regime.json`, and 28 stock JSONs) against source Parquet/JSON artifacts confirms **0 JSON-vs-Parquet numeric serialization discrepancies** (`total_discrepancies = 0`, `nan_token_files = {}`), while surfacing **7 structural reporting, presentation, and upstream audit-alignment issues** across dashboard JSONs and UI rendering.

All 16 tests across `tests/test_remediation_dashboard_reporting.py`, `tests/test_frontend_contract.py`, `tests/test_frontend_report_render.py`, and `tests/test_frontend_app_contract.py` pass (`16 passed in 1.51s`).

---

## Issue-by-Issue Audit & Remediation Log

### Issue 1: Missing Rejected-Signal Reason Breakdown & Upstream `recent_trades[-60:]` Truncation Visibility

- **Original finding:** The Forward Validation Paper Portfolio UI displayed only a single aggregate `rejected_signals` count (`20 / 20 / 19 = 59` across `1D` / `5D` / `10D`) without showing *why* signals were rejected (`REJECTED_NON_BUY_SIGNAL_NO TRADE`, `REJECTED_NON_BUY_SIGNAL_HOLD`), nor alerting the user that `paper_portfolio.recent_trades` is truncated to `60` events (omitting `40` of `59` rejected signals).
- **Root cause:** `frontend/index.html` had no dedicated summary container between `#tbody-fv-paper` and `#tbody-fv-audit`, `frontend/app.js` (`renderForwardValidation`) did not aggregate `t.reason` from `pp.recent_trades`, and `frontend/report.js` (`rrAppendixHtml`) omitted `rejected_signals` and rejection reasons in its Paper Portfolio table. Upstream `backend/news_intel/forward_validation.py` (`L1174`, `L1318`) slices `all_trades[-60:]`.
- **Affected files & records:**
  - `docs/data/forward_validation.json` (`paper_portfolio.strategies.{1D,5D,10D}.rejected_signals`: `20, 20, 19`; `recent_trades`: `60` rows containing `19` rejections: `18 × REJECTED_NON_BUY_SIGNAL_NO TRADE`, `1 × REJECTED_NON_BUY_SIGNAL_HOLD`).
  - `data/predictions/paper_trades.parquet` (`219` rows: `42` `SIGNAL_REJECTED`, `15` legacy `REJECTED`, with `50 × REJECTED_NON_BUY_SIGNAL_NO TRADE`, `5 × REJECTED_NON_BUY_SIGNAL_WATCH`, `2 × REJECTED_NON_BUY_SIGNAL_HOLD`).
  - `frontend/index.html`, `frontend/app.js`, `frontend/report.js`.
- **Remediation:**
  - Added [`reconcile_forward_validation`](backend/audit/dashboard_reconciler.py) (`_extract_rejection_breakdown`) to cross-check `strategy_rejected_signal_counts` (`59`), `recent_trades` rejections (`19`), truncation gap (`40`), and `paper_trades.parquet` legacy rows (`15`).
  - Added `#fv-rejection-summary` in [`frontend/index.html`](frontend/index.html#L913-L915) and populated per-reason badges + truncation disclosure in [`frontend/app.js`](frontend/app.js#L2290-L2316).
  - Updated [`frontend/report.js`](frontend/report.js#L884-L931) (`rrAppendixHtml`) to include the `Rejected` column and a concise rejected-signals reason + truncation line below the Paper Portfolio table.
- **Tests:** `test_dashboard_reconciler_live_datasets_and_read_only`, `test_frontend_app_js_rejection_breakdown_cutoff_and_unavailable`, `test_frontend_report_js_rejection_breakdown_and_cutoff`, `test_frontend_index_html_has_rejection_container_and_docs_index_untouched`.
- **Evidence of successful execution:** Node DOM tests verify `#fv-rejection-summary` renders `"Rejected Signals Breakdown (59 total across 1D/5D/10D (showing 19 in recent_trades window; 40 earlier rejections truncated in upstream JSON)): ... REJECTED_NON_BUY_SIGNAL_NO TRADE: 18 REJECTED_NON_BUY_SIGNAL_HOLD: 1"`, and `report.js` renders `"Rejected signals (59 total across 1D/5D/10D; 19 in recent_trades window): REJECTED_NON_BUY_SIGNAL_NO TRADE: 18, REJECTED_NON_BUY_SIGNAL_HOLD: 1 (40 earlier rejections truncated in JSON)."`
- **Remaining limitations:** Upstream `forward_validation.json` only ships `recent_trades[-60:]` (`19` of `59` rejections), so exact reason counts for the `40` earlier truncated rejections cannot be reconstructed by the browser without reading `paper_trades.parquet` or extending the frozen upstream exporter.

---

### Issue 2: Inconsistent Missing-Value Presentation (`N/A`, `—`, `Pending`, `Unavailable` vs `UNAVAILABLE`)

- **Original finding:** Missing or unmatured metrics in `frontend/app.js` rendered inconsistently across panels as `"N/A"`, `"—"`, `"Pending"`, or `"Unavailable"` instead of the standardized `"UNAVAILABLE"` token, and `net_pnl_inr: 0` or missing benchmark cells risked appearing as `"—"` or ambiguous values.
- **Root cause:** Multiple renderers in `frontend/app.js` (`renderBundle`, `updateRegimeBannerDOM`, `renderQuantAndCorporateIntelSection`, `renderWalkForwardBacktestTable`, `renderAllStockPredictionsMatrix`, `renderPhase5Intelligence`, `renderPhase5ExperimentsTable`, `renderForwardValidation`) used ad-hoc fallback literals instead of `"UNAVAILABLE"`.
- **Affected files & records:**
  - `frontend/app.js` (`L448`, `L1075`, `L1118-L1258`, `L1334-L1473`, `L2000-L2345`).
  - `docs/data/BEL.json` (and 14 other non-quant stock JSONs where `quantitative_prediction` is `null`, plus 23 stock JSONs where `data_quality` is `null`).
  - `docs/data/forward_validation.json` (`5D`/`10D` unmatured metrics, `nifty_50_buy_hold` = `UNAVAILABLE`, `naive_momentum_1d` = `INSUFFICIENT_DATA`).
- **Remediation:**
  - Replaced `"N/A"`, `"Pending"`, `"Unavailable"`, and placeholder `"—"` fallbacks across all quant, Phase 5, backtest, regime, and forward-validation renderers in [`frontend/app.js`](frontend/app.js) with `"UNAVAILABLE"`, while preserving `fmtNum`/`fmtPct`/`fmtCr` behavior where existing contract tests expect it.
  - Added [`reconcile_forward_validation`](backend/audit/dashboard_reconciler.py) and [`reconcile_phase5_intelligence`](backend/audit/dashboard_reconciler.py) checks verifying that `INSUFFICIENT_DATA` horizons and `UNAVAILABLE` benchmarks contain `null` numeric metrics (no fabricated or leaked numbers).
- **Tests:** `test_frontend_app_js_rejection_breakdown_cutoff_and_unavailable`, `test_dashboard_reconciler_detects_synthetic_mismatches`, `test_app_regime_and_data_quality_contract`, `test_report_missing_data_low_quality_stale_and_unsafe_urls`.
- **Evidence of successful execution:** Node test on `BEL.json` confirms `qp-action-pill` = `"ML: UNAVAILABLE"`, `qp-kpi-strip` and `ci-kpi-strip` contain `"UNAVAILABLE"` and `"N/A" not in res["belKpiHtml"]`, and `tbody-fv-paper` renders `"UNAVAILABLE"` for missing `nifty_50_buy_hold` / `naive_momentum_1d` metrics.
- **Remaining limitations:** Static HTML initial loading placeholders in `frontend/index.html` (`₹0.00` before `loadStockResearch()` resolves) and `fmtNum`/`fmtPct`/`fmtCr` (`"—"` for table cells in legacy annual/quarterly financials) remain unchanged to preserve existing visual layout and `test_frontend_contract.py` expectations. `docs/index.html` intentionally remains unbuilt per boundary rules.

---

### Issue 3: Data Cutoff Timestamps, Observation Date & Sample-Size Visibility Gaps

- **Original finding:**
  1. `market-regime-strip` omitted `regime.observation_date` (`2026-11-18`).
  2. `renderWalkForwardBacktestTable` did not surface `active_trades_count` (`n=72`) in the ML Ensemble KPI card or walk-forward topbar pill, and labeled the timestamp `Backtest Freshness` rather than `Backtest Cutoff / Generated`.
  3. `renderAllStockPredictionsMatrix` omitted `model_version` (`v3.2.0`) in its freshness subtitle.
  4. `renderPhase5ExperimentsTable` omitted `generated_at` in the PIT note and omitted `folds` (`6 folds`) and `1D` `active_trades` (`n=16` / `n=17`) sample counts in the table rows.
  5. `renderForwardValidation` omitted `fv.generated_at` (`Snapshot Cutoff`) in `#fv-subtitle` and omitted `min_sample_threshold` (`min 10`) alongside `matured_count / pending_count` in `#tbody-fv-horizons`.
  6. `renderPhase5Intelligence` omitted `latest_prediction_at` in `#p5-freshness-sub`.
- **Root cause:** UI template strings in `frontend/app.js` and `frontend/report.js` did not render all available metadata fields from the published JSONs.
- **Affected files & records:** `frontend/app.js`, `frontend/report.js`, `docs/data/{regime,backtest_summary,latest_predictions,phase5_experiments,phase5_intelligence,forward_validation}.json`.
- **Remediation:**
  - Updated [`frontend/app.js`](frontend/app.js) to display:
    - Regime `observation_date` (`2026-11-18`) in `#mr-regime-badge`.
    - Backtest sample size (`n=72` / `n=302`) in `#mr-walkforward-pill` and `#quant-bt-kpis`, plus `Backtest Cutoff / Generated`.
    - Model version (`v3.2.0`) in `#quant-pred-freshness`.
    - Phase 5 `Quant Cutoff` and `Latest Pred` timestamps in `#p5-freshness-sub`.
    - Phase 5 experiments `Feature Cutoff`, `Generated`, and row-level `folds (n=...)` badges.
    - Forward validation `Snapshot Cutoff` in `#fv-subtitle` and `(min 10)` in `#tbody-fv-horizons`.
  - Updated [`frontend/report.js`](frontend/report.js#L848-L872) to display `bt.generated_at` and `fv.model_version` / `fv.feature_set_version` in the Technical Appendix.
- **Tests:** `test_frontend_app_js_rejection_breakdown_cutoff_and_unavailable`, `test_frontend_report_js_rejection_breakdown_and_cutoff`, `test_dashboard_reconciler_live_datasets_and_read_only`.
- **Evidence of successful execution:** Verified via Node assertions checking `"2026-11-18"`, `"n=72"`, `"Backtest Cutoff / Generated"`, `"Feature Cutoff:"`, `"6 folds (n=16)"`, `"Quant Cutoff:"`, `"Latest Pred:"`, `"Snapshot Cutoff:"`, `"(min 10)"`, and `"Model 5.3.0-pilot"`.
- **Remaining limitations:** `phase5_experiments.json` itself has `latest_feature_date = "2026-11-16"` while `daily_features.parquet` extends to `2026-11-18` (flagged by `reconcile_phase5_experiments`); re-running experiments is outside this workstream's read-only scope.

---

### Issue 4: Model Version Drift (`5.0.0-pilot` vs `5.3.0-pilot` vs `3.2.0`)

- **Original finding:** Published dashboard JSONs report three distinct `model_version` strings simultaneously:
  - `docs/data/phase5_intelligence.json`: `5.0.0-pilot`
  - `docs/data/forward_validation.json`: `5.3.0-pilot`
  - `docs/data/latest_predictions.json` & `docs/data/{SYMBOL}.json` (`quantitative_prediction`): `3.2.0`
- **Root cause:** Upstream `backend/news_intel/pipeline.py:177` and `backend/models.py:766` stamp `5.0.0-pilot` for Phase 5 intelligence, whereas `backend/news_intel/forward_validation.py:121,504` stamps `5.3.0-pilot`, and `backend/prediction/` stamps `3.2.0` for quantitative base predictions.
- **Affected files & records:** `docs/data/phase5_intelligence.json`, `docs/data/forward_validation.json`, `docs/data/latest_predictions.json`, 13 stock JSONs with `quantitative_prediction`.
- **Remediation:**
  - `backend/audit/dashboard_reconciler.py` integrates `audit_metadata_version_drift` and cross-checks `qp_model_versions` across all 28 stock JSONs, surfacing the exact drift in `sections.model_version_drift` and `sections.phase5_intelligence.findings`.
  - UI now explicitly renders the artifact's `model_version` on the Phase 5 header, Forward Validation header, All-Stock Predictions matrix, and Technical Appendix so users never confuse versions.
- **Tests:** `test_dashboard_reconciler_live_datasets_and_read_only`, `test_frontend_app_js_rejection_breakdown_cutoff_and_unavailable`, `test_frontend_report_js_rejection_breakdown_and_cutoff`.
- **Evidence of successful execution:** `reconcile_dashboard_artifacts` reports `unique_versions_detected: ["5.0.0-pilot", "5.3.0-pilot"]` (`severity: "HIGH"`) and `qp_model_versions: {"3.2.0": 13}`.
- **Remaining limitations:** Unifying `model_version` in `pipeline.py` / `models.py` / `forward_validation.py` belongs to frozen upstream pipeline code and was not modified.

---

### Issue 5: Upstream Look-Ahead, Duplicate-Run Monitoring Inflation & Broken `NIFTY_50` Benchmark

- **Original finding:**
  1. `forward_validation.json` reports `20` `1D` `MATURED` outcomes, whereas `backend/audit/data_integrity.py` classifies `15` of `30` canonical `1D` outcomes as `INVALID_LOOKAHEAD` (`10` matured runs at `2026-11-17T10:25:29+05:30` and `2026-11-18T17:32:47+05:30`, plus `5` pending at `2026-11-19T17:28:47+05:30`), leaving only `10` look-ahead-free matured `1D` outcomes. The sole executed paper trade (`BIRLACABLE` 10D) comes from the `2026-11-17T10:25:29` intraday look-ahead snapshot.
  2. `phase5_intelligence.json` (`monitoring`) reports `total_logged_predictions = 90` and `evaluated_outcomes_count = 75` (`drift_status = "OK"`), inflated by `45` duplicate `(symbol, observation_date)` rows in `prediction_log.parquet` (vs `30` unique evaluated pairs).
  3. `forward_validation.json` reports `paper_portfolio.benchmarks.nifty_50_buy_hold.status = "UNAVAILABLE"` despite `indices_daily.parquet` containing `495` `"NIFTY_50"` rows (due to `"NIFTY 50"` space-vs-underscore mismatch in `forward_validation.py`).
- **Root cause:** Frozen upstream `backend/news_intel/forward_validation.py` and `backend/news_intel/monitoring.py`.
- **Affected files & records:** `docs/data/forward_validation.json`, `docs/data/phase5_intelligence.json`, `data/predictions/{forward_predictions,forward_outcomes,prediction_log,paper_trades}.parquet`, `data/processed/indices_daily.parquet`.
- **Remediation:**
  - `backend/audit/dashboard_reconciler.py` reconciles and surfaces all three issues automatically (`derived_canonical_1d_audit_counts`, `parquet_duplicate_sym_obs_rows`, `nifty_benchmark.parquet_has_NIFTY_50`).
  - Frontend continues to display `UNAVAILABLE` honestly for `nifty_50_buy_hold` rather than fabricating a benchmark figure.
- **Tests:** `test_dashboard_reconciler_live_datasets_and_read_only`.
- **Evidence of successful execution:** Reconciler confirms `derived_canonical_1d_audit_counts == {"INVALID_LOOKAHEAD": 15, "MATURED": 10, "PENDING": 5}`, `parquet_duplicate_sym_obs_rows == 45`, and `parquet_has_NIFTY_50 is True` while `json_status == "UNAVAILABLE"`.
- **Remaining limitations:** Because frozen prediction/forward-validation pipelines cannot be edited in this workstream, published JSON values retain the upstream un-deduplicated/look-ahead aggregates until upstream remediation branches are integrated and artifacts regenerated.

---

### Issue 6: Stock JSON Bundle Freshness Skew & Zero-Sentinel Cash Flows

- **Original finding:**
  1. `27/28` stock JSONs (`docs/data/*.json`, all except `TATAELXSI`) contain `latest_ocf_cr = 0.0` and `latest_fcf_cr = 0.0` zero-sentinels.
  2. `8` non-core stock JSONs (`BHARTIARTL`, `HDFCBANK`, `ICICIBANK`, `INFY`, `LT`, `SBIN`, `SUNPHARMA`, `TRENT`) have `fetched_at = 2026-11-16` while embedding `quantitative_prediction.observation_date = 2026-11-18`.
  3. `2` symbols in `latest_predictions.json` (`HFCL`, `KPITTECH`) have no standalone `docs/data/{SYMBOL}.json` bundle, and `15/28` stock JSONs lack `quantitative_prediction` (`23/28` lack `data_quality`).
- **Root cause:** `scripts/generate_static_data.py` only forces refresh for `CORE_5_SYMBOLS`, while quantitative updates patch `quantitative_prediction` onto existing cached bundles.
- **Affected files & records:** `docs/data/*.json` (28 stock bundles), `docs/data/latest_predictions.json`, `data/processed/data_quality.parquet`.
- **Remediation:**
  - `reconcile_stock_jsons` in [`backend/audit/dashboard_reconciler.py`](backend/audit/dashboard_reconciler.py) audits all 28 stock bundles against `latest_predictions.json`, `data_quality.parquet`, and `corporate_intelligence.parquet`, flagging the 27 zero-sentinel files and 8 `fetched_at`-skewed files.
  - Confirmed `frontend/report.js:594` renders zero-sentinel cash flows as `UNAVAILABLE` with explanatory note, and `frontend/app.js` resets `Data Quality: UNAVAILABLE` and `ML: UNAVAILABLE` when `null`.
- **Tests:** `test_dashboard_reconciler_live_datasets_and_read_only`, `test_report_renderer_real_data_and_integrity_via_node`, `test_app_regime_and_data_quality_contract`.
- **Evidence of successful execution:** `reconcile_stock_jsons` verified `28` stock files (`13` with `quantitative_prediction`, `5` with `data_quality`, `0` probability/score mismatches vs `latest_predictions.json` / `data_quality.parquet`).
- **Remaining limitations:** `fetched_at` skew on the 8 cached bundles persists until `FORCE_REFRESH=1` regenerates `docs/data/*.json`.

---

### Issue 7: `phase5_intelligence.json` Stock vs Horizon Date Skew (`2026-11-19` vs `2026-11-18`)

- **Original finding:** All 5 pilot stocks in `docs/data/phase5_intelligence.json` carry `stock.observation_date = "2026-11-19"` while all 15 `horizon_outlooks` (`1D`, `5D`, `10D` × 5 stocks) carry `model_prediction_date = "2026-11-18"`.
- **Root cause:** News collection ran on `2026-11-19` (`last_news_collection_at = 2026-11-19T17:29:00+05:30`) while quantitative model predictions were anchored to the latest EOD bar (`2026-11-18`).
- **Affected files & records:** `docs/data/phase5_intelligence.json` (`stocks.{RELIANCE,TCS,TATAELXSI,ITC,BIRLACABLE}`).
- **Remediation:** `reconcile_phase5_intelligence` detects and reports `horizon_date_gaps_count = 15`, and verifies `horizon_outlooks[h].model_prediction_date` matches `latest_predictions.json` (`2026-11-18`). `frontend/app.js` renders `o.model_prediction_date` (`2026-11-18`) per horizon row and separates `News Collected` vs `Quant Cutoff` in `#p5-freshness-sub`.
- **Tests:** `test_dashboard_reconciler_live_datasets_and_read_only`, `test_frontend_app_js_rejection_breakdown_cutoff_and_unavailable`.
- **Evidence of successful execution:** Reconciler confirms `horizon_date_gaps_count == 15` with `0` mismatches against `latest_predictions.json`.
- **Remaining limitations:** Date difference reflects asynchronous news-vs-EOD-price cadence in upstream `pipeline.py`.

---

## Verification Command & Exact Output

```bash
/home/antigravity/.venv/bin/pytest tests/test_remediation_dashboard_reporting.py tests/test_frontend_contract.py tests/test_frontend_report_render.py tests/test_frontend_app_contract.py -v
```

Result: **`16 passed in 1.51s`** (0 failed, 0 skipped).
