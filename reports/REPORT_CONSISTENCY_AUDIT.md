# BharatAlpha AI — Report Consistency & Metric Reconciliation Audit

**Branch:** `feature/data-integrity-audit`  
**Base Commit SHA (`main`):** `8f328b8cdb8f7a1e508c1c47a648849945873013`  
**Audit Date:** 2026-11-18  
**Scope:** Read-only root-cause investigation of reporting discrepancies between `docs/data/forward_validation.json` (Report A) and `docs/data/phase5_intelligence.json` (Report B) without modifying production prediction logic, thresholds, schedules, or historical snapshots.

---

## 1. Executive Summary & Answers to the 10 Investigation Questions

1. **Why do the model versions differ?**
   - Both `5.3.0-pilot` (`backend/news_intel/forward_validation.py:121, 504`, `backend/models.py:801, 878, 999`) and `5.0.0-pilot` (`backend/news_intel/pipeline.py:177`, `backend/models.py:766`) are **hardcoded subsystem schema labels**, not dynamic reads from a central model registry. The underlying quantitative ensemble model is `ml_ensemble` version `3.2.0` (`backend/prediction/ensemble_model.py:71`), while `prediction_log.parquet` also retains 15 legacy `3.0.0` rows from an earlier run on `2026-11-16`.
2. **Are both reports reading the same prediction records?**
   - **No.** Report A reads `data/predictions/forward_predictions.parquet` (55 rows across the 5-stock Phase 5 pilot universe) and `data/predictions/forward_outcomes.parquet` (165 rows). Report B's `monitoring` block reads `data/predictions/prediction_log.parquet` (75 rows across the 15-stock quantitative universe).
3. **Are historical proxy observations included in one report?**
   - Neither report merges `walk_forward_predictions.parquet` or `phase5_experiments.json` into its headline counts, **but** `structured_events.parquet` contains 156 historical RSS backfill articles (`collection_timestamp = NaN`) plus 18 late-collected articles (`collection_timestamp > prediction_timestamp`, `effective_trading_date <= observation_date`) that enter `compute_pit_news_features()` because it filters only on `publication_timestamp <= as_of_ts`.
4. **Are prospective observations correctly separated from historical records?**
   - **No.** In `forward_predictions.parquet`, 5 canonical `INTRADAY` snapshots (`obs=2026-11-16`, generated `2026-11-17T10:25:29+05:30`) were recorded **70 minutes after** the `2026-11-17 09:15 IST` entry open (`INVALID` / look-ahead contaminated), and 5 canonical `POST_MARKET` snapshots (`obs=2026-11-17`, generated `2026-11-18T17:32:47+05:30`) were recorded **after** the `2026-11-18 15:30 IST` close (`HISTORICAL` for 1D, `INVALID` for 5D/10D).
5. **Are duplicate snapshots counted consistently?**
   - **No.** Report B (`monitoring.py`) performs **zero deduplication** on `prediction_log.parquet` (counting 4 runs on `2026-11-16` × 15 stocks = 60 rows as independent observations). Report A deduplicates by `(symbol, observation_date, nse_session)` (`deduplicate_forward_predictions`, L202), which removes 35 same-session retries but still keeps both `POST_MARKET` and `INTRADAY` for `obs=2026-11-16` as "canonical", doubling the 1D matured count from **5** to **10**.
6. **Are outcome maturity rules identical?**
   - **No.** Report A (`evaluate_forward_outcomes`, `forward_validation.py:293`) evaluates **open-to-close** returns (`open[t+1] → close[t+H]`) and deducts delivery transaction costs. Report B (`DuckDBStore.evaluate_realized_outcomes`, `duckdb_store.py:357`) evaluates **close-to-close** returns (`close[t] → close[t+1]`) with no transaction cost deduction.
7. **Are both reports using the same evaluation dates?**
   - Both reports evaluate `observation_date = 2026-11-16` outcomes realized on `2026-11-17` (the latest date in `equities_daily.parquet`), while `2026-11-17` predictions remain pending `2026-11-18` Bhavcopy ingestion. However, `phase5_intelligence.json` stamps `stocks[*].observation_date = "2026-11-18"` (wall-clock `now`) even though its quant inputs come from `2026-11-17`.
8. **Are different feature sets or model configurations being reported?**
   - Yes. Report A evaluates the combined Phase 5 multi-horizon signal (`237` quant features + `17` news/cross-stock features = `237+17`). Report B's `monitoring` block evaluates raw `prob_next_day_up` from the 237-feature `ml_ensemble` quant model (`3.2.0` mixed with `3.0.0`).
9. **Are the metrics calculated from different populations?**
   - Yes: Report A's 1D metrics use 10 canonical rows (5 pilot stocks × 2 sessions on `2026-11-16`). Report B's header says `evaluated_outcomes_count = 60` (15 quant stocks × 4 runs on `2026-11-16`), but `monitoring.py:50` slices `recent = eval_df.tail(50)`—silently dropping 10 rows (`BHARTIARTL` ×4, `BIRLACABLE` ×4, `HDFCBANK` ×2).
10. **Can each displayed metric be traced back to its source records?**
    - **Yes.** Every metric in both reports has been deterministically reproduced from the exact Parquet rows and code paths documented in Section 2 below.

---

## 2. Detailed Metric-by-Metric Comparison Table

| Metric Name | Report A (`forward_validation.json`) | Report B (`phase5_intelligence.json` → `monitoring`) | Source Dataset(s) | Evaluation Population | Calculation Method & Code Path | Explanation of Discrepancy | Acceptable? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Version** | `5.3.0-pilot` | `5.0.0-pilot` (underlying log: `3.2.0` ×60, `3.0.0` ×15) | Hardcoded literals in Python models/pipelines | Subsystem metadata | `forward_validation.py:121,504` vs `pipeline.py:177` | Hardcoded strings in two modules; neither reads `ml_ensemble.version` (`3.2.0`). | **No (Bug)** — requires unified version registry. |
| **Feature-Set Version** | `237+17` | Not explicitly labelled in `monitoring` (uses 237 quant features) | `forward_predictions.parquet` vs `prediction_log.parquet` | 5 pilot stocks (combined) vs 15 quant stocks (quant-only) | `ForwardValidationPayload.feature_set_version` | Report A evaluates combined quant+news; Report B monitoring evaluates quant-only log. | **Acceptable only if explicitly labelled** as distinct layers. |
| **Total Stored / Logged Predictions** | `55` (`total_immutable_snapshots`) | `75` (`total_logged_predictions`) | `forward_predictions.parquet` (55) vs `prediction_log.parquet` (75) | 5 pilot stocks (11 runs) vs 15 quant stocks (5 runs) | `len(snaps_df)` vs `len(pred_df)` | Different Parquet stores, universes (5 vs 15), and run histories. | **Legitimate difference**, but must not be presented as same store. |
| **Canonical / Deduplicated Predictions** | `20` (`canonical_snapshots_count`; `35` duplicates) | Not reported (all `75` treated as canonical) | `forward_predictions.parquet` (`~is_duplicate_run`) vs `prediction_log.parquet` | 20 canonical snapshots vs 75 un-deduped rows | `deduplicate_forward_predictions` (`(symbol, obs_date, session)`) vs no dedup in `monitoring.py` | `prediction_log.parquet` keys on `prediction_id` with `run_tag`; `forward_predictions` still keeps 2 sessions per date. | **No (Bug)** — `monitoring.py` lacks dedup; `forward_predictions` needs target-session dedup. |
| **Evaluated / Matured Outcomes** | `10` (`matured_outcomes_total`) | `60` (`evaluated_outcomes_count`) | `forward_outcomes.parquet` vs `prediction_log.parquet` | 10 canonical 1D rows (5 stocks × 2 sessions) vs 60 rows (15 stocks × 4 runs on `11-16`) | `(canonical_outs_df["status"] == "MATURED").sum()` vs `pred_df["realized_next_day_return_pct"].notna().sum()` | Report B counts 4 runs on `2026-11-16` × 15 stocks = 60; Report A counts 2 sessions × 5 pilot stocks = 10 (5 prospective + 5 look-ahead `INTRADAY`). | **No (Bug)** — both inflate counts via duplicate/post-open runs. |
| **Pending Outcomes** | `50` (`pending_outcomes_total`: 10×1D + 20×5D + 20×10D) | `15` implicit (`75 - 60`, 1D only) | `forward_outcomes.parquet` vs `prediction_log.parquet` | 20 canonical × 3 horizons minus 10 matured vs 15 stocks on `2026-11-17` | `(canonical_outs_df["status"] == "PENDING").sum()` | Report A sums pending rows across 3 horizons (1D/5D/10D); Report B tracks 15 pending rows on `2026-11-17`. | **Legitimate structural difference**, requires per-horizon breakdown. |
| **1D Directional Accuracy** | `40.0%` | `60.0%` (deduped 15 stocks: `73.33%`; pilot-5: `80.0%`) | `forward_outcomes` + `forward_predictions` vs `prediction_log` | 10 canonical pilot rows (`open[t+1]→close[t+1]`) vs `eval_df.tail(50)` of 60 quant rows (`close[t]→close[t+1]`) | `compute_forward_horizon_metrics` (`forward_validation.py:553`) vs `_compute_horizon_metrics` (`monitoring.py:50`) | Different stock universes (5 vs 15), different return definitions (open→close vs close→close), `tail(50)` truncation in Report B, and duplicate runs. | **No (Bug in Report B `tail(50)` & lack of dedup; look-ahead rows in Report A).** |
| **1D ROC-AUC** | `0.6667` | `0.7452` | Same as above | `n=10` (Report A) vs `n=50` `tail(50)` slice (Report B) | `roc_auc_score(y_up, p_up)` | Same population, return-definition, and `tail(50)` differences as accuracy. | **No (Bug)** — same root causes. |
| **1D Brier Score** | `0.2478` | `0.2436` | Same as above | `n=10` vs `n=50` (`tail(50)`) | `brier_score_loss(y_up, p_up)` | Same root causes. | **No (Bug)** — same root causes. |
| **1D Rank IC** | `0.5908` | `-0.0671` | Same as above | `n=10` vs `n=50` (`tail(50)`) | Spearman rank correlation `spearmanr(p_up, y_gross)` | On `2026-11-17`, open→close returns on the 5 pilot stocks ranked positively with combined `prob_1d_up` (`0.5908`), whereas close→close returns across `tail(50)` of the 15-stock quant log had near-zero rank correlation (`-0.0671`). | **Explained by return definition & population**, compounded by `tail(50)` and duplicate runs. |
| **1D Gross / Net Return** | `+0.35%` gross / `-1.80%` net | Not computed in `monitoring` | `forward_outcomes.parquet` | 10 canonical 1D rows | Sum/mean of `gross_return_pct` and `net_return_pct` (`forward_validation.py:570-585`) | Only Report A computes execution-aware gross/net returns after delivery costs. | **Acceptable (feature of Report A)**, except `n=10` includes 5 `INTRADAY` duplicates. |
| **5D & 10D Evaluation Status** | `INSUFFICIENT_DATA` (`0` matured, `20` pending each) | `INSUFFICIENT_DATA` (`0` evaluated) | `forward_outcomes.parquet` vs `prediction_log.parquet` | `0` matured 5D/10D rows | `n_eval < MIN_FORWARD_SAMPLES (10)` | Consistent: collection started `2026-11-16`, so 5D and 10D horizons have not elapsed. | **Yes (Acceptable & Consistent).** |

---

## 3. Exact Record-Level Traceability

### 3.1 The 60 Evaluated Rows in `prediction_log.parquet` (Report B)
- **4 runs on `observation_date = 2026-11-16` × 15 stocks = 60 rows:**
  1. `2026-11-16T05:42:38Z` — `ml_ensemble` `3.0.0` (15 rows, no run tag suffix)
  2. `2026-11-16T06:47:59Z` — `ml_ensemble` `3.2.0` (15 rows, suffix `_20261116T064759Z`)
  3. `2026-11-16T07:03:07Z` — `ml_ensemble` `3.2.0` (15 rows, suffix `_20261116T070307Z`)
  4. `2026-11-16T13:34:55Z` — `ml_ensemble` `3.2.0` (15 rows, suffix `_20261116T133455Z`)
- All three `3.2.0` runs have identical `prob_next_day_up` per symbol.
- `monitoring.py:50` (`recent = eval_df.tail(50)`) drops the first 10 rows in sorted order (`BHARTIARTL` ×4, `BIRLACABLE` ×4, `HDFCBANK` ×2), leaving 50 rows (`3.2.0` ×38 + `3.0.0` ×12) on which `30 / 50 = 60.0%` accuracy and `0.7452` ROC-AUC are calculated.

### 3.2 The 10 Matured Canonical 1D Rows in `forward_predictions.parquet` / `forward_outcomes.parquet` (Report A)
All 10 rows have `observation_date = 2026-11-16` and `entry_date = exit_date = 2026-11-17`:
- **5 Genuinely Prospective (`POST_MARKET`, `2026-11-16T19:23:27.388296+05:30` < `2026-11-17T09:15:00+05:30`):**
  - `fwd_2026-11-16_BIRLACABLE` (`prob_1d_up=0.4806`, `gross=-1.9817%`, `net=-2.4971%`, correct down call)
  - `fwd_2026-11-16_ITC` (`prob_1d_up=0.5080`, `gross=-0.6268%`, `net=-0.9821%`, incorrect up call)
  - `fwd_2026-11-16_RELIANCE` (`prob_1d_up=0.5183`, `gross=+2.0911%`, `net=+1.7376%`, correct up call)
  - `fwd_2026-11-16_TATAELXSI` (`prob_1d_up=0.5116`, `gross=-1.2586%`, `net=-1.6284%`, incorrect up call)
  - `fwd_2026-11-16_TCS` (`prob_1d_up=0.4939`, `gross=+0.4843%`, `net=+0.1211%`, incorrect down call)
  - **Prospective-only 1D result (`n=5`):** `2 / 5 = 40.0%` directional accuracy (`statistically_sufficient = False` since `n=5 < 10`).
- **5 Look-Ahead Contaminated & Duplicate-Target (`INTRADAY`, `2026-11-17T10:25:29.884689+05:30` >= `2026-11-17T09:15:00+05:30`):**
  - `fwd_BIRLACABLE_2026-11-16_20261117T102529`, `fwd_ITC_2026-11-16_20261117T102529`, `fwd_RELIANCE_2026-11-16_20261117T102529`, `fwd_TATAELXSI_2026-11-16_20261117T102529`, `fwd_TCS_2026-11-16_20261117T102529`.

---

## 4. Authoritative Source & Recommended Engineering Remediation

1. **Authoritative Source for Prospective Validation:** `forward_predictions.parquet` + `forward_outcomes.parquet` filtered through `backend/audit/classification.py::classify_forward_predictions` (`prediction_timestamp < entry_open`, deduplicated per `(symbol, target_entry_date, horizon)`).
2. **Do Not Combine Reports:** `phase5_intelligence.json -> monitoring` is a legacy 15-stock quant-only close-to-close log and must be explicitly labelled as such, deduplicated by `(symbol, observation_date)`, filtered to a single `model_version`, and freed from silent `tail(50)` truncation.
