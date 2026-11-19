# BharatAlpha AI — Dashboard UX, Explainability & Data-Contract Audit

**Audit Scope:** `frontend/index.html`, `frontend/app.js`, `frontend/report.js`, `frontend/styles.css`, `scripts/build_github_pages.py`, `docs/index.html`, `docs/data/*.json`, `backend/models.py`, and `tests/test_frontend_*.py`.  
**Model Freeze Compliance:** Read-only with respect to the prediction engine. Zero changes to model training, feature engineering, calibration, or inference logic.

---

## 1. Executive Summary of Findings

| Area | Status Before Fix | Resolution Applied (UI / Contract Only) |
|---|---|---|
| Market Regime Ribbon (`updateRegimeBannerDOM`) | **Broken field mapping**: read `nifty_50_return_1d_pct`, `india_vix`, `advances_count`, `declines_count` instead of schema fields `nifty_50_day_change_pct`, `india_vix_close`, `advance_count`, `decline_count`. | Mapped canonical `MarketRegimeSnapshot` fields (`app.js:1040-1057`). |
| Data Quality Gate (`#mr-gate-badge`) | **Stale state**: skipped DOM update when `data_quality` was `null` (21 non-pilot stocks, e.g. `BEL.json`), leaving `"Data Quality Gate: ACTIVE"` or prior stock's score. | Reset explicitly to `"Data Quality: UNAVAILABLE"` whenever `quality` is `null` (`app.js:1058-1068`). |
| TreeSHAP Attribution Units (`formatShapDriverItem`) | **Unit mislabeling**: multiplied LightGBM `pred_contrib=True` log-odds delta (`contribution`) by `100` and labeled it `bps`. | Formatted raw SHAP log-odds delta to 4 decimals with `SHAP` label (`app.js:1071-1081`, `index.html:293-297`). |
| Backtest / Universe Header Hardcoding | Hardcoded `"7 Expanding Folds"` (`index.html:597`) while `backtest_summary.json` had `6`; hardcoded `"27"` badge (`index.html:634`) while `latest_predictions.json` had `15`. | Dynamic fold count in `#quant-bt-heading` (`app.js:1278-1281`) and dynamic row count in `#quant-pred-count-badge` (`app.js:1388-1391`). |
| Historical Proxy vs. Rolling Monitor vs. Prospective Evidence | `phase5_experiments.json` lacked `HISTORICAL PROXY` tag & `historical_pit_note`; `phase5_intelligence.monitoring` (rolling log, `n=75`, `OK`) sat above `forward_validation` (prospective 5-stock pilot, `n=20`, `WARNING`) without scope distinction; `news_provenance.is_genuinely_forward_collected` (`false`, `0` post-cutoff events) was hidden. | Added `ROLLING MODEL MONITOR` scope pill, `HISTORICAL PROXY — NOT LIVE FORWARD RESULTS` pill + `historical_pit_note` banner (`#p5-experiments-pit-note`), `PROSPECTIVE PILOT` + `#fv-news-provenance-banner` in `index.html`/`app.js` and `npTxt` in `report.js`. |
| Serverless Fallback (`scripts/build_github_pages.py`) | `liveBrowserResearch` fabricated `1.5%` R&D, `14.2%` 5Y Profit CAGR, `Piotroski 7`, `RSI 58.5`, `12.4%` 1Y return, `50/15/15/20` shareholding, and fixed ATR bands. | Replaced all uncollected live-scrape metrics with `null` / `UNAVAILABLE` and guarded `app.js` renderers against `null` values (`renderBundle`, `renderCatalystsRisksNews`, `drawRDCanvas`). |
| Frontend Contract Tests | `test_frontend_contract.py` & `test_frontend_report_render.py` tested `report.js` only; zero execution of `app.js` against `docs/data/*.json`. | Added `tests/test_frontend_app_contract.py` executing `app.js` renderers in Node against `regime.json`, `backtest_summary.json`, `latest_predictions.json`, `phase5_experiments.json`, `phase5_intelligence.json`, `forward_validation.json`, `RELIANCE.json`, and `BEL.json`. |

---

## 2. Detailed UX & Explainability Surface Inventory

### A. Model Status & Freeze Transparency
- **Global Freeze Banner** (`frontend/index.html:56-61`): Visible across all tabs (`MODEL FREEZE & AUDIT PHASE`, `17 Nov – 8 Dec 2026 review`, research validation only).
- **Phase 5 & Forward Validation Headers** (`frontend/app.js:1945-1953`, `2127-2142`): Renders `model_version` (`5.0.0-pilot` / `5.3.0-pilot`), `feature_set_version` (`237+17`), `last_news_collection_at`, `last_quant_update_at`, and `pilot_universe`.
- **Per-Stock Quant Status** (`frontend/app.js:1100-1124`, `frontend/report.js:819-831`): Renders `observation_date`, `target_session_date`, `generated_at`, `model_name`, `model_version`, `model_agreement_score`, and `abstention_reason`.

### B. Prediction Explainability
- **1D / 5D / 10D Multi-Horizon Outlook**:
  - `report.js:142-188, 704-743`: Independently displays 1D, 5D, and 10D model availability, calibrated probability, gross & net expected return, 1D MAE downside, freshness, and Phase 5 combined signal (`combinedProb`, `combinedSignal`, `decision`). Explicitly separates **Probability**, **Expected Return**, and **Confidence Score**.
  - `app.js:1998-2056`: Displays per-pilot-stock 1D/5D/10D table with `quant_prob_positive`, `news_signal`, `cross_stock_signal`, `confidence`, `decision`, `expected_gross_return_pct / expected_net_return_pct`, `model_prediction_date`, and `data_freshness`.
- **Feature Attribution (TreeSHAP)**:
  - `app.js:1071-1081, 1171-1184, 1419-1444` and `report.js:834-836`: Renders `top_bullish_drivers` and `top_bearish_drivers` with feature name, family, raw feature value, and exact LightGBM `pred_contrib` SHAP contribution.
- **News Categorization, Epistemic Separation & Warnings**:
  - `report.js:327-455`: Groups events into 6 categories (`Company`, `Promoter / Management`, `Industry`, `Competitor`, `Regulatory`, `Global Macro`), filters generic price-page headlines (`RR_GENERIC_PRICE_RE`), separates `Verified fact` from `Model interpretation`, and flags low-confidence/headline-only items.
  - Warnings: `p5-warnings-banner`, `fv-warnings-banner`, `rrRisks`, and `rrSources` surface `data_quality.warnings`, `data_quality_warnings`, `monitoring_alerts`, and `failure_alerts`.

### C. Performance Evaluation, Benchmarks, Data Quality & Paper Portfolio
- **Historical Walk-Forward Backtest** (`backtest_summary.json` -> `app.js:1278-1384`, `report.js:838-855`): Objectively ranks models by `annualized_sharpe_net` (`naive_momentum` vs `ml_ensemble` vs baselines & Nifty 50 buy-and-hold), displaying cost drag, drawdown, win rate, ROC AUC, and Brier score.
- **Historical Proxy Experiments A–E** (`phase5_experiments.json` -> `app.js:2092-2142`): Explicitly tagged `HISTORICAL PROXY — NOT LIVE FORWARD RESULTS` with `latest_feature_date`, `execution_mode`, and `historical_pit_note`.
- **Rolling Model Monitor vs. Prospective Forward Validation**:
  - `p5-monitoring-kpis` (`app.js:1964-2001`) is scoped as `ROLLING MODEL MONITOR` (`prediction_log.parquet`, showing `n` per horizon).
  - `sec-forward-validation` (`app.js:2144-2240`, `report.js:857-911`) is scoped as `PROSPECTIVE PILOT`, displaying `canonical_snapshots_count` vs `total_immutable_snapshots`, `matured_count / pending_count`, `min_sample_threshold` (10), `INSUFFICIENT_DATA` / `WARNING` / `OK` status, `vs_naive_momentum_net_diff_pct`, and `news_provenance` (`is_genuinely_forward_collected`, `events_after_prediction_cutoff`, `provenance_note`).
- **Paper Portfolio Transparency** (`app.js:2242-2330`, `report.js:871-908`): Displays hypothetical disclaimer, ₹10,00,000 capital / 10% position / 80% exposure limits, strategy vs. benchmark rows (`naive_momentum*`, `nifty_50_buy_hold`, `cash_no_trade`), signal counts (`Rec / Exec / Open / Mat / Rej`), and last 20 audit-trail events.

---

## 3. Known Remaining UX Notes (Non-Blocking / Out of Scope for Model Freeze)
1. **Legacy `Next-Day Guidance` vs. Quant Ensemble Action**: Section `01` (`gd-action-banner`) shows the 5-agent heuristic `next_day_guidance.action`, while `Quick Summary` / `Detailed Research` (`report.js`) defaults to hiding legacy sections (`styles.css:1048-1049`) and computes `rrAssessment` transparently across fundamentals, technicals, and quant probabilities.
2. **Static Prose in 1-Month / 1-Year Conditions** (`frontend/index.html:1004-1035`): Macro/regime condition checklists below the 5-day table are general NSE research checklists rather than stock-specific.
3. **GitHub Pages Bundle Rebuild**: `docs/index.html` must only be rebuilt via `python scripts/build_github_pages.py` during the coordinated release step.
