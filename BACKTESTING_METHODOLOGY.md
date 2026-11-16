# BharatAlpha AI v3.2 — Backtesting Methodology & Leakage Audit

## 1. Executive Summary of Audit Findings (v3.0 → v3.2)

An end-to-end quantitative audit of `backend/backtesting/walk_forward.py`, `backend/prediction/ensemble_model.py`, `backend/features/feature_engine.py`, and `scripts/run_phase3_backtest.py` identified four material sources of look-ahead bias and evaluation distortion in the prior v3.0/v3.1 pipeline:

1. **Global Feature-Selection Leakage**: `scripts/run_phase3_backtest.py` previously passed `feature_catalog.json["selected_features"]` (computed across the entire dataset, including out-of-sample test dates `2025-11-07` to `2026-11-16`) into `run_walk_forward_backtest`. Every walk-forward fold indirectly benefited from out-of-sample Spearman IC screening.
2. **Insufficient Purge Gap for 5-Day Labels**: `walk_forward.py` previously used `purge_gap_days=1` while `BharatAlphaQuantEnsemble` trained `lgb_reg_5d_` and `xgb_reg_5d_` on `target_5d_ret` (a 5-trading-session forward return). The final 4 training sessions overlapped the test window.
3. **Unpurged In-Fold Screening & Hardcoded Calibration**: `_select_in_fold_features` scored `target_5d_ret` without purging the trailing 5 sessions, and `fit()` used a hardcoded `calibration_slope_ = 0.85` rather than fitting Platt scaling on a chronologically separated validation slice.
4. **Same-Day Post-Market Filing Visibility**: `_build_pit_fundamental_frame` in `backend/features/feature_engine.py` set `pub_dt` to the filing calendar date even though NSE/SEBI LODR filings are timestamped `18:00 IST` (post-market close at `15:30 IST`), making quarterly/annual numbers visible on the filing day itself rather than `t + 1`.

All four issues have been corrected in v3.2.

---

## 2. Chronological Walk-Forward, Purge & Embargo Design

- **Expanding-Window Chronological Splits**:
  - `initial_train_dates = 240` trading sessions
  - `step_dates = 42` trading sessions (~2 months per out-of-sample fold)
  - All splits are measured strictly in **trading sessions** (`all_dates` from `equities_daily.parquet` / `daily_features.parquet`), never calendar days.
- **5-Session Purge + 1-Session Embargo**:
  - `purge_gap_days = 5` trading sessions between `train_end_date` (`cursor - 1`) and `test_start_date`, protecting the longest prediction horizon (`target_5d_ret`).
  - `embargo_sessions = 1` additional trading session after the purge window (`test_start_idx = cursor + purge_gap_days + embargo_sessions`), preventing serial-correlation leakage at the fold boundary.
  - `cursor` advances by `step_dates` each fold so out-of-sample test windows are contiguous and non-overlapping.
- **Strictly In-Fold Feature Selection & Collinearity Pruning**:
  - `run_phase3_backtest.py` passes all `237` candidate feature columns (`get_candidate_feature_columns`) into `run_walk_forward_backtest`.
  - Inside each training fold (`train_slice`), `BharatAlphaQuantEnsemble._select_in_fold_features` purges the trailing `purge_sessions` (`5` sessions) when scoring `target_5d_ret`, applies two-half temporal IC-sign stability screening, prunes collinear features (`|corr| > 0.85`), and selects `top_k_features` (`40` by default).
- **Chronological Inner Validation & Platt Calibration**:
  - `BharatAlphaQuantEnsemble.fit()` splits `train_slice` chronologically (`80%` `inner_train`, `5`-session purge gap, `20%` `inner_val`).
  - Sub-model ensemble weights (`w_lgb`, `w_xgb`, `w_lin`, `w_pri`) and Platt sigmoid calibration (`LogisticRegression` on `logit(p_raw)`) are fitted exclusively on `inner_val`.

---

## 3. Point-in-Time (PIT) Data Alignment

- **Price, Volume & Regime Features**: Computed strictly from lagging/current session bars (`t` and earlier) with `shift(1)` or rolling windows ending at `t`.
- **SEBI LODR Fundamental & Shareholding Filings**: Statutory filing timestamps (`18:00 IST`) are shifted by `+1` calendar day in `_build_pit_fundamental_frame` so post-market filings on session `t` become available no earlier than session `t + 1`.
- **Target Isolation**: `target_next_day_ret`, `target_next_day_up`, `target_5d_ret`, `target_next_day_mae`, and `target_next_day_mfe` are excluded from `get_candidate_feature_columns`.

---

## 4. Automated Verification

- `tests/test_walk_forward_integrity.py`:
  - Verifies 5-session purge + 1-session embargo protects the 5D horizon with zero duplicate `(symbol, observation_date)` rows across folds.
  - Verifies **planted future-leak invariance**: injecting `target_next_day_ret` into `test_slice` features produces bit-for-bit identical out-of-sample predictions (`atol=1e-6`), proving zero test-fold leakage into feature selection, scaling, or calibration.
  - Verifies post-market `18:00 IST` filings on `2025-05-15` are invisible on `2025-05-15` and visible on `2025-05-16`.
