# BharatAlpha Benchmark & Statistical Evaluation Methodology

## 1. Purpose & Separation from Frozen Prediction Logic
`backend/backtesting/benchmark_eval.py` is an independent, read-only evaluation module completely decoupled from frozen prediction, feature-engineering, and news-intelligence code. It consumes persisted prediction snapshots (`data/predictions/forward_predictions.parquet`) and market price histories (`data/processed/equities_daily.parquet`, `data/processed/indices_daily.parquet`) to produce apples-to-apples benchmark comparisons and uncertainty-quantified statistical metrics (`data/backtests/benchmark_comparison.json`).

## 2. Look-Ahead Exclusion & Session Deduplication
1. **Strict Prediction-Timestamp Cutoff**:
   - Entry is permitted only at `open[t+1]` on the first NSE trading session strictly **after** `max(observation_date, prediction_timestamp[:10])`.
   - If a snapshot has `observation_date = 2026-11-17` but `prediction_timestamp = 2026-11-18T17:32:47+05:30`, `2026-11-18` is **excluded** as an entry date because the prediction was generated after `2026-11-18` market close.
2. **Daily Canonical Deduplication**:
   - `deduplicate_daily_canonical_snapshots()` filters `is_duplicate_run == True` and selects at most **one** canonical snapshot per `(symbol, observation_date)` (preferring `POST_MARKET` > `INTRADAY` > `PRE_MARKET`, tie-broken by latest `prediction_timestamp`).
   - This prevents same-day multi-session double-counting in return compounding, Sharpe ratios, and effective sample sizes.

## 3. Horizon-Aligned, Cost-Matched Benchmarks
All active and passive strategies are evaluated over the **exact same** `[entry_date, exit_date]` trading windows (`open[t+1]` to `close[t+H]` for `H ∈ {1, 5, 10}`):
- **BharatAlpha (`bharat_alpha`)**: Uses `prob_{h}_up` and `recommendation_{h}`.
- **Naive Momentum (`naive_momentum`)**: Uses horizon-matched `benchmark_prob_{h}_up` (`1D` fallback to `benchmark_prob_up`; never cross-horizon).
- **Equal-Weight Pilot Universe (`equal_weight_universe`)**: Equal-weight long across all evaluated pilot symbols over the identical `[entry_date, exit_date]` windows.
- **Nifty 50 Buy-and-Hold (`nifty_50_buy_hold`)**: Normalizes `index_code` via `str.upper().replace(" ", "_") == "NIFTY_50"` (resolving the `NIFTY_50` vs `NIFTY 50` mismatch in `indices_daily.parquet`) and computes `open[entry_date] -> close[exit_date]` returns over the exact same windows.
- **Cash No-Trade (`cash_no_trade`)**: `0.0%` gross/net return, `0.0%` max drawdown.

All equity and index strategies apply the same statutory Indian CNC delivery round-trip cost (`DEFAULT_CNC_ROUNDTRIP_COST_PCT = 0.347%`) so gross-to-net comparisons are strictly comparable. Corporate-action splits/bonuses are adjusted via `adj_close / close` ratios between `entry_idx` and `exit_idx`.

## 4. Statistical Metrics, Confidence Intervals & `INSUFFICIENT_DATA` Gating
- **Daily-Portfolio Compounding & Sharpe**: Returns are averaged across active positions per `entry_date` before compounding (`prod(1 + r_daily / 100) - 1`) and annualizing (`mean(r_daily) / std(r_daily) * sqrt(252)`).
- **Confidence Intervals (95%)**:
  - **Directional Accuracy & Win Rate**: Wilson score 95% CI (`wilson_ci`).
  - **ROC-AUC, Brier Score, Spearman Rank IC**: Non-parametric percentile bootstrap 95% CI (`bootstrap_ci`, 500 resamples).
- **Effective Sample Size & `INSUFFICIENT_DATA` Gating**:
  - Statistical evaluation requires **both** `matured_observations >= 30` (`MIN_EVAL_OBSERVATIONS`) and `unique_observation_dates >= 10` (`MIN_EVAL_UNIQUE_DATES`).
  - When either threshold is not met, `status` is set to `"INSUFFICIENT_DATA"` with an explicit `insufficiency_reason`, and inferential/annualized metrics (`directional_accuracy_pct`, `roc_auc`, `brier_score`, `rank_ic`, `net_return_pct`, `daily_portfolio_sharpe`, etc.) are gated to `null`—never fabricated or presented as statistically meaningful.
