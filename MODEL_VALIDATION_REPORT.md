# BharatAlpha AI v3.2 — Quantitative Model Validation Report

## 1. Audit Verdict

**Promotion Decision: DO NOT PROMOTE `ml_ensemble`. Retain `naive_momentum` as the active benchmark.**

Once global feature-selection leakage (`feature_catalog.json["selected_features"]` computed over OOS dates), 5D label overlap (`purge_gap_days=1` -> `5` trading sessions + `1` embargo session), post-market filing visibility, and 100%-single-stock concentration were removed, **`ml_ensemble` failed to demonstrate any out-of-sample predictive edge** over the 6-fold walk-forward period (`2025-11-14` to `2026-11-13`, 15 NSE stocks, 3,705 OOS observations per model).

---

## 2. Why v3.0/v3.1 Showed `+19.75%` Net Return Despite `-6.77 bps` Trade Expectancy

Diagnostic decomposition of the published v3.0/v3.1 `docs/data/backtest_summary.json` revealed two structural distortions:
1. **Portfolio Concentration Distortion**: On `69` of `99` active days, `ml_ensemble` held only **1 stock** (`active_cnt = 1`), which received **100% portfolio weight** (`+29.56%` cumulative on `<=2`-stock days). On the `23` days with `>2` active stocks (`~103` of `199` stock-days), average daily portfolio return was `-0.3331%` (cumulative `-7.66%`). Unweighted stock-day expectancy (`-6.77 bps`) and profit factor (`0.918`) reflected the losing multi-stock days, while concentrated single-stock days inflated the portfolio curve.
2. **Single-Stock Outlier (`HFCL`)**: `HFCL` alone (`39` trades, `+123.99 bps` mean net return, `+48.36%` sum, driven by `+9.87%` on `2026-05-22` and `+7.49%` on `2026-05-26`) accounted for all net gains. **10 of 14 traded symbols** had negative mean net return (`SBIN -425.21 bps`, `KPITTECH -128.86 bps`, `TRENT -93.09 bps`, `HDFCBANK -53.27 bps`, `BHARTIARTL -38.51 bps`, `ITC -31.42 bps`, `INFY -29.95 bps`).

---

## 3. Clean Out-of-Sample Walk-Forward Comparison (v3.2)

- **Configuration**: Expanding window (`240` initial train sessions, `42`-session step, `5`-session purge, `1`-session embargo), strictly in-fold feature selection from `237` candidates, `top_k_features = 40`, `next_open_delivery_capped` execution (`09:15 AM IST` open, 20% max position weight, CNC costs, conservative SL/TP).
- **Benchmarks**: Nifty 50 Buy-and-Hold: **-2.84%** | Equal-Weight Universe Buy-and-Hold: **+0.76%**

| Model | Dir. Acc % | HC Hit % | ROC AUC | Brier | Log Loss | Calib. ECE | Rank IC (1D) | Gross % | Cost % | Net % | Net Sharpe | Max DD % | Profit Factor | Expectancy (bps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`naive_momentum`** | **52.6%** | **54.4%** | 0.510 | **0.2495** | — | — | **+0.0263** | **+14.52%** | 5.12% | **+9.39%** | **+0.93** | -8.44% | — | — |
| `linear_ridge_logistic` | 51.1% | 51.3% | **0.512** | 0.2584 | — | — | +0.0150 | +8.48% | 12.83% | -4.36% | -0.29 | -19.50% | — | — |
| `ml_ensemble` (`k=40`) | 50.7% | 49.3% | 0.475 | 0.2521 | 0.6974 | 0.0435 | -0.0402 | -5.48% | 1.28% | -6.76% | -2.47 | -6.79% | 0.461 | -50.68 |
| `technical_rule` | 51.9% | 52.4% | 0.511 | 0.2530 | — | — | +0.0135 | +0.34% | 12.57% | -12.23% | -0.88 | -18.22% | — | — |

---

## 4. Nested Walk-Forward Feature-Count Ablation (`20`, `40`, `80`)

| Configuration | ROC AUC | Brier | Log Loss | Calib. ECE | Rank IC (1D) | Jaccard Stability | Net Return % | Net Sharpe | Expectancy (bps) | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed `top_k = 20` | 0.4691 | 0.2520 | 0.6971 | 0.0481 | -0.0467 | 0.6007 | -1.67% | -0.965 | -30.91 | 0.616 |
| Fixed `top_k = 40` | 0.4754 | 0.2521 | 0.6974 | 0.0435 | -0.0402 | 0.5532 | -6.76% | -2.466 | -50.68 | 0.461 |
| Fixed `top_k = 80` | 0.4754 | 0.2532 | 0.6995 | 0.0520 | -0.0448 | 0.6231 | -15.22% | -2.124 | -31.91 | 0.609 |
| **Nested CV (`k` chosen in-fold)** (`[40,20,40,20,80,80]`) | 0.4789 | 0.2512 | 0.6956 | 0.0435 | — | — | +0.47% | +0.129 | +3.11 | 1.035 |

### Interpretation
- Every fixed `k` (`20`, `40`, `80`) has **negative Rank IC (`-0.040` to `-0.047`)** and **AUC < 0.48** out-of-sample.
- Nested CV (`+0.47%` net, `+0.129` Sharpe, `1.035` PF) barely breaks even and still trails `naive_momentum` (`+9.39%` net, `+0.93` Sharpe) and Equal-Weight Buy-and-Hold (`+0.76%`) while exhibiting `AUC = 0.4789`. We do **not** claim this as a reliable edge.

---

## 5. Market Regime Filter Evaluation (Out-of-Sample `ml_ensemble`)

| Regime Filter Variant | Active Stock-Days | Net Return % | Net Sharpe | Max Drawdown % | Profit Factor | Expectancy (bps) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_block_high_vol` | 72 | -6.76% | -2.466 | -6.79% | 0.461 | -50.68 |
| `block_bear_and_high_vol` | 56 | -5.99% | -2.391 | -6.02% | 0.396 | -57.87 |
| `bull_and_sideways_only` | 53 | -5.78% | -2.471 | -5.81% | 0.358 | -59.10 |
| `nifty_above_sma50_only` | 0 | 0.00% | 0.000 | 0.00% | 0.000 | 0.00 |

Regime filtering alone does not rescue `ml_ensemble` because the underlying signal (`Rank IC = -0.0402`) lacks cross-sectional or directional skill.

---

## 6. Cost & Infrastructure Compliance
- `ENABLE_GEMINI_EXTRACTOR=false` and `ENABLE_BIGQUERY=false` remain disabled by default.
- All storage and backtesting remain 100% local in Parquet + DuckDB + GitHub Pages (`docs/`).
