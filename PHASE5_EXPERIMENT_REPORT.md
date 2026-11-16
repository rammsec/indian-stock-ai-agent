# BharatAlpha AI — Phase 5.2 Controlled Experiment Report (Experiments A–E)

## 1. Experimental Setup (Strict v3.2 Integrity)
- **Pilot Universe (`PILOT_UNIVERSE_5`)**: `RELIANCE`, `TCS`, `TATAELXSI`, `ITC`, `BIRLACABLE`
- **Walk-Forward Protocol**: 6 expanding-window chronological folds (`initial_train_dates=240`, `step_dates=42`)
  - **1D**: trained on `target_next_day_ret > 0` with `5`-session purge + `1`-session embargo
  - **5D**: trained on `target_5d_ret > 0` with `5`-session purge + `1`-session embargo
  - **10D**: trained on `target_10d_ret > 0` with `10`-session purge + `1`-session embargo (protects 10-day forward return labels from any train/test overlap)
- **In-Fold Feature Selection**: Top-40 features selected strictly inside each training fold (`BharatAlphaQuantEnsemble`).
- **Execution & Costs**:
  - **1D**: `next_open_delivery_capped` (09:15 IST next-session open), full Indian statutory CNC costs + ATR/Amihud slippage.
  - **5D & 10D**: Non-overlapping `H`-session holding rule per symbol (`H=5` for 5D, `H=10` for 10D, `prob >= 0.535`, 20% max position weight, VIX < 21), full Indian statutory CNC delivery costs + slippage.
- **Benchmark**: `naive_momentum` evaluated under identical dates, symbols, non-overlapping holding rules, and costs.
- **Data Integrity Note**: Historical Google News RSS does not exist years in the past. Rather than fabricating historical news articles, Experiments B–E reconstruct genuine historical point-in-time proxies (`ne_*`, `inst_*`, `rel_*`, `ms_*`) into the 17 Phase 5 news/cross-stock features. Live RSS/Gemini extraction accumulates forward-only in `data/predictions/structured_events.parquet`.

---

## 2A. Out-of-Sample Classification & Ranking Metrics (1D, 5D, 10D)

*(Classification metrics measure directional discrimination/calibration only and are reported separately from investment returns.)*

| Experiment | Features | 1D AUC | 1D Acc% | 1D Brier | 1D IC | 5D AUC | 5D Acc% | 5D Brier | 5D IC | 10D AUC | 10D Acc% | 10D Brier | 10D IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A: Quant 237 Only** | 237 | 0.4647 | 52.23% | 0.2520 | -0.0342 | 0.4466 | 50.45% | 0.2642 | -0.0693 | 0.4218 | 52.19% | 0.2695 | -0.1598 |
| **B: + Company News** | 245 | 0.4647 | 52.23% | 0.2520 | -0.0342 | 0.4466 | 50.45% | 0.2642 | -0.0693 | 0.4218 | 52.19% | 0.2695 | -0.1598 |
| **C: + Company & Promoter** | 247 | 0.4648 | 52.23% | 0.2520 | -0.0342 | 0.4456 | 50.45% | 0.2641 | -0.0714 | 0.4272 | 52.19% | 0.2689 | -0.1517 |
| **D: + Industry & Competitor** | 250 | 0.4648 | 52.23% | 0.2520 | -0.0342 | 0.4456 | 50.45% | 0.2641 | -0.0714 | 0.4272 | 52.19% | 0.2689 | -0.1517 |
| **E: + All News, Macro & Cross-Stock** | 254 | 0.4643 | 51.26% | 0.2527 | -0.0374 | 0.4502 | 49.22% | 0.2645 | -0.0678 | 0.4315 | 52.10% | 0.2687 | -0.1442 |
| **Benchmark: `naive_momentum`** | Rule | **0.4829** | **52.96%** | **0.2506** | **-0.0129** | **0.4833** | **55.47%** | **0.2479** | **0.0000** | **0.4732** | **60.52%** | **0.2417** | **0.0000** |

---

## 2B. Execution-Aware Investment Returns (After Indian CNC Costs & Non-Overlapping Rules)

| Experiment | 1D Trades | 1D Net% | 1D Sharpe | 1D MaxDD | 5D Trades | 5D Gross% | 5D Cost% | 5D Net% | 5D Sharpe | 10D Trades | 10D Gross% | 10D Cost% | 10D Net% | 10D Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A: Quant 237 Only** | 16 | -1.74% | -1.092 | -2.98% | 44 | -4.89% | 3.22% | -8.10% | -1.379 | 29 | -9.99% | 2.03% | -12.02% | -2.354 |
| **B: + Company News** | 16 | -1.74% | -1.092 | -2.98% | 44 | -4.89% | 3.22% | -8.10% | -1.379 | 29 | -9.99% | 2.03% | -12.02% | -2.354 |
| **C: + Company & Promoter** | 16 | -1.74% | -1.092 | -2.98% | 43 | -5.71% | 3.11% | -8.82% | -1.531 | 29 | -9.99% | 2.03% | -12.02% | -2.354 |
| **D: + Industry & Competitor** | 16 | -1.74% | -1.092 | -2.98% | 43 | -5.71% | 3.11% | -8.82% | -1.531 | 29 | -9.99% | 2.03% | -12.02% | -2.354 |
| **E: + All News, Macro & Cross-Stock** | 17 | -1.03% | -0.624 | -2.28% | 43 | -10.38% | 2.96% | -13.34% | -2.340 | 29 | -9.99% | 2.03% | -12.02% | -2.354 |
| **Benchmark: `naive_momentum`** | — | **+0.20%** | **+0.064** | -3.45% | 15 | **+5.67%** | 1.48% | **+4.19%** | **+0.540** | 1 | -0.51% | 0.10% | -0.61% | -1.040 |

---

## 3. Honest Assessment of Incremental Predictive Value

1. **No Proven ML Edge Yet Over `naive_momentum`**: On the 5-stock pilot historical sample, none of Experiments A–E outperform `naive_momentum` on 1D (`+0.20%` net, `+0.064` Sharpe), 5D (`+4.19%` net, `+0.540` Sharpe), or 10D (`-0.61%` net).
2. **Impact of Adding News/Cross-Stock Features (Exp E vs Exp A)**:
   - On 1D, Exp E reduced net loss (`-1.03%` vs `-1.74%`) and max drawdown (`-2.28%` vs `-2.98%`), and modestly lifted 5D/10D AUC (`0.4502` vs `0.4466`; `0.4315` vs `0.4218`).
   - However, under realistic 5D/10D execution with non-overlapping holding rules and CNC delivery costs, all ML experiments remain negative (`-8.10%` to `-13.34%` on 5D; `-12.02%` on 10D).
3. **Conclusion**: The Phase 5.2 infrastructure (strict per-horizon live probabilities & returns without 1D substitution or 5D scaling, `UNAVAILABLE` safety when horizon models are absent, CONFIRMED/INFERRED/UNKNOWN promoter classification, horizon-specific monitoring, and execution-aware 1D/5D/10D validation) is **operational and leakage-free**. Live trading must remain **disabled** until 6–12 months of forward `structured_events.parquet` accumulation demonstrates statistically significant out-of-sample alpha over `naive_momentum`.


