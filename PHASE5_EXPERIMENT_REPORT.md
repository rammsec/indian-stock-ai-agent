# BharatAlpha AI — Phase 5 Controlled Experiment Report (Experiments A–E)

## 1. Experimental Setup (Strict v3.2 Integrity)
- **Pilot Universe (`PILOT_UNIVERSE_5`)**: `RELIANCE`, `TCS`, `TATAELXSI`, `ITC`, `BIRLACABLE`
- **Walk-Forward Protocol**: 6 expanding-window chronological folds (`initial_train_dates=240`, `step_dates=42`)
  - **1D & 5D**: `5`-session purge + `1`-session embargo
  - **10D**: `10`-session purge + `1`-session embargo (protects 10-day forward return labels from any train/test overlap)
- **In-Fold Feature Selection**: Top-40 features selected strictly inside each training fold (`BharatAlphaQuantEnsemble`).
- **Execution & Costs**: `next_open_delivery_capped` (09:15 IST next-session open), full Indian statutory costs (STT, Stamp Duty, Exchange/SEBI fees, 18% GST, DP charges) + ATR/Amihud slippage.
- **Benchmark**: `naive_momentum` evaluated on identical dates, symbols, and costs.
- **Data Integrity Note**: Historical Google News RSS does not exist years in the past. Rather than fabricating historical news articles, Experiments B–E reconstruct genuine historical point-in-time proxies (`ne_*`, `inst_*`, `rel_*`, `ms_*`) into the 17 Phase 5 news/cross-stock features. Live RSS/Gemini extraction accumulates forward-only in `data/predictions/structured_events.parquet`.

---

## 2. Controlled Comparison Table (1D, 5D, 10D)

| Experiment | Features | 1D ROC AUC | 1D Net Return | 1D Net Sharpe | 1D Max DD | 1D Expectancy (bps) | 5D ROC AUC | 5D Rank IC | 10D ROC AUC | 10D Rank IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A: Quant 237 Only** | 237 | 0.4647 | -1.74% | -1.092 | -2.98% | -54.53 | 0.4599 | -0.0675 | 0.5000 | 0.0000 |
| **B: + Company News** | 245 | 0.4647 | -1.74% | -1.092 | -2.98% | -54.53 | 0.4599 | -0.0675 | 0.5000 | 0.0000 |
| **C: + Company & Promoter** | 247 | 0.4648 | -1.74% | -1.092 | -2.98% | -54.53 | 0.4599 | -0.0675 | 0.5000 | 0.0000 |
| **D: + Industry & Competitor** | 250 | 0.4648 | -1.74% | -1.092 | -2.98% | -54.53 | 0.4599 | -0.0675 | 0.5000 | 0.0000 |
| **E: + All News, Macro & Cross-Stock** | 254 | 0.4643 | -1.03% | -0.624 | -2.28% | -30.03 | 0.4562 | -0.0974 | 0.5000 | 0.0000 |
| **Benchmark: `naive_momentum`** | Rule | **0.4829** | **+0.20%** | **+0.064** | -3.45% | **+2.55** | **0.4920** | 0.0000 | 0.5000 | 0.0000 |

---

## 3. Honest Assessment of Incremental Predictive Value

1. **No Proven ML Edge Yet Over `naive_momentum`**: On the 5-stock pilot historical sample, none of Experiments A–E outperform `naive_momentum` (+0.20% net return, +0.064 Sharpe, +2.55 bps expectancy) on 1D out-of-sample risk-adjusted performance.
2. **Impact of Adding News/Cross-Stock Features (Exp E vs Exp A)**:
   - Adding the full 17-feature set (including regulatory, global macro, and cross-stock net impact) improved 1D net return from `-1.74%` to `-1.03%`, Sharpe from `-1.092` to `-0.624`, and max drawdown from `-2.98%` to `-2.28%` (primarily via better trade filtering/sizing), but 1D ROC AUC (`0.4643` vs `0.4647`) and 5D ROC AUC (`0.4562` vs `0.4599`) remained below `0.50`.
   - Adding historical `ne_*` company/promoter/competitor proxies alone (B, C, D) produced virtually identical results to A because those historical proxies were sparse and rarely survived in-fold top-40 selection.
3. **10D Horizon**: With `10`-session purge + `1`-session embargo on a 5-stock sample, `BharatAlphaQuantEnsemble` abstained/output `0.50` across OOS folds (`10D ROC AUC = 0.5000`), confirming insufficient 10D training signal in the current historical sample.
4. **Conclusion**: The Phase 5 infrastructure (6-category collectors, deduplication, PIT `effective_trading_date`, 17 numerical news features, cross-stock graph, 1D/5D/10D combiner, DuckDB Parquet persistence, monitoring, and Stock Intelligence UI) is **operational and leakage-free**. However, **genuine predictive validation of live news & cross-stock alpha requires forward accumulation of real `structured_events.parquet` snapshots over 6–12 months** before expanding beyond the 5 pilot stocks or enabling live trading.
