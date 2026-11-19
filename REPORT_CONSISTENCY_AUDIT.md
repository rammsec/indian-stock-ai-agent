# Report Consistency Audit

## 1. Model Version Drift (`5.0.0-pilot` vs `5.3.0-pilot`)
| Artifact / Code Location | Reported `model_version` | Line / Field |
|---|---|---|
| `backend/news_intel/pipeline.py` | `5.0.0-pilot` | L177 (`run_phase5_intelligence_pipeline`) |
| `backend/models.py` (`Phase5IntelligencePayload`) | `5.0.0-pilot` | L766 |
| `data/predictions/phase5_intelligence.json` | `5.0.0-pilot` | L3 |
| `docs/data/phase5_intelligence.json` | `5.0.0-pilot` | L3 |
| `backend/news_intel/forward_validation.py` | `5.3.0-pilot` | L121, L504 |
| `backend/models.py` (`ForwardPredictionSnapshot`, `ForwardHorizonEvaluation`, `ForwardValidationPayload`) | `5.3.0-pilot` | L801, L878, L999 |
| `data/predictions/forward_validation.json` | `5.3.0-pilot` | L3 |
| `docs/data/forward_validation.json` | `5.3.0-pilot` | L3 |
| `data/predictions/forward_predictions.parquet` | `5.3.0-pilot` | 65 / 65 rows |

## 2. Monitoring & Forward Evaluation Count Consistency
- **`phase5_intelligence.json` (`monitoring`)**: Reports `total_logged_predictions: 90` and `evaluated_outcomes_count: 75` (`drift_status: OK`).
  - **Root cause**: `backend/news_intel/monitoring.py:135` (`build_monitoring_snapshot`) reads `prediction_log.parquet` without deduplicating `(symbol, observation_date)`. On `2026-11-16` alone there are 60 rows (4 runs × 15 symbols), inflating `evaluated_outcomes_count` from 30 unique `(symbol, observation_date)` pairs (15 from `2026-11-16`, 15 from `2026-11-17`) to 75.
- **`forward_validation.json`**: Reports `total_immutable_snapshots: 65`, `canonical_snapshots_count: 30`, `duplicate_snapshots_count: 35`, `matured_outcomes_total: 20` (1D), `pending_outcomes_total: 70` (10 1D + 30 5D + 30 10D).
  - **Audit finding**: Of the 20 canonical `1D` outcomes marked `MATURED`, **10** (runs `2026-11-17T10:25:29+05:30` and `2026-11-18T17:32:47+05:30`) violate `prediction_timestamp <= entry_date 09:15 IST` (`INVALID_LOOKAHEAD`), leaving only **10** genuinely look-ahead-free matured `1D` outcomes. Of the 10 canonical `1D` outcomes marked `PENDING`, **5** (run `2026-11-19T17:28:47+05:30`) will also violate look-ahead once `2026-11-19` EOD bars arrive.

## 3. Benchmark Consistency (`NIFTY_50` vs `NIFTY 50`)
- `forward_validation.json` reports `paper_portfolio.benchmarks.nifty_50_buy_hold.status = "UNAVAILABLE"` and `forward_outcomes.parquet` has `0 / 195` non-null `benchmark_return_pct`.
- **Root cause**: `backend/news_intel/forward_validation.py:344` and `L1137` filter `index_code == "NIFTY 50"` (space), whereas `data/processed/indices_daily.parquet` stores `"NIFTY_50"` (underscore, 495 rows).
