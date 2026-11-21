# BharatAlpha — Research Integrity & Untouched Holdout Protocol (Phase 11.2)

## 1. Holdout Date Lifecycle
Every trading session date in `data/lab/sealed_holdout_registry.json` has an explicit lifecycle state:

1. **`AVAILABLE`**: Eligible for training, feature discovery, or validation.
2. **`SEALED`**: Reserved for final evaluation of a specific research cycle; blocked from tuning or parameter selection.
3. **`EXPOSED`**: Final evaluation results on this date have been accessed and recorded. Cannot be reused as an independent unseen test.
4. **`RETIRED`**: Permanently retired from independent out-of-sample validation.

## 2. How Future Untouched Test Data Is Collected
- **Daily 5m Archiver** (`.github/workflows/daily-5m-archive.yml`): Runs Monday–Friday at `15:50 IST` (`10:20 UTC`), appending new NSE trading sessions to `data/intraday/{SYM}_5m.parquet` without overwriting complete historical sessions.
- **Forward-Only Untouched Holdout Accumulation**:
  - All historical dates up to `2026-11-20` that were evaluated in `CYCLE-002` and `CYCLE-003` are permanently marked `EXPOSED` in `data/lab/sealed_holdout_registry.json`.
  - Future independent out-of-sample qualification requires **newly archived NSE sessions after `2026-11-20`** that remain `AVAILABLE` (never touched during training/tuning), are transitioned to `SEALED` prior to final evaluation, and are evaluated once.
  - Until at least `20–30` genuinely untouched forward sessions accumulate, any cycle reusing historical `EXPOSED` dates must set `independent_holdout_valid: false` and status `EXPOSED_HOLDOUT_REUSED`.

## 3. Benchmark Separation Policy
- `BENCHMARK-CASH-NO-TRADE` (`0.00%`) and `5-Stock Equal-Weight Buy & Hold` are **reference benchmarks**, never active trading strategies.
- Cash preservation is never counted toward `qualified_strategies` or `best_active_strategy`.
