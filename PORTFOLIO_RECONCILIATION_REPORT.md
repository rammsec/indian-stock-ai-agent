# Portfolio Reconciliation Report — Workstream 4 (Paper Portfolio Remediation)

**Module:** [`backend/backtesting/event_ledger_portfolio.py`](backend/backtesting/event_ledger_portfolio.py) (`ledger_version="ledger_v1.0.0"`)
**Regression Suite:** [`tests/test_remediation_paper_portfolio.py`](tests/test_remediation_paper_portfolio.py)
**Freeze Compliance:** [`backend/news_intel/forward_validation.py`](backend/news_intel/forward_validation.py), [`backend/news_intel/combiner.py`](backend/news_intel/combiner.py), and all original `data/predictions/*.parquet` files remain completely unmodified.

---

## Executive Summary

The remediated event-ledger paper portfolio simulator (`ledger_v1.0.0`) replaces ad-hoc snapshot replay with a deterministic, hash-chained immutable event ledger (`SIGNAL_RECEIVED`, `SIGNAL_REJECTED`, `ENTRY_EXECUTED`, `POSITION_MTM`, `POSITION_EXITED`) that enforces strict prediction-before-entry chronology, eliminates `[-60:]` persistence truncation, reconciles `cash + MTM_value == total_equity` and accrued CNC delivery costs at every event step, and aligns `NIFTY_50`, `naive_momentum`, and `cash_no_trade` benchmarks on identical execution dates.

---

## Issue 1: Look-Ahead Execution & Same-Day Multi-Session Signal Ambiguity

- **Original finding:**
  - The sole executed trade in production (`10D` `BIRLACABLE`, `prediction_id="fwd_BIRLACABLE_2026-11-16_20261117T102529"`) had `prediction_timestamp="2026-11-17T10:25:29.884689+05:30"` (`nse_session="INTRADAY"`), which occurred **1h 10m 29s after** its simulated entry at `2026-11-17T09:15:00+05:30` (`open=300.25`), while the valid post-close snapshot for `2026-11-16` (`fwd_2026-11-16_BIRLACABLE`, `19:23:27 IST`) emitted `HOLD`.
  - Across `65` persisted rows in `data/predictions/forward_predictions.parquet` (`30` canonical after `is_duplicate_run` filtering), `15/30` canonical snapshots have `prediction_timestamp >= scheduled_entry_open (09:15 IST)`: `5` on `2026-11-16` (`INTRADAY`, `2026-11-17T10:25:29+05:30`), `5` on `2026-11-17` (`POST_MARKET`, `2026-11-18T17:32:47+05:30`), and `5` on `2026-11-18` (`POST_MARKET`, `2026-11-19T17:28:47+05:30`).
- **Root cause:**
  - [`backend/news_intel/forward_validation.py`](backend/news_intel/forward_validation.py#L631-L993) (`_simulate_strategy_timeline`) scheduled `entry_dt` solely from `observation_date` (`b_idx + 1`) without checking `prediction_timestamp < entry_open (09:15 IST)`, and sorted only by `["observation_date", "symbol"]` (`L671`), leaving multiple `nse_session` snapshots per `(symbol, observation_date)` competing nondeterministically.
- **Affected files & records:**
  - `backend/news_intel/forward_validation.py` (`_simulate_strategy_timeline`, `deduplicate_forward_predictions`).
  - `data/predictions/forward_predictions.parquet`: `65` total rows (`35` `is_duplicate_run=True`, `30` canonical, `15` look-ahead canonical rows including `fwd_BIRLACABLE_2026-11-16_20261117T102529`).
  - `data/predictions/paper_trades.parquet`: `1` look-ahead `ENTRY_EXECUTED` row (`BIRLACABLE` `10D` on `2026-11-17`).
- **Remediation:**
  - Implemented [`_classify_signals`](backend/backtesting/event_ledger_portfolio.py#L343-L450) in [`backend/backtesting/event_ledger_portfolio.py`](backend/backtesting/event_ledger_portfolio.py):
    1. Rejects all `is_duplicate_run == True` snapshots and exact duplicate `prediction_id` rows with `SIGNAL_REJECTED` (`reason="REJECTED_DUPLICATE_RUN"`).
    2. Rejects any snapshot where `prediction_timestamp` is missing/unparseable or `prediction_timestamp >= entry_open (09:15:00+05:30)` with `SIGNAL_REJECTED` (`reason="REJECTED_LOOKAHEAD"`).
    3. Deduplicates multiple valid canonical sessions on the same `(symbol, observation_date)` deterministically by earliest valid `prediction_timestamp`, rejecting superseded runs with `SIGNAL_REJECTED` (`reason="REJECTED_SAME_DAY_DUPLICATE"`).
- **Tests:**
  - `test_lookahead_and_duplicate_run_and_same_day_rejection` and `test_persisted_repo_data_reconciliation_and_replay` in [`tests/test_remediation_paper_portfolio.py`](tests/test_remediation_paper_portfolio.py).
- **Evidence of successful execution:**
  - Running `run_event_ledger_from_data_dir("data", strict_canonical=True)` classifies all `65` snapshots cleanly per horizon: `35` `REJECTED_DUPLICATE_RUN`, `15` `REJECTED_LOOKAHEAD` (including `fwd_BIRLACABLE_2026-11-16_20261117T102529`), `10` `REJECTED_NON_BUY_SIGNAL`, `5` `pending_next_session_count`, and `0` look-ahead entries executed (`ending_equity_inr = ₹1,000,000.00`).
  - Running with `strict_canonical=False` reproduces the historical `10D` `BIRLACABLE` entry (`₹1,000,782.68` MTM equity) for reconciliation verification.
- **Remaining limitations:**
  - `forward_validation.py` remains frozen per programme rules, so live `forward_validation.json` continues to reflect legacy behaviour until downstream integration adopts `event_ledger_portfolio.py`.
  - Under `strict_canonical=True`, `0` model `BUY` trades are active across the `3`-day forward sample (`2026-11-16`..`2026-11-18`) because expected net returns from the frozen combiner are negative.

---

## Issue 2: Event Ledger Truncation (`[-60:]`) & Non-Reproducible Persistence

- **Original finding:**
  - `data/predictions/paper_trades.parquet` contained `219` rows polluted by `84` rows referencing `is_duplicate_run=True` snapshots and `15` legacy `action="REJECTED"` rows (`event_date=NaN`), while `build_forward_validation_payload` (`forward_validation.py:1318`) persisted only `paper_report.recent_trades` (`all_trades[-60:]` at `L1174`), dropping **`91` of `151`** canonical events (`50/50` `1D` events and `41/50` `5D` events) and emitting no per-session `POSITION_MTM` events.
- **Root cause:**
  - Persistence relied on a UI-sized `recent_trades = all_trades[-60:]` slice merged via `upsert_paper_trades` without an immutable sequence, hash chain, or full replay reconstruction capability.
- **Affected files & records:**
  - `backend/news_intel/forward_validation.py` (`L1174`, `L1318`).
  - `backend/storage/duckdb_store.py` (`upsert_paper_trades`, `L593-L602`).
  - `data/predictions/paper_trades.parquet` (`219` rows; `91` canonical events missing).
- **Remediation:**
  - Designed [`LedgerEvent`](backend/backtesting/event_ledger_portfolio.py#L75-L120) (`ledger_version="ledger_v1.0.0"`) covering all `5` lifecycle actions (`SIGNAL_RECEIVED`, `SIGNAL_REJECTED`, `ENTRY_EXECUTED`, `POSITION_MTM`, `POSITION_EXITED`) with zero truncation.
  - Added SHA-256 hash chaining (`compute_event_hash`, `prev_event_hash` → `event_hash`, monotonic `event_seq`) and Parquet round-trip persistence (`events_to_dataframe`, `dataframe_to_events`, `persist_ledger_parquet`).
  - Implemented [`replay_strategy_from_events`](backend/backtesting/event_ledger_portfolio.py#L1147-L1343) and [`replay_portfolio_from_events`](backend/backtesting/event_ledger_portfolio.py#L1346-L1354) to reconstruct cash, positions, MTM, accrued costs, exposure, drawdowns, and daily equity curves strictly from stored immutable events, rejecting any tampered or broken chain.
- **Tests:**
  - `test_ledger_replay_reproducibility_zero_truncation_and_tamper_detection` in [`tests/test_remediation_paper_portfolio.py`](tests/test_remediation_paper_portfolio.py).
- **Evidence of successful execution:**
  - On persisted `data/`, `strict_canonical=True` emits `375` hash-chained events (`125` per horizon: `1D`, `5D`, `10D`) and `strict_canonical=False` emits `362` events (`120` `1D`, `120` `5D`, `122` `10D`) with `zero_truncation_events_count == len(all_events)` and `replay_exact_match == True`.
  - In synthetic multi-day tests (> 60 events written to Parquet and reloaded), `replay_portfolio_from_events` reproduces identical cash, equity, MTM, accrued exit costs, drawdowns, and equity curves, while mutating `cash_after_inr` raises `ValueError` (`Tampered event_hash`).
- **Remaining limitations:**
  - `data/predictions/paper_trades.parquet` was intentionally left untouched to obey the read-only freeze on `data/predictions/*.parquet`. Full ledger persistence uses `persist_ledger_parquet(...)` targeting a dedicated ledger Parquet path when integrated.

---

## Issue 3: Cash, Invested Cost vs. MTM Exposure & Accrued CNC Exit Cost Accounting

- **Original finding:**
  - `forward_validation.py` deducted CNC delivery round-trip costs (`0.347%` default, `0.5154%` for `BIRLACABLE`) only upon `POSITION_EXITED`, omitting accrued exit costs on open positions (`BIRLACABLE` MTM equity `₹1,000,782.68` vs. true net-of-cost liquidation value `₹1,000,267.28`, overstating open return by `+₹515.40` / `+0.05%`).
  - `forward_validation.py:857, 958` also divided `invested_cost` by `mtm_equity`, inflating `peak_exposure_pct` to `10.02%` when `BIRLACABLE` fell `-1.98%` on `2026-11-17` (actual MTM exposure `9.82%`), and silently stranded positions if `close[exit_idx]` was missing (`L888`).
- **Root cause:**
  - Absence of explicit `mtm_value_after_inr`, `accrued_exit_cost_after_inr`, `net_liquidation_equity_after_inr`, and step-level invariant checks in event records, plus missing close-fallback logic on exit bars.
- **Affected files & records:**
  - `backend/news_intel/forward_validation.py` (`L857`, `L888`, `L894-L897`, `L958`).
  - `data/predictions/forward_validation.json` (`paper_portfolio.strategies.10D`).
- **Remediation:**
  - Every `LedgerEvent` and EOD curve point in `event_ledger_portfolio.py` records and verifies `4` accounting invariants via [`_reconciliation_error`](backend/backtesting/event_ledger_portfolio.py#L453-L469):
    1. `cash_after_inr + mtm_value_after_inr == total_equity_after_inr`
    2. `invested_cost_after_inr + unrealized_pnl_total_after_inr == mtm_value_after_inr`
    3. `starting_capital + realized_pnl_total + unrealized_pnl_total == total_equity_after_inr`
    4. `total_equity_after_inr - accrued_exit_cost_after_inr == net_liquidation_equity_after_inr`
  - Separates `peak_exposure_cost_pct` (`10.02%` in `10D` replay) from `peak_exposure_mtm_pct` (`10.07%` in `10D` replay) and adds `FALLBACK_NEXT_VALID_CLOSE_EXIT` if a scheduled exit bar has a missing/non-positive close.
- **Tests:**
  - `test_portfolio_cash_mtm_and_accrued_cost_reconciliation` in [`tests/test_remediation_paper_portfolio.py`](tests/test_remediation_paper_portfolio.py).
- **Evidence of successful execution:**
  - Across both synthetic and persisted runs, `max_reconciliation_error_inr == 0.0` (`all_invariants_ok == True`).
  - In `10D` replay on persisted `data/`:
    - Entry `2026-11-17` (`open=300.25`): `cash=₹900,000.00`, `mtm=₹100,000.00`, `total_equity=₹1,000,000.00`, `accrued_exit_cost=₹515.40`, `net_liquidation_equity=₹999,484.60`.
    - EOD `2026-11-17` (`close=294.30`): `cash=₹900,000.00`, `mtm=₹98,018.32`, `total_equity=₹998,018.32`, `net_liquidation_equity=₹997,502.92`.
    - EOD `2026-11-18` (`close=302.60`): `cash=₹900,000.00`, `mtm=₹100,782.68`, `total_equity=₹1,000,782.68` (`+0.08%`), `net_liquidation_equity=₹1,000,267.28` (`+0.03%`).
- **Remaining limitations:**
  - The simulator uses daily OHLC bars (`open[t+1]` entry, `close[t+H]` exit) without intraday stop-loss/take-profit triggers, matching the Phase 5.4 horizon evaluation convention rather than [`backend/backtesting/execution_simulator.py`](backend/backtesting/execution_simulator.py).

---

## Issue 4: Broken `NIFTY_50` Lookup & Unaligned Benchmarks

- **Original finding:**
  - `forward_validation.py:344, 1137` queried `index_code == "NIFTY 50"`, whereas `data/processed/indices_daily.parquet` stores `"NIFTY_50"`. As a result, `nifty_50_buy_hold` was falsely reported as `UNAVAILABLE` (`null`), and `naive_momentum` at root level was hardcoded to `1D` (`INSUFFICIENT_DATA`) while `5D` and `10D` were active (`+0.12%`).
  - `nifty_50_buy_hold` entered at `close[first_signal_date]` instead of `open[first_entry_date]` (`t+1`), misaligning entry timing vs. portfolio trades.
- **Root cause:**
  - Unnormalized string comparison (`"NIFTY 50"` vs. `"NIFTY_50"`) and misaligned benchmark start date / entry price convention.
- **Affected files & records:**
  - `backend/news_intel/forward_validation.py` (`L344`, `L1126`, `L1137-L1154`).
  - `data/processed/indices_daily.parquet` (`index_code="NIFTY_50"`).
  - `data/predictions/forward_validation.json` (`paper_portfolio.benchmarks.nifty_50_buy_hold`).
- **Remediation:**
  - Added [`normalize_index_code`](backend/backtesting/event_ledger_portfolio.py#L202-L204) mapping both `"NIFTY 50"` and `"NIFTY_50"` to `"NIFTY_50"`.
  - Integrated `nifty_50_buy_hold` (entering at `open[first_entry_date]` with `80%` exposure and `5 bps` cost drag), per-horizon `nifty_50_matched_windows` (replicating exact trade entry/exit dates and lot sizes), horizon-matched `naive_momentum_{1d,5d,10d}` run through the identical event-ledger simulator, and date-aligned `cash_no_trade`.
- **Tests:**
  - `test_aligned_benchmarks_and_index_normalization` and `test_persisted_repo_data_reconciliation_and_replay` in [`tests/test_remediation_paper_portfolio.py`](tests/test_remediation_paper_portfolio.py).
- **Evidence of successful execution:**
  - On persisted `data/` (`2026-11-17` `open=25,277.30` to `2026-11-18` `close=25,227.95`):
    - `nifty_50_buy_hold`: `status="ACTIVE"`, `index_code_matched="NIFTY_50"`, `ending_equity_inr=₹998,038.91`, `net_return_pct=-0.20%`, `max_drawdown_pct=-0.50%`.
    - `nifty_50_matched_windows["10D"]` (replay mode, matching `BIRLACABLE` ₹1L lot): `ending_equity_inr=₹999,804.77` (`-0.02%`).
    - `naive_momentum_1d`: `INSUFFICIENT_DATA` (`0` entries); `naive_momentum_5d` & `naive_momentum_10d`: `ACTIVE`, `2` open positions (`ITC`, `RELIANCE`), `ending_equity_inr=₹1,001,238.62` (`+0.12%` MTM, `net_liquidation_equity_inr=₹1,000,527.52` / `+0.05%` after `₹711.10` accrued CNC exit cost).
    - `cash_no_trade`: `ending_equity_inr=₹1,000,000.00` (`0.00%`) aligned over `2026-11-17`..`2026-11-18`.
- **Remaining limitations:**
  - Only `2` post-entry NSE sessions (`2026-11-17`, `2026-11-18`) exist in `equities_daily.parquet` and `indices_daily.parquet` after `2026-11-16`, so `5D` and `10D` positions remain unmatured (`0` closed trades) until additional sessions accumulate.

---

## Verification Summary

- Command: `/home/antigravity/.venv/bin/pytest tests/test_remediation_paper_portfolio.py -v`
- Result: **5 passed in 2.23s** (`test_lookahead_and_duplicate_run_and_same_day_rejection`, `test_portfolio_cash_mtm_and_accrued_cost_reconciliation`, `test_ledger_replay_reproducibility_zero_truncation_and_tamper_detection`, `test_aligned_benchmarks_and_index_normalization`, `test_persisted_repo_data_reconciliation_and_replay`).
