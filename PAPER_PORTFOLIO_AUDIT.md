# Paper Portfolio Audit — BharatAlpha AI (Phase 5.3 / 5.4)

**Audit Scope:** Read-only verification of [`backend/news_intel/forward_validation.py`](backend/news_intel/forward_validation.py), `data/predictions/paper_trades.parquet`, `data/predictions/forward_predictions.parquet`, `data/predictions/forward_outcomes.parquet`, and `docs/data/forward_validation.json`.
**Audit Engine:** [`backend/backtesting/portfolio_audit.py`](backend/backtesting/portfolio_audit.py) (`run_paper_portfolio_audit`, `run_audit_from_data_dir`).
**Freeze Compliance:** Frozen prediction/signal generation and `forward_validation.py` remain untouched.

---

## 1. Executive Summary

1. **Why Trades Are Sparse (Not a Simulator Bug):** Across `65` persisted rows in `forward_predictions.parquet` (`30` canonical after `is_duplicate_run` filtering; `15` stock-days across 5 pilot symbols on `2026-11-16`, `2026-11-17`, and `2026-11-18`), the frozen signal combiner (`build_multi_horizon_outlooks` in `backend/news_intel/combiner.py:156-165`) emitted only **1 `BUY` signal** (`BIRLACABLE` 10D on `2026-11-16`). Expected net returns are negative on `30/30` 1D snapshots, `30/30` 5D snapshots, and `29/30` 10D snapshots.
2. **Look-Ahead Finding on the Single Executed 10D Trade (`BIRLACABLE`):** The sole `BUY` snapshot (`fwd_BIRLACABLE_2026-11-16_20261117T102529`, `nse_session="INTRADAY"`) has `prediction_timestamp = 2026-11-17T10:25:29.884689+05:30`, which is **1h 10m 29s after** its simulated execution at `2026-11-17T09:15:00+05:30` (`open=300.25`). Meanwhile, the genuine post-close snapshot for the same stock/date (`fwd_2026-11-16_BIRLACABLE`, `prediction_timestamp = 2026-11-16T19:23:27.388296+05:30`) rated 10D as **`HOLD`** (`prob_10d_up=0.5542`, `expected_10d_return_pct=-0.023`). Under `strict_canonical` look-ahead enforcement (`prediction_timestamp < entry_open`), `10/30` canonical snapshots (`5` on `2026-11-16` INTRADAY at `2026-11-17T10:25:29` and `5` on `2026-11-17` POST_MARKET at `2026-11-18T17:32:47`) are rejected as `REJECTED_LOOKAHEAD`, leaving **0 active trades** across 1D/5D/10D.
3. **`NIFTY_50` Benchmark Broken by Index Code Mismatch:** `forward_validation.py:344` and `1137` filter `index_code == "NIFTY 50"`, whereas `data/processed/indices_daily.parquet` stores `"NIFTY_50"` (`backend/config.py:53`). Consequently, `nifty_50_buy_hold` is falsely reported as `UNAVAILABLE` and all `195` rows in `forward_outcomes.parquet` have `benchmark_return_pct = None`. When aligned to `"NIFTY_50"` with next-open entry (`2026-11-17` `open=25,277.30` → `2026-11-18` `close=25,227.95`, 80% exposure, 5 bps drag), Nifty 50 returned **`-0.20%`** (`₹998,038.91`, max DD `-0.50%`).
4. **`paper_trades.parquet` Persistence Gaps:** `build_forward_validation_payload` (`forward_validation.py:1318`) only persists `paper_report.recent_trades` (`all_trades[-60:]` at `L1174`), dropping **`91` of `151`** simulated events (`50/50` 1D events and `41/50` 5D events), while `paper_trades.parquet` (`219` rows) retains `84` rows tied to `is_duplicate_run=True` snapshots and `15` legacy `action="REJECTED"` rows (`event_date=NaN`).

---

## 2. Snapshot Hygiene & Signal Rejection Breakdown

| Metric | Production Replay (`forward_validation.py`) | Strict Canonical (`portfolio_audit.py`) | Notes |
|---|---:|---:|---|
| Total snapshots in Parquet | 65 | 65 | 5 pilot symbols × 2026-11-16..18 |
| Duplicate runs (`is_duplicate_run=True`) | 35 | 35 | Excluded before simulation |
| Canonical snapshots evaluated | 30 | 30 | 2 sessions/day × 5 stocks × 3 days |
| Look-ahead snapshots (`pred_ts >= entry_open`) | 10 (not flagged) | 10 (`REJECTED_LOOKAHEAD`) | `2026-11-16` INTRADAY (5) + `2026-11-17` POST_MARKET (5) |
| **1D** Received / Executed / Rejected / Pending `t+1` | 30 / 0 / 20 / 10 | 30 / 0 / 20 / 10 | Replay: `20 NON_BUY`; Strict: `10 NON_BUY + 10 LOOKAHEAD` |
| **5D** Received / Executed / Rejected / Pending `t+1` | 30 / 0 / 20 / 10 | 30 / 0 / 20 / 10 | Replay: `20 NON_BUY`; Strict: `10 NON_BUY + 10 LOOKAHEAD` |
| **10D** Received / Executed / Rejected / Pending `t+1` | 30 / 1 / 19 / 10 | 30 / 0 / 20 / 10 | Replay executes `BIRLACABLE`; Strict rejects it (`LOOKAHEAD`) |

### Rejection Classification Taxonomy (`backend/backtesting/portfolio_audit.py`)

| Rejection Code | Production Equivalent (`forward_validation.py`) | Trigger Condition |
|---|---|---|
| `REJECTED_NON_BUY_SIGNAL` | `REJECTED_NON_BUY_SIGNAL_{rec}` (`L723`) | Decision is `NO TRADE`, `WATCH`, or `HOLD` |
| `REJECTED_UNAVAILABLE_MODEL` | `REJECTED_UNAVAILABLE_MODEL` (`L704`) | Decision is `UNAVAILABLE` |
| `REJECTED_LOOKAHEAD` | *(Not implemented)* | `prediction_timestamp >= entry_date 09:15 IST` |
| `REJECTED_SAME_DAY_DUPLICATE` | *(Not implemented)* | Second valid canonical session on same `(symbol, observation_date)` |
| `REJECTED_OVERLAPPING_POSITION` | `REJECTED_OVERLAPPING_POSITION` (`L742`) | `symbol in open_positions` |
| `REJECTED_INSUFFICIENT_CASH` | `REJECTED_INSUFFICIENT_CASH` (`L762`) | `cash < 1,000.0` |
| `REJECTED_MAX_EXPOSURE` | `REJECTED_MAX_EXPOSURE_80PCT` (`L781`) | `max(0, equity * 0.80 - invested_cost) < 1,000.0` |
| `REJECTED_MIN_LOT_CAPITAL` | `REJECTED_MIN_LOT_CAPITAL` (`L801`) | `min(equity * 0.10, headroom, cash) < 1,000.0` |
| `REJECTED_INVALID_ENTRY_PRICE` | `REJECTED_INVALID_ENTRY_PRICE` (`L823`) | `open[t+1]` is `NaN` or `<= 0` |
| `REJECTED_MISSING_OBSERVATION_BAR` | *(Silently omitted at `L983-991`)* | Symbol has no rows in `equities_daily` |

---

## 3. Accounting, Horizon Exits, Costs & Persistence Verification

### Cash & MTM Conservation (`10D` Replay — `BIRLACABLE`)
- **Starting Capital:** `₹1,000,000.00`
- **Entry (`2026-11-17` open):** `entry_price = 300.25`, `allocated_capital_inr = ₹100,000.00` (`10.0%`), `cash = ₹900,000.00`, `equity = ₹1,000,000.00`.
- **EOD `2026-11-17` MTM:** `close = 294.30` → `unrealized_pnl = 100,000 * (294.30 / 300.25 - 1) = -₹1,981.68`, `equity = ₹998,018.32`, drawdown `-0.20%`.
- **EOD `2026-11-18` MTM:** `close = 302.60` → `unrealized_pnl = 100,000 * (302.60 / 300.25 - 1) = +₹782.68`, `equity = ₹1,000,782.68` (`+0.08%`).
- **Accounting Invariant:** `cash + invested_cost + unrealized_pnl == starting_capital + realized_pnl + unrealized_pnl == ending_capital` holds with **`0.0` INR** error.
- **Unrealized Cost Gap:** CNC round-trip cost (`0.5154%` = `₹515.40` from snapshot `metadata.roundtrip_delivery_cost_pct`) is deducted only at `POSITION_EXITED`. Open MTM reports `total_cost_inr = ₹0.00` and `net_return_pct = +0.08%`, which overstates liquidation value by `₹515.40` (liquidation net `+₹267.28` / `+0.03%`).

### Horizon Exit Indexing Verification
- `forward_validation.py:835` (`exit_idx = _base_idx + hold_n`) with `_entry_idx = _base_idx + 1` (`L990`) matches `evaluate_forward_outcomes` (`L417`, `exit_idx = entry_idx + (hold_n - 1)`):
  - `1D` (`hold_n=1`): enters `t+1` open, exits `t+1` close.
  - `5D` (`hold_n=5`): enters `t+1` open, exits `t+5` close.
  - `10D` (`hold_n=10`): enters `t+1` open, exits `t+10` close.
- **Edge Case Note:** If `close[exit_idx]` is missing or `<= 0` when `sess_dt >= exit_date`, `forward_validation.py:888` silently leaves the position open forever rather than marking `INVALID` or exiting at the next valid close.

### `paper_trades.parquet` Persistence Audit (`219` rows)
- `action_counts_in_parquet`: `SIGNAL_RECEIVED=159`, `SIGNAL_REJECTED=42`, `REJECTED=15` (legacy), `POSITION_OPEN=2` (`2026-11-17` and `2026-11-18` snapshots of the same open position), `ENTRY_EXECUTED=1`.
- `duplicate_run_rows_in_parquet`: `84` rows reference `is_duplicate_run=True` snapshots from prior runs before deduplication.
- `dropped_by_truncation_count`: `91` of `151` canonical events are never written to `paper_trades.parquet` because `build_forward_validation_payload` (`L1318`) passes `paper_report.recent_trades` (`all_trades[-60:]` at `L1174`), discarding all `50` `1D` events and `41` of `50` `5D` events.

---

## 4. Fair Benchmark Alignment (`2026-11-16` to `2026-11-18`)

| Benchmark | `forward_validation.json` | Audited (`portfolio_audit.py`) | Diagnostic Finding |
|---|---:|---:|---|
| `naive_momentum` (root alias) | `null` (`INSUFFICIENT_DATA`) | — | Hardcoded alias to `1D` (`L1126`), masking active `5D`/`10D` |
| `naive_momentum_1d` | `null` (`INSUFFICIENT_DATA`) | `null` (`INSUFFICIENT_DATA`) | Max `benchmark_prob_1d_up` is `0.5294 < 0.535` |
| `naive_momentum_5d` | `+0.12%` (`₹1,001,238.62`) | `+0.12%` (`₹1,001,238.62`) | 2 open on `2026-11-18`: `ITC` (`+1,749.57`), `RELIANCE` (`-510.95`) |
| `naive_momentum_10d` | `+0.12%` (`₹1,001,238.62`) | `+0.12%` (`₹1,001,238.62`) | Same 2 open entries (`20%` exposure, `0.0%` DD) |
| `nifty_50_buy_hold` | `null` (`UNAVAILABLE`) | **`-0.20%` (`₹998,038.91`)** | Fixed `"NIFTY_50"` lookup + next-open `2026-11-17` (`25,277.30 → 25,227.95`), DD `-0.50%` |
| `cash_no_trade` | `0.00%` (`₹1,000,000.00`) | `0.00%` (`₹1,000,000.00`) | Risk-free baseline |
