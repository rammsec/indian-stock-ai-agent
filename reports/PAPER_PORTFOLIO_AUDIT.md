# BharatAlpha AI — Paper Research Portfolio Audit

**Branch:** `feature/data-integrity-audit`  
**Base Commit SHA (`main`):** `8f328b8cdb8f7a1e508c1c47a648849945873013`  
**Audit Date:** 2026-11-18  
**Inspected Code & Data:** `backend/news_intel/forward_validation.py` (`_simulate_strategy_timeline` L631–1042, `run_paper_research_portfolio` L1045–1175), `backend/news_intel/combiner.py` (`build_multi_horizon_outlooks` L110–185), `backend/news_intel/pipeline.py` (L114), `data/predictions/paper_trades.parquet` (179 rows), `docs/data/forward_validation.json`.

---

## 1. Published Portfolio State Overview

| Horizon | `signals_received` | `entries_executed` | `rejected_signals` | `open_positions_count` | `starting_capital_inr` | `cash_inr` | `invested_capital_inr` | `unrealized_pnl_inr` | `ending_capital_inr` | `net_return_pct` | `status` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **1D** | 20 | 0 | 10 | 0 | ₹10,00,000.00 | ₹10,00,000.00 | ₹0.00 | ₹0.00 | ₹10,00,000.00 | `0.00%` | `INSUFFICIENT_DATA` |
| **5D** | 20 | 0 | 10 | 0 | ₹10,00,000.00 | ₹10,00,000.00 | ₹0.00 | ₹0.00 | ₹10,00,000.00 | `0.00%` | `INSUFFICIENT_DATA` |
| **10D** | 20 | 1 | 9 | 1 | ₹10,00,000.00 | ₹9,00,000.00 | ₹1,00,000.00 | `-₹1,981.68` | ₹9,98,018.32 | `-0.20%` | `ACTIVE` |

---

## 2. Answers to the 10 Portfolio Investigation Questions

### Q1–Q3. Why did 1D and 5D generate no entries? Were signals rejected due to confidence thresholds or `HOLD`/`WATCH`/`NO TRADE` recommendations?
- **Exact Rule (`backend/news_intel/forward_validation.py:712`):** `_simulate_strategy_timeline` only executes an entry when `rec == "BUY"` (`recommendation_1d` for 1D, `recommendation_5d` for 5D, `recommendation_10d` for 10D). Any non-`BUY` signal is rejected at market open with `reason = f"REJECTED_NON_BUY_SIGNAL_{rec}"`.
- **Decision Thresholds (`backend/news_intel/combiner.py:18-41, 156-166`):**
  - `BUY` requires **all four** conditions simultaneously:
    1. `trade_eligible == True` (where `pipeline.py:114` sets `trade_eligible = (quant_pred["recommended_action"] != "NO TRADE")`; only `TCS` has `recommended_action == "NO TRADE"`).
    2. `regime_label != "High Volatility"` (current regime is `"Weak Bull"`).
    3. `comb_prob >= buy_prob` (`0.54` for 1D and 5D; `0.55` for 10D).
    4. `expected_net_return_pct > 0.05` **and** `combined_signal == "BULLISH"` (`comb_score >= 0.06`).
- **Why all 20 canonical 1D signals were non-`BUY`:**
  - Across all 20 canonical snapshots, `prob_1d_up` ranges from `0.4372` (`BIRLACABLE`) to `0.5232` (`RELIANCE`), below the `0.54` 1D `buy_prob` threshold.
  - Consequently, the 20 canonical 1D recommendations are `14 × NO TRADE` and `6 × WATCH` (`RELIANCE` ×4, `ITC` ×2).
  - For the 10 `obs=2026-11-16` snapshots evaluated at the `2026-11-17` open, all 10 were rejected (`8 × REJECTED_NON_BUY_SIGNAL_NO TRADE`, `2 × REJECTED_NON_BUY_SIGNAL_WATCH`). The 10 `obs=2026-11-17` snapshots have no `2026-11-18` bar in `equities_daily.parquet` yet, so they remain unscheduled after `SIGNAL_RECEIVED`.
- **Why all 20 canonical 5D signals were non-`BUY`:**
  - Across all 20 canonical snapshots, `prob_5d_up` ranges from `0.4266` (`RELIANCE`) to `0.5368` (`BIRLACABLE`), below the `0.54` 5D `buy_prob` threshold, and `expected_5d_return_pct` is negative (`-0.109%` to `-0.447%`).
  - The 20 canonical 5D recommendations are `16 × NO TRADE` and `4 × WATCH` (`BIRLACABLE` ×4). At the `2026-11-17` open, the 10 `obs=2026-11-16` signals were rejected (`8 × REJECTED_NON_BUY_SIGNAL_NO TRADE`, `2 × REJECTED_NON_BUY_SIGNAL_WATCH`).

### Q4. Were there price-data or execution eligibility issues?
- For `observation_date = 2026-11-16`, `equities_daily.parquet` has valid `2026-11-17` OHLCV bars for all 5 pilot stocks (`BIRLACABLE`, `ITC`, `RELIANCE`, `TATAELXSI`, `TCS`), so no `2026-11-16` signals were dropped for missing price data.
- For `observation_date = 2026-11-17`, `equities_daily.parquet` max date is `2026-11-17` (`b_idx + 1 < len(grp)` is `False` at `forward_validation.py:988`), so all 10 `obs=2026-11-17` signals logged `SIGNAL_RECEIVED` and are awaiting the `2026-11-18` NSE session bar.

### Q5. Why did 10D generate exactly one entry?
- On `observation_date = 2026-11-16`, `BIRLACABLE` had two canonical snapshots:
  1. **`fwd_2026-11-16_BIRLACABLE`** (`POST_MARKET`, `2026-11-16T19:23:27.388296+05:30` — genuinely prospective):
     - `news_score = 0.0`, `prob_10d_up = 0.5542` (>= `0.55`), but `expected_10d_return_pct = -0.023%` (<= `0.05%`), yielding **`recommendation_10d = "HOLD"`**.
     - Rejected at `2026-11-17` open with `REJECTED_NON_BUY_SIGNAL_HOLD`.
  2. **`fwd_BIRLACABLE_2026-11-16_20261117T102529`** (`INTRADAY`, `2026-11-17T10:25:29.884689+05:30` — look-ahead contaminated):
     - Collected positive `BIRLACABLE` articles on the morning of `2026-11-17` (`10:19:13 IST`), raising `news_score` from `0.0` to `0.1850`, `prob_10d_up` to `0.5593` (>= `0.55`), `expected_10d_return_pct` to `+0.058%` (> `0.05%`), and `comb_score` above `0.06` (`BULLISH`), flipping **`recommendation_10d = "BUY"`**.
     - Executed at `2026-11-17` open (`ENTRY_EXECUTED`, `BUY_SIGNAL_EXECUTED_AT_NEXT_OPEN`).

### Q6. Was its entry made on the correct next eligible trading session?
- **Calendar Date:** Yes (`signal_date = 2026-11-16` → next NSE trading session `entry_date = 2026-11-17` at `open = ₹300.25`).
- **Chronological Look-Ahead Violation (`PORT_LOOKAHEAD_CONTAMINATED_ENTRY`):** **No.** Because `_simulate_strategy_timeline` keys signals only by `observation_date` (`2026-11-16`) without verifying `prediction_timestamp < entry_open`, the `INTRADAY` snapshot generated at `2026-11-17 10:25:29 IST` was allowed to "enter" retroactively at `2026-11-17 09:15:00 IST` (`70 minutes` earlier), after `BIRLACABLE` had already gapped down from its `2026-11-16` close (`₹309.00`) to `₹300.25`.

### Q7. Is the current valuation based on a valid latest price?
- **Yes.** In `equities_daily.parquet`, the latest bar for `BIRLACABLE` is `2026-11-17` with `open = ₹300.25`, `close = ₹294.30`, `adj_close = ₹294.30` (`adj_f = 1.0`).
- Mark-to-market unrealized return:
  $$\text{unrl\_ret} = \left(\frac{294.30}{300.25} - 1\right) \times 100\% = -1.9816819\%$$
- On `allocated_capital_inr = ₹1,00,000.00`, `unrealized_pnl_inr` = `₹1,00,000 × (-0.019816819)` = **`-₹1,981.68`**, matching `forward_validation.json` and `paper_trades.parquet` to the paisa.

### Q8. Are transaction costs included correctly?
- **Exit-Only Deduction Convention (`forward_validation.py:820-831, 938-947`):** Round-trip delivery transaction cost (`roundtrip_delivery_cost_pct = 0.5154%` for `BIRLACABLE`, stored in `pos["cost_pct"]`) is deducted **upon position exit** (`n_ret = g_ret - c_pct`, `c_inr = pos["capital"] * (c_pct / 100.0)`).
- While the 10D position remains open (`POSITION_OPEN`), `total_cost_inr = ₹0.00` and `unrealized_pnl_inr = -₹1,981.68` is stated **gross of estimated round-trip exit/entry delivery costs** (~`₹515.40`).

### Q9. Are cash, invested capital and equity reconciled?
- **Yes (`PORT_CASH_INVESTED_RECONCILIATION` & `PORT_EQUITY_RECONCILIATION` both `PASS`):**
  - **1D & 5D:** `cash (₹10,00,000.00) + invested (₹0.00) + unrealized (₹0.00) = ending_capital (₹10,00,000.00)`.
  - **10D:**
    - `cash_inr (₹9,00,000.00) + invested_capital_inr (₹1,00,000.00) = starting_capital_inr (₹10,00,000.00) + realized_pnl_inr (₹0.00)`
    - `cash_inr (₹9,00,000.00) + invested_capital_inr (₹1,00,000.00) + unrealized_pnl_inr (-₹1,981.68) = ending_capital_inr (₹9,98,018.32)`
    - `net_return_pct = (998,018.32 / 1,000,000.00 - 1) × 100% = -0.1982% → -0.20%`.
    - `peak_exposure_pct = (100,000.00 / 998,018.32) × 100% = 10.02%` (within the `10%` initial allocation rule and `80%` portfolio exposure cap).

### Q10. Are duplicate predictions being counted as multiple signals?
- **Yes (`PORT_DUPLICATE_SIGNAL_COUNTING`):**
  1. **In `forward_validation.json` (`strategies[*].signals_received = 20`):** Because `canonical_snaps_df` has `20` rows (`5` pilot stocks × `2` dates × `2` sessions per date), each stock-date is counted **twice** in `signals_received` (`20` instead of `10`) and twice in `rejected_signals` (`10` on `2026-11-16` instead of `5`).
  2. **In `data/predictions/paper_trades.parquet` (`179` rows):** `build_forward_validation_payload` calls `st.upsert_paper_trades(paper_report.recent_trades)` (`duckdb_store.py:553`), which keys on `(strategy_horizon, prediction_id, symbol, signal_date, action)`. Consequently, `paper_trades.parquet` retains historical `SIGNAL_RECEIVED` and legacy `action="REJECTED"` rows from earlier un-deduplicated runs (`32` rows for 1D, `52` for 5D, `55` for 10D `SIGNAL_RECEIVED`, plus `15` legacy `action="REJECTED"` rows with `event_date = NaN`).

---

## 3. Record-Level Ledger of Active 10D Trade (`BIRLACABLE`)

| Field | `ENTRY_EXECUTED` Row | `POSITION_OPEN` Row |
| :--- | :--- | :--- |
| `strategy_horizon` | `10D` | `10D` |
| `prediction_id` | `fwd_BIRLACABLE_2026-11-16_20261117T102529` | `fwd_BIRLACABLE_2026-11-16_20261117T102529` |
| `symbol` | `BIRLACABLE` | `BIRLACABLE` |
| `signal_date` | `2026-11-16` | `2026-11-16` |
| `event_date` | `2026-11-17` | `2026-11-17` |
| `entry_date` / `entry_price` | `2026-11-17` / `₹300.25` | `2026-11-17` / `₹300.25` |
| `allocated_capital_inr` / `weight_pct` | `₹1,00,000.00` / `10.0%` | `₹1,00,000.00` / `10.0%` |
| `unrealized_pnl_inr` | `₹0.00` (at open) | `-₹1,981.68` (at `₹294.30` close) |
| `cash_after_inr` | `₹9,00,000.00` | `₹9,00,000.00` |
| `invested_after_inr` | `₹1,00,000.00` | `₹1,00,000.00` |
| `equity_after_inr` | `₹10,00,000.00` | `₹9,98,018.32` |
| **Audit Classification** | **`INVALID` (Look-Ahead Entry)** — `prediction_timestamp` (`2026-11-17 10:25:29 IST`) > `entry_open` (`2026-11-17 09:15:00 IST`) | Same |
