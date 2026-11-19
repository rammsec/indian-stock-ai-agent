# Portfolio Risk Simulation Report — BharatAlpha AI

**Scope:** Risk, exposure, concentration, drawdown, execution-realism, and edge-case assessment of the Phase 5.3/5.4 paper portfolio (`backend/news_intel/forward_validation.py`) compared against the Phase 3 walk-forward execution simulator (`backend/backtesting/execution_simulator.py`) and audited via [`backend/backtesting/portfolio_audit.py`](backend/backtesting/portfolio_audit.py).

---

## 1. Portfolio Risk Parameters & Observed Exposure

| Parameter | Configured Limit | Observed (`production_replay` 10D) | Observed (`strict_canonical` 10D) | Audit Assessment |
|---|---:|---:|---:|---|
| Starting Capital | `₹1,000,000.00` | `₹1,000,000.00` | `₹1,000,000.00` | Verified |
| Max Single-Position Weight | `10.00%` (`MAX_POSITION_PCT = 0.10`) | `10.00%` at entry (`₹100,000.00`) | `0.00%` | Enforced at entry (`eq_curr * 0.10`) |
| Max Portfolio Exposure | `80.00%` (`MAX_EXPOSURE_PCT = 0.80`) | `10.02%` (cost) / `10.07%` (MTM) | `0.00%` | Under-utilized due to signal sparsity |
| Min Lot Capital | `₹1,000.00` | `₹100,000.00` | — | Enforced (`REJECTED_MIN_LOT_CAPITAL`) |
| Max Drawdown | Uncapped | `-0.20%` (`2026-11-17` EOD) | `0.00%` | Peak-to-trough tracked at exits + EOD MTM |
| Open Positions / Matured | — | `1` open / `0` matured | `0` open / `0` matured | Sample size (`n=0` matured) insufficient for Sharpe/win-rate |

### Cost-Basis vs. Mark-to-Market Exposure Distortion
In `forward_validation.py:857, 958`, `peak_exposure_pct` is calculated as:
$$\text{exposure\_pct} = \frac{\sum \text{capital}_{\text{cost}}}{\text{cash} + \sum \text{capital}_{\text{cost}} + \sum \text{unrealized\_pnl}} \times 100$$
Mixing a **cost-basis numerator** with a **mark-to-market denominator** caused `BIRLACABLE` (sized at `10.00%` on entry) to report `peak_exposure_pct = 10.02%` on `2026-11-17` when its price dropped `-1.98%`, even though its true MTM portfolio share fell to **`9.82%`**. On `2026-11-18` (`close = 302.60`), true MTM exposure rose to **`10.07%`** while cost-basis exposure fell to **`9.99%`**. `portfolio_audit.py` separates `peak_exposure_cost_pct` (`10.02%`) from `peak_exposure_mtm_pct` (`10.07%`) and tracks symbol/sector MTM concentration (`10.07%` in `Optical Fiber & Telecom Cables`).

---

## 2. Concentration & Sector Risk Profile

- **Universe Size Constraint:** The Phase 5 pilot covers 5 stocks across 4 sectors (`RELIANCE` — Energy/Telecom/Retail; `TCS` & `TATAELXSI` — IT/ER&D; `ITC` — FMCG; `BIRLACABLE` — Optical Fiber/Telecom Cables).
- **Maximum Reachable Exposure:** Because `open_positions` allows at most 1 concurrent position per symbol (`10%` cap) across 5 symbols, the effective **maximum achievable portfolio exposure is `50.0%`**, well below the `80.0%` ceiling (`MAX_EXPOSURE_PCT`).
- **Sector Overlap Risk:** `TCS` and `TATAELXSI` both map to Information Technology (`NIFTY_IT`), and `RELIANCE` + `BIRLACABLE` share telecom/fiber capex exposure (`BIRLACABLE` is a Jio FTTH supplier in `backend/news_intel/relationships.py:21, 138`). No sector cap exists in `forward_validation.py`; up to `20%` (`2/5` symbols) could concentrate in a single sector/theme.
- **Liquidity & Volatility Risk (`BIRLACABLE`):** `BIRLACABLE` is a small-cap telecom cable stock carrying `roundtrip_delivery_cost_pct = 0.5154%` (`51.54 bps`, vs `34.7 bps` default) due to higher ATR and Amihud illiquidity (`backend/prediction/cost_model.py`). In `production_replay`, `BIRLACABLE` is also the single most volatile pilot stock (`-1.98%` on `2026-11-17`, `+2.30%` on `2026-11-18`).

---

## 3. Execution Realism Gap: `forward_validation.py` vs. `execution_simulator.py`

| Risk Control | `execution_simulator.py` (Walk-Forward OOS) | `forward_validation.py` (Paper Portfolio) | Risk Impact |
|---|---|---|---|
| Opening Gap Filter | Skips `> +2.5%` gap-up / `< -3.0%` gap-down (`L193-200`) | **None** | Vulnerable to chasing extreme gap-ups/downs at `09:15` open |
| Stop-Loss / Take-Profit | `1.5 × ATR` SL / `2.5 × ATR` TP + conservative OHLC ambiguity (`L206-259, L308-333`) | **None** (unconditional `t+H` close hold) | Unbounded tail loss over 5D/10D holds (e.g., earnings shock) |
| Intraday Cost Split | 50% on entry, 50% on exit (`L209, L253`) | 100% deducted at `POSITION_EXITED` (`L894-897`) | Open MTM overstates net equity by `₹515.40` (`0.05%`) per `₹1L` lot |
| Corporate Action Adj. | Relies on pre-adjusted targets | Dynamic `adj_close / close` ratio (`L253-274`) | Correctly prevents split/dividend false drawdowns |

---

## 4. Edge-Case & Operational Risk Audit

1. **Look-Ahead Risk in Multi-Session Snapshots:** `deduplicate_forward_predictions` (`forward_validation.py:228-246`) preserves multiple `nse_session` snapshots per `(symbol, observation_date)` (`PRE_MARKET`, `INTRADAY`, `POST_MARKET`). In `forward_predictions.parquet`, `10/30` canonical snapshots have `prediction_timestamp >= entry_date 09:15 IST` (`2026-11-16` INTRADAY generated `2026-11-17T10:25:29`; `2026-11-17` POST_MARKET generated `2026-11-18T17:32:47`). `portfolio_audit.py` in `strict_canonical` mode rejects these via `REJECTED_LOOKAHEAD`.
2. **Same-Day Signal Ordering Dependency:** When `2` canonical sessions exist for the same `(symbol, observation_date)`, `forward_validation.py:671` sorts only by `["observation_date", "symbol"]` (not `prediction_timestamp`). If both sessions emit `BUY`, entry selection depends on Parquet row order rather than chronological precedence.
3. **Trading Holidays:** `all_sessions` and `sym_map[sym]` operate on actual NSE trading bar dates (skipping holidays such as `2026-11-09` / `2026-11-10` and weekends), preventing phantom zero-return bars.
4. **Missing Price Handling:**
   - Missing `open[t+1]` or `<= 0` → safely rejected (`REJECTED_INVALID_ENTRY_PRICE`).
   - Missing `observation_date` bar in `equities_daily` (e.g., holiday signal) → `forward_validation.py:983` silently drops the signal from `pending_entries_by_date` without logging a rejection, whereas `portfolio_audit.py` rolls forward to the next trading bar (`observation_date > sess_dt`) if history exists or emits `REJECTED_MISSING_OBSERVATION_BAR`.
   - Missing `close[exit_idx]` at maturity → `forward_validation.py:888` fails to settle the position, leaving capital permanently locked in `open_positions`.
