# BharatAlpha AI v3.2 — Next-Day Execution Simulation

## 1. Explicit Execution Assumptions (`backend/backtesting/execution_simulator.py`)

In v3.0/v3.1, walk-forward evaluation assumed frictionless entry at `close[t]` with `100%` portfolio allocation even when only a single stock triggered a signal. v3.2 replaces this with [`ExecutionConfig`](file:///home/antigravity/rammsec/indian-stock-ai-agent/backend/backtesting/execution_simulator.py#L38) and [`simulate_oos_execution`](file:///home/antigravity/rammsec/indian-stock-ai-agent/backend/backtesting/execution_simulator.py#L58):

1. **Signal Generation & Earliest Entry**: Predictions are generated post-close on session `t` (~`18:00 IST`). Earliest possible execution is **09:15 AM IST** on session `t + 1` at `open[t+1]`.
2. **Opening Gap Filter**: Overnight gap `(open[t+1] / close[t] - 1) * 100` is checked before entry:
   - Skip if gap up exceeds `+2.5%` (`SKIPPED_GAP_UP`) to avoid chasing exhausted opens.
   - Skip if gap down exceeds `-3.0%` (`SKIPPED_GAP_DOWN`) to avoid entering broken gap-downs.
3. **Target & Stop-Loss Sequencing (Conservative Daily-OHLC Handling)**:
   - Stop-loss: `entry_price * (1 - 1.5 * ATR_14_pct / 100)`
   - Take-profit: `entry_price * (1 + 2.5 * ATR_14_pct / 100)`
   - Gap-through open: if `open[d] <= sl_price`, exit at `open[d]`; if `open[d] >= tp_price`, exit at `open[d]`.
   - **Ambiguous Daily OHLC**: Daily OHLC bars cannot establish whether intraday `low[d]` or `high[d]` occurred first. When `low[d] <= sl_price` **and** `high[d] >= tp_price` on the same bar, the simulator **conservatively assumes stop-loss was hit first** (`AMBIGUOUS_OHLC_CONSERVATIVE_SL`).
4. **Indian Statutory Costs & Liquidity Slippage (`backend/prediction/cost_model.py`)**:
   - **CNC Delivery**: `~34.7 bps` round-trip (0.10% STT buy + 0.10% STT sell, 0.015% stamp duty, NSE exchange/SEBI fees, 18% GST, DP charge, and ATR/Amihud liquidity-adjusted slippage).
   - **MIS Intraday**: `~19.6 bps` round-trip (zero-brokerage/capped Rs 20 leg, 0.025% sell-side STT, stamp duty, exchange/SEBI fees, GST, slippage), `max_hold_sessions = 1`.
5. **Position Sizing & Portfolio Exposure**:
   - `max_position_weight = 0.20` (20% capital cap per stock; eliminates single-stock 100% concentration distortion).
   - `max_portfolio_exposure = 1.00` (unallocated capital sits in cash at 0% return).

---

## 2. Empirical Comparison Under Alternative Execution Assumptions (`ml_ensemble`, OOS 2025-11-14 to 2026-11-13)

| Execution Mode | Active Stock-Days | Executed Entries | Skipped (Gap Up / Down) | SL / TP Exits | Net Return | Net Sharpe | Max Drawdown | Profit Factor | Expectancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `same_close_legacy` (100% cap, no SL/TP) | 71 | 18 | 0 / 0 | 0 / 0 | -28.12% | -2.684 | -28.21% | 0.318 | -78.64 bps |
| `next_open_delivery_capped` (09:15 open, 20% cap, CNC) | 72 | 20 | 0 / 2 | 8 / 1 | -6.76% | -2.466 | -6.79% | 0.461 | -50.68 bps |
| `next_open_intraday_mis` (09:15 open, 20% cap, 1D MIS) | 25 | 25 | 2 / 5 | 1 / 0 | -3.54% | -1.810 | -3.54% | 0.362 | -70.58 bps |

### Key Takeaway
Capping single-stock position size at 20% (`next_open_delivery_capped`) reduces drawdown from `-28.21%` to `-6.79%`, and switching to 1-day MIS lowers per-trade cost drag, **but `ml_ensemble` remains unprofitable across every execution assumption**.
