# Pine Script v5 Research & Translation Workflow

## Objective
Enable rapid visual prototyping of intraday 5-minute strategies in **TradingView Pine Script v5**, followed by strict, zero-lookahead translation into deterministic Python modules under `trader/strategies/`.

---

## 1. Pine Script Research Rules (TradingView)
1. **Strict Bar-Close Evaluation**: All strategy entries and exits must evaluate only on confirmed closed 5-minute bars (`barstate.isconfirmed == true`). Never use `calc_on_every_tick = true`.
2. **No Repainting / No Future `request.security()`**: Any higher-timeframe reference must use `lookahead = barmerge.lookahead_off` with `[1]` historical offset.
3. **Intraday Session & Square-Off**: Explicitly restrict entries to `09:20–14:45 IST` and force `strategy.close_all()` at `15:15 IST`.
4. **Realistic Friction**: Configure `commission_type = strategy.commission.percent, commission_value = 0.05` and `slippage = 2` ticks in Pine Script before exporting trade logs.

---

## 2. Translation Protocol (Pine Script → Python `BaseStrategy`)

| Pine Script v5 Construct | Python Equivalent (`trader/`) |
|---|---|
| `ta.vwap(hlc3)` | `compute_intraday_vwap(df)` (resets at `09:15 IST`, zero `.bfill()`) |
| `ta.ema(close, 9)` | `df["ema_9"]` via `ewm(span=9, adjust=False)` |
| `ta.rsi(close, 14)` | `df["rsi_14"]` via Wilder's RMA |
| `ta.atr(14)` | `df["atr_14"]` via True Range RMA |
| `strategy.entry(..., stop=sl, limit=tp)` | `StrategySignal(action="ENTER_LONG", stop_loss=sl, target=tp)` |
| Next-bar market fill | `IntradayBacktester(execution_mode="NEXT_BAR_OPEN", slippage_bps=5.0)` |

---

## 3. Mandatory Validation Gate Before Paper Approval
Every translated strategy must pass:
1. **Incremental Invariance Test**: `test_indicator_incremental_invariance_across_all_features` (zero lookahead).
2. **Chronological Split**: Evaluated via `create_chronological_split()` (60% Train / 20% Validation / 20% Unseen Test).
3. **Purged K-Fold**: Validated via `PurgedKFold(n_splits=5, purge_bars=6, embargo_bars=6)` to prevent overlapping-bar leakage.
4. **Baseline Comparison**: Must be benchmarked via `compare_strategy_vs_baseline()` against Buy & Hold after full Indian statutory charges.
