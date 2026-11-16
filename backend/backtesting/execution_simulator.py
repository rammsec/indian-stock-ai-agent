"""Realistic Next-Day Indian Equity Execution Simulator (v3.2 — Priority 2).

Models NSE cash-equity execution assumptions after a prediction is generated
post-close on session `t` (~18:00 IST):
1. Entry occurs no earlier than 09:15 AM IST on session `t + 1` (at `open[t+1]`,
   or optionally `close[t]` for legacy baseline comparison).
2. Explicit opening gap filter: if overnight gap `(open[t+1] / close[t] - 1) * 100`
   exceeds `max_gap_up_pct` or falls below `-max_gap_down_pct`, entry is skipped
   (`SKIPPED_GAP_UP` / `SKIPPED_GAP_DOWN`).
3. Target and Stop-Loss sequencing with conservative daily-OHLC ambiguity handling:
   - If `open[d] <= sl_price`: gapped through stop-loss -> exit at `open[d]`.
   - Else if `open[d] >= tp_price`: gapped through target -> exit at `open[d]`.
   - Else if BOTH `low[d] <= sl_price` AND `high[d] >= tp_price` on the same daily
     bar (daily OHLC cannot establish intraday order): conservatively assume
     stop-loss was hit first (`AMBIGUOUS_OHLC_CONSERVATIVE_SL`) -> exit at `sl_price`.
   - Else if `low[d] <= sl_price`: exit at `sl_price`.
   - Else if `high[d] >= tp_price`: exit at `tp_price`.
   - Else: hold until signal exit or `max_hold_sessions`, exiting at `close[d]`.
4. Statutory Indian transaction costs (`CNC` delivery or `MIS` intraday) plus
   liquidity-adjusted slippage (`compute_row_cost_series`).
5. Portfolio position sizing & exposure limits:
   - `max_position_weight` (default 20% of capital per stock).
   - `max_portfolio_exposure` (default 100% total invested; unallocated capital sits in cash at 0%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..prediction.cost_model import compute_row_cost_series


@dataclass(frozen=True)
class ExecutionConfig:
    """Explicit execution assumptions for backtest trade simulation."""

    name: str = "next_open_delivery_capped"
    entry_timing: Literal["next_open", "same_close_legacy"] = "next_open"
    cost_mode: Literal["delivery", "intraday"] = "delivery"
    max_hold_sessions: int = 10
    max_gap_up_pct: float = 2.5
    max_gap_down_pct: float = 3.0
    stop_loss_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    use_sl_tp: bool = True
    max_position_weight: float = 0.20
    max_portfolio_exposure: float = 1.00
    long_prob_entry: float = 0.535
    long_prob_hold: float = 0.502
    min_exp_1d_pct: float = 0.08
    block_high_vol_regime: bool = True


def simulate_oos_execution(
    oos_df: pd.DataFrame,
    ohlc_df: pd.DataFrame | None = None,
    config: ExecutionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Simulate realistic next-day execution on out-of-sample predictions.

    `oos_df` contains signal rows at session `t` (`observation_date`) with
    `prob_next_day_up`, `expected_next_day_return_pct`, `expected_5d_return_pct`,
    `target_next_day_ret`, `pm_atr_14_pct`, `regime_label`, and `roundtrip_cost_pct`.

    If `ohlc_df` is provided (columns: `symbol`, `observation_date`, `open`, `high`,
    `low`, `close`), next-session `t+1` OHLC is matched chronologically per symbol
    to simulate 09:15 AM IST `open[t+1]` entry, opening gaps, and conservative SL/TP.
    """
    cfg = config or ExecutionConfig()
    df = oos_df.sort_values(["symbol", "observation_date"]).copy().reset_index(drop=True)

    cost_s = compute_row_cost_series(df, mode=cfg.cost_mode)
    df["roundtrip_cost_pct"] = cost_s.values

    # Build lookup of next-session OHLC per symbol if available
    next_ohlc_map: dict[tuple[str, str], dict[str, float]] = {}
    if ohlc_df is not None and not ohlc_df.empty:
        eq = ohlc_df.sort_values(["symbol", "observation_date"]).copy()
        for sym, grp in eq.groupby("symbol", sort=False):
            grp = grp.reset_index(drop=True)
            for i in range(len(grp) - 1):
                obs_t = str(grp.loc[i, "observation_date"])
                nxt = grp.loc[i + 1]
                c_t = float(grp.loc[i, "close"]) if pd.notna(grp.loc[i, "close"]) else np.nan
                o_n = float(nxt["open"]) if pd.notna(nxt["open"]) else c_t
                h_n = float(nxt["high"]) if pd.notna(nxt["high"]) else max(o_n, float(nxt["close"]))
                l_n = float(nxt["low"]) if pd.notna(nxt["low"]) else min(o_n, float(nxt["close"]))
                c_n = float(nxt["close"]) if pd.notna(nxt["close"]) else o_n
                next_ohlc_map[(str(sym), obs_t)] = {
                    "prev_close": c_t,
                    "next_date": str(nxt["observation_date"]),
                    "open": o_n,
                    "high": h_n,
                    "low": l_n,
                    "close": c_n,
                }

    positions = np.zeros(len(df), dtype=float)
    gross_ret_arr = np.zeros(len(df), dtype=float)
    cost_arr = np.zeros(len(df), dtype=float)
    exit_reason_arr = [""] * len(df)

    stats = {
        "total_signals_triggered": 0,
        "executed_trades": 0,
        "skipped_gap_up": 0,
        "skipped_gap_down": 0,
        "ambiguous_ohlc_conservative_sl": 0,
        "sl_exits": 0,
        "tp_exits": 0,
        "signal_or_max_hold_exits": 0,
        "missing_next_ohlc_fallback": 0,
    }

    p_up = pd.to_numeric(df["prob_next_day_up"], errors="coerce").fillna(0.50).values
    exp_r1 = pd.to_numeric(df["expected_next_day_return_pct"], errors="coerce").fillna(0.0).values
    exp_r5 = pd.to_numeric(df.get("expected_5d_return_pct", 0.0), errors="coerce").fillna(0.0).values
    y_r1 = pd.to_numeric(df["target_next_day_ret"], errors="coerce").fillna(0.0).values
    atr_s = pd.to_numeric(df.get("pm_atr_14_pct", 1.6), errors="coerce").fillna(1.6).clip(0.4, 8.0).values

    for _, grp in df.groupby("symbol", sort=False):
        idxs = grp.index.values
        in_pos = False
        hold_days = 0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0

        for i, row_idx in enumerate(idxs):
            sym = str(df.loc[row_idx, "symbol"])
            obs_t = str(df.loc[row_idx, "observation_date"])
            pr = float(p_up[row_idx])
            er1 = float(exp_r1[row_idx])
            er5 = float(exp_r5[row_idx])
            reg = str(df.loc[row_idx, "regime_label"]) if "regime_label" in df.columns else "Sideways"
            rt_cost = float(cost_s.iloc[row_idx])
            atr_pct = float(atr_s[row_idx])

            regime_ok = (reg != "High Volatility") if cfg.block_high_vol_regime else True
            nxt = next_ohlc_map.get((sym, obs_t))

            if cfg.entry_timing == "same_close_legacy" or nxt is None:
                if nxt is None and cfg.entry_timing == "next_open":
                    stats["missing_next_ohlc_fallback"] += 1
                # Legacy close[t] -> close[t+1]
                if not in_pos:
                    can_enter = (
                        (pr >= cfg.long_prob_entry)
                        and (er1 >= cfg.min_exp_1d_pct or er5 >= rt_cost * 1.15)
                        and regime_ok
                    )
                    if can_enter:
                        stats["total_signals_triggered"] += 1
                        stats["executed_trades"] += 1
                        in_pos = True
                        hold_days = 1
                        positions[row_idx] = 1.0
                        gross_ret_arr[row_idx] = float(y_r1[row_idx])
                        cost_arr[row_idx] = rt_cost * 0.5
                        exit_reason_arr[row_idx] = "ENTRY_CLOSE"
                else:
                    keep_holding = (pr >= cfg.long_prob_hold) and (hold_days < cfg.max_hold_sessions) and regime_ok
                    if keep_holding:
                        hold_days += 1
                        positions[row_idx] = 1.0
                        gross_ret_arr[row_idx] = float(y_r1[row_idx])
                    else:
                        prev_idx = idxs[i - 1]
                        cost_arr[prev_idx] += rt_cost * 0.5
                        in_pos = False
                        hold_days = 0
                continue

            # Realistic Next-Day 09:15 AM IST Open Execution
            prev_c = nxt["prev_close"]
            o_n, h_n, l_n, c_n = nxt["open"], nxt["high"], nxt["low"], nxt["close"]
            gap_pct = ((o_n / prev_c) - 1.0) * 100.0 if (prev_c and prev_c > 0) else 0.0

            if not in_pos:
                can_enter = (
                    (pr >= cfg.long_prob_entry)
                    and (er1 >= cfg.min_exp_1d_pct or er5 >= rt_cost * 1.15)
                    and regime_ok
                )
                if not can_enter:
                    continue
                stats["total_signals_triggered"] += 1

                if gap_pct > cfg.max_gap_up_pct:
                    stats["skipped_gap_up"] += 1
                    exit_reason_arr[row_idx] = "SKIPPED_GAP_UP"
                    continue
                if gap_pct < -cfg.max_gap_down_pct:
                    stats["skipped_gap_down"] += 1
                    exit_reason_arr[row_idx] = "SKIPPED_GAP_DOWN"
                    continue

                stats["executed_trades"] += 1
                in_pos = True
                hold_days = 1
                entry_price = o_n
                sl_price = entry_price * (1.0 - (cfg.stop_loss_atr_mult * atr_pct) / 100.0)
                tp_price = entry_price * (1.0 + (cfg.target_atr_mult * atr_pct) / 100.0)
                positions[row_idx] = 1.0
                cost_arr[row_idx] = rt_cost * 0.5

                # Evaluate intraday SL / TP / Close on entry session t+1
                if cfg.cost_mode == "intraday":
                    # Intraday MIS must exit by close of session t+1
                    exit_px, reason = _resolve_bar_exit(o_n, h_n, l_n, c_n, sl_price, tp_price, cfg.use_sl_tp, force_close=True)
                    _record_exit_stat(stats, reason)
                    gross_ret_arr[row_idx] = ((exit_px / entry_price) - 1.0) * 100.0
                    cost_arr[row_idx] += rt_cost * 0.5
                    exit_reason_arr[row_idx] = reason
                    in_pos = False
                    hold_days = 0
                else:
                    exit_px, reason = _resolve_bar_exit(o_n, h_n, l_n, c_n, sl_price, tp_price, cfg.use_sl_tp, force_close=False)
                    if reason != "HOLD":
                        _record_exit_stat(stats, reason)
                        gross_ret_arr[row_idx] = ((exit_px / entry_price) - 1.0) * 100.0
                        cost_arr[row_idx] += rt_cost * 0.5
                        exit_reason_arr[row_idx] = reason
                        in_pos = False
                        hold_days = 0
                    else:
                        # Mark-to-market from open[t+1] to close[t+1]
                        gross_ret_arr[row_idx] = ((c_n / entry_price) - 1.0) * 100.0
                        exit_reason_arr[row_idx] = "HOLD"
            else:
                # Already holding from prior session (mark-to-market base is prev_c)
                keep_holding = (pr >= cfg.long_prob_hold) and (hold_days < cfg.max_hold_sessions) and regime_ok
                if not keep_holding:
                    # Exit at next open[t+1]
                    positions[row_idx] = 1.0
                    gross_ret_arr[row_idx] = ((o_n / prev_c) - 1.0) * 100.0 if prev_c > 0 else 0.0
                    cost_arr[row_idx] = rt_cost * 0.5
                    exit_reason_arr[row_idx] = "SIGNAL_OR_MAX_HOLD_OPEN_EXIT"
                    stats["signal_or_max_hold_exits"] += 1
                    in_pos = False
                    hold_days = 0
                else:
                    hold_days += 1
                    positions[row_idx] = 1.0
                    exit_px, reason = _resolve_bar_exit(o_n, h_n, l_n, c_n, sl_price, tp_price, cfg.use_sl_tp, force_close=False)
                    if reason != "HOLD":
                        _record_exit_stat(stats, reason)
                        gross_ret_arr[row_idx] = ((exit_px / prev_c) - 1.0) * 100.0 if prev_c > 0 else 0.0
                        cost_arr[row_idx] = rt_cost * 0.5
                        exit_reason_arr[row_idx] = reason
                        in_pos = False
                        hold_days = 0
                    else:
                        gross_ret_arr[row_idx] = ((c_n / prev_c) - 1.0) * 100.0 if prev_c > 0 else 0.0
                        exit_reason_arr[row_idx] = "HOLD"

        if in_pos and len(idxs) > 0:
            last_idx = idxs[-1]
            cost_arr[last_idx] += float(cost_s.iloc[last_idx]) * 0.5

    df["position"] = positions
    df["gross_trade_ret_pct"] = gross_ret_arr
    df["cost_drag_pct"] = cost_arr
    df["net_trade_ret_pct"] = df["gross_trade_ret_pct"] - df["cost_drag_pct"]
    df["exit_reason"] = exit_reason_arr

    # Portfolio-level daily aggregation with explicit position sizing & exposure cap
    daily_rows: list[dict[str, Any]] = []
    for obs_dt, g in df.groupby("observation_date", sort=True):
        act_g = g[g["position"] > 0]
        n_act = len(act_g)
        if n_act == 0:
            daily_rows.append(
                {
                    "observation_date": obs_dt,
                    "active_cnt": 0,
                    "portfolio_exposure": 0.0,
                    "gross_ret_pct": 0.0,
                    "cost_drag_pct": 0.0,
                    "net_ret_pct": 0.0,
                }
            )
        else:
            w_per_pos = min(cfg.max_position_weight, cfg.max_portfolio_exposure / n_act)
            tot_exp = w_per_pos * n_act
            daily_rows.append(
                {
                    "observation_date": obs_dt,
                    "active_cnt": n_act,
                    "portfolio_exposure": round(tot_exp, 4),
                    "gross_ret_pct": float(act_g["gross_trade_ret_pct"].sum() * w_per_pos),
                    "cost_drag_pct": float(act_g["cost_drag_pct"].sum() * w_per_pos),
                    "net_ret_pct": float(act_g["net_trade_ret_pct"].sum() * w_per_pos),
                }
            )

    daily_port = pd.DataFrame(daily_rows).sort_values("observation_date").reset_index(drop=True)
    stats["avg_portfolio_exposure_pct"] = round(float(daily_port["portfolio_exposure"].mean() * 100.0), 2) if len(daily_port) else 0.0
    stats["max_position_weight_pct"] = round(cfg.max_position_weight * 100.0, 1)
    stats["execution_mode"] = cfg.name
    return df, daily_port, stats


def _resolve_bar_exit(
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    sl_price: float,
    tp_price: float,
    use_sl_tp: bool,
    force_close: bool,
) -> tuple[float, str]:
    if use_sl_tp:
        if open_p <= sl_price:
            return open_p, "GAP_BELOW_SL"
        if open_p >= tp_price:
            return open_p, "GAP_ABOVE_TP"
        hit_sl = low_p <= sl_price
        hit_tp = high_p >= tp_price
        if hit_sl and hit_tp:
            return sl_price, "AMBIGUOUS_OHLC_CONSERVATIVE_SL"
        if hit_sl:
            return sl_price, "STOP_LOSS_HIT"
        if hit_tp:
            return tp_price, "TARGET_HIT"
    if force_close:
        return close_p, "INTRADAY_CLOSE_EXIT"
    return close_p, "HOLD"


def _record_exit_stat(stats: dict[str, int], reason: str) -> None:
    if reason == "AMBIGUOUS_OHLC_CONSERVATIVE_SL":
        stats["ambiguous_ohlc_conservative_sl"] += 1
        stats["sl_exits"] += 1
    elif reason in ("STOP_LOSS_HIT", "GAP_BELOW_SL"):
        stats["sl_exits"] += 1
    elif reason in ("TARGET_HIT", "GAP_ABOVE_TP"):
        stats["tp_exits"] += 1
    elif reason == "INTRADAY_CLOSE_EXIT":
        stats["signal_or_max_hold_exits"] += 1
