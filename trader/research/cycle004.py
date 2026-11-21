"""CYCLE-004: Multi-Day CNC Delivery Swing Simulator (₹0 Brokerage + Full Indian Delivery Statutory Costs)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import IST_TZ, LAB_DEFAULT_UNIVERSE
from ..data.loader import CACHE_DIR
from ..data.schema import Candle5M
from ..features.indicators import compute_all_features


@dataclass(frozen=True)
class CNCCostBreakdown:
    brokerage_inr: float
    stt_inr: float
    exchange_txn_inr: float
    sebi_inr: float
    stamp_duty_inr: float
    gst_inr: float
    dp_charge_inr: float
    total_statutory_inr: float
    total_friction_inr: float


def calculate_cnc_delivery_costs(side: str, turnover: float) -> CNCCostBreakdown:
    """Realistic Indian Equity Delivery (CNC) charges: ₹0 brokerage + 0.1% STT + Stamp + DP charge."""
    brokerage = 0.0
    stt = round(turnover * 0.001, 2)
    exch = round(turnover * 0.0000297, 2)
    sebi = round(turnover * 0.000001, 2)
    stamp = round(turnover * 0.00015, 2) if side == "BUY" else 0.0
    gst = round((brokerage + exch + sebi) * 0.18, 2)
    dp = 15.93 if side == "SELL" else 0.0
    statutory = round(stt + exch + sebi + stamp + gst + dp, 2)
    return CNCCostBreakdown(
        brokerage_inr=brokerage,
        stt_inr=stt,
        exchange_txn_inr=exch,
        sebi_inr=sebi,
        stamp_duty_inr=stamp,
        gst_inr=gst,
        dp_charge_inr=dp,
        total_statutory_inr=statutory,
        total_friction_inr=round(brokerage + statutory, 2),
    )


def _load_full_1h_features(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p_file = CACHE_DIR / f"{sym}_1h.parquet"
        df = pd.read_parquet(p_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST_TZ)
        df = df.sort_values("timestamp").reset_index(drop=True)
        candles = [
            Candle5M(
                symbol=sym,
                exchange="NSE",
                timestamp=r.timestamp.to_pydatetime(),
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=int(r.volume),
            )
            for r in df.itertuples(index=False)
        ]
        out[sym] = compute_all_features(candles)
    return out


def run_cnc_swing_backtest(
    eligible_dates: list[date],
    strategy_id: str,
    strategy_name: str,
    mode: str = "ORB_SWING",
    initial_capital_inr: float = 50_000.0,
    slippage_bps: float = 5.0,
    max_open_positions: int = 2,
    max_hold_days: int = 5,
    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE,
) -> dict[str, Any]:
    """Run chronological multi-day CNC delivery swing backtest on 1H bars (no 15:15 square-off)."""
    features_by_sym = _load_full_1h_features(symbols)
    elig_set = set(eligible_dates)

    # Build unified chronological bar timeline across eligible dates
    timeline: list[tuple[pd.Timestamp, str, int]] = []
    for sym, f_df in features_by_sym.items():
        for idx in range(len(f_df)):
            ts = f_df.iloc[idx]["timestamp"]
            if ts.date() in elig_set:
                timeline.append((ts, sym, idx))
    timeline.sort(key=lambda x: (x[0], x[1]))

    # Group by timestamp for synchronized bar-by-bar execution
    bars_by_ts: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for ts, sym, idx in timeline:
        bars_by_ts.setdefault(ts, []).append((sym, idx))

    cash = initial_capital_inr
    positions: dict[str, dict[str, Any]] = {}
    pending_signals: dict[str, dict[str, Any]] = {}
    last_close: dict[str, float] = {}
    trades: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []

    total_brokerage = 0.0
    total_statutory = 0.0
    total_slippage = 0.0

    daily_equity_map: dict[date, float] = {}
    daily_trades_map: dict[date, int] = {}
    daily_fees_map: dict[date, float] = {}
    bh_start_px: dict[str, float] = {}
    bh_end_px: dict[str, float] = {}

    for ts in sorted(bars_by_ts.keys()):
        curr_d = ts.date()
        bar_items = bars_by_ts[ts]

        # 1. Execute pending next-bar-open orders
        for sym, idx in bar_items:
            f_df = features_by_sym[sym]
            row = f_df.iloc[idx]
            open_px = float(row["open"])
            if sym not in bh_start_px:
                bh_start_px[sym] = open_px
            bh_end_px[sym] = float(row["close"])
            last_close[sym] = float(row["close"])

            if sym in pending_signals:
                sig = pending_signals.pop(sym)
                if sig["action"] == "EXIT" and sym in positions:
                    pos = positions.pop(sym)
                    exec_px = round(open_px * (1.0 - slippage_bps / 10000.0), 2)
                    slip_inr = round((open_px - exec_px) * pos["qty"], 2)
                    turnover = exec_px * pos["qty"]
                    cost = calculate_cnc_delivery_costs("SELL", turnover)
                    cash += turnover - cost.total_friction_inr
                    gross_pnl = (exec_px - pos["entry_px"]) * pos["qty"] + slip_inr + pos["entry_slip"]
                    net_pnl = (exec_px - pos["entry_px"]) * pos["qty"] - pos["entry_fees"] - cost.total_friction_inr
                    total_brokerage += cost.brokerage_inr
                    total_statutory += cost.total_statutory_inr
                    total_slippage += slip_inr
                    trades.append({"net_pnl": round(net_pnl, 2), "gross_pnl": round(gross_pnl, 2), "fees": round(pos["entry_fees"] + cost.total_friction_inr, 2)})
                    orders.append({"symbol": sym, "side": "SELL", "qty": pos["qty"], "price": exec_px, "fees": cost.total_friction_inr})
                    daily_trades_map[curr_d] = daily_trades_map.get(curr_d, 0) + 1
                    daily_fees_map[curr_d] = daily_fees_map.get(curr_d, 0.0) + cost.total_friction_inr

                elif sig["action"] == "BUY" and sym not in positions and len(positions) < max_open_positions:
                    alloc = min(cash, initial_capital_inr / max_open_positions)
                    exec_px = round(open_px * (1.0 + slippage_bps / 10000.0), 2)
                    qty = int(alloc // (exec_px * 1.003))
                    if qty >= 1:
                        turnover = exec_px * qty
                        cost = calculate_cnc_delivery_costs("BUY", turnover)
                        if cash >= turnover + cost.total_friction_inr:
                            slip_inr = round((exec_px - open_px) * qty, 2)
                            cash -= turnover + cost.total_friction_inr
                            total_brokerage += cost.brokerage_inr
                            total_statutory += cost.total_statutory_inr
                            total_slippage += slip_inr
                            positions[sym] = {
                                "qty": qty,
                                "entry_px": exec_px,
                                "entry_date": curr_d,
                                "entry_fees": cost.total_friction_inr,
                                "entry_slip": slip_inr,
                                "sl": sig["sl"],
                                "tp": sig["tp"],
                            }
                            orders.append({"symbol": sym, "side": "BUY", "qty": qty, "price": exec_px, "fees": cost.total_friction_inr})
                            daily_fees_map[curr_d] = daily_fees_map.get(curr_d, 0.0) + cost.total_friction_inr

        # 2. Check intra-bar SL / TP and max_hold_days
        for sym, idx in bar_items:
            if sym not in positions:
                continue
            pos = positions[sym]
            row = features_by_sym[sym].iloc[idx]
            low_px = float(row["low"])
            high_px = float(row["high"])
            close_px = float(row["close"])
            exit_px = None
            if low_px <= pos["sl"]:
                exit_px = pos["sl"]
            elif high_px >= pos["tp"]:
                exit_px = pos["tp"]
            elif (curr_d - pos["entry_date"]).days >= max_hold_days and ts.strftime("%H:%M") >= "15:15":
                exit_px = close_px

            if exit_px is not None:
                positions.pop(sym)
                slip_exec = round(exit_px * (1.0 - slippage_bps / 10000.0), 2)
                slip_inr = round((exit_px - slip_exec) * pos["qty"], 2)
                turnover = slip_exec * pos["qty"]
                cost = calculate_cnc_delivery_costs("SELL", turnover)
                cash += turnover - cost.total_friction_inr
                net_pnl = (slip_exec - pos["entry_px"]) * pos["qty"] - pos["entry_fees"] - cost.total_friction_inr
                total_brokerage += cost.brokerage_inr
                total_statutory += cost.total_statutory_inr
                total_slippage += slip_inr
                trades.append({"net_pnl": round(net_pnl, 2), "fees": round(pos["entry_fees"] + cost.total_friction_inr, 2)})
                orders.append({"symbol": sym, "side": "SELL", "qty": pos["qty"], "price": slip_exec, "fees": cost.total_friction_inr})
                daily_trades_map[curr_d] = daily_trades_map.get(curr_d, 0) + 1
                daily_fees_map[curr_d] = daily_fees_map.get(curr_d, 0.0) + cost.total_friction_inr

        # 3. Generate signals at bar close (strictly <= current bar)
        for sym, idx in bar_items:
            if idx < 15:
                continue
            f_df = features_by_sym[sym]
            row = f_df.iloc[idx]
            close_px = float(row["close"])
            ema9 = float(row["ema_9"])
            ema21 = float(row["ema_21"])
            rsi = float(row["rsi_14"])
            atr = max(float(row["atr_14"]), close_px * 0.005)
            vol = float(row["volume"])
            avg_vol = float(f_df.iloc[idx - 10:idx]["volume"].mean())

            if sym in positions:
                if ema9 < ema21 * 0.997:
                    pending_signals[sym] = {"action": "EXIT"}
            else:
                if mode == "ORB_SWING":
                    same_day = f_df.iloc[max(0, idx - 7):idx]
                    same_day = same_day[same_day["timestamp"].dt.date == curr_d]
                    hhmm = ts.strftime("%H:%M")
                    if hhmm in ("10:15", "11:15", "12:15") and not same_day.empty:
                        orb_high = float(same_day.iloc[0]["high"])
                        if close_px > orb_high and ema9 > ema21 and 52.0 <= rsi <= 70.0 and vol >= 1.25 * max(1.0, avg_vol):
                            pending_signals[sym] = {
                                "action": "BUY",
                                "sl": round(close_px - 1.5 * atr, 2),
                                "tp": round(close_px + 3.5 * atr, 2),
                            }
                else:  # TREND_SWING
                    prev_6h_high = float(f_df.iloc[idx - 6:idx]["high"].max())
                    if close_px > prev_6h_high and ema9 > ema21 and 52.0 <= rsi <= 70.0 and vol >= 1.3 * max(1.0, avg_vol):
                        pending_signals[sym] = {
                            "action": "BUY",
                            "sl": round(close_px - 1.5 * atr, 2),
                            "tp": round(close_px + 4.0 * atr, 2),
                        }

        # End-of-day mark-to-market equity
        mtm_equity = cash + sum(pos["qty"] * last_close.get(s, pos["entry_px"]) for s, pos in positions.items())
        daily_equity_map[curr_d] = round(mtm_equity, 2)

    # Liquidate any open position at final close for exact accounting
    for sym, pos in list(positions.items()):
        px = last_close.get(sym, pos["entry_px"])
        slip_exec = round(px * (1.0 - slippage_bps / 10000.0), 2)
        slip_inr = round((px - slip_exec) * pos["qty"], 2)
        turnover = slip_exec * pos["qty"]
        cost = calculate_cnc_delivery_costs("SELL", turnover)
        cash += turnover - cost.total_friction_inr
        net_pnl = (slip_exec - pos["entry_px"]) * pos["qty"] - pos["entry_fees"] - cost.total_friction_inr
        total_brokerage += cost.brokerage_inr
        total_statutory += cost.total_statutory_inr
        total_slippage += slip_inr
        trades.append({"net_pnl": round(net_pnl, 2), "fees": round(pos["entry_fees"] + cost.total_friction_inr, 2)})
        orders.append({"symbol": sym, "side": "SELL", "qty": pos["qty"], "price": slip_exec, "fees": cost.total_friction_inr})
    if eligible_dates:
        daily_equity_map[eligible_dates[-1]] = round(cash, 2)

    # Build daily_runs series
    daily_runs: list[dict[str, Any]] = []
    prev_eq = initial_capital_inr
    peak = initial_capital_inr
    max_dd = 0.0
    for d in eligible_dates:
        eq = daily_equity_map.get(d, prev_eq)
        day_pnl = round(eq - prev_eq, 2)
        peak = max(peak, eq)
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        daily_runs.append({
            "date": d.isoformat(),
            "net_pnl": day_pnl,
            "return_pct": round((day_pnl / initial_capital_inr) * 100.0, 2),
            "trades": daily_trades_map.get(d, 0),
            "fees": round(daily_fees_map.get(d, 0.0), 2),
            "cumulative_equity": eq,
        })
        prev_eq = eq

    final_eq = round(cash, 2)
    net_pnl_total = round(final_eq - initial_capital_inr, 2)
    total_fees = round(total_brokerage + total_statutory, 2)
    gross_pnl_total = round(net_pnl_total + total_fees, 2)
    net_ret_pct = round((net_pnl_total / initial_capital_inr) * 100.0, 2)
    gross_ret_pct = round((gross_pnl_total / initial_capital_inr) * 100.0, 2)

    bh_rets = [((bh_end_px[s] - bh_start_px[s]) / bh_start_px[s]) * 100.0 for s in bh_start_px if bh_start_px[s] > 0]
    bm_ret_pct = round(float(np.mean(bh_rets)), 2) if bh_rets else 0.0

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = round((len(wins) / len(pnls)) * 100.0, 2) if pnls else 0.0
    pf = round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) < 0 else (10.0 if wins else 0.0)
    sharpe = round((float(np.mean(pnls)) / float(np.std(pnls)) * math.sqrt(252)), 2) if len(pnls) > 1 and np.std(pnls) > 0 else 0.0
    downside = float(np.std(losses)) if len(losses) > 1 else 1e-4
    sortino = round((float(np.mean(pnls)) / downside * math.sqrt(252)), 2) if len(pnls) > 1 and downside > 0 else 0.0

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "timeframe_minutes": 60,
        "performance_category": "CNC_DELIVERY_SWING",
        "statistical_sufficiency": "VALIDATED_SAMPLE" if (len(eligible_dates) >= 30 and len(trades) >= 20) else "STATISTICALLY_INSUFFICIENT",
        "start_date": eligible_dates[0].isoformat(),
        "end_date": eligible_dates[-1].isoformat(),
        "sessions_evaluated": len(eligible_dates),
        "initial_capital_inr": initial_capital_inr,
        "final_equity_inr": final_eq,
        "gross_pnl_inr": gross_pnl_total,
        "gross_return_pct": gross_ret_pct,
        "net_pnl_inr": net_pnl_total,
        "total_return_pct": net_ret_pct,
        "benchmark_return_pct": bm_ret_pct,
        "alpha_vs_buy_and_hold_pct": round(net_ret_pct - bm_ret_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": pf,
        "expectancy_inr": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        "total_trades": len(trades),
        "win_rate_pct": win_rate,
        "total_fees_inr": total_fees,
        "total_brokerage_inr": round(total_brokerage, 2),
        "total_statutory_inr": round(total_statutory, 2),
        "total_slippage_inr": round(total_slippage, 2),
        "slippage_bps": slippage_bps,
        "orders": orders,
        "daily_runs": daily_runs,
    }
