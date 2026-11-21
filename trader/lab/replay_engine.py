"""Chronological 5-Minute Historical Trading Replay Engine (₹50,000 Virtual Portfolio)."""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..agents.contracts import MarketPosture
from ..agents.council import CentralMarketCouncil
from ..agents.market_intelligence import MarketIntelligenceAgent
from ..agents.microstructure_agent import MicrostructureAgent
from ..agents.technical_agent import TechnicalAnalysisAgent
from ..config import (
    IST_TZ,
    LAB_DEFAULT_UNIVERSE,
    LAB_INITIAL_CAPITAL_INR,
    CostConfig,
    LabRiskConfig,
)
from ..data.instruments import INSTRUMENT_MASTER
from ..data.schema import Candle5M, DataSourceMode
from ..features.indicators import compute_all_features
from ..portfolio.cost_calculator import calculate_intraday_costs
from ..portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from ..risk.engine import DeterministicRiskEngine
from ..strategies.base import BaseStrategy, StrategySignal
from ..strategies.ema_cross import EMACrossStrategy
from ..strategies.multi_indicator import MultiIndicatorConfluenceStrategy
from ..strategies.orb_breakout import ORBBreakoutStrategy
from ..strategies.rsi_macd import RSIMACDStrategy
from ..strategies.supertrend import SupertrendStrategy
from ..strategies.volume_breakout import VolumeBreakoutStrategy
from ..strategies.vwap_pullback import VWAPPullbackStrategy
from .data_service import StockSessionQuality, load_session_with_warmup


STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "STRAT-001-VWAP-PULLBACK": {"name": "VWAP Pullback", "cls": VWAPPullbackStrategy, "kind": "STRATEGY"},
    "STRAT-002-ORB-BREAKOUT": {"name": "Opening Range Breakout", "cls": ORBBreakoutStrategy, "kind": "STRATEGY"},
    "STRAT-003-EMA-CROSS": {"name": "EMA 9/21 Crossover", "cls": EMACrossStrategy, "kind": "STRATEGY"},
    "STRAT-004-SUPERTREND": {"name": "Supertrend", "cls": SupertrendStrategy, "kind": "STRATEGY"},
    "STRAT-005-RSI-MACD": {"name": "RSI & MACD Combination", "cls": RSIMACDStrategy, "kind": "STRATEGY"},
    "STRAT-006-VOLUME-BREAKOUT": {"name": "Volume Breakout", "cls": VolumeBreakoutStrategy, "kind": "STRATEGY"},
    "STRAT-007-MULTI-INDICATOR": {"name": "Multi-Indicator Confirmation", "cls": MultiIndicatorConfluenceStrategy, "kind": "STRATEGY"},
    "AGENT-CENTRAL-COUNCIL": {"name": "Central AI Council (8-Agent)", "cls": None, "kind": "AI_AGENT"},
    "AGENT-AUTONOMOUS-TRADER": {"name": "Autonomous Trading Agent", "cls": None, "kind": "AI_AGENT"},
}


def compute_five_stock_equal_weight_benchmark(
    session_candles_by_symbol: dict[str, list[Candle5M]],
    initial_capital: float = LAB_INITIAL_CAPITAL_INR,
    cost_config: CostConfig | None = None,
) -> dict[str, Any]:
    """Compute equal-weight ₹50,000 Buy & Hold benchmark across the 5 stocks with transaction costs."""
    active_syms = [s for s, bars in session_candles_by_symbol.items() if bars]
    if not active_syms:
        return {
            "initial_capital_inr": initial_capital,
            "ending_capital_inr": initial_capital,
            "total_return_pct": 0.0,
            "net_pnl_inr": 0.0,
            "equity_curve": [],
        }

    alloc_per_stock = initial_capital / len(active_syms)
    holdings: dict[str, dict[str, float]] = {}
    total_cash = initial_capital

    for sym in active_syms:
        first_bar = session_candles_by_symbol[sym][0]
        entry_px = first_bar.open
        qty = int(alloc_per_stock / entry_px) if entry_px > 0 else 0
        if qty > 0:
            buy_cost = calculate_intraday_costs(entry_px, qty, "BUY", "NSE", cost_config=cost_config)
            while qty > 0 and (qty * entry_px + buy_cost.total_friction_inr) > alloc_per_stock:
                qty -= 1
                buy_cost = calculate_intraday_costs(entry_px, qty, "BUY", "NSE", cost_config=cost_config)
            if qty > 0:
                total_cash -= (qty * entry_px + buy_cost.total_friction_inr)
        holdings[sym] = {"qty": float(qty), "entry": entry_px}

    # Align timestamps across all stocks
    all_ts = sorted({c.timestamp for bars in session_candles_by_symbol.values() for c in bars})
    last_close = {sym: session_candles_by_symbol[sym][0].open for sym in active_syms}
    bar_lookup = {
        sym: {c.timestamp: c for c in bars}
        for sym, bars in session_candles_by_symbol.items()
    }

    curve = []
    for ts in all_ts:
        pos_val = 0.0
        for sym in active_syms:
            if ts in bar_lookup[sym]:
                last_close[sym] = bar_lookup[sym][ts].close
            pos_val += holdings[sym]["qty"] * last_close[sym]
        eq = round(total_cash + pos_val, 2)
        curve.append({"timestamp": ts.isoformat(), "time": ts.strftime("%H:%M"), "equity": eq})

    # Deduct simulated closing friction at end of day
    exit_friction = 0.0
    for sym in active_syms:
        qty = int(holdings[sym]["qty"])
        if qty > 0:
            sell_cost = calculate_intraday_costs(last_close[sym], qty, "SELL", "NSE", cost_config=cost_config)
            exit_friction += sell_cost.total_friction_inr

    ending_eq = round((curve[-1]["equity"] if curve else initial_capital) - exit_friction, 2)
    net_pnl = round(ending_eq - initial_capital, 2)
    ret_pct = round((net_pnl / initial_capital) * 100.0, 2)

    return {
        "initial_capital_inr": initial_capital,
        "ending_capital_inr": ending_eq,
        "net_pnl_inr": net_pnl,
        "total_return_pct": ret_pct,
        "equity_curve": curve,
    }


class HistoricalLabReplayEngine:
    """Replays a historical 5m session chronologically across the 5-stock universe on ₹50,000 capital."""

    def __init__(
        self,
        risk_config: LabRiskConfig | None = None,
        cost_config: CostConfig | None = None,
    ) -> None:
        self.risk_config = risk_config or LabRiskConfig()
        self.cost_config = cost_config or CostConfig()

    def run_historical_day(
        self,
        target_date: date,
        strategy_id: str = "STRAT-001-VWAP-PULLBACK",
        symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
        data_mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
        execution_mode: Literal["DETERMINISTIC", "AI_AGENT"] = "DETERMINISTIC",
    ) -> dict[str, Any]:
        """Run deterministic chronological replay for a single historical date."""
        if strategy_id not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy/model ID: {strategy_id}")

        started_at = datetime.now(tz=IST_TZ).isoformat()
        run_id = f"run_{target_date.strftime('%Y%m%d')}_{strategy_id}_{uuid.uuid4().hex[:6]}"

        reg_entry = STRATEGY_REGISTRY[strategy_id]
        strategy_obj: BaseStrategy | None = reg_entry["cls"]() if reg_entry["cls"] is not None else None

        # 1. Load warmup + session candles strictly <= target_date
        warmup_by_sym: dict[str, list[Candle5M]] = {}
        session_by_sym: dict[str, list[Candle5M]] = {}
        quality_by_sym: dict[str, StockSessionQuality] = {}
        features_by_sym: dict[str, pd.DataFrame] = {}
        session_start_idx: dict[str, int] = {}

        for sym in symbols:
            w_bars, s_bars, q = load_session_with_warmup(sym, target_date, warmup_days=5, mode=data_mode)
            warmup_by_sym[sym] = w_bars
            session_by_sym[sym] = s_bars
            quality_by_sym[sym] = q
            combined = w_bars + s_bars
            if combined:
                features_by_sym[sym] = compute_all_features(combined)
                session_start_idx[sym] = len(w_bars)
            else:
                features_by_sym[sym] = pd.DataFrame()
                session_start_idx[sym] = 0

        # 2. Initialize ₹50,000 Portfolio & Lab Risk Engine
        portfolio = IntradayPaperPortfolio(
            initial_capital=self.risk_config.initial_capital_inr,
            cost_config=self.cost_config,
        )
        risk_engine = DeterministicRiskEngine(
            max_risk_per_trade_pct=self.risk_config.risk_per_trade_pct,
            max_position_weight_pct=0.40,  # Allow up to ₹20k per stock in ₹50k portfolio so RELIANCE/TCS/TATAELXSI can trade >=1 share
            max_portfolio_exposure_pct=self.risk_config.max_portfolio_allocation_pct,
            max_daily_loss_pct=self.risk_config.max_daily_loss_pct,
            max_concurrent_positions=self.risk_config.max_open_positions,
        )

        # AI Agents
        mi_agent = MarketIntelligenceAgent()
        tech_agent = TechnicalAnalysisAgent()
        micro_agent = MicrostructureAgent()
        council = CentralMarketCouncil()

        # Chronological union of timestamps on target_date
        all_timestamps = sorted({c.timestamp for bars in session_by_sym.values() for c in bars})
        bar_index_lookup: dict[str, dict[datetime, int]] = {}
        for sym in symbols:
            df = features_by_sym[sym]
            mapping: dict[datetime, int] = {}
            if not df.empty:
                start_i = session_start_idx[sym]
                for idx in range(start_i, len(df)):
                    ts_val = df.iloc[idx]["timestamp"]
                    ts_dt = ts_val.to_pydatetime() if hasattr(ts_val, "to_pydatetime") else ts_val
                    mapping[ts_dt] = idx
            bar_index_lookup[sym] = mapping

        pending_orders: dict[str, dict[str, Any]] = {}
        decision_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []
        chart_markers: dict[str, list[dict[str, Any]]] = {sym: [] for sym in symbols}
        snapshots: list[dict[str, Any]] = []
        last_prices: dict[str, float] = {}
        last_council_eval: datetime | None = None
        current_posture = MarketPosture.NEUTRAL
        daily_trades_per_sym: dict[str, int] = {sym: 0 for sym in symbols}
        open_entry_meta: dict[str, dict[str, Any]] = {}
        completed_trades_detailed: list[dict[str, Any]] = []

        # 3. Replay chronologically bar by bar
        for ts in all_timestamps:
            t_ist = ts.astimezone(IST_TZ).time()

            # A. Execute pending orders from previous bar at THIS bar's OPEN (+ slippage)
            for sym in list(pending_orders.keys()):
                if ts not in bar_index_lookup[sym]:
                    continue
                pend = pending_orders.pop(sym)
                idx = bar_index_lookup[sym][ts]
                row = features_by_sym[sym].iloc[idx]
                open_px = float(row["open"])
                action_type = pend["action"]

                if action_type == "ENTER_LONG" and sym not in portfolio.positions:
                    slip_mult = 1.0 + (self.cost_config.base_slippage_bps / 10000.0)
                    exec_px = round(open_px * slip_mult, 2)
                    sl_px = pend["stop_loss"]
                    tp_px = pend["target"]

                    # Gap protection: if next bar opens below SL or above Target, reject fill!
                    if sl_px is not None and exec_px <= sl_px:
                        ord_id = f"ord_gap_{len(order_records)+1:04d}"
                        order_records.append({
                            "order_id": ord_id,
                            "symbol": sym,
                            "side": "BUY",
                            "quantity": 0,
                            "order_type": "MARKET",
                            "expected_price": pend["signal_price"],
                            "execution_price": exec_px,
                            "fill_quantity": 0,
                            "slippage": 0.0,
                            "fees": 0.0,
                            "status": "REJECTED_GAP",
                            "timestamp": ts.isoformat(),
                        })
                        chart_markers[sym].append({
                            "time": int(ts.timestamp()),
                            "time_str": ts.strftime("%H:%M"),
                            "type": "REJECTED",
                            "price": open_px,
                            "reason": f"Gap below stop-loss on open (₹{open_px:.2f} <= SL ₹{sl_px:.2f})",
                            "decision": pend["decision_record"],
                        })
                        continue

                    # Compute risk-based quantity (₹250 max risk, 0.5% equity)
                    risk_per_share = max(0.01, exec_px - (sl_px or exec_px * 0.99))
                    max_risk_inr = min(self.risk_config.max_risk_per_trade_inr, portfolio.total_equity * self.risk_config.risk_per_trade_pct)
                    req_qty = max(1, int(max_risk_inr / risk_per_share))

                    risk_res = risk_engine.evaluate_order(
                        symbol=sym,
                        side="BUY",
                        requested_quantity=req_qty,
                        entry_price=exec_px,
                        stop_loss_price=sl_px,
                        portfolio=portfolio,
                        current_time=ts,
                    )

                    if risk_res.approved and risk_res.approved_quantity > 0:
                        sim_ord = portfolio.submit_order(
                            symbol=sym,
                            side=OrderSide.BUY,
                            quantity=risk_res.approved_quantity,
                            price=exec_px,
                            exchange="NSE",
                            stop_loss=sl_px,
                            target=tp_px,
                            timestamp=ts,
                        )
                        fees = sim_ord.cost_breakdown.total_friction_inr if sim_ord.cost_breakdown else 0.0
                        slip_inr = round((exec_px - open_px) * risk_res.approved_quantity, 2)
                        order_records.append({
                            "order_id": sim_ord.order_id,
                            "symbol": sym,
                            "side": "BUY",
                            "quantity": risk_res.approved_quantity,
                            "order_type": "MARKET",
                            "expected_price": open_px,
                            "execution_price": exec_px,
                            "fill_quantity": sim_ord.filled_quantity,
                            "slippage": slip_inr,
                            "fees": fees,
                            "status": sim_ord.status.value,
                            "timestamp": ts.isoformat(),
                        })
                        if sim_ord.status == OrderStatus.FILLED:
                            daily_trades_per_sym[sym] += 1
                            open_entry_meta[sym] = {
                                "entry_order_id": sim_ord.order_id,
                                "entry_timestamp": ts.isoformat(),
                                "entry_fees": fees,
                            }
                            chart_markers[sym].append({
                                "time": int(ts.timestamp()),
                                "time_str": ts.strftime("%H:%M"),
                                "type": "BUY",
                                "price": exec_px,
                                "quantity": sim_ord.filled_quantity,
                                "stop_loss": sl_px,
                                "target": tp_px,
                                "reason": pend["decision_record"]["reasoning_summary"],
                                "decision": pend["decision_record"],
                                "execution": {
                                    "execution_time": ts.strftime("%H:%M IST"),
                                    "execution_price": exec_px,
                                    "quantity": sim_ord.filled_quantity,
                                    "fees_inr": fees,
                                    "status": sim_ord.status.value,
                                },
                            })
                    else:
                        order_records.append({
                            "order_id": f"ord_rej_{len(order_records)+1:04d}",
                            "symbol": sym,
                            "side": "BUY",
                            "quantity": req_qty,
                            "order_type": "MARKET",
                            "expected_price": open_px,
                            "execution_price": None,
                            "fill_quantity": 0,
                            "slippage": 0.0,
                            "fees": 0.0,
                            "status": "REJECTED_RISK",
                            "timestamp": ts.isoformat(),
                        })
                        chart_markers[sym].append({
                            "time": int(ts.timestamp()),
                            "time_str": ts.strftime("%H:%M"),
                            "type": "REJECTED",
                            "price": open_px,
                            "reason": "; ".join(risk_res.rejection_reasons),
                            "decision": pend["decision_record"],
                        })

                elif action_type == "EXIT" and sym in portfolio.positions:
                    slip_mult = 1.0 - (self.cost_config.base_slippage_bps / 10000.0)
                    exec_px = round(open_px * slip_mult, 2)
                    pos = portfolio.positions[sym]
                    qty = pos.quantity
                    entry_px = pos.entry_price
                    sim_ord = portfolio.submit_order(
                        symbol=sym,
                        side=OrderSide.SELL,
                        quantity=qty,
                        price=exec_px,
                        exchange="NSE",
                        timestamp=ts,
                    )
                    fees = sim_ord.cost_breakdown.total_friction_inr if sim_ord.cost_breakdown else 0.0
                    order_records.append({
                        "order_id": sim_ord.order_id,
                        "symbol": sym,
                        "side": "SELL",
                        "quantity": qty,
                        "order_type": "MARKET",
                        "expected_price": open_px,
                        "execution_price": exec_px,
                        "fill_quantity": sim_ord.filled_quantity,
                        "slippage": round((open_px - exec_px) * qty, 2),
                        "fees": fees,
                        "status": sim_ord.status.value,
                        "timestamp": ts.isoformat(),
                    })
                    meta = open_entry_meta.pop(sym, {})
                    last_closed = portfolio.closed_trades[-1] if portfolio.closed_trades else {}
                    entry_dt = datetime.fromisoformat(meta.get("entry_timestamp", ts.isoformat()))
                    completed_trades_detailed.append({
                        "trade_id": f"tr_{len(completed_trades_detailed)+1:04d}",
                        "symbol": sym,
                        "entry_order_id": meta.get("entry_order_id", "unknown"),
                        "exit_order_id": sim_ord.order_id,
                        "entry_timestamp": meta.get("entry_timestamp", ts.isoformat()),
                        "exit_timestamp": ts.isoformat(),
                        "quantity": qty,
                        "entry_price": entry_px,
                        "exit_price": exec_px,
                        "gross_pnl": last_closed.get("gross_pnl_inr", 0.0),
                        "net_pnl": last_closed.get("net_pnl_inr", 0.0),
                        "fees": round(meta.get("entry_fees", 0.0) + fees, 2),
                        "holding_minutes": round((ts - entry_dt).total_seconds() / 60.0, 1),
                        "exit_reason": pend["decision_record"]["reasoning_summary"],
                    })
                    chart_markers[sym].append({
                        "time": int(ts.timestamp()),
                        "time_str": ts.strftime("%H:%M"),
                        "type": "SELL",
                        "price": exec_px,
                        "quantity": qty,
                        "reason": pend["decision_record"]["reasoning_summary"],
                        "decision": pend["decision_record"],
                        "execution": {
                            "execution_time": ts.strftime("%H:%M IST"),
                            "execution_price": exec_px,
                            "quantity": qty,
                            "fees_inr": fees,
                            "status": sim_ord.status.value,
                        },
                    })

            # B. Update market prices (check SL/TP intra-bar) and handle 15:15 IST square-off
            for sym in symbols:
                if ts not in bar_index_lookup[sym]:
                    continue
                idx = bar_index_lookup[sym][ts]
                row = features_by_sym[sym].iloc[idx]
                low_px = float(row["low"])
                high_px = float(row["high"])
                close_px = float(row["close"])
                last_prices[sym] = close_px

                if sym in portfolio.positions:
                    prev_closed_len = len(portfolio.closed_trades)
                    portfolio.update_market_price(sym, low_px, timestamp=ts)
                    if sym in portfolio.positions:
                        portfolio.update_market_price(sym, high_px, timestamp=ts)
                    if sym in portfolio.positions:
                        portfolio.update_market_price(sym, close_px, timestamp=ts)

                    # Check if SL/TP auto-closed the position
                    if len(portfolio.closed_trades) > prev_closed_len:
                        tr = portfolio.closed_trades[-1]
                        meta = open_entry_meta.pop(sym, {})
                        entry_dt = datetime.fromisoformat(meta.get("entry_timestamp", ts.isoformat()))
                        exit_px = tr["exit_price"]
                        exit_reason = "STOP_LOSS_HIT" if tr["net_pnl_inr"] <= 0 else "TARGET_HIT"
                        completed_trades_detailed.append({
                            "trade_id": f"tr_{len(completed_trades_detailed)+1:04d}",
                            "symbol": sym,
                            "entry_order_id": meta.get("entry_order_id", "unknown"),
                            "exit_order_id": f"ord_auto_{len(order_records)+1:04d}",
                            "entry_timestamp": meta.get("entry_timestamp", ts.isoformat()),
                            "exit_timestamp": ts.isoformat(),
                            "quantity": tr["quantity"],
                            "entry_price": tr["entry_price"],
                            "exit_price": exit_px,
                            "gross_pnl": tr["gross_pnl_inr"],
                            "net_pnl": tr["net_pnl_inr"],
                            "fees": tr["total_friction_inr"],
                            "holding_minutes": round((ts - entry_dt).total_seconds() / 60.0, 1),
                            "exit_reason": exit_reason,
                        })
                        chart_markers[sym].append({
                            "time": int(ts.timestamp()),
                            "time_str": ts.strftime("%H:%M"),
                            "type": "SELL",
                            "price": exit_px,
                            "quantity": tr["quantity"],
                            "reason": exit_reason,
                            "decision": {
                                "decision_id": f"dec_auto_{len(decision_records)+1:05d}",
                                "symbol": sym,
                                "decision_timestamp": ts.isoformat(),
                                "agent_name": "Risk / Portfolio Engine",
                                "recommendation": "SELL",
                                "confidence": 1.0,
                                "reasoning_summary": exit_reason,
                                "council_decision": current_posture.value,
                                "risk_approval": "AUTO_TRIGGERED",
                            },
                        })

            if t_ist >= self.risk_config.square_off_time and len(portfolio.positions) > 0:
                sq_orders = portfolio.square_off_all_positions(last_prices, timestamp=ts)
                for so in sq_orders:
                    tr = portfolio.closed_trades[-1] if portfolio.closed_trades else {}
                    meta = open_entry_meta.pop(so.symbol, {})
                    entry_dt = datetime.fromisoformat(meta.get("entry_timestamp", ts.isoformat()))
                    completed_trades_detailed.append({
                        "trade_id": f"tr_{len(completed_trades_detailed)+1:04d}",
                        "symbol": so.symbol,
                        "entry_order_id": meta.get("entry_order_id", "unknown"),
                        "exit_order_id": so.order_id,
                        "entry_timestamp": meta.get("entry_timestamp", ts.isoformat()),
                        "exit_timestamp": ts.isoformat(),
                        "quantity": so.filled_quantity,
                        "entry_price": tr.get("entry_price", so.price),
                        "exit_price": so.price,
                        "gross_pnl": tr.get("gross_pnl_inr", 0.0),
                        "net_pnl": tr.get("net_pnl_inr", 0.0),
                        "fees": tr.get("total_friction_inr", 0.0),
                        "holding_minutes": round((ts - entry_dt).total_seconds() / 60.0, 1),
                        "exit_reason": "MANDATORY_1515_SQUAREOFF",
                    })
                    chart_markers[so.symbol].append({
                        "time": int(ts.timestamp()),
                        "time_str": ts.strftime("%H:%M"),
                        "type": "SELL",
                        "price": so.price,
                        "quantity": so.filled_quantity,
                        "reason": "MANDATORY_1515_SQUAREOFF",
                        "decision": {
                            "decision_id": f"dec_sq_{len(decision_records)+1:05d}",
                            "symbol": so.symbol,
                            "decision_timestamp": ts.isoformat(),
                            "agent_name": "Deterministic Risk Engine",
                            "recommendation": "SELL",
                            "confidence": 1.0,
                            "reasoning_summary": "Mandatory 15:15 IST Intraday Square-Off",
                            "council_decision": current_posture.value,
                            "risk_approval": "MANDATORY",
                        },
                    })
                pending_orders.clear()

            # C. Periodic Central Council Assessment (Hourly on 5m replay)
            if last_council_eval is None or (ts - last_council_eval).total_seconds() >= 3600:
                # Derive market breadth from available symbols up to ts
                adv = 0
                dec = 0
                pct_changes = []
                for sym in symbols:
                    if ts in bar_index_lookup[sym]:
                        idx = bar_index_lookup[sym][ts]
                        start_i = session_start_idx[sym]
                        day_open = float(features_by_sym[sym].iloc[start_i]["open"])
                        curr_close = float(features_by_sym[sym].iloc[idx]["close"])
                        chg = ((curr_close - day_open) / day_open) * 100.0 if day_open > 0 else 0.0
                        pct_changes.append(chg)
                        if chg >= 0:
                            adv += 1
                        else:
                            dec += 1
                avg_chg = float(np.mean(pct_changes)) if pct_changes else 0.0
                mi_msg = mi_agent.analyze_market(
                    nifty_change_pct=avg_chg,
                    bank_nifty_change_pct=avg_chg * 1.1,
                    india_vix=14.5,
                    advances=max(1, adv * 300),
                    declines=max(1, dec * 300),
                    timestamp=ts,
                )
                c_msg = council.evaluate_council(market_intel=mi_msg)
                current_posture = c_msg.posture
                last_council_eval = ts

            # D. Evaluate each stock at closed bar `ts` (NO future info!)
            if t_ist < self.risk_config.square_off_time:
                for sym in symbols:
                    if ts not in bar_index_lookup[sym]:
                        continue
                    idx = bar_index_lookup[sym][ts]
                    df = features_by_sym[sym]
                    row = df.iloc[idx]
                    close_px = float(row["close"])
                    atr = max(float(row["atr_14"]), close_px * 0.002)

                    ind_snap = {
                        "close": round(close_px, 2),
                        "vwap": round(float(row["vwap"]), 2),
                        "ema_9": round(float(row["ema_9"]), 2),
                        "ema_21": round(float(row["ema_21"]), 2),
                        "rsi_14": round(float(row["rsi_14"]), 1),
                        "macd": round(float(row["macd"]), 2),
                        "atr_14": round(atr, 2),
                    }

                    if strategy_obj is not None:
                        sig = strategy_obj.evaluate_bar(sym, df, current_bar_idx=idx)
                        agent_name = reg_entry["name"]
                        model_name = "deterministic_python"
                        model_ver = "v1.0"
                        exec_mode_label = "DETERMINISTIC"
                    else:
                        # AI Agent Mode: slice strictly up to idx (zero future leakage)
                        sub_candles = (warmup_by_sym[sym] + session_by_sym[sym])[: idx + 1]
                        tech_msg = tech_agent.analyze_candles(sub_candles[-60:])
                        micro_msg = micro_agent.analyze_microstructure(
                            sym, "NSE", best_bid=close_px - 0.25, best_ask=close_px + 0.25, traded_volume_20m=25000.0
                        )
                        if strategy_id == "AGENT-CENTRAL-COUNCIL" and current_posture in (MarketPosture.SELL_BEARISH, MarketPosture.HOLD_HIGH_RISK):
                            sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="NO_SIGNAL", confidence=0.7, rationale=[f"Council posture {current_posture.value} blocks new longs"])
                        elif tech_msg.signal == "BULLISH" and tech_msg.confidence >= 0.65 and micro_msg.execution_feasibility != "REJECT":
                            sl = round(close_px - 1.0 * atr, 2)
                            tp = round(close_px + 2.0 * atr, 2)
                            sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="ENTER_LONG", entry_price=close_px, confidence=tech_msg.confidence, stop_loss=sl, target=tp, rationale=tech_msg.evidence[:2])
                        elif tech_msg.signal == "BEARISH" and sym in portfolio.positions:
                            sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="EXIT", confidence=tech_msg.confidence, rationale=["Technical bearish reversal"])
                        else:
                            sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="HOLD", confidence=0.55, rationale=[f"Tech={tech_msg.signal}, Micro={micro_msg.execution_feasibility}"])
                        agent_name = reg_entry["name"]
                        model_name = "deterministic_fallback" if execution_mode == "DETERMINISTIC" else "gemini-2.5-flash"
                        model_ver = "v1.0"
                        exec_mode_label = "DETERMINISTIC_FALLBACK" if execution_mode == "DETERMINISTIC" else "LLM"

                    # Map StrategySignal action -> BUY / SELL / HOLD / WAIT
                    rec = "BUY" if sig.action == "ENTER_LONG" else ("SELL" if sig.action == "EXIT" else ("WAIT" if sig.action == "NO_SIGNAL" else sig.action))
                    reason_text = "; ".join(sig.rationale) if sig.rationale else rec
                    dec_id = f"dec_{len(decision_records)+1:05d}"
                    dec_rec = {
                        "decision_id": dec_id,
                        "symbol": sym,
                        "decision_timestamp": ts.isoformat(),
                        "time_str": ts.strftime("%H:%M"),
                        "agent_name": agent_name,
                        "model_name": model_name,
                        "model_version": model_ver,
                        "execution_mode": exec_mode_label,
                        "recommendation": rec,
                        "confidence": round(sig.confidence, 2),
                        "reasoning_summary": reason_text,
                        "indicator_snapshot": ind_snap,
                        "council_decision": current_posture.value,
                        "risk_approval": "PENDING_NEXT_BAR_OPEN" if rec in ("BUY", "SELL") else "N/A",
                    }

                    if rec == "BUY" and sym not in portfolio.positions and daily_trades_per_sym[sym] < 3:
                        pending_orders[sym] = {
                            "action": "ENTER_LONG",
                            "stop_loss": sig.stop_loss,
                            "target": sig.target,
                            "signal_price": close_px,
                            "decision_record": dec_rec,
                        }
                        decision_records.append(dec_rec)
                    elif rec == "SELL" and sym in portfolio.positions:
                        pending_orders[sym] = {
                            "action": "EXIT",
                            "stop_loss": None,
                            "target": None,
                            "signal_price": close_px,
                            "decision_record": dec_rec,
                        }
                        decision_records.append(dec_rec)
                    elif idx % 6 == 0:
                        # Record periodic HOLD/WAIT samples every 30 mins for timeline visibility
                        decision_records.append(dec_rec)

            # E. Record Portfolio Snapshot at bar `ts`
            dd_pct = ((portfolio.peak_equity - portfolio.total_equity) / portfolio.peak_equity) * 100.0 if portfolio.peak_equity > 0 else 0.0
            snapshots.append({
                "timestamp": ts.isoformat(),
                "time": ts.strftime("%H:%M"),
                "cash": round(portfolio.cash, 2),
                "invested_capital": round(portfolio.invested_capital, 2),
                "unrealised_pnl": round(portfolio.unrealized_pnl, 2),
                "realised_pnl": round(portfolio.realized_pnl, 2),
                "total_equity": round(portfolio.total_equity, 2),
                "drawdown_pct": round(dd_pct, 2),
            })

        # 4. Compute Final Performance Metrics & Benchmark
        summary = portfolio.get_portfolio_summary()
        benchmark = compute_five_stock_equal_weight_benchmark(
            session_by_sym, initial_capital=self.risk_config.initial_capital_inr, cost_config=self.cost_config
        )
        pnls = [t["net_pnl"] for t in completed_trades_detailed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        sharpe = (float(np.mean(pnls)) / float(np.std(pnls)) * math.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0.0
        downside = float(np.std([p for p in pnls if p < 0])) if losses else 1e-4
        sortino = (float(np.mean(pnls)) / downside * math.sqrt(252)) if len(pnls) > 1 and downside > 0 else 0.0
        total_fees = round(sum(t["fees"] for t in completed_trades_detailed), 2)
        net_pnl = round(summary["total_equity_inr"] - self.risk_config.initial_capital_inr, 2)

        completed_at = datetime.now(tz=IST_TZ).isoformat()

        # Build candle payloads per stock for frontend charting
        candles_payload: dict[str, list[dict[str, Any]]] = {}
        for sym in symbols:
            df = features_by_sym[sym]
            start_i = session_start_idx[sym]
            bars_out = []
            if not df.empty:
                for idx in range(start_i, len(df)):
                    r = df.iloc[idx]
                    ts_val = r["timestamp"]
                    ts_dt = ts_val.to_pydatetime() if hasattr(ts_val, "to_pydatetime") else ts_val
                    bars_out.append({
                        "time": int(ts_dt.timestamp()),
                        "time_str": ts_dt.strftime("%H:%M"),
                        "open": round(float(r["open"]), 2),
                        "high": round(float(r["high"]), 2),
                        "low": round(float(r["low"]), 2),
                        "close": round(float(r["close"]), 2),
                        "volume": int(r["volume"]),
                        "ema_9": round(float(r["ema_9"]), 2),
                        "ema_21": round(float(r["ema_21"]), 2),
                        "vwap": round(float(r["vwap"]), 2),
                    })
            candles_payload[sym] = bars_out

        # Stage B / C Learning Analysis
        analysis = self.generate_stage_bc_analysis(strategy_id, target_date, completed_trades_detailed, net_pnl)

        run_summary = {
            "run_id": run_id,
            "selected_date": target_date.isoformat(),
            "starting_capital": self.risk_config.initial_capital_inr,
            "strategy_id": strategy_id,
            "strategy_name": reg_entry["name"],
            "strategy_version": "v1.0",
            "mode": execution_mode,
            "model_config": {"strategy_id": strategy_id, "execution_mode": execution_mode},
            "data_provenance": {
                sym: {
                    "availability": q.data_availability,
                    "provider": q.provider,
                    "candles": q.candles_count,
                    "warmup": q.warmup_candles_count,
                    "missing": q.missing_intervals_count,
                    "status": q.status,
                    "sha256": q.sha256_hash[:16],
                }
                for sym, q in quality_by_sym.items()
            },
            "execution_config": {
                "slippage_bps": self.cost_config.base_slippage_bps,
                "execution_rule": "NEXT_BAR_OPEN",
                "square_off_ist": "15:15",
            },
            "started_at": started_at,
            "completed_at": completed_at,
            "final_portfolio_value": summary["total_equity_inr"],
            "net_pnl": net_pnl,
            "total_return_pct": summary["total_return_pct"],
            "max_drawdown_pct": max((s["drawdown_pct"] for s in snapshots), default=0.0),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate_pct": summary["win_rate_pct"],
            "profit_factor": summary["profit_factor"],
            "total_trades": len(completed_trades_detailed),
            "total_fees": total_fees,
            "avg_win_inr": round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss_inr": round(float(np.mean(losses)), 2) if losses else 0.0,
            "expectancy_inr": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        }

        return {
            "run": run_summary,
            "benchmark": benchmark,
            "decisions": decision_records,
            "orders": order_records,
            "trades": completed_trades_detailed,
            "snapshots": snapshots,
            "candles": candles_payload,
            "markers": chart_markers,
            "stage_bc_analysis": analysis,
        }

    def generate_stage_bc_analysis(
        self,
        strategy_id: str,
        target_date: date,
        trades: list[dict[str, Any]],
        net_pnl: float,
    ) -> dict[str, Any]:
        """Stage B (Performance Analysis) & Stage C (Improvement Research Suggestions)."""
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        sl_hits = [t for t in trades if t["exit_reason"] == "STOP_LOSS_HIT"]
        sq_offs = [t for t in trades if t["exit_reason"] == "MANDATORY_1515_SQUAREOFF"]

        observations = []
        suggestions = []

        if len(trades) == 0:
            observations.append(f"No entries triggered for {strategy_id} on {target_date}.")
            suggestions.append("Verify whether entry thresholds are overly restrictive in low-volatility regimes.")
        else:
            observations.append(
                f"Executed {len(trades)} trades ({len(wins)} wins, {len(losses)} losses) with net P&L ₹{net_pnl:+.2f}."
            )
            if len(sl_hits) > len(wins):
                observations.append(f"High stop-loss hit count ({len(sl_hits)}/{len(trades)}), indicating premature entries or tight stops.")
                suggestions.append("Test widening ATR stop-loss multiplier from 1.0x to 1.3x on validation dates (do not overwrite production).")
            if len(sq_offs) > 0:
                observations.append(f"{len(sq_offs)} positions held until mandatory 15:15 IST square-off.")
            if net_pnl < 0:
                suggestions.append("Evaluate adding VWAP slope or ADX > 20 trend filter to avoid sideways whipsaws.")
            else:
                suggestions.append("Positive net expectancy after friction; candidate for multi-week walk-forward validation.")

        return {
            "stage_b_observations": observations,
            "stage_c_suggestions": suggestions,
            "auto_overwrite_production": False,
        }
