"""Chronological 5-Minute Historical Trading Replay Engine (₹50,000 Virtual Portfolio)."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..agents.contracts import MarketPosture
from ..agents.council import CentralMarketCouncil
from ..agents.market_intelligence import MarketIntelligenceAgent
from ..agents.microstructure_agent import MicrostructureAgent
from ..agents.router import ModelRouter
from ..agents.technical_agent import TechnicalAnalysisAgent
from ..config import (
    IST_TZ,
    LAB_DEFAULT_UNIVERSE,
    LAB_INITIAL_CAPITAL_INR,
    CostConfig,
    LabRiskConfig,
)
from ..data.instruments import INSTRUMENT_MASTER
from ..data.resampler import resample_5m_candles, to_candle5m_compatible
from ..data.schema import Candle5M, DataSourceMode
from ..features.indicators import compute_all_features
from ..portfolio.cost_calculator import calculate_intraday_costs
from ..portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from ..risk.engine import DeterministicRiskEngine
from ..strategies.base import BaseStrategy, StrategySignal
from ..strategies.ema_cross import EMACrossStrategy
from ..strategies.mtf_strategies import (
    CashNoTradeBenchmarkStrategy,
    LowFrequencyTrendFollowingStrategy,
    MTF30mBreakoutVolumeStrategy,
    MTF15mRelativeStrengthLeaderStrategy,
    MTF15mVWAPTrendContinuationStrategy,
    MTF30mEMATrendStrategy,
)
from ..strategies.multi_indicator import MultiIndicatorConfluenceStrategy
from ..strategies.orb_breakout import ORBBreakoutStrategy
from ..strategies.rsi_macd import RSIMACDStrategy
from ..strategies.supertrend import SupertrendStrategy
from ..strategies.volume_breakout import VolumeBreakoutStrategy
from ..strategies.vwap_pullback import VWAPPullbackStrategy
from .data_service import StockSessionQuality, load_session_with_warmup


STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "STRAT-001-VWAP-PULLBACK": {"name": "VWAP Pullback", "cls": VWAPPullbackStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-002-ORB-BREAKOUT": {"name": "Opening Range Breakout", "cls": ORBBreakoutStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-003-EMA-CROSS": {"name": "EMA 9/21 Crossover", "cls": EMACrossStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-004-SUPERTREND": {"name": "Supertrend", "cls": SupertrendStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-005-RSI-MACD": {"name": "RSI & MACD Combination", "cls": RSIMACDStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-006-VOLUME-BREAKOUT": {"name": "Volume Breakout", "cls": VolumeBreakoutStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-007-MULTI-INDICATOR": {"name": "Multi-Indicator Confirmation", "cls": MultiIndicatorConfluenceStrategy, "kind": "STRATEGY", "timeframe_minutes": 5},
    "STRAT-008-15M-RS-LEADER": {"name": "15m Relative Strength Leader", "cls": MTF15mRelativeStrengthLeaderStrategy, "kind": "STRATEGY", "timeframe_minutes": 15},
    "STRAT-009-15M-VWAP-TREND": {"name": "15m VWAP Trend Continuation", "cls": MTF15mVWAPTrendContinuationStrategy, "kind": "STRATEGY", "timeframe_minutes": 15},
    "STRAT-010-30M-EMA-TREND": {"name": "30m EMA Trend Pullback", "cls": MTF30mEMATrendStrategy, "kind": "STRATEGY", "timeframe_minutes": 30},
    "STRAT-011-30M-BREAKOUT-VOL": {"name": "30m Breakout Volume Confirmation", "cls": MTF30mBreakoutVolumeStrategy, "kind": "STRATEGY", "timeframe_minutes": 30},
    "STRAT-012-LOW-FREQ-TREND": {"name": "30m Low-Frequency Trend Following", "cls": LowFrequencyTrendFollowingStrategy, "kind": "STRATEGY", "timeframe_minutes": 30},
    "BENCHMARK-CASH-NO-TRADE": {"name": "Cash / No-Trade Baseline", "cls": CashNoTradeBenchmarkStrategy, "kind": "BENCHMARK", "timeframe_minutes": 30},
    "AGENT-CENTRAL-COUNCIL": {"name": "Central AI Council (8-Agent)", "cls": None, "kind": "AI_AGENT", "timeframe_minutes": 5},
    "AGENT-AUTONOMOUS-TRADER": {"name": "Autonomous Trading Agent", "cls": None, "kind": "AI_AGENT", "timeframe_minutes": 5},
}


def classify_session_regime(candles: list[Candle5M]) -> dict[str, str]:
    """Classify a historical session's market regime from actual price/volatility behavior."""
    if not candles or len(candles) < 5:
        return {"trend_regime": "UNKNOWN", "volatility_regime": "UNKNOWN", "summary": "INSUFFICIENT_DATA"}

    day_open = candles[0].open
    day_close = candles[-1].close
    ret_pct = ((day_close - day_open) / day_open) * 100.0 if day_open > 0 else 0.0
    bar_rets = [
        abs((candles[i].close - candles[i - 1].close) / candles[i - 1].close) * 100.0
        for i in range(1, len(candles))
        if candles[i - 1].close > 0
    ]
    avg_bar_move = float(np.mean(bar_rets)) if bar_rets else 0.0

    trend = "TRENDING" if abs(ret_pct) >= 0.65 else "SIDEWAYS"
    vol = "HIGH_VOLATILITY" if avg_bar_move >= 0.25 else "LOW_VOLATILITY"
    return {
        "trend_regime": trend,
        "volatility_regime": vol,
        "session_return_pct": f"{ret_pct:+.2f}%",
        "summary": f"{trend}_{vol}",
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
        self._session_feature_cache: dict[tuple[str, date, str, int], tuple[list[Candle5M], list[Candle5M], StockSessionQuality, pd.DataFrame]] = {}

    def run_historical_day(
        self,
        target_date: date,
        strategy_id: str = "STRAT-001-VWAP-PULLBACK",
        symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
        data_mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
        execution_mode: Literal["DETERMINISTIC", "AI_AGENT"] = "DETERMINISTIC",
        db: Any | None = None,
        router: ModelRouter | None = None,
        timeframe_minutes: Literal[5, 15, 30] | None = None,
    ) -> dict[str, Any]:
        """Run deterministic or genuine AI chronological replay for a single historical date."""
        if strategy_id not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy/model ID: {strategy_id}")

        started_at = datetime.now(tz=IST_TZ).isoformat()
        run_id = f"run_{target_date.strftime('%Y%m%d')}_{strategy_id}_{uuid.uuid4().hex[:6]}"

        reg_entry = STRATEGY_REGISTRY[strategy_id]
        tf_min = int(timeframe_minutes or reg_entry.get("timeframe_minutes", 5))
        if tf_min not in (5, 15, 30):
            raise ValueError(f"Unsupported timeframe_minutes: {tf_min}")
        strategy_obj: BaseStrategy | None = reg_entry["cls"]() if reg_entry["cls"] is not None else None

        # 1. Load warmup + session candles strictly <= target_date (using in-memory feature cache)
        warmup_by_sym: dict[str, list[Candle5M]] = {}
        session_by_sym: dict[str, list[Candle5M]] = {}
        quality_by_sym: dict[str, StockSessionQuality] = {}
        features_by_sym: dict[str, pd.DataFrame] = {}
        session_start_idx: dict[str, int] = {}

        for sym in symbols:
            cache_key = (sym, target_date, data_mode.value, tf_min)
            if cache_key in self._session_feature_cache:
                w_bars, s_bars, q, f_df = self._session_feature_cache[cache_key]
            else:
                w_bars_5m, s_bars_5m, q = load_session_with_warmup(sym, target_date, warmup_days=5, mode=data_mode)
                if tf_min == 5:
                    w_bars, s_bars = w_bars_5m, s_bars_5m
                else:
                    w_bars = to_candle5m_compatible(resample_5m_candles(w_bars_5m, target_minutes=tf_min, include_incomplete=False))
                    s_bars = to_candle5m_compatible(resample_5m_candles(s_bars_5m, target_minutes=tf_min, include_incomplete=False))
                combined = w_bars + s_bars
                f_df = compute_all_features(combined) if combined else pd.DataFrame()
                self._session_feature_cache[cache_key] = (w_bars, s_bars, q, f_df)

            warmup_by_sym[sym] = w_bars
            session_by_sym[sym] = s_bars
            quality_by_sym[sym] = q
            features_by_sym[sym] = f_df
            session_start_idx[sym] = len(w_bars) if not f_df.empty else 0

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
        active_router = router or ModelRouter()
        council = CentralMarketCouncil(router=active_router)
        total_ai_cost_usd: float = 0.0

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
                prev_closed_len = len(portfolio.closed_trades)
                sq_orders = portfolio.square_off_all_positions(last_prices, timestamp=ts)
                new_closed_trades = portfolio.closed_trades[prev_closed_len:]
                for so, tr in zip(sq_orders, new_closed_trades):
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
                # NOTE: Historical India VIX & FII/DII intraday feeds are unavailable in 5m OHLCV cache;
                # pass neutral 0.0 sentinel only when deterministic council runs, and explicitly tag VIX UNAVAILABLE in snapshot.
                if strategy_obj is None and execution_mode == "AI_AGENT":
                    council_snap = {
                        "council_ts": ts.isoformat(),
                        "avg_chg": round(avg_chg, 4),
                        "adv": adv,
                        "dec": dec,
                    }
                    c_hash = hashlib.sha256(json.dumps(council_snap, sort_keys=True).encode()).hexdigest()
                    c_cached = db.get_cached_ai_response(c_hash) if db is not None else None
                    if c_cached is not None:
                        current_posture = MarketPosture(c_cached["response_text"])
                    else:
                        mi_msg = mi_agent.analyze_market(
                            nifty_change_pct=avg_chg,
                            bank_nifty_change_pct=avg_chg,
                            india_vix=15.0,
                            advances=max(1, adv),
                            declines=max(1, dec),
                            timestamp=ts,
                        )
                        c_msg = council.evaluate_council(market_intel=mi_msg)
                        current_posture = c_msg.posture
                        if db is not None:
                            db.save_cached_ai_response({
                                "input_hash": c_hash,
                                "provider": "google-gemini" if "LLM" in c_msg.model_used else "deterministic_fallback",
                                "model_id": c_msg.model_used,
                                "model_version": "v1.0",
                                "execution_mode": "LLM" if "LLM" in c_msg.model_used else "DETERMINISTIC_FALLBACK",
                                "request_timestamp": ts.isoformat(),
                                "response_timestamp": datetime.now(tz=IST_TZ).isoformat(),
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "actual_cost_usd": 0.0,
                                "response_text": current_posture.value,
                                "input_snapshot": council_snap,
                            })
                else:
                    # Pure deterministic posture from 5-stock breadth
                    if adv >= 4 and avg_chg > 0.3:
                        current_posture = MarketPosture.BUY_BULLISH
                    elif dec >= 4 and avg_chg < -0.3:
                        current_posture = MarketPosture.SELL_BEARISH
                    else:
                        current_posture = MarketPosture.NEUTRAL
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
                        "volume": int(row["volume"]),
                        "vix_status": "UNAVAILABLE",
                        "microstructure_l1_status": "UNAVAILABLE",
                    }

                    if strategy_obj is not None:
                        sig = strategy_obj.evaluate_bar(sym, df, current_bar_idx=idx)
                        agent_name = reg_entry["name"]
                        provider_name = "local_python"
                        model_name = "deterministic_python"
                        model_ver = "v1.0"
                        exec_mode_label = "DETERMINISTIC"
                        ai_tokens_in = 0
                        ai_tokens_out = 0
                        ai_cost = 0.0
                        req_ts = ts.isoformat()
                        resp_ts = ts.isoformat()
                    else:
                        # AI Agent Mode: slice strictly up to idx (zero future leakage)
                        sub_candles = (warmup_by_sym[sym] + session_by_sym[sym])[: idx + 1]
                        tech_msg = tech_agent.analyze_candles(sub_candles[-60:])
                        actual_vol_20m = sum(c.volume for c in sub_candles[-4:])

                        # Deterministic baseline signal (without fabricated L1 bid/ask spreads)
                        if strategy_id == "AGENT-CENTRAL-COUNCIL" and current_posture in (MarketPosture.SELL_BEARISH, MarketPosture.HOLD_HIGH_RISK):
                            det_sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="NO_SIGNAL", confidence=0.7, rationale=[f"Council posture {current_posture.value} blocks new longs"])
                        elif tech_msg.signal == "BULLISH" and tech_msg.confidence >= 0.65 and actual_vol_20m > 0:
                            sl = round(close_px - 1.0 * atr, 2)
                            tp = round(close_px + 2.0 * atr, 2)
                            det_sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="ENTER_LONG", entry_price=close_px, confidence=tech_msg.confidence, stop_loss=sl, target=tp, rationale=tech_msg.evidence[:2])
                        elif tech_msg.signal == "BEARISH" and sym in portfolio.positions:
                            det_sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="EXIT", confidence=tech_msg.confidence, rationale=["Technical bearish reversal"])
                        else:
                            det_sig = StrategySignal(strategy_id=strategy_id, symbol=sym, timestamp=ts, action="HOLD", confidence=0.55, rationale=[f"Tech={tech_msg.signal} (VIX/L1_Microstructure=UNAVAILABLE)"])

                        sig = det_sig
                        agent_name = reg_entry["name"]
                        model_ver = "v1.0"
                        ai_tokens_in = 0
                        ai_tokens_out = 0
                        ai_cost = 0.0
                        req_ts = datetime.now(tz=IST_TZ).isoformat()
                        resp_ts = req_ts

                        if execution_mode == "AI_AGENT":
                            snap_payload = {
                                "strategy_id": strategy_id,
                                "symbol": sym,
                                "timestamp": ts.isoformat(),
                                "indicators": ind_snap,
                                "tech_signal": tech_msg.signal,
                                "council_posture": current_posture.value,
                                "in_position": sym in portfolio.positions,
                            }
                            input_hash = hashlib.sha256(json.dumps(snap_payload, sort_keys=True).encode()).hexdigest()
                            cached = db.get_cached_ai_response(input_hash) if db is not None else None
                            if cached is not None:
                                provider_name = cached["provider"]
                                model_name = cached["model_id"]
                                exec_mode_label = cached["execution_mode"]
                                ai_tokens_in = int(cached["input_tokens"])
                                ai_tokens_out = int(cached["output_tokens"])
                                ai_cost = 0.0  # Cached hit incurs $0 incremental API cost
                            else:
                                prompt = (
                                    f"Evaluate {sym} at {ts.strftime('%H:%M IST')}. "
                                    f"Close={close_px}, VWAP={ind_snap['vwap']}, EMA9={ind_snap['ema_9']}, "
                                    f"EMA21={ind_snap['ema_21']}, RSI={ind_snap['rsi_14']}, Vol20m={actual_vol_20m:.0f}. "
                                    f"Tech={tech_msg.signal}, Council={current_posture.value}. "
                                    f"Note: Historical VIX and L1 Bid/Ask are UNAVAILABLE. "
                                    f"Return concise reasoning for action={det_sig.action}."
                                )
                                inv_res = active_router.invoke_agent(
                                    task_level=1,
                                    agent_name=strategy_id.lower(),
                                    prompt=prompt,
                                    deterministic_fn=lambda: "; ".join(det_sig.rationale),
                                )
                                resp_ts = datetime.now(tz=IST_TZ).isoformat()
                                model_name = inv_res.model_used
                                exec_mode_label = inv_res.execution_mode
                                provider_name = "google-gemini" if inv_res.execution_mode == "LLM" else "deterministic_fallback"
                                ai_tokens_in = inv_res.input_tokens
                                ai_tokens_out = inv_res.output_tokens
                                ai_cost = inv_res.cost_usd
                                total_ai_cost_usd += ai_cost
                                if inv_res.execution_mode == "LLM" and inv_res.text:
                                    sig.rationale = [inv_res.text[:240]]
                                if db is not None:
                                    db.save_cached_ai_response({
                                        "input_hash": input_hash,
                                        "provider": provider_name,
                                        "model_id": model_name,
                                        "model_version": model_ver,
                                        "execution_mode": exec_mode_label,
                                        "request_timestamp": req_ts,
                                        "response_timestamp": resp_ts,
                                        "input_tokens": ai_tokens_in,
                                        "output_tokens": ai_tokens_out,
                                        "actual_cost_usd": ai_cost,
                                        "response_text": "; ".join(sig.rationale),
                                        "input_snapshot": snap_payload,
                                    })
                        else:
                            provider_name = "local_python"
                            model_name = "deterministic_python"
                            exec_mode_label = "DETERMINISTIC"

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
                        "provider": provider_name,
                        "model_name": model_name,
                        "model_version": model_ver,
                        "execution_mode": exec_mode_label,
                        "request_timestamp": req_ts,
                        "response_timestamp": resp_ts,
                        "input_tokens": ai_tokens_in,
                        "output_tokens": ai_tokens_out,
                        "actual_cost_usd": ai_cost,
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
        total_brokerage_inr = round(
            sum(o.cost_breakdown.brokerage_inr for o in portfolio.orders if o.cost_breakdown is not None), 2
        )
        total_statutory_inr = round(max(0.0, total_fees - total_brokerage_inr), 2)
        total_slippage_inr = round(sum(o.get("slippage", 0.0) for o in order_records), 2)
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
            "timeframe_minutes": tf_min,
            "mode": execution_mode,
            "model_config": {"strategy_id": strategy_id, "execution_mode": execution_mode, "timeframe_minutes": tf_min},
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
            "total_brokerage_inr": total_brokerage_inr,
            "total_statutory_inr": total_statutory_inr,
            "total_slippage_inr": total_slippage_inr,
            "ai_api_cost_usd": round(total_ai_cost_usd, 6),
            "avg_holding_minutes": round(float(np.mean([t["holding_minutes"] for t in completed_trades_detailed])), 1) if completed_trades_detailed else 0.0,
            "avg_win_inr": round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss_inr": round(float(np.mean(losses)), 2) if losses else 0.0,
            "expectancy_inr": round(float(np.mean(pnls)), 2) if pnls else 0.0,
            "regime": classify_session_regime(session_by_sym.get(symbols[0], [])) if symbols else {},
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

    def run_date_range(
        self,
        start_date: date,
        end_date: date,
        strategy_id: str = "STRAT-001-VWAP-PULLBACK",
        symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
        data_mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
        execution_mode: Literal["DETERMINISTIC", "AI_AGENT"] = "DETERMINISTIC",
        db: Any | None = None,
        router: ModelRouter | None = None,
        eligible_dates: list[date] | None = None,
        excluded_dates_info: list[dict[str, Any]] | None = None,
        timeframe_minutes: Literal[5, 15, 30] | None = None,
    ) -> dict[str, Any]:
        """Run multi-day chronological evaluation across explicitly eligible dates (or all NSE trading days in range)."""
        from ..config import is_nse_trading_day

        if eligible_dates is not None:
            target_dates = sorted(d for d in eligible_dates if start_date <= d <= end_date and is_nse_trading_day(d))
        else:
            target_dates = []
            curr = start_date
            while curr <= end_date:
                if is_nse_trading_day(curr):
                    target_dates.append(curr)
                curr += timedelta(days=1)

        tf_min = int(timeframe_minutes or STRATEGY_REGISTRY[strategy_id].get("timeframe_minutes", 5))
        daily_runs: list[dict[str, Any]] = []
        cumulative_equity = self.risk_config.initial_capital_inr
        cumulative_benchmark = self.risk_config.initial_capital_inr
        total_trades = 0
        total_fees = 0.0
        total_brokerage = 0.0
        total_statutory = 0.0
        total_slippage = 0.0
        total_ai_cost = 0.0
        all_pnls: list[float] = []
        all_orders: list[dict[str, Any]] = []
        all_trades: list[dict[str, Any]] = []

        for curr in target_dates:
            res = self.run_historical_day(
                target_date=curr,
                strategy_id=strategy_id,
                symbols=symbols,
                data_mode=data_mode,
                execution_mode=execution_mode,
                db=db,
                router=router,
                timeframe_minutes=tf_min,  # type: ignore[arg-type]
            )
            r = res["run"]
            has_bars = any(p["candles"] > 0 for p in r["data_provenance"].values())
            if has_bars:
                cumulative_equity += r["net_pnl"]
                cumulative_benchmark += res["benchmark"]["net_pnl_inr"]
                total_trades += r["total_trades"]
                total_fees += r["total_fees"]
                total_brokerage += r.get("total_brokerage_inr", 0.0)
                total_statutory += r.get("total_statutory_inr", 0.0)
                total_slippage += r.get("total_slippage_inr", 0.0)
                total_ai_cost += r.get("ai_api_cost_usd", 0.0)
                all_pnls.extend(t["net_pnl"] for t in res["trades"])
                all_orders.extend(res["orders"])
                all_trades.extend(res["trades"])
                daily_runs.append({
                    "date": curr.isoformat(),
                    "net_pnl": r["net_pnl"],
                    "return_pct": r["total_return_pct"],
                    "benchmark_return_pct": res["benchmark"]["total_return_pct"],
                    "trades": r["total_trades"],
                    "win_rate_pct": r["win_rate_pct"],
                    "fees": r["total_fees"],
                    "brokerage_inr": r.get("total_brokerage_inr", 0.0),
                    "statutory_inr": r.get("total_statutory_inr", 0.0),
                    "slippage_inr": r.get("total_slippage_inr", 0.0),
                    "cumulative_equity": round(cumulative_equity, 2),
                    "cumulative_benchmark": round(cumulative_benchmark, 2),
                    "regime": r.get("regime", {}).get("summary", "UNKNOWN"),
                })

        total_gross_pnl = 0.0
        total_turnover = 0.0
        all_holding_mins: list[float] = []
        peak_cum = self.risk_config.initial_capital_inr
        max_cum_dd = 0.0

        for d_idx, d_run in enumerate(daily_runs):
            peak_cum = max(peak_cum, d_run["cumulative_equity"])
            dd = ((peak_cum - d_run["cumulative_equity"]) / peak_cum) * 100.0 if peak_cum > 0 else 0.0
            max_cum_dd = max(max_cum_dd, dd)

        wins = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]
        win_rate = round((len(wins) / len(all_pnls)) * 100.0, 2) if all_pnls else 0.0
        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else (10.0 if gross_wins > 0 else 0.0)
        sharpe = round((float(np.mean(all_pnls)) / float(np.std(all_pnls)) * math.sqrt(252)), 2) if len(all_pnls) > 1 and np.std(all_pnls) > 0 else 0.0
        downside = float(np.std(losses)) if len(losses) > 1 else 1e-4
        sortino = round((float(np.mean(all_pnls)) / downside * math.sqrt(252)), 2) if len(all_pnls) > 1 and downside > 0 else 0.0

        net_pnl_total = round(cumulative_equity - self.risk_config.initial_capital_inr, 2)
        gross_pnl_total = round(net_pnl_total + total_fees, 2)
        net_ret_pct = round((net_pnl_total / self.risk_config.initial_capital_inr) * 100.0, 2)
        gross_ret_pct = round((gross_pnl_total / self.risk_config.initial_capital_inr) * 100.0, 2)
        bm_ret_pct = round(((cumulative_benchmark - self.risk_config.initial_capital_inr) / self.risk_config.initial_capital_inr) * 100.0, 2)
        alpha_pct = round(net_ret_pct - bm_ret_pct, 2)
        sufficiency = "VALIDATED_SAMPLE" if (len(daily_runs) >= 30 and total_trades >= 20) else "STATISTICALLY_INSUFFICIENT"
        perf_cat = (
            "CASH_BENCHMARK"
            if STRATEGY_REGISTRY[strategy_id]["kind"] == "BENCHMARK"
            else (
                "DETERMINISTIC_STRATEGY"
                if STRATEGY_REGISTRY[strategy_id]["kind"] == "STRATEGY"
                else ("GENUINE_AI" if (execution_mode == "AI_AGENT" and total_ai_cost > 0) else "DETERMINISTIC_FALLBACK")
            )
        )

        return {
            "strategy_id": strategy_id,
            "strategy_name": STRATEGY_REGISTRY[strategy_id]["name"],
            "timeframe_minutes": tf_min,
            "performance_category": perf_cat,
            "statistical_sufficiency": sufficiency,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sessions_evaluated": len(daily_runs),
            "initial_capital_inr": self.risk_config.initial_capital_inr,
            "final_equity_inr": round(cumulative_equity, 2),
            "gross_pnl_inr": gross_pnl_total,
            "gross_return_pct": gross_ret_pct,
            "net_pnl_inr": net_pnl_total,
            "total_return_pct": net_ret_pct,
            "benchmark_final_inr": round(cumulative_benchmark, 2),
            "benchmark_return_pct": bm_ret_pct,
            "cash_only_return_pct": 0.0,
            "alpha_vs_buy_and_hold_pct": alpha_pct,
            "max_drawdown_pct": round(max_cum_dd, 2),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "profit_factor": profit_factor,
            "expectancy_inr": round(float(np.mean(all_pnls)), 2) if all_pnls else 0.0,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "total_fees_inr": round(total_fees, 2),
            "total_brokerage_inr": round(total_brokerage, 2),
            "total_statutory_inr": round(total_statutory, 2),
            "total_slippage_inr": round(total_slippage, 2),
            "total_ai_cost_usd": round(total_ai_cost, 6),
            "slippage_bps": self.cost_config.base_slippage_bps,
            "orders": all_orders,
            "trades": all_trades,
            "daily_runs": daily_runs,
        }

    def run_slippage_sensitivity(
        self,
        start_date: date,
        end_date: date,
        strategy_id: str = "STRAT-001-VWAP-PULLBACK",
        slippage_bps_list: tuple[float, ...] = (5.0, 10.0, 20.0),
        symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
        data_mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
        eligible_dates: list[date] | None = None,
        timeframe_minutes: Literal[5, 15, 30] | None = None,
    ) -> list[dict[str, Any]]:
        """Run multi-day backtest across configured slippage bps assumptions without double-counting."""
        from dataclasses import replace

        results = []
        orig_cost = self.cost_config
        try:
            for bps in slippage_bps_list:
                self.cost_config = replace(orig_cost, base_slippage_bps=bps)
                res = self.run_date_range(
                    start_date,
                    end_date,
                    strategy_id=strategy_id,
                    symbols=symbols,
                    data_mode=data_mode,
                    eligible_dates=eligible_dates,
                    timeframe_minutes=timeframe_minutes,
                )
                results.append({
                    "slippage_bps": bps,
                    "gross_return_pct": res["gross_return_pct"],
                    "net_return_pct": res["total_return_pct"],
                    "net_pnl_inr": res["net_pnl_inr"],
                    "total_fees_inr": res["total_fees_inr"],
                    "total_brokerage_inr": res["total_brokerage_inr"],
                    "total_statutory_inr": res["total_statutory_inr"],
                    "total_slippage_inr": res["total_slippage_inr"],
                    "total_trades": res["total_trades"],
                    "win_rate_pct": res["win_rate_pct"],
                })
        finally:
            self.cost_config = orig_cost
        return results

    def run_walk_forward(
        self,
        ordered_dates: list[date],
        strategy_id: str = "STRAT-001-VWAP-PULLBACK",
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
        data_mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
    ) -> dict[str, Any]:
        """Evaluate strategy across chronological Train, Validation, and Unseen Test date splits."""
        n = len(ordered_dates)
        if n < 3:
            raise ValueError("Need at least 3 trading dates for Train/Validation/Unseen splits")
        t_end = max(1, int(n * train_ratio))
        v_end = max(t_end + 1, int(n * (train_ratio + val_ratio)))
        train_dates = ordered_dates[:t_end]
        val_dates = ordered_dates[t_end:v_end]
        test_dates = ordered_dates[v_end:] or [ordered_dates[-1]]

        return {
            "strategy_id": strategy_id,
            "train": self.run_date_range(train_dates[0], train_dates[-1], strategy_id, symbols, data_mode, eligible_dates=train_dates),
            "validation": self.run_date_range(val_dates[0], val_dates[-1], strategy_id, symbols, data_mode, eligible_dates=val_dates),
            "unseen_test": self.run_date_range(test_dates[0], test_dates[-1], strategy_id, symbols, data_mode, eligible_dates=test_dates),
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
