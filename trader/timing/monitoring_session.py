"""CYCLE-010 Phase 3 — Continuous Bar-by-Bar Intraday Candle Monitoring Session.

Executes a strictly causal, event-driven intraday session from 09:15 IST to 15:30 IST:
1. Pre-open baseline (< 09:15 IST): loads prior-day warmup bars, Central Directional
   Hypothesis (UNCALIBRATED), and Human Authority Instruction.
2. Chronological bar-by-bar loop for every newly completed candle (ts_open_ist, ts_close_ist):
   - Step A: Fills any PendingOrder from bar t-1 close at bar t open (₹0.05 tick rounded)
     under atomic 7-point MandatoryExecutionGate protection + gap-invalidation check.
   - Step B: Evaluates intrabar OHLC stops & targets (conservative stop-first precedence)
     and updates DeterministicRiskEngine MTM drawdown circuit breaker.
   - Step C: Enforces 15:05 IST ACTIVE->REDUCING wind-down and 15:15 IST mandatory square-off.
   - Step D: Fetches strictly causal evidence (ts_close_ist <= bar_t.ts_close_ist), invokes
     stateful CandleTimingAgent.evaluate_bar_lifecycle() (all 9 canonical calls), ratchets
     trailing stops on HOLD, validates candidates through MandatoryExecutionGate, and queues
     approved orders for bar t+1 open.
3. Phase 2 Post-Session Outcome Review: runs evaluate_session_outcomes() ONLY after the
   session completes, computing forward MFE/MAE, missed moves, and saved losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from trader.authority import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    MandatoryExecutionGate,
    PermissionType,
)
from trader.config import (
    DEFAULT_INITIAL_CAPITAL_INR,
    INTRADAY_MANDATORY_SQUAREOFF_TIME,
    INTRADAY_REDUCING_WIND_DOWN_TIME,
    IST_TZ,
)
from trader.data.market_feed import BaseMarketFeed, TimeframeLiteral
from trader.evaluation.outcome_evaluator import SessionOutcomeReport, evaluate_session_outcomes
from trader.intelligence.chart_overlays import (
    StrategyProfileLiteral,
    compute_chart_overlays,
    infer_auto_market_direction,
)
from trader.intelligence.data_contract import ProvenanceTier
from trader.portfolio.cost_calculator import calculate_intraday_costs
from trader.portfolio.ledger import IntradayPaperPortfolio, round_to_nse_tick
from trader.risk.engine import DeterministicRiskEngine, RiskState
from trader.timing.candle_timing_agent import (
    ActiveSetupContext,
    AgentCall,
    CandleTimingAgent,
    PositionContext,
    TimingAssessment,
)
from trader.timing.central_directional_agent import (
    CONFIDENCE_CALIBRATION_STATUS,
    DirectionalBias,
    DirectionalHypothesis,
)
from trader.timing.paper_feed_session import evidence_to_agent_df


@dataclass(frozen=True)
class PendingOrder:
    """Order queued at bar t close to execute causally at bar t+1 open."""

    symbol: str
    action_type: ActionType
    requested_quantity: int
    signal_bar_close_ist: datetime
    stop_loss: float | None
    target: float | None
    candle_assessment: TimingAssessment | None
    reason: str
    reservation_order_id: str | None = None


@dataclass(frozen=True)
class BarDecisionRecord:
    """Immutable bar-by-bar monitoring record persisted for 100% of completed session bars."""

    bar_index: int
    ts_open_ist: str
    ts_close_ist: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    raw_agent_call: str
    effective_call: str
    intended_direction: str
    position_side_after: str | None
    position_qty_after: int
    stop_loss_after: float | None
    target_after: float | None
    risk_state: str
    gate_status: str | None
    rejection_reasons: tuple[str, ...]
    detected_patterns: tuple[str, ...]
    active_scenarios: tuple[str, ...]
    cash_inr: float
    total_equity_inr: float
    realized_pnl_inr: float
    unrealized_pnl_inr: float
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar_index": self.bar_index,
            "ts_open_ist": self.ts_open_ist,
            "ts_close_ist": self.ts_close_ist,
            "symbol": self.symbol,
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "volume": self.volume,
            "raw_agent_call": self.raw_agent_call,
            "effective_call": self.effective_call,
            "intended_direction": self.intended_direction,
            "position_side_after": self.position_side_after,
            "position_qty_after": self.position_qty_after,
            "stop_loss_after": round(self.stop_loss_after, 2) if self.stop_loss_after is not None else None,
            "target_after": round(self.target_after, 2) if self.target_after is not None else None,
            "risk_state": self.risk_state,
            "gate_status": self.gate_status,
            "rejection_reasons": list(self.rejection_reasons),
            "detected_patterns": list(self.detected_patterns),
            "active_scenarios": list(self.active_scenarios),
            "cash_inr": round(self.cash_inr, 2),
            "total_equity_inr": round(self.total_equity_inr, 2),
            "realized_pnl_inr": round(self.realized_pnl_inr, 2),
            "unrealized_pnl_inr": round(self.unrealized_pnl_inr, 2),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class MonitoringSessionResult:
    """Complete output of a full-day continuous intraday candle monitoring session."""

    symbol: str
    session_date: str
    timeframe: str
    provider_name: str
    provenance_status: str
    central_direction: str
    confidence_calibration_status: str
    paper_only: bool
    warmup_bars_count: int
    session_bars_count: int
    decisions: tuple[BarDecisionRecord, ...]
    closed_trades: tuple[dict[str, Any], ...]
    final_cash_inr: float
    final_equity_inr: float
    realized_pnl_inr: float
    total_friction_inr: float
    final_risk_state: str
    ledger_integrity_ok: bool
    risk_audit_chain_ok: bool
    authority_audit_chain_ok: bool
    outcome_report: SessionOutcomeReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "session_date": self.session_date,
            "timeframe": self.timeframe,
            "provider_name": self.provider_name,
            "provenance_status": self.provenance_status,
            "central_direction": self.central_direction,
            "confidence_calibration_status": self.confidence_calibration_status,
            "paper_only": self.paper_only,
            "warmup_bars_count": self.warmup_bars_count,
            "session_bars_count": self.session_bars_count,
            "decisions": [d.to_dict() for d in self.decisions],
            "closed_trades": list(self.closed_trades),
            "final_cash_inr": round(self.final_cash_inr, 2),
            "final_equity_inr": round(self.final_equity_inr, 2),
            "realized_pnl_inr": round(self.realized_pnl_inr, 2),
            "total_friction_inr": round(self.total_friction_inr, 2),
            "final_risk_state": self.final_risk_state,
            "ledger_integrity_ok": self.ledger_integrity_ok,
            "risk_audit_chain_ok": self.risk_audit_chain_ok,
            "authority_audit_chain_ok": self.authority_audit_chain_ok,
            "outcome_report": self.outcome_report.to_dict(),
        }


def _build_default_hypothesis(
    *,
    symbol: str,
    session_date: date,
    direction: DirectionalBias,
    as_of_ist: datetime,
) -> DirectionalHypothesis:
    return DirectionalHypothesis(
        symbol=symbol,
        direction=direction,
        confidence=0.70 if direction != DirectionalBias.NEUTRAL else 0.30,
        prob_next_day_up=0.65 if direction == DirectionalBias.BULLISH else (0.35 if direction == DirectionalBias.BEARISH else 0.50),
        expected_next_day_return_pct=1.0 if direction == DirectionalBias.BULLISH else (-1.0 if direction == DirectionalBias.BEARISH else 0.0),
        news_sentiment_score=0.15 if direction == DirectionalBias.BULLISH else (-0.15 if direction == DirectionalBias.BEARISH else 0.0),
        market_regime="USER_SELECTED_SESSION",
        observation_date=session_date,
        as_of_ist=as_of_ist,
        rationale=(f"Pre-open directional bias set to {direction.value}",),
    )


def evaluate_causal_5m_mtf_trend(causal_df: pd.DataFrame) -> Literal["BULLISH", "BEARISH", "NEUTRAL"]:
    """Causally aggregate completed 1m bars into 5m buckets and return 5m EMA trend bias.

    Strictly uses only completed bars already present in `causal_df` (zero lookahead).
    Returns "NEUTRAL" until at least 5 completed 5m bars exist.
    """
    if causal_df is None or causal_df.empty or "close" not in causal_df.columns:
        return "NEUTRAL"
    if isinstance(causal_df.index, pd.DatetimeIndex):
        ts_idx = causal_df.index
    elif "ts_open_ist" in causal_df.columns:
        ts_idx = pd.DatetimeIndex(pd.to_datetime(causal_df["ts_open_ist"]))
    elif "timestamp" in causal_df.columns:
        ts_idx = pd.DatetimeIndex(pd.to_datetime(causal_df["timestamp"]))
    else:
        return "NEUTRAL"

    closes = pd.Series(causal_df["close"].astype(float).values, index=ts_idx)
    bars_5m = closes.resample("5min", origin="start_day", offset="15min").last().dropna()
    if len(bars_5m) < 5:
        return "NEUTRAL"
    ema9 = bars_5m.ewm(span=9, adjust=False).mean().iloc[-1]
    last_close = float(bars_5m.iloc[-1])
    if len(bars_5m) >= 21:
        ema21 = bars_5m.ewm(span=21, adjust=False).mean().iloc[-1]
        if last_close >= ema9 and ema9 >= ema21:
            return "BULLISH"
        if last_close <= ema9 and ema9 <= ema21:
            return "BEARISH"
        return "NEUTRAL"
    if last_close > ema9:
        return "BULLISH"
    if last_close < ema9:
        return "BEARISH"
    return "NEUTRAL"


def compute_chandelier_trailing_stop(
    *,
    side: Literal["LONG", "SHORT"],
    entry_price: float,
    current_close: float,
    current_stop_loss: float | None,
    highest_price: float,
    lowest_price: float,
    atr_14: float,
    initial_risk_distance: float,
    round_trip_fee_per_share: float = 0.0,
    chandelier_atr_mult: float = 2.2,
    breakeven_trigger_r: float = 1.2,
) -> tuple[float | None, bool]:
    """Compute monotonic Chandelier ATR trailing stop with fee-aware breakeven lock.

    Returns (new_stop_loss, breakeven_locked).
    - Never widens risk (strictly monotonic ratchet toward profit).
    - Locks in entry +/- (round_trip_fee_per_share + 0.05) once favorable excursion >= breakeven_trigger_r * R.
    - Clamps strictly inside current_close by at least 1 NSE tick (₹0.05) so it never self-triggers above market.
    """
    if atr_14 <= 0.0 and initial_risk_distance <= 0.0:
        return current_stop_loss, False

    breakeven_locked = False
    candidates: list[float] = []
    if current_stop_loss is not None:
        candidates.append(float(current_stop_loss))

    if side == "LONG":
        if initial_risk_distance > 0.0 and highest_price >= entry_price + breakeven_trigger_r * initial_risk_distance:
            be_floor = round_to_nse_tick(entry_price + max(0.0, round_trip_fee_per_share) + 0.05)
            candidates.append(be_floor)
            breakeven_locked = True
        if atr_14 > 0.0:
            ch_stop = round_to_nse_tick(highest_price - chandelier_atr_mult * atr_14)
            if ch_stop > 0.0:
                candidates.append(ch_stop)
        if not candidates:
            return None, breakeven_locked
        best = max(candidates)
        max_valid = round_to_nse_tick(max(0.05, current_close - 0.05))
        if current_stop_loss is not None:
            best = max(float(current_stop_loss), min(best, max_valid))
        else:
            best = min(best, max_valid)
        return round_to_nse_tick(best), breakeven_locked

    # SHORT
    if initial_risk_distance > 0.0 and lowest_price <= entry_price - breakeven_trigger_r * initial_risk_distance:
        be_cap = round_to_nse_tick(max(0.05, entry_price - max(0.0, round_trip_fee_per_share) - 0.05))
        candidates.append(be_cap)
        breakeven_locked = True
    if atr_14 > 0.0:
        ch_stop = round_to_nse_tick(lowest_price + chandelier_atr_mult * atr_14)
        candidates.append(ch_stop)
    if not candidates:
        return None, breakeven_locked
    best = min(candidates)
    min_valid = round_to_nse_tick(current_close + 0.05)
    if current_stop_loss is not None:
        best = min(float(current_stop_loss), max(best, min_valid))
    else:
        best = max(best, min_valid)
    return round_to_nse_tick(best), breakeven_locked


class IntradayCandleMonitoringSession:
    """Continuous bar-by-bar intraday candle monitoring & paper-trading session engine."""

    def __init__(
        self,
        feed: BaseMarketFeed,
        *,
        timing_agent: CandleTimingAgent | None = None,
        risk_engine: DeterministicRiskEngine | None = None,
        portfolio: IntradayPaperPortfolio | None = None,
        execution_gate: MandatoryExecutionGate | None = None,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL_INR,
    ) -> None:
        self.feed = feed
        self.timing_agent = timing_agent or CandleTimingAgent()
        self.risk_engine = risk_engine or DeterministicRiskEngine()
        self.portfolio = portfolio or IntradayPaperPortfolio(initial_capital=initial_capital)
        self.execution_gate = execution_gate or MandatoryExecutionGate(risk_engine=self.risk_engine)

    def run_session(
        self,
        *,
        symbol: str,
        session_date: date,
        timeframe: TimeframeLiteral = "1h",
        direction: DirectionalBias = DirectionalBias.BULLISH,
        hypothesis: DirectionalHypothesis | None = None,
        human_permission: PermissionType = PermissionType.ALLOW_LONG,
        human_instruction: AuthorityInstruction | None = None,
        mid_session_human_instructions: Mapping[datetime, AuthorityInstruction] | None = None,
        requested_quantity: int = 10,
        warmup_lookback_bars: int = 100,
        min_reward_to_friction_ratio: float | None = None,
        cooldown_bars_after_exit: int | None = None,
        max_entries_per_session: int | None = None,
        require_mtf_trend_alignment: bool | None = None,
        min_winner_rr_multiple: float | None = None,
        enable_chandelier_trailing: bool | None = None,
        chandelier_atr_mult: float = 2.2,
        breakeven_trigger_r: float = 1.2,
        min_hold_bars_before_agent_exit: int | None = None,
        strategy_profile: StrategyProfileLiteral = "BALANCED_ALL",
        min_adx_for_entry: float | None = None,
        block_vwap_2sd_chase: bool | None = None,
        auto_direction: bool = False,
    ) -> MonitoringSessionResult:
        """Run a full causal intraday monitoring session from 09:15 IST to 15:30 IST."""
        sym_u = symbol.strip().upper()
        session_open_ist = datetime.combine(session_date, time(9, 15), tzinfo=IST_TZ)
        session_close_ist = datetime.combine(session_date, time(15, 30), tzinfo=IST_TZ)

        # CYCLE-012..018 timeframe-aware entry & trailing-exit defaults (active by default on 1m)
        eff_min_rtf = (
            float(min_reward_to_friction_ratio)
            if min_reward_to_friction_ratio is not None
            else (3.0 if timeframe == "1m" else 0.0)
        )
        eff_cooldown_bars = (
            int(cooldown_bars_after_exit)
            if cooldown_bars_after_exit is not None
            else (5 if timeframe == "1m" else 0)
        )
        eff_max_entries = (
            max_entries_per_session
            if max_entries_per_session is not None
            else (4 if timeframe == "1m" else None)
        )
        eff_require_mtf = (
            bool(require_mtf_trend_alignment)
            if require_mtf_trend_alignment is not None
            else (timeframe == "1m")
        )
        eff_min_rr = (
            float(min_winner_rr_multiple)
            if min_winner_rr_multiple is not None
            else (2.5 if timeframe == "1m" else 0.0)
        )
        eff_chandelier = (
            bool(enable_chandelier_trailing)
            if enable_chandelier_trailing is not None
            else (timeframe == "1m")
        )
        eff_min_hold_bars = (
            int(min_hold_bars_before_agent_exit)
            if min_hold_bars_before_agent_exit is not None
            else (6 if timeframe == "1m" else 0)
        )
        eff_min_adx = (
            float(min_adx_for_entry)
            if min_adx_for_entry is not None
            else (18.0 if timeframe == "1m" else 0.0)
        )
        eff_block_vwap_chase = (
            bool(block_vwap_2sd_chase)
            if block_vwap_2sd_chase is not None
            else (timeframe == "1m")
        )

        self.portfolio.reset_daily_session(session_date)
        self.risk_engine.reset_daily_session(session_date)

        # 1. Pre-open warmup strictly before 09:15 IST (zero session lookahead)
        warmup_ev = self.feed.get_bars(sym_u, timeframe, session_open_ist, lookback_bars=warmup_lookback_bars)
        warmup_count = warmup_ev.bar_count

        # 2. Load session bar schedule strictly for iteration (agent ONLY sees causal slices via feed.get_bars)
        session_bar_budget = 400 if timeframe == "1m" else 100
        full_day_ev = self.feed.get_bars(
            sym_u, timeframe, session_close_ist, lookback_bars=warmup_lookback_bars + session_bar_budget
        )
        if full_day_ev.provenance_status == ProvenanceTier.UNAVAILABLE or full_day_ev.frame.empty:
            raise ValueError(f"No historical {timeframe} bars available for {sym_u} on {session_date.isoformat()}")

        full_df = full_day_ev.frame.copy()
        bar_dates = pd.to_datetime(full_df["ts_open_ist"]).dt.tz_convert(IST_TZ).dt.date
        session_bars = full_df.loc[bar_dates == session_date].reset_index(drop=True)
        if session_bars.empty:
            raise ValueError(f"No {timeframe} session bars found for {sym_u} on {session_date.isoformat()}")

        # 3. Register pre-open Central Directional Hypothesis & Human Authority Instruction
        hyp = hypothesis or _build_default_hypothesis(
            symbol=sym_u,
            session_date=session_date,
            direction=direction,
            as_of_ist=session_open_ist,
        )
        self.execution_gate.set_central_instruction(hyp.to_authority_instruction(expires_after_hours=8.0))

        h_instr = human_instruction or AuthorityInstruction(
            instruction_id=f"HUMAN-{sym_u}-{session_date.isoformat()}",
            authority=AuthoritySource.HUMAN,
            scope=sym_u,
            permission=human_permission,
            valid_from=session_open_ist - timedelta(minutes=15),
            expires_at=session_close_ist + timedelta(minutes=15),
            reason=f"User session authority: {human_permission.value}",
        )
        self.execution_gate.set_human_instruction(h_instr)

        sorted_mid_updates = sorted(
            [(ts.astimezone(IST_TZ), instr) for ts, instr in (mid_session_human_instructions or {}).items()],
            key=lambda x: x[0],
        )
        mid_update_idx = 0

        pending_order: PendingOrder | None = None
        current_human_permission: PermissionType = h_instr.permission
        entries_opened_this_session = 0
        last_exit_bar_index: int | None = None
        active_initial_risk: float = 0.0
        active_setup: ActiveSetupContext | None = None
        decisions: list[BarDecisionRecord] = []
        candidate_levels: list[tuple[float | None, float | None]] = []
        bars_held = 0

        # 4. Chronological Bar-by-Bar Monitoring Loop (09:15 -> 15:30 IST)
        for bar_idx, row in session_bars.iterrows():
            ts_open = pd.Timestamp(row["ts_open_ist"]).to_pydatetime().astimezone(IST_TZ)
            ts_close = pd.Timestamp(row["ts_close_ist"]).to_pydatetime().astimezone(IST_TZ)
            o_px = float(row["open"])
            h_px = float(row["high"])
            l_px = float(row["low"])
            c_px = float(row["close"])
            vol = float(row["volume"])
            bar_events_notes: list[str] = []

            # Apply any scheduled mid-session human instruction updates up to ts_open
            while mid_update_idx < len(sorted_mid_updates) and sorted_mid_updates[mid_update_idx][0] <= ts_open:
                mid_instr = sorted_mid_updates[mid_update_idx][1]
                reg_res = self.execution_gate.set_human_instruction(mid_instr, current_time=ts_open)
                if reg_res.accepted:
                    current_human_permission = mid_instr.permission
                    if mid_instr.permission == PermissionType.STOP_ALL:
                        if pending_order is not None and pending_order.reservation_order_id:
                            self.execution_gate.release_cash_reservation(pending_order.reservation_order_id)
                        pending_order = None
                        self.execution_gate.invoke_safe_position_management(
                            symbol=sym_u,
                            portfolio=self.portfolio,
                            current_time=ts_open,
                        )
                        bar_events_notes.append(f"HUMAN_STOP_ALL_SAFE_MANAGEMENT:{mid_instr.instruction_id}")
                else:
                    bar_events_notes.append(f"HUMAN_INSTRUCTION_REJECTED:{reg_res.reason}")
                mid_update_idx += 1

            # STEP A: At bar_t.ts_open_ist, execute any PendingOrder queued at bar_{t-1}.ts_close_ist
            if pending_order is not None:
                fill_px = round_to_nse_tick(o_px)
                sl_tick = round_to_nse_tick(pending_order.stop_loss) if pending_order.stop_loss is not None else None
                tp_tick = round_to_nse_tick(pending_order.target) if pending_order.target is not None else None

                gap_cancel_reason: str | None = None
                if pending_order.action_type == ActionType.ENTRY_LONG:
                    if sl_tick is not None and fill_px <= sl_tick:
                        gap_cancel_reason = f"PENDING_ORDER_CANCELLED_GAP_STOP:open={fill_px:.2f}<=sl={sl_tick:.2f}"
                    elif tp_tick is not None and fill_px >= tp_tick:
                        gap_cancel_reason = f"PENDING_ORDER_CANCELLED_GAP_TARGET:open={fill_px:.2f}>=tp={tp_tick:.2f}"
                elif pending_order.action_type == ActionType.ENTRY_SHORT:
                    if sl_tick is not None and fill_px >= sl_tick:
                        gap_cancel_reason = f"PENDING_ORDER_CANCELLED_GAP_STOP:open={fill_px:.2f}>=sl={sl_tick:.2f}"
                    elif tp_tick is not None and fill_px <= tp_tick:
                        gap_cancel_reason = f"PENDING_ORDER_CANCELLED_GAP_TARGET:open={fill_px:.2f}<=tp={tp_tick:.2f}"

                if gap_cancel_reason is not None:
                    if pending_order.reservation_order_id:
                        self.execution_gate.release_cash_reservation(pending_order.reservation_order_id)
                    bar_events_notes.append(gap_cancel_reason)
                else:
                    gate_exec, executed = self.execution_gate.execute_with_gate_protection(
                        symbol=sym_u,
                        action_type=pending_order.action_type,
                        current_time=ts_open,
                        portfolio=self.portfolio,
                        requested_quantity=pending_order.requested_quantity,
                        price=fill_px,
                        candle_assessment=pending_order.candle_assessment,
                        stop_loss_price=sl_tick,
                        target_price=tp_tick,
                        reservation_order_id=pending_order.reservation_order_id,
                    )
                    if executed:
                        bar_events_notes.append(
                            f"PENDING_ORDER_FILLED:{pending_order.action_type.value}@{fill_px:.2f}"
                        )
                        if pending_order.action_type.is_new_entry:
                            bars_held = 0
                            entries_opened_this_session += 1
                            active_initial_risk = abs(fill_px - sl_tick) if sl_tick is not None else max(0.05, fill_px * 0.005)
                        elif pending_order.action_type.is_exit:
                            last_exit_bar_index = int(bar_idx)
                            active_initial_risk = 0.0
                    else:
                        bar_events_notes.append(
                            f"PENDING_ORDER_REJECTED_AT_OPEN:{','.join(gate_exec.rejection_reasons) or 'UNFILLED'}"
                        )
                pending_order = None

            # Apply any scheduled mid-session human instruction updates up to ts_close
            while mid_update_idx < len(sorted_mid_updates) and sorted_mid_updates[mid_update_idx][0] <= ts_close:
                mid_instr = sorted_mid_updates[mid_update_idx][1]
                reg_res = self.execution_gate.set_human_instruction(mid_instr, current_time=ts_close)
                if reg_res.accepted:
                    current_human_permission = mid_instr.permission
                    if mid_instr.permission == PermissionType.STOP_ALL:
                        self.execution_gate.invoke_safe_position_management(
                            symbol=sym_u,
                            portfolio=self.portfolio,
                            current_time=ts_close,
                        )
                        bar_events_notes.append(f"HUMAN_STOP_ALL_SAFE_MANAGEMENT:{mid_instr.instruction_id}")
                else:
                    bar_events_notes.append(f"HUMAN_INSTRUCTION_REJECTED:{reg_res.reason}")
                mid_update_idx += 1

            # STEP B: Intrabar OHLC stop-loss / target check (conservative stop-first) & MTM DD check
            # Decision 4: RiskState.HALTED implements EXIT_ONLY by default (blocks new entries,
            # continues managing existing positions via SL/TP/approved exits & 15:15 square-off).
            if sym_u in self.portfolio.positions:
                self.portfolio.update_candle_bar(
                    symbol=sym_u,
                    open_price=o_px,
                    high_price=h_px,
                    low_price=l_px,
                    close_price=c_px,
                    timestamp=ts_close,
                    round_ticks=True,
                )
                if sym_u not in self.portfolio.positions:
                    last_exit_bar_index = int(bar_idx)
                if self.risk_engine.check_daily_loss_limit(self.portfolio):
                    bar_events_notes.append("RISK_HALTED_EXIT_ONLY_ACTIVE:drawdown_breach_blocks_new_entries")

            # STEP C: NSE Time Boundaries (15:05 REDUCING, 15:15 Mandatory Square-Off)
            if self.risk_engine.state == RiskState.ACTIVE and ts_close.time() >= INTRADAY_REDUCING_WIND_DOWN_TIME:
                self.risk_engine.state = RiskState.REDUCING

            squared_off_side: str | None = None
            if ts_close.time() >= INTRADAY_MANDATORY_SQUAREOFF_TIME and sym_u in self.portfolio.positions:
                open_pos = self.portfolio.positions[sym_u]
                open_side = open_pos.side
                sq_action = ActionType.EXIT_SHORT if open_side == "SHORT" else ActionType.EXIT_LONG
                _, sq_executed = self.execution_gate.execute_with_gate_protection(
                    symbol=sym_u,
                    action_type=sq_action,
                    current_time=ts_close,
                    portfolio=self.portfolio,
                    requested_quantity=open_pos.quantity,
                    price=round_to_nse_tick(c_px),
                )
                if sq_executed:
                    squared_off_side = open_side
                    last_exit_bar_index = int(bar_idx)
                    bar_events_notes.append(f"MANDATORY_1515_SQUAREOFF_EXECUTED:{sq_action.value}")

            # STEP D: Fetch strictly causal evidence <= ts_close and evaluate stateful CandleTimingAgent
            causal_ev = self.feed.get_bars(sym_u, timeframe, ts_close, lookback_bars=warmup_lookback_bars + session_bar_budget)
            causal_df = evidence_to_agent_df(causal_ev)

            # Decision 3: AUTO is a Central-authorized regime mode; yields immediately if an explicit
            # Central hypothesis was provided, and never overrides Human restrictions.
            if auto_direction and central_hypothesis is None:
                auto_mtf = evaluate_causal_5m_mtf_trend(causal_df)
                auto_ov = compute_chart_overlays(causal_df, session_date)
                inferred = infer_auto_market_direction(auto_mtf, auto_ov, c_px)
                new_bias = (
                    DirectionalBias.BULLISH
                    if inferred == "BULLISH"
                    else (DirectionalBias.BEARISH if inferred == "BEARISH" else DirectionalBias.NEUTRAL)
                )
                if new_bias != hyp.direction:
                    hyp = _build_default_hypothesis(
                        symbol=sym_u,
                        session_date=session_date,
                        direction=new_bias,
                        as_of_ist=ts_close,
                    )
                    self.execution_gate.set_central_instruction(
                        hyp.to_authority_instruction(expires_after_hours=8.0),
                        current_time=ts_close,
                    )
                    bar_events_notes.append(f"AUTO_DIRECTION_SWITCH:{new_bias.value}")

            pos_ctx: PositionContext | None = None
            if sym_u in self.portfolio.positions:
                p_obj = self.portfolio.positions[sym_u]
                bars_held += 1
                pos_ctx = PositionContext(
                    side=p_obj.side,
                    entry_price=p_obj.entry_price,
                    current_stop_loss=(
                        p_obj.stop_loss
                        if p_obj.stop_loss is not None
                        else round_to_nse_tick(p_obj.entry_price * (1.015 if p_obj.side == "SHORT" else 0.985))
                    ),
                    target_price=p_obj.target,
                    highest_price=p_obj.highest_price,
                    lowest_price=p_obj.lowest_price,
                    bars_held=bars_held,
                )

            lifecycle = self.timing_agent.evaluate_bar_lifecycle(
                hypothesis=hyp,
                bars_primary=causal_df,
                primary_timeframe=timeframe,
                as_of_ist=ts_close,
                position=pos_ctx,
                active_setup=active_setup if pos_ctx is None else None,
            )
            active_setup = lifecycle.updated_setup

            raw_call = lifecycle.call
            effective_call = raw_call
            gate_status: str | None = None
            rej_reasons: tuple[str, ...] = ()
            rationale_list = bar_events_notes + list(lifecycle.rationale)

            if squared_off_side is None and eff_chandelier and sym_u in self.portfolio.positions:
                cur_pos = self.portfolio.positions[sym_u]
                atr_val = float(getattr(lifecycle.timing_assessment, "atr_14", 0.0) or 0.0)
                if atr_val <= 0.0 and len(causal_df) >= 2:
                    atr_val = float((causal_df["high"] - causal_df["low"]).tail(14).mean())
                e_side = "SELL" if cur_pos.side == "SHORT" else "BUY"
                x_side = "BUY" if cur_pos.side == "SHORT" else "SELL"
                est_rt_fee_per_sh = (
                    calculate_intraday_costs(round_to_nse_tick(cur_pos.entry_price), cur_pos.quantity, e_side).total_friction_inr
                    + calculate_intraday_costs(round_to_nse_tick(c_px), cur_pos.quantity, x_side).total_friction_inr
                ) / max(1, cur_pos.quantity)
                ch_sl, be_locked = compute_chandelier_trailing_stop(
                    side=cur_pos.side,
                    entry_price=cur_pos.entry_price,
                    current_close=c_px,
                    current_stop_loss=cur_pos.stop_loss,
                    highest_price=cur_pos.highest_price,
                    lowest_price=cur_pos.lowest_price,
                    atr_14=atr_val,
                    initial_risk_distance=active_initial_risk,
                    round_trip_fee_per_share=est_rt_fee_per_sh,
                    chandelier_atr_mult=chandelier_atr_mult,
                    breakeven_trigger_r=breakeven_trigger_r,
                )
                if ch_sl is not None and ch_sl != cur_pos.stop_loss:
                    cur_pos.stop_loss = ch_sl
                    rationale_list.append(
                        f"CHANDELIER_TRAIL_SL:{ch_sl:.2f}" + (" (BREAKEVEN_LOCKED)" if be_locked else "")
                    )

                if (
                    raw_call in (AgentCall.EXIT_LONG, AgentCall.COVER_SHORT)
                    and eff_min_hold_bars > 0
                    and bars_held < eff_min_hold_bars
                    and self.risk_engine.state == RiskState.ACTIVE
                    and current_human_permission not in (PermissionType.EXIT_ONLY, PermissionType.STOP_ALL)
                ):
                    effective_call = AgentCall.HOLD
                    rationale_list.append(
                        f"CHANDELIER_HOLD_SUPPRESSED_EARLY_EXIT:bars_held={bars_held}<{eff_min_hold_bars}"
                    )

            if squared_off_side is not None:
                effective_call = AgentCall.COVER_SHORT if squared_off_side == "SHORT" else AgentCall.EXIT_LONG
                rationale_list.append(f"Mandatory 15:15 IST intraday square-off executed ({effective_call.value}).")
            elif effective_call == AgentCall.HOLD and sym_u in self.portfolio.positions:
                if lifecycle.recommended_trailing_stop is not None:
                    cur_pos = self.portfolio.positions[sym_u]
                    new_sl = round_to_nse_tick(lifecycle.recommended_trailing_stop)
                    if cur_pos.stop_loss is None:
                        cur_pos.stop_loss = new_sl
                    elif cur_pos.side == "SHORT":
                        cur_pos.stop_loss = min(cur_pos.stop_loss, new_sl)
                    else:
                        cur_pos.stop_loss = max(cur_pos.stop_loss, new_sl)
            elif effective_call in (AgentCall.OPEN_LONG, AgentCall.OPEN_SHORT, AgentCall.EXIT_LONG, AgentCall.COVER_SHORT):
                action_map = {
                    AgentCall.OPEN_LONG: ActionType.ENTRY_LONG,
                    AgentCall.OPEN_SHORT: ActionType.ENTRY_SHORT,
                    AgentCall.EXIT_LONG: ActionType.EXIT_LONG,
                    AgentCall.COVER_SHORT: ActionType.EXIT_SHORT,
                }
                act = action_map[effective_call]
                qty_req = (
                    self.portfolio.positions[sym_u].quantity
                    if (act.is_exit and sym_u in self.portfolio.positions)
                    else requested_quantity
                )
                sl_cand = lifecycle.timing_assessment.invalidation_price if act.is_new_entry else None
                tp_cand = lifecycle.timing_assessment.target_price if act.is_new_entry else None
                if act.is_new_entry and eff_min_rr > 0.0 and sl_cand is not None:
                    risk_dist = abs(c_px - sl_cand)
                    if risk_dist > 0.0:
                        min_tp = (
                            round_to_nse_tick(c_px + risk_dist * eff_min_rr)
                            if act == ActionType.ENTRY_LONG
                            else round_to_nse_tick(max(0.05, c_px - risk_dist * eff_min_rr))
                        )
                        if tp_cand is None:
                            tp_cand = min_tp
                        elif act == ActionType.ENTRY_LONG:
                            tp_cand = max(tp_cand, min_tp)
                        else:
                            tp_cand = min(tp_cand, min_tp)

                gate_out = self.execution_gate.evaluate_gate(
                    symbol=sym_u,
                    action_type=act,
                    current_time=ts_close,
                    portfolio=self.portfolio,
                    candle_assessment=lifecycle.timing_assessment if act.is_new_entry else None,
                    requested_quantity=qty_req,
                    entry_price=round_to_nse_tick(c_px),
                    stop_loss_price=sl_cand,
                    target_price=tp_cand,
                    dry_run=True,
                )
                gate_status = gate_out.status
                rej_reasons = gate_out.rejection_reasons
                if gate_out.approved and ts_close.time() < INTRADAY_MANDATORY_SQUAREOFF_TIME:
                    approved_qty = (
                        gate_out.risk_evaluation.approved_quantity
                        if gate_out.risk_evaluation is not None
                        else qty_req
                    )
                    discipline_rejects: list[str] = []
                    if act.is_new_entry:
                        if eff_require_mtf:
                            mtf_trend = evaluate_causal_5m_mtf_trend(causal_df)
                            expected_trend = "BULLISH" if act == ActionType.ENTRY_LONG else "BEARISH"
                            if mtf_trend != expected_trend:
                                discipline_rejects.append(
                                    f"MTF_5M_TREND_MISALIGNED:5m={mtf_trend},req={expected_trend}"
                                )
                        if eff_min_adx > 0.0 or eff_block_vwap_chase or strategy_profile != "BALANCED_ALL":
                            ov = compute_chart_overlays(causal_df, session_date)
                            want_dir = "BULLISH" if act == ActionType.ENTRY_LONG else "BEARISH"
                            if eff_min_adx > 0.0 and ov.adx_14 is not None and ov.adx_14 < eff_min_adx:
                                discipline_rejects.append(
                                    f"ADX_CHOPPY_MARKET:adx={ov.adx_14:.1f}<{eff_min_adx:.1f}"
                                )
                            if eff_block_vwap_chase:
                                if act == ActionType.ENTRY_LONG and ov.vwap_stretch == "OVEREXTENDED_ABOVE_2SD":
                                    discipline_rejects.append("VWAP_2SD_OVEREXTENDED_CHASE:LONG_ABOVE_+2SD")
                                elif act == ActionType.ENTRY_SHORT and ov.vwap_stretch == "OVEREXTENDED_BELOW_2SD":
                                    discipline_rejects.append("VWAP_2SD_OVEREXTENDED_CHASE:SHORT_BELOW_-2SD")
                            if strategy_profile == "ORB_BREAKOUT":
                                req_orb = "ORB_BREAKOUT_UP" if act == ActionType.ENTRY_LONG else "ORB_BREAKOUT_DOWN"
                                if not ov.orb_ready or ov.orb_state != req_orb:
                                    discipline_rejects.append(f"PROFILE_ORB_REQUIRED:{ov.orb_state}!={req_orb}")
                            elif strategy_profile == "SUPERTREND_ADX":
                                if ov.supertrend_direction != want_dir or (ov.adx_14 is not None and ov.adx_14 < 20.0):
                                    discipline_rejects.append(
                                        f"PROFILE_SUPERTREND_ADX_MISMATCH:st={ov.supertrend_direction},adx={ov.adx_14}"
                                    )
                            elif strategy_profile == "LIQUIDITY_REVERSAL":
                                req_sw = "BULLISH_SWEEP" if act == ActionType.ENTRY_LONG else "BEARISH_SWEEP"
                                fvg_ok = ov.recent_fvg is not None and ov.recent_fvg.direction == want_dir
                                if ov.liquidity_sweep != req_sw and not ov.near_key_level and not fvg_ok:
                                    discipline_rejects.append("PROFILE_LIQUIDITY_REVERSAL_NO_LEVEL_OR_SWEEP")
                        if eff_max_entries is not None and entries_opened_this_session >= eff_max_entries:
                            discipline_rejects.append(
                                f"DAILY_ENTRY_CAP_REACHED:{entries_opened_this_session}/{eff_max_entries}"
                            )
                        if (
                            eff_cooldown_bars > 0
                            and last_exit_bar_index is not None
                            and (int(bar_idx) - last_exit_bar_index) < eff_cooldown_bars
                        ):
                            discipline_rejects.append(
                                f"POST_EXIT_COOLDOWN_ACTIVE:{int(bar_idx) - last_exit_bar_index}<{eff_cooldown_bars}"
                            )
                        if eff_min_rtf > 0.0 and tp_cand is not None and approved_qty > 0:
                            entry_side = "BUY" if act == ActionType.ENTRY_LONG else "SELL"
                            exit_side = "SELL" if act == ActionType.ENTRY_LONG else "BUY"
                            est_rt_cost = (
                                calculate_intraday_costs(round_to_nse_tick(c_px), approved_qty, entry_side).total_friction_inr
                                + calculate_intraday_costs(round_to_nse_tick(tp_cand), approved_qty, exit_side).total_friction_inr
                            )
                            expected_gross = abs(tp_cand - c_px) * approved_qty
                            if expected_gross < eff_min_rtf * est_rt_cost:
                                discipline_rejects.append(
                                    f"FRICTION_EDGE_TOO_LOW:exp={expected_gross:.2f}<{eff_min_rtf:.1f}x{est_rt_cost:.2f}"
                                )

                    if discipline_rejects:
                        effective_call = AgentCall.REJECT
                        gate_status = "REJECTED"
                        rej_reasons = tuple(discipline_rejects)
                    else:
                        res_id: str | None = None
                        if act.is_new_entry and approved_qty > 0:
                            res_id = f"PEND-{sym_u}-{int(bar_idx)}"
                            self.execution_gate.reserve_cash_for_pending_order(
                                res_id,
                                sym_u,
                                round_to_nse_tick(c_px) * approved_qty,
                            )
                        pending_order = PendingOrder(
                            symbol=sym_u,
                            action_type=act,
                            requested_quantity=approved_qty,
                            signal_bar_close_ist=ts_close,
                            stop_loss=sl_cand,
                            target=tp_cand,
                            candle_assessment=lifecycle.timing_assessment if act.is_new_entry else None,
                            reason=f"Queued from {raw_call.value} at {ts_close.isoformat()}",
                            reservation_order_id=res_id,
                        )
                else:
                    effective_call = AgentCall.REJECT
                    if not rej_reasons and ts_close.time() >= INTRADAY_MANDATORY_SQUAREOFF_TIME:
                        rej_reasons = ("PAST_1515_MANDATORY_SQUAREOFF_WINDOW",)

            intended_dir = (
                "LONG"
                if hyp.direction == DirectionalBias.BULLISH
                else ("SHORT" if hyp.direction == DirectionalBias.BEARISH else "NEUTRAL")
            )
            pos_after = self.portfolio.positions.get(sym_u)
            cand_sl = lifecycle.timing_assessment.invalidation_price
            cand_tp = lifecycle.timing_assessment.target_price
            candidate_levels.append((cand_sl, cand_tp))

            decisions.append(
                BarDecisionRecord(
                    bar_index=int(bar_idx),
                    ts_open_ist=ts_open.isoformat(),
                    ts_close_ist=ts_close.isoformat(),
                    symbol=sym_u,
                    open=o_px,
                    high=h_px,
                    low=l_px,
                    close=c_px,
                    volume=vol,
                    raw_agent_call=raw_call.value,
                    effective_call=effective_call.value,
                    intended_direction=intended_dir,
                    position_side_after=pos_after.side if pos_after else None,
                    position_qty_after=pos_after.quantity if pos_after else 0,
                    stop_loss_after=pos_after.stop_loss if pos_after else None,
                    target_after=pos_after.target if pos_after else None,
                    risk_state=self.risk_engine.state.value,
                    gate_status=gate_status,
                    rejection_reasons=rej_reasons,
                    detected_patterns=tuple(p.value for p in lifecycle.timing_assessment.detected_patterns),
                    active_scenarios=lifecycle.structure.active_scenarios,
                    cash_inr=self.portfolio.cash,
                    total_equity_inr=self.portfolio.total_equity,
                    realized_pnl_inr=self.portfolio.realized_pnl,
                    unrealized_pnl_inr=self.portfolio.unrealized_pnl,
                    rationale=tuple(rationale_list),
                )
            )

        # Release any lingering pending reservation at end of session
        if pending_order is not None and pending_order.reservation_order_id:
            self.execution_gate.release_cash_reservation(pending_order.reservation_order_id)

        # 5. Phase 2: Post-Session Counterfactual Outcome Review (strictly after session ends)
        eval_inputs = [
            {
                "bar_index": d.bar_index,
                "timestamp_ist": d.ts_close_ist,
                "call": d.effective_call,
                "intended_direction": d.intended_direction,
                "reference_price": d.close,
                "stop_loss": d.stop_loss_after if d.stop_loss_after is not None else cand_sl,
                "target": d.target_after if d.target_after is not None else cand_tp,
                "quantity": d.position_qty_after or requested_quantity,
            }
            for d, (cand_sl, cand_tp) in zip(decisions, candidate_levels)
        ]
        outcome_report = evaluate_session_outcomes(
            symbol=sym_u,
            decision_records=eval_inputs,
            session_bars=session_bars,
            default_eval_quantity=requested_quantity,
        )

        # 6. Verify all 3 cryptographic SHA-256 chains (full prev_hash linkage)
        ledger_ok, _ = self.portfolio.verify_ledger_integrity()
        risk_ok = bool(self.risk_engine.verify_audit_chain())
        auth_ok = self.execution_gate.verify_audit_chain()
        total_friction = sum(float(t.get("total_friction_inr", 0.0)) for t in self.portfolio.closed_trades)

        return MonitoringSessionResult(
            symbol=sym_u,
            session_date=session_date.isoformat(),
            timeframe=timeframe,
            provider_name=self.feed.provider_name,
            provenance_status=full_day_ev.provenance_status.value,
            central_direction=hyp.direction.value,
            confidence_calibration_status=CONFIDENCE_CALIBRATION_STATUS,
            paper_only=True,
            warmup_bars_count=warmup_count,
            session_bars_count=len(session_bars),
            decisions=tuple(decisions),
            closed_trades=tuple(self.portfolio.closed_trades),
            final_cash_inr=self.portfolio.cash,
            final_equity_inr=self.portfolio.total_equity,
            realized_pnl_inr=self.portfolio.realized_pnl,
            total_friction_inr=round(total_friction, 2),
            final_risk_state=self.risk_engine.state.value,
            ledger_integrity_ok=ledger_ok,
            risk_audit_chain_ok=risk_ok,
            authority_audit_chain_ok=auth_ok,
            outcome_report=outcome_report,
        )
