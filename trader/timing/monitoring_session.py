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
from trader.intelligence.data_contract import ProvenanceTier
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
    ) -> MonitoringSessionResult:
        """Run a full causal intraday monitoring session from 09:15 IST to 15:30 IST."""
        sym_u = symbol.strip().upper()
        session_open_ist = datetime.combine(session_date, time(9, 15), tzinfo=IST_TZ)
        session_close_ist = datetime.combine(session_date, time(15, 30), tzinfo=IST_TZ)

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
                self.execution_gate.set_human_instruction(sorted_mid_updates[mid_update_idx][1])
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
                    )
                    if executed:
                        bar_events_notes.append(
                            f"PENDING_ORDER_FILLED:{pending_order.action_type.value}@{fill_px:.2f}"
                        )
                        if pending_order.action_type.is_new_entry:
                            bars_held = 0
                    else:
                        bar_events_notes.append(
                            f"PENDING_ORDER_REJECTED_AT_OPEN:{','.join(gate_exec.rejection_reasons) or 'UNFILLED'}"
                        )
                pending_order = None

            # Apply any scheduled mid-session human instruction updates up to ts_close
            while mid_update_idx < len(sorted_mid_updates) and sorted_mid_updates[mid_update_idx][0] <= ts_close:
                self.execution_gate.set_human_instruction(sorted_mid_updates[mid_update_idx][1])
                mid_update_idx += 1

            # STEP B: Intrabar OHLC stop-loss / target check (conservative stop-first) & MTM DD check
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
                self.risk_engine.check_daily_loss_limit(self.portfolio)

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
                    bar_events_notes.append(f"MANDATORY_1515_SQUAREOFF_EXECUTED:{sq_action.value}")

            # STEP D: Fetch strictly causal evidence <= ts_close and evaluate stateful CandleTimingAgent
            causal_ev = self.feed.get_bars(sym_u, timeframe, ts_close, lookback_bars=warmup_lookback_bars + 100)
            causal_df = evidence_to_agent_df(causal_ev)

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

            if squared_off_side is not None:
                effective_call = AgentCall.COVER_SHORT if squared_off_side == "SHORT" else AgentCall.EXIT_LONG
                rationale_list.append(f"Mandatory 15:15 IST intraday square-off executed ({effective_call.value}).")
            elif raw_call == AgentCall.HOLD and sym_u in self.portfolio.positions:
                if lifecycle.recommended_trailing_stop is not None:
                    cur_pos = self.portfolio.positions[sym_u]
                    new_sl = round_to_nse_tick(lifecycle.recommended_trailing_stop)
                    if cur_pos.stop_loss is None:
                        cur_pos.stop_loss = new_sl
                    elif cur_pos.side == "SHORT":
                        cur_pos.stop_loss = min(cur_pos.stop_loss, new_sl)
                    else:
                        cur_pos.stop_loss = max(cur_pos.stop_loss, new_sl)
            elif raw_call in (AgentCall.OPEN_LONG, AgentCall.OPEN_SHORT, AgentCall.EXIT_LONG, AgentCall.COVER_SHORT):
                action_map = {
                    AgentCall.OPEN_LONG: ActionType.ENTRY_LONG,
                    AgentCall.OPEN_SHORT: ActionType.ENTRY_SHORT,
                    AgentCall.EXIT_LONG: ActionType.EXIT_LONG,
                    AgentCall.COVER_SHORT: ActionType.EXIT_SHORT,
                }
                act = action_map[raw_call]
                qty_req = (
                    self.portfolio.positions[sym_u].quantity
                    if (act.is_exit and sym_u in self.portfolio.positions)
                    else requested_quantity
                )
                sl_cand = lifecycle.timing_assessment.invalidation_price if act.is_new_entry else None
                tp_cand = lifecycle.timing_assessment.target_price if act.is_new_entry else None

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
                )
                gate_status = gate_out.status
                rej_reasons = gate_out.rejection_reasons
                if gate_out.approved and ts_close.time() < INTRADAY_MANDATORY_SQUAREOFF_TIME:
                    approved_qty = (
                        gate_out.risk_evaluation.approved_quantity
                        if gate_out.risk_evaluation is not None
                        else qty_req
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

        # 6. Verify all 3 cryptographic SHA-256 chains
        ledger_ok, _ = self.portfolio.verify_ledger_integrity()
        risk_ok = bool(self.risk_engine.verify_audit_chain())
        auth_trail = self.execution_gate.audit_trail
        auth_ok = all(rec.verify_hash() for rec in auth_trail)
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
