"""Mandatory Central Agent & Human Authority Execution Gate.

Enforces strict authority hierarchy:
1. Human instructions (highest priority for restrictions; cannot override Risk Engine)
2. Central Agent trading instructions & permissions
3. Deterministic Risk Engine controls (absolute veto on capital/drawdown limits)
4. Independent Candle Timing Agent (advisory only; never self-authorizing)

Revalidates all 7 mandatory gates atomically under lock immediately prior to
every paper execution to prevent stale permissions and race conditions.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Any, Callable, Literal, Sequence

from ..config import IST_TZ
from ..portfolio.ledger import IntradayPaperPortfolio
from ..risk.engine import DeterministicRiskEngine, RiskEvaluationResult
from ..timing.candle_timing_agent import TimingAssessment, TimingDecision
from ..timing.central_directional_agent import DirectionalBias, DirectionalHypothesis
from .audit import AuthorityAuditRecord
from .instructions import ActionType, AuthorityInstruction, AuthoritySource, PermissionType


class AuthorityGateRejectionError(RuntimeError):
    """Raised when an action is rejected by the mandatory authority execution gate."""


@dataclass(frozen=True)
class GateEvaluationOutcome:
    """Result of mandatory execution gate evaluation."""

    approved: bool
    status: Literal["APPROVED", "REJECTED"]
    action_type: ActionType
    symbol: str
    rejection_reasons: tuple[str, ...]
    human_instruction: AuthorityInstruction | None
    central_instruction: AuthorityInstruction | None
    candle_assessment: TimingAssessment | None
    risk_evaluation: RiskEvaluationResult | None
    audit_record_hash: str
    disagreement_explanation: str | None = None


class MandatoryExecutionGate:
    """Thread-safe, race-condition protected authority execution gate."""

    _GENESIS_HASH = "0" * 64

    def __init__(
        self,
        *,
        risk_engine: DeterministicRiskEngine | None = None,
        safe_position_handler: Callable[[str, IntradayPaperPortfolio, datetime], None] | None = None,
    ) -> None:
        self.risk_engine = risk_engine or DeterministicRiskEngine()
        self._safe_position_handler = safe_position_handler or self._default_safe_position_handler
        self._lock = threading.RLock()

        # Active instructions stores: keyed by normalized scope ("*" or uppercase symbol)
        self._human_instructions: dict[str, AuthorityInstruction] = {}
        self._central_instructions: dict[str, AuthorityInstruction] = {}

        # Cryptographically hash-chained audit log
        self._audit_log: list[AuthorityAuditRecord] = []

    # --------------------------------------------------------------------------
    # Instruction Management (Human & Central Agent)
    # --------------------------------------------------------------------------

    def set_human_instruction(self, instruction: AuthorityInstruction) -> None:
        """Register or update an active human authority instruction."""
        if instruction.authority != AuthoritySource.HUMAN:
            raise ValueError(f"Instruction authority must be HUMAN; got {instruction.authority.value}")
        with self._lock:
            self._human_instructions[instruction.normalized_scope] = instruction

    def clear_human_instruction(self, scope: str = "*") -> None:
        """Clear human instruction for scope."""
        with self._lock:
            norm = "*" if scope in ("*", "PORTFOLIO", "ALL") else scope.strip().upper()
            self._human_instructions.pop(norm, None)

    def emergency_human_stop(
        self,
        *,
        reason: str = "HUMAN_EMERGENCY_STOP",
        current_time: datetime,
        scope: str = "*",
        portfolio: IntradayPaperPortfolio | None = None,
        valid_duration_minutes: int = 1440,
    ) -> AuthorityInstruction:
        """Immediately issue a Human STOP_ALL instruction and trigger safe position management."""
        if current_time.tzinfo is None:
            raise ValueError(f"current_time must be timezone-aware; got {current_time!r}")
        now_ist = current_time.astimezone(IST_TZ)
        from datetime import timedelta

        instr = AuthorityInstruction(
            instruction_id=f"HUMAN-STOP-{uuid.uuid4().hex[:8]}",
            authority=AuthoritySource.HUMAN,
            scope=scope,
            permission=PermissionType.STOP_ALL,
            valid_from=now_ist - timedelta(seconds=1),
            expires_at=now_ist + timedelta(minutes=valid_duration_minutes),
            reason=reason,
            version="1.0",
        )
        with self._lock:
            self.set_human_instruction(instr)
            if portfolio is not None:
                self.invoke_safe_position_management(
                    symbol=scope,
                    portfolio=portfolio,
                    current_time=now_ist,
                )
        return instr

    def set_central_instruction(self, instruction: AuthorityInstruction) -> None:
        """Register or update an active Central Agent authority instruction."""
        if instruction.authority != AuthoritySource.CENTRAL_AGENT:
            raise ValueError(f"Instruction authority must be CENTRAL_AGENT; got {instruction.authority.value}")
        with self._lock:
            self._central_instructions[instruction.normalized_scope] = instruction

    def clear_central_instruction(self, scope: str = "*") -> None:
        """Clear central instruction for scope."""
        with self._lock:
            norm = "*" if scope in ("*", "PORTFOLIO", "ALL") else scope.strip().upper()
            self._central_instructions.pop(norm, None)

    def _get_applicable_instruction(
        self,
        store: dict[str, AuthorityInstruction],
        symbol: str,
        current_time: datetime,
    ) -> tuple[AuthorityInstruction | None, list[str]]:
        """Find active instruction applying to symbol (symbol-specific takes precedence over '*')."""
        sym_u = symbol.strip().upper()
        candidates = []
        if sym_u in store:
            candidates.append(store[sym_u])
        if "*" in store:
            candidates.append(store["*"])

        if not candidates:
            return None, ["NO_INSTRUCTION_FOUND"]

        rejection_reasons = []
        for cand in candidates:
            if not cand.is_valid_at(current_time):
                if current_time < cand.valid_from:
                    rejection_reasons.append(f"INSTRUCTION_NOT_YET_VALID:{cand.instruction_id}")
                else:
                    rejection_reasons.append(f"INSTRUCTION_EXPIRED:{cand.instruction_id}")
                continue
            return cand, []

        return None, rejection_reasons

    # --------------------------------------------------------------------------
    # Safe Position Management
    # --------------------------------------------------------------------------

    def _default_safe_position_handler(
        self,
        symbol: str,
        portfolio: IntradayPaperPortfolio,
        current_time: datetime,
    ) -> None:
        """Default safe management: square off open long and short positions to eliminate risk."""
        sym_u = symbol.strip().upper()
        if sym_u == "*":
            # Square off all open positions
            open_keys = list(portfolio.positions.keys())
            for k in open_keys:
                pos = portfolio.positions.get(k)
                if pos and pos.quantity > 0:
                    px = getattr(pos, "current_price", pos.entry_price)
                    if getattr(pos, "side", "LONG") == "SHORT":
                        portfolio.submit_order(
                            symbol=pos.symbol,
                            side="BUY",
                            quantity=pos.quantity,
                            price=px,
                            exchange=pos.exchange,
                            timestamp=current_time,
                            position_effect="COVER_SHORT",
                        )
                    else:
                        portfolio.submit_order(
                            symbol=pos.symbol,
                            side="SELL",
                            quantity=pos.quantity,
                            price=px,
                            exchange=pos.exchange,
                            timestamp=current_time,
                            position_effect="EXIT_LONG",
                        )
        else:
            pos = portfolio.positions.get(sym_u)
            if pos and pos.quantity > 0:
                px = getattr(pos, "current_price", pos.entry_price)
                if getattr(pos, "side", "LONG") == "SHORT":
                    portfolio.submit_order(
                        symbol=pos.symbol,
                        side="BUY",
                        quantity=pos.quantity,
                        price=px,
                        exchange=pos.exchange,
                        timestamp=current_time,
                        position_effect="COVER_SHORT",
                    )
                else:
                    portfolio.submit_order(
                        symbol=pos.symbol,
                        side="SELL",
                        quantity=pos.quantity,
                        price=px,
                        exchange=pos.exchange,
                        timestamp=current_time,
                        position_effect="EXIT_LONG",
                    )

    def invoke_safe_position_management(
        self,
        symbol: str,
        portfolio: IntradayPaperPortfolio,
        current_time: datetime,
    ) -> None:
        """Execute configured safe position management (e.g. on STOP_ALL)."""
        with self._lock:
            self._safe_position_handler(symbol, portfolio, current_time)

    # --------------------------------------------------------------------------
    # The Mandatory 7-Point Execution Gate
    # --------------------------------------------------------------------------

    def evaluate_gate(
        self,
        *,
        symbol: str,
        action_type: ActionType,
        current_time: datetime,
        portfolio: IntradayPaperPortfolio,
        candle_assessment: TimingAssessment | None = None,
        requested_quantity: int = 0,
        entry_price: float = 0.0,
        stop_loss_price: float | None = None,
        target_price: float | None = None,
    ) -> GateEvaluationOutcome:
        """Evaluate all 7 mandatory gates atomically under lock immediately prior to execution."""
        sym_u = symbol.strip().upper()
        if current_time.tzinfo is None:
            raise ValueError(f"current_time must be timezone-aware; got {current_time!r}")
        now_ist = current_time.astimezone(IST_TZ)

        rejection_reasons: list[str] = []
        disagreement_str: str | None = None

        with self._lock:
            # GATE 1: Current Human Permission (fail closed if an explicitly registered human instruction expired)
            human_instr, human_errs = self._get_applicable_instruction(self._human_instructions, sym_u, now_ist)
            if human_instr is not None:
                allowed, r_reason = human_instr.allows_action(action_type)
                if not allowed:
                    rejection_reasons.append(f"HUMAN_PERMISSION_DENIED:{r_reason}")
            elif action_type.is_new_entry and any(
                e.startswith(("INSTRUCTION_EXPIRED:", "INSTRUCTION_NOT_YET_VALID:")) for e in human_errs
            ):
                rejection_reasons.append(f"HUMAN_INSTRUCTION_EXPIRED_OR_INVALID:{','.join(human_errs)}")

            # GATE 2: Current Central Agent Permission
            central_instr, central_errs = self._get_applicable_instruction(self._central_instructions, sym_u, now_ist)

            # For new entries, Central Agent permission is MANDATORY (fail closed if missing or expired)
            if action_type.is_new_entry:
                if central_instr is None:
                    rejection_reasons.append(
                        f"CENTRAL_AGENT_PERMISSION_MISSING_OR_EXPIRED:{','.join(central_errs) or 'NO_RECORD'}"
                    )
                else:
                    allowed, c_reason = central_instr.allows_action(action_type)
                    if not allowed:
                        rejection_reasons.append(f"CENTRAL_PERMISSION_DENIED:{c_reason}")

            # GATE 3: Instruction Freshness, Expiry & Time Boundaries
            if human_instr is not None and not human_instr.is_valid_at(now_ist):
                rejection_reasons.append(f"HUMAN_INSTRUCTION_EXPIRED:{human_instr.instruction_id}")
            if central_instr is not None and not central_instr.is_valid_at(now_ist):
                rejection_reasons.append(f"CENTRAL_INSTRUCTION_EXPIRED:{central_instr.instruction_id}")

            # GATE 4: Symbol & Directional Restrictions
            # (Checked inside allows_action above)

            # GATE 5: Candle Agent Subordination & Conflict Resolution
            # Candle Agent MUST NOT independently authorize execution.
            if action_type.is_new_entry:
                if candle_assessment is None:
                    rejection_reasons.append("CANDLE_TIMING_ASSESSMENT_MISSING")
                else:
                    # Candle Agent must confirm timing
                    if candle_assessment.decision != TimingDecision.ENTER_NOW:
                        rejection_reasons.append(
                            f"CANDLE_TIMING_NOT_CONFIRMED:decision={candle_assessment.decision.value}"
                        )

                    # Verify action-to-candle directional alignment (even under ALLOW_BOTH)
                    expected_action_bias = (
                        DirectionalBias.BULLISH if action_type == ActionType.ENTRY_LONG else DirectionalBias.BEARISH
                    )
                    if candle_assessment.candle_bias.value != expected_action_bias.value:
                        disagreement_str = (
                            f"Disagreement: Action={action_type.value} ({expected_action_bias.value}) vs "
                            f"Candle={candle_assessment.candle_bias.value} "
                            f"({candle_assessment.alignment_status.value})"
                        )
                        rejection_reasons.append(f"CANDLE_ACTION_DIRECTION_MISMATCH:{disagreement_str}")

                    # Check for Disagreement with Central Agent
                    if central_instr is not None and central_instr.permission != PermissionType.ALLOW_BOTH:
                        c_dir = (
                            DirectionalBias.BULLISH
                            if central_instr.permission == PermissionType.ALLOW_LONG
                            else (
                                DirectionalBias.BEARISH
                                if central_instr.permission == PermissionType.ALLOW_SHORT
                                else DirectionalBias.NEUTRAL
                            )
                        )
                        if candle_assessment.candle_bias.value != c_dir.value:
                            disagreement_str = (
                                f"Disagreement: Central={c_dir.value} vs Candle={candle_assessment.candle_bias.value} "
                                f"({candle_assessment.alignment_status.value})"
                            )
                            rejection_reasons.append(f"CENTRAL_CANDLE_DISAGREEMENT:{disagreement_str}")

            # GATE 6: Current Position and Available Capital
            pos = portfolio.positions.get(sym_u)
            current_qty = pos.quantity if pos else 0
            current_side = getattr(pos, "side", "LONG") if pos else None

            if action_type == ActionType.ENTRY_LONG:
                if pos is not None and current_side == "SHORT":
                    rejection_reasons.append(
                        f"OPPOSITE_POSITION_OPEN:cannot ENTRY_LONG while SHORT {current_qty} shares in {sym_u}"
                    )
                required_cash = requested_quantity * entry_price
                if required_cash > portfolio.cash:
                    rejection_reasons.append(
                        f"INSUFFICIENT_CASH:required=Rs {required_cash:,.2f}, available=Rs {portfolio.cash:,.2f}"
                    )
            elif action_type == ActionType.ENTRY_SHORT:
                if pos is not None and current_side == "LONG":
                    rejection_reasons.append(
                        f"OPPOSITE_POSITION_OPEN:cannot ENTRY_SHORT while LONG {current_qty} shares in {sym_u}"
                    )
                required_margin = requested_quantity * entry_price
                if required_margin > portfolio.cash:
                    rejection_reasons.append(
                        f"INSUFFICIENT_CASH:required=Rs {required_margin:,.2f}, available=Rs {portfolio.cash:,.2f}"
                    )
            elif action_type == ActionType.EXIT_LONG:
                if requested_quantity <= 0 or pos is None or current_side != "LONG" or current_qty < requested_quantity:
                    rejection_reasons.append(
                        f"INSUFFICIENT_POSITION_TO_EXIT:requested={requested_quantity}, open_long={current_qty if current_side == 'LONG' else 0}"
                    )
            elif action_type == ActionType.EXIT_SHORT:
                if requested_quantity <= 0 or pos is None or current_side != "SHORT" or current_qty < requested_quantity:
                    rejection_reasons.append(
                        f"INSUFFICIENT_POSITION_TO_EXIT:requested={requested_quantity}, open_short={current_qty if current_side == 'SHORT' else 0}"
                    )
            elif action_type == ActionType.RISK_EXIT:
                if requested_quantity <= 0 or current_qty < requested_quantity:
                    rejection_reasons.append(
                        f"INSUFFICIENT_POSITION_TO_EXIT:requested={requested_quantity}, open={current_qty}"
                    )

            # GATE 7: Deterministic Risk Engine Approval (Human/Central CANNOT override Risk limits)
            risk_eval: RiskEvaluationResult | None = None
            if action_type.is_new_entry:
                if requested_quantity <= 0 or entry_price <= 0:
                    rejection_reasons.append(
                        f"INVALID_ENTRY_PARAMETERS:requested_quantity={requested_quantity}, entry_price={entry_price}"
                    )
                else:
                    if stop_loss_price is not None:
                        sl = stop_loss_price
                    elif action_type == ActionType.ENTRY_SHORT:
                        sl = round(entry_price * 1.015, 2)
                    else:
                        sl = round(entry_price * 0.985, 2)
                    side_str = "BUY" if action_type == ActionType.ENTRY_LONG else "SELL"
                    risk_eval = self.risk_engine.evaluate_order(
                        symbol=sym_u,
                        side=side_str,
                        requested_quantity=requested_quantity,
                        entry_price=entry_price,
                        stop_loss_price=sl,
                        portfolio=portfolio,
                        current_time=now_ist,
                        action_type=action_type.value,
                        include_entry_friction_in_cash_cap=True,
                    )
                    if not risk_eval.approved:
                        for reason in risk_eval.rejection_reasons:
                            rejection_reasons.append(f"RISK_ENGINE_REJECTION:{reason}")

            # Final Gate Decision
            approved = len(rejection_reasons) == 0
            final_status: Literal["APPROVED", "REJECTED"] = "APPROVED" if approved else "REJECTED"

            # Record in Immutable SHA-256 Hash-Chained Audit Ledger
            prev_h = self._audit_log[-1].record_hash if self._audit_log else self._GENESIS_HASH
            audit_rec = AuthorityAuditRecord.create(
                audit_id=f"AUDIT-{uuid.uuid4().hex[:12]}",
                timestamp_ist=now_ist,
                symbol=sym_u,
                action_type=action_type,
                human_instruction=human_instr,
                central_instruction=central_instr,
                candle_recommendation=candle_assessment.decision.value if candle_assessment else None,
                candle_bias=candle_assessment.candle_bias.value if candle_assessment else None,
                disagreement_explanation=disagreement_str,
                permission_evaluation_passed=not any(r.startswith(("HUMAN_", "CENTRAL_")) for r in rejection_reasons),
                risk_evaluation_passed=(risk_eval.approved if risk_eval else True),
                final_execution_decision=final_status,
                rejection_reasons=tuple(rejection_reasons),
                prev_hash=prev_h,
            )
            self._audit_log.append(audit_rec)

            return GateEvaluationOutcome(
                approved=approved,
                status=final_status,
                action_type=action_type,
                symbol=sym_u,
                rejection_reasons=tuple(rejection_reasons),
                human_instruction=human_instr,
                central_instruction=central_instr,
                candle_assessment=candle_assessment,
                risk_evaluation=risk_eval,
                audit_record_hash=audit_rec.record_hash,
                disagreement_explanation=disagreement_str,
            )

    def execute_with_gate_protection(
        self,
        *,
        symbol: str,
        action_type: ActionType,
        current_time: datetime,
        portfolio: IntradayPaperPortfolio,
        requested_quantity: int,
        price: float,
        candle_assessment: TimingAssessment | None = None,
        stop_loss_price: float | None = None,
        target_price: float | None = None,
    ) -> tuple[GateEvaluationOutcome, bool]:
        """Atomically evaluate the 7 mandatory gates and immediately execute paper order if approved.

        Eliminates any race window between gate evaluation and paper order placement.
        Returns (GateEvaluationOutcome, executed: bool).
        """
        with self._lock:
            # 1. Gate revalidation immediately prior to execution
            outcome = self.evaluate_gate(
                symbol=symbol,
                action_type=action_type,
                current_time=current_time,
                portfolio=portfolio,
                candle_assessment=candle_assessment,
                requested_quantity=requested_quantity,
                entry_price=price,
                stop_loss_price=stop_loss_price,
                target_price=target_price,
            )

            if not outcome.approved:
                return outcome, False

            # 2. Execution under the same atomic lock with explicit position_effect, stop_loss, and target
            sym_u = symbol.strip().upper()
            pos = portfolio.positions.get(sym_u)
            if action_type == ActionType.ENTRY_LONG:
                side = "BUY"
                pos_effect = "OPEN_LONG"
            elif action_type == ActionType.ENTRY_SHORT:
                side = "SELL"
                pos_effect = "OPEN_SHORT"
            elif action_type == ActionType.EXIT_LONG:
                side = "SELL"
                pos_effect = "EXIT_LONG"
            elif action_type == ActionType.EXIT_SHORT:
                side = "BUY"
                pos_effect = "COVER_SHORT"
            else:
                # ActionType.RISK_EXIT matches the existing position's direction
                if pos is not None and getattr(pos, "side", "LONG") == "SHORT":
                    side = "BUY"
                    pos_effect = "COVER_SHORT"
                else:
                    side = "SELL"
                    pos_effect = "EXIT_LONG"

            qty = outcome.risk_evaluation.approved_quantity if outcome.risk_evaluation else requested_quantity
            effective_stop_loss = stop_loss_price
            if effective_stop_loss is None and action_type.is_new_entry and price > 0:
                effective_stop_loss = (
                    round(price * 1.015, 2) if action_type == ActionType.ENTRY_SHORT else round(price * 0.985, 2)
                )

            order = portfolio.submit_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                stop_loss=effective_stop_loss,
                target=target_price,
                timestamp=current_time,
                position_effect=pos_effect,
            )
            executed = order.status.value in ("FILLED", "PARTIALLY_FILLED")
            return outcome, executed

    @property
    def audit_trail(self) -> tuple[AuthorityAuditRecord, ...]:
        """Return immutable copy of the audit trail."""
        with self._lock:
            return tuple(self._audit_log)
