"""Mandatory Central Agent & Human Authority Execution Gate (REMED-01 Hardened).

Enforces strict 5-tier authority hierarchy:
1. Human Authority (highest priority; fail-closed if missing/invalid for new entries)
2. Central Agent directional instructions & freshness
3. Candle Timing Agent (advisory only; never self-authorizing)
4. Deterministic Risk Engine (absolute veto on capital/drawdown/exposure limits)
5. Paper Execution Ledger (atomic under RLock with CashReservationLedger holds)

Revalidates all mandatory gates atomically under lock immediately prior to
every paper execution to prevent stale permissions, replay attacks, and race conditions.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Literal

from ..config import IST_TZ
from ..portfolio.ledger import IntradayPaperPortfolio
from ..risk.engine import DeterministicRiskEngine, RiskEvaluationResult, RiskState
from ..timing.candle_timing_agent import TimingAssessment, TimingDecision
from ..timing.central_directional_agent import DirectionalBias
from .audit import AuthorityAuditRecord
from .instructions import ActionType, AuthorityInstruction, AuthoritySource, PermissionType


class AuthorityGateRejectionError(RuntimeError):
    """Raised when an action is rejected by the mandatory authority execution gate."""


@dataclass(frozen=True)
class InstructionRegistrationResult:
    """Structured, auditable outcome of registering a Human or Central instruction."""

    accepted: bool
    reason: str
    instruction_id: str
    authority: str
    scope: str
    permission: str
    timestamp_ist: str
    active_instruction_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "instruction_id": self.instruction_id,
            "authority": self.authority,
            "scope": self.scope,
            "permission": self.permission,
            "timestamp_ist": self.timestamp_ist,
            "active_instruction_id": self.active_instruction_id,
        }


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
        max_central_staleness_minutes: int | None = None,
    ) -> None:
        self.risk_engine = risk_engine or DeterministicRiskEngine()
        self._safe_position_handler = safe_position_handler or self._default_safe_position_handler
        self._max_central_staleness_minutes = max_central_staleness_minutes
        self._lock = threading.RLock()

        # Active instruction stores: keyed by normalized scope ("*" or uppercase symbol)
        self._human_instructions: dict[str, AuthorityInstruction] = {}
        self._central_instructions: dict[str, AuthorityInstruction] = {}

        # Replay & out-of-order protection
        self._seen_instruction_ids: set[str] = set()
        self._instruction_audit_log: list[InstructionRegistrationResult] = []

        # Cryptographically hash-chained gate audit log
        self._audit_log: list[AuthorityAuditRecord] = []

    # --------------------------------------------------------------------------
    # Instruction Management (Human & Central Agent) — Decision 2 Hardened
    # --------------------------------------------------------------------------

    def _register_instruction(
        self,
        store: dict[str, AuthorityInstruction],
        expected_source: AuthoritySource,
        instruction: AuthorityInstruction,
        *,
        current_time: datetime | None = None,
    ) -> InstructionRegistrationResult:
        now_ist = (
            current_time.astimezone(IST_TZ)
            if (current_time is not None and current_time.tzinfo is not None)
            else datetime.now(IST_TZ)
        )
        scope = instruction.normalized_scope

        with self._lock:
            existing = store.get(scope)
            active_id = existing.instruction_id if existing is not None else None

            if instruction.authority != expected_source:
                res = InstructionRegistrationResult(
                    accepted=False,
                    reason=(
                        f"INVALID_AUTHORITY_SOURCE:expected={expected_source.value},"
                        f"got={instruction.authority.value}"
                    ),
                    instruction_id=instruction.instruction_id,
                    authority=instruction.authority.value,
                    scope=scope,
                    permission=instruction.permission.value,
                    timestamp_ist=now_ist.isoformat(),
                    active_instruction_id=active_id,
                )
                self._instruction_audit_log.append(res)
                return res

            # 1. Duplicate / replayed instruction_id rejection
            if instruction.instruction_id in self._seen_instruction_ids:
                res = InstructionRegistrationResult(
                    accepted=False,
                    reason=f"DUPLICATE_INSTRUCTION_ID:{instruction.instruction_id}",
                    instruction_id=instruction.instruction_id,
                    authority=instruction.authority.value,
                    scope=scope,
                    permission=instruction.permission.value,
                    timestamp_ist=now_ist.isoformat(),
                    active_instruction_id=active_id,
                )
                self._instruction_audit_log.append(res)
                return res

            # 2. Stale / already-expired instruction rejection (when current_time supplied)
            if current_time is not None and current_time.tzinfo is not None:
                if now_ist > instruction.expires_at:
                    res = InstructionRegistrationResult(
                        accepted=False,
                        reason=f"STALE_EXPIRED_INSTRUCTION:{instruction.instruction_id}",
                        instruction_id=instruction.instruction_id,
                        authority=instruction.authority.value,
                        scope=scope,
                        permission=instruction.permission.value,
                        timestamp_ist=now_ist.isoformat(),
                        active_instruction_id=active_id,
                    )
                    self._instruction_audit_log.append(res)
                    return res

            # 3. Delayed / out-of-order instruction rejection (cannot overwrite newer valid_from)
            if existing is not None and instruction.valid_from < existing.valid_from:
                res = InstructionRegistrationResult(
                    accepted=False,
                    reason=(
                        f"DELAYED_OUT_OF_ORDER_INSTRUCTION:"
                        f"incoming={instruction.valid_from.isoformat()}<active={existing.valid_from.isoformat()}"
                    ),
                    instruction_id=instruction.instruction_id,
                    authority=instruction.authority.value,
                    scope=scope,
                    permission=instruction.permission.value,
                    timestamp_ist=now_ist.isoformat(),
                    active_instruction_id=active_id,
                )
                self._instruction_audit_log.append(res)
                return res

            store[scope] = instruction
            self._seen_instruction_ids.add(instruction.instruction_id)
            res = InstructionRegistrationResult(
                accepted=True,
                reason="ACCEPTED",
                instruction_id=instruction.instruction_id,
                authority=instruction.authority.value,
                scope=scope,
                permission=instruction.permission.value,
                timestamp_ist=now_ist.isoformat(),
                active_instruction_id=instruction.instruction_id,
            )
            self._instruction_audit_log.append(res)
            return res

    def set_human_instruction(
        self,
        instruction: AuthorityInstruction,
        *,
        current_time: datetime | None = None,
    ) -> InstructionRegistrationResult:
        """Register or update an active human authority instruction (structured rejection on replay/delay)."""
        if instruction.authority != AuthoritySource.HUMAN:
            raise ValueError(f"Instruction authority must be HUMAN; got {instruction.authority.value}")
        return self._register_instruction(
            self._human_instructions,
            AuthoritySource.HUMAN,
            instruction,
            current_time=current_time,
        )

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
        """Immediately issue a Human STOP_ALL instruction, release pending reservations, and trigger safe management."""
        if current_time.tzinfo is None:
            raise ValueError(f"current_time must be timezone-aware; got {current_time!r}")
        now_ist = current_time.astimezone(IST_TZ)

        with self._lock:
            norm_scope = "*" if scope in ("*", "PORTFOLIO", "ALL") else scope.strip().upper()
            existing = self._human_instructions.get(norm_scope)
            # Ensure STOP_ALL valid_from is never earlier than an existing active instruction
            base_valid_from = now_ist - timedelta(seconds=1)
            if existing is not None and existing.valid_from > base_valid_from:
                base_valid_from = existing.valid_from

            instr = AuthorityInstruction(
                instruction_id=f"HUMAN-STOP-{uuid.uuid4().hex[:8]}",
                authority=AuthoritySource.HUMAN,
                scope=scope,
                permission=PermissionType.STOP_ALL,
                valid_from=base_valid_from,
                expires_at=now_ist + timedelta(minutes=valid_duration_minutes),
                reason=reason,
                version="1.0",
            )
            self.set_human_instruction(instr, current_time=now_ist)
            if norm_scope == "*":
                self.risk_engine.cash_ledger.clear()
            if portfolio is not None:
                self.invoke_safe_position_management(
                    symbol=scope,
                    portfolio=portfolio,
                    current_time=now_ist,
                )
        return instr

    def set_central_instruction(
        self,
        instruction: AuthorityInstruction,
        *,
        current_time: datetime | None = None,
    ) -> InstructionRegistrationResult:
        """Register or update an active Central Agent authority instruction (structured rejection on replay/delay)."""
        if instruction.authority != AuthoritySource.CENTRAL_AGENT:
            raise ValueError(f"Instruction authority must be CENTRAL_AGENT; got {instruction.authority.value}")
        return self._register_instruction(
            self._central_instructions,
            AuthoritySource.CENTRAL_AGENT,
            instruction,
            current_time=current_time,
        )

    def clear_central_instruction(self, scope: str = "*") -> None:
        """Clear central instruction for scope."""
        with self._lock:
            norm = "*" if scope in ("*", "PORTFOLIO", "ALL") else scope.strip().upper()
            self._central_instructions.pop(norm, None)

    def get_active_human_instruction(self, symbol: str, current_time: datetime) -> AuthorityInstruction | None:
        """Return currently valid Human instruction for symbol (or None)."""
        with self._lock:
            instr, _ = self._get_applicable_instruction(self._human_instructions, symbol, current_time)
            return instr

    def get_active_central_instruction(self, symbol: str, current_time: datetime) -> AuthorityInstruction | None:
        """Return currently valid Central instruction for symbol (or None)."""
        with self._lock:
            instr, _ = self._get_applicable_instruction(self._central_instructions, symbol, current_time)
            return instr

    def _get_applicable_instruction(
        self,
        store: dict[str, AuthorityInstruction],
        symbol: str,
        current_time: datetime,
    ) -> tuple[AuthorityInstruction | None, list[str]]:
        """Find active instruction applying to symbol.

        HARD FAIL-CLOSED PRECEDENCE:
        If a symbol-specific instruction exists in `store` and is expired or not-yet-valid,
        NEVER fall back to the broader portfolio-wide `"*"` instruction.
        """
        sym_u = symbol.strip().upper()
        if sym_u in store:
            cand = store[sym_u]
            if not cand.is_valid_at(current_time):
                if current_time < cand.valid_from:
                    return None, [f"INSTRUCTION_NOT_YET_VALID:{cand.instruction_id}"]
                return None, [f"INSTRUCTION_EXPIRED:{cand.instruction_id}"]
            return cand, []

        if "*" in store:
            cand = store["*"]
            if not cand.is_valid_at(current_time):
                if current_time < cand.valid_from:
                    return None, [f"INSTRUCTION_NOT_YET_VALID:{cand.instruction_id}"]
                return None, [f"INSTRUCTION_EXPIRED:{cand.instruction_id}"]
            return cand, []

        return None, ["NO_INSTRUCTION_FOUND"]

    # --------------------------------------------------------------------------
    # Cash Reservation Coordination (Multi-Symbol Capital Protection)
    # --------------------------------------------------------------------------

    def reserve_cash_for_pending_order(self, order_id: str, symbol: str, amount_inr: float) -> None:
        """Reserve cash in DeterministicRiskEngine.cash_ledger for a pending order."""
        with self._lock:
            self.risk_engine.cash_ledger.reserve(order_id, symbol.strip().upper(), amount_inr)

    def release_cash_reservation(self, order_id: str) -> float:
        """Release a pending order's cash reservation."""
        with self._lock:
            return self.risk_engine.cash_ledger.release(order_id)

    def clear_cash_reservations(self) -> None:
        """Clear all pending cash reservations."""
        with self._lock:
            self.risk_engine.cash_ledger.clear()

    # --------------------------------------------------------------------------
    # Safe Position Management (Fully Audited via RISK_EXIT)
    # --------------------------------------------------------------------------

    def _default_safe_position_handler(
        self,
        symbol: str,
        portfolio: IntradayPaperPortfolio,
        current_time: datetime,
    ) -> None:
        """Default safe management: square off open long/short positions via audited RISK_EXIT."""
        sym_u = symbol.strip().upper()
        target_symbols = list(portfolio.positions.keys()) if sym_u == "*" else [sym_u]
        for k in target_symbols:
            pos = portfolio.positions.get(k)
            if pos and pos.quantity > 0:
                px = getattr(pos, "current_price", pos.entry_price)
                self.execute_with_gate_protection(
                    symbol=pos.symbol,
                    action_type=ActionType.RISK_EXIT,
                    current_time=current_time,
                    portfolio=portfolio,
                    requested_quantity=pos.quantity,
                    price=px,
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
        dry_run: bool = False,
    ) -> GateEvaluationOutcome:
        """Evaluate all 7 mandatory gates atomically under lock immediately prior to execution."""
        sym_u = symbol.strip().upper()
        if current_time.tzinfo is None:
            raise ValueError(f"current_time must be timezone-aware; got {current_time!r}")
        now_ist = current_time.astimezone(IST_TZ)

        rejection_reasons: list[str] = []
        disagreement_str: str | None = None

        with self._lock:
            # GATE 1: Human Authority (Decision 1: STRICT FAIL-CLOSED for new entries; exits remain safe)
            human_instr, human_errs = self._get_applicable_instruction(self._human_instructions, sym_u, now_ist)
            if human_instr is not None:
                allowed, r_reason = human_instr.allows_action(action_type)
                if not allowed:
                    rejection_reasons.append(f"HUMAN_PERMISSION_DENIED:{r_reason}")
            elif action_type.is_new_entry:
                if any(e.startswith(("INSTRUCTION_EXPIRED:", "INSTRUCTION_NOT_YET_VALID:")) for e in human_errs):
                    rejection_reasons.append(f"HUMAN_INSTRUCTION_EXPIRED_OR_INVALID:{','.join(human_errs)}")
                else:
                    rejection_reasons.append(f"HUMAN_INSTRUCTION_MISSING:{','.join(human_errs) or 'NO_RECORD'}")

            # GATE 2: Central Agent Authority & Freshness (fail-closed for new entries)
            central_instr, central_errs = self._get_applicable_instruction(self._central_instructions, sym_u, now_ist)
            if action_type.is_new_entry:
                if central_instr is None:
                    rejection_reasons.append(
                        f"CENTRAL_AGENT_PERMISSION_MISSING_OR_EXPIRED:{','.join(central_errs) or 'NO_RECORD'}"
                    )
                else:
                    allowed, c_reason = central_instr.allows_action(action_type)
                    if not allowed:
                        rejection_reasons.append(f"CENTRAL_PERMISSION_DENIED:{c_reason}")
                    if self._max_central_staleness_minutes is not None:
                        age_min = (now_ist - central_instr.valid_from).total_seconds() / 60.0
                        if age_min > self._max_central_staleness_minutes:
                            rejection_reasons.append(
                                f"CENTRAL_INSTRUCTION_STALE:age_min={age_min:.1f}>max={self._max_central_staleness_minutes}"
                            )

            # GATE 3: Candle Agent Subordination & Conflict Resolution (never self-authorizing)
            if action_type.is_new_entry:
                if candle_assessment is None:
                    rejection_reasons.append("CANDLE_TIMING_ASSESSMENT_MISSING")
                else:
                    if candle_assessment.decision != TimingDecision.ENTER_NOW:
                        rejection_reasons.append(
                            f"CANDLE_TIMING_NOT_CONFIRMED:decision={candle_assessment.decision.value}"
                        )

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

            # GATE 4: Position Direction & Exit Quantity Consistency
            pos = portfolio.positions.get(sym_u)
            current_qty = pos.quantity if pos else 0
            current_side = getattr(pos, "side", "LONG") if pos else None

            if action_type == ActionType.ENTRY_LONG:
                if pos is not None and current_side == "SHORT":
                    rejection_reasons.append(
                        f"OPPOSITE_POSITION_OPEN:cannot ENTRY_LONG while SHORT {current_qty} shares in {sym_u}"
                    )
            elif action_type == ActionType.ENTRY_SHORT:
                if pos is not None and current_side == "LONG":
                    rejection_reasons.append(
                        f"OPPOSITE_POSITION_OPEN:cannot ENTRY_SHORT while LONG {current_qty} shares in {sym_u}"
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

            # GATE 5 & 6 (REMED-01 Fix): Deterministic Risk Engine Sizing & Limits FIRST,
            # THEN check unreserved available cash against the post-sized `approved_quantity`!
            # (Decision 4: HALTED blocks new entries and implements EXIT_ONLY for existing positions.)
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
                        dry_run=dry_run,
                    )
                    if not risk_eval.approved:
                        for reason in risk_eval.rejection_reasons:
                            rejection_reasons.append(f"RISK_ENGINE_REJECTION:{reason}")
                    else:
                        # Gate 6 post-sizing unreserved cash verification
                        sized_qty = risk_eval.approved_quantity
                        available_unreserved = max(
                            0.0, float(portfolio.cash) - self.risk_engine.cash_ledger.total_reserved_inr
                        )
                        required_notional = sized_qty * entry_price
                        if required_notional > available_unreserved + 1e-6:
                            rejection_reasons.append(
                                f"INSUFFICIENT_UNRESERVED_CASH:required=Rs {required_notional:,.2f}, "
                                f"available=Rs {available_unreserved:,.2f}"
                            )
            else:
                # Audit all exits through Risk Engine with force_reduce_only=True so HALTED -> EXIT_ONLY works cleanly
                if requested_quantity > 0 and entry_price > 0 and pos is not None:
                    exit_side = "BUY" if (action_type == ActionType.EXIT_SHORT or current_side == "SHORT") else "SELL"
                    risk_eval = self.risk_engine.evaluate_order(
                        symbol=sym_u,
                        side=exit_side,
                        requested_quantity=requested_quantity,
                        entry_price=entry_price,
                        stop_loss_price=None,
                        portfolio=portfolio,
                        current_time=now_ist,
                        force_reduce_only=True,
                        action_type=action_type.value,
                        dry_run=dry_run,
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
        reservation_order_id: str | None = None,
    ) -> tuple[GateEvaluationOutcome, bool]:
        """Atomically revalidate all mandatory gates and immediately execute paper order if approved.

        Releases any pending `reservation_order_id` atomically before sizing/execution so the
        order's own reservation does not block itself, and ensures release on rejection as well.
        """
        with self._lock:
            if reservation_order_id is not None:
                self.risk_engine.cash_ledger.release(reservation_order_id)

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
                dry_run=False,
            )

            if not outcome.approved:
                return outcome, False

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
                symbol=sym_u,
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
        """Return immutable copy of the gate evaluation audit trail."""
        with self._lock:
            return tuple(self._audit_log)

    @property
    def instruction_audit_trail(self) -> tuple[InstructionRegistrationResult, ...]:
        """Return immutable copy of the instruction registration/rejection audit trail."""
        with self._lock:
            return tuple(self._instruction_audit_log)

    def verify_audit_chain(self) -> bool:
        """Verify BOTH per-record SHA-256 hashes AND contiguous prev_hash linkage from genesis."""
        with self._lock:
            expected_prev = self._GENESIS_HASH
            for rec in self._audit_log:
                if rec.prev_hash != expected_prev:
                    return False
                if not rec.verify_hash():
                    return False
                expected_prev = rec.record_hash
            return True
