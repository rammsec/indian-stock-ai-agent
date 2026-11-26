"""Comprehensive Unit Tests for Central Agent & Human Authority Control (Addendum).

Tests all 12+ mandatory authority and execution gate requirements:
1. Central BUY + Candle BUY -> Approved
2. Central BUY + Candle WAIT -> Blocked (Candle unconfirmed)
3. Central BUY + Candle SHORT -> Blocked (Disagreement / Contradiction)
4. Central NO_NEW_ENTRIES -> Blocked for new entries, exits permitted
5. Central instruction expiry -> Blocked (fail closed)
6. Human STOP during pending entry -> Blocked
7. Human STOP immediately before execution -> Blocked (atomic gate revalidation)
8. Human STOP with open paper position -> Safe position management invoked
9. Central Agent unavailable (missing instruction) -> Blocked (fail closed)
10. Risk Engine rejection -> Blocked (Human/Central cannot override risk limits)
11. Conflicting Human and Central instructions -> Human restriction takes precedence
12. Repeated / delayed instructions -> Checked for validity and expiry
13. EXIT_ONLY -> Blocks new exposure, permits position reductions
14. Candle Agent cannot self-authorize -> Rejects unapproved independent entries
15. SHA-256 hash-chained immutable audit ledger -> Tamper-evident & verifiable
16. Zero broker integration / real-money execution check
"""

from __future__ import annotations

import time as py_time
from datetime import datetime, timedelta
from typing import Any

import pytest

from trader.authority import (
    ActionType,
    AuthorityAuditRecord,
    AuthorityInstruction,
    AuthoritySource,
    MandatoryExecutionGate,
    PermissionType,
)
from trader.config import DEFAULT_INITIAL_CAPITAL_INR, IST_TZ
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine
from trader.timing.candle_intelligence import CandleBias
from trader.timing.candle_timing_agent import AlignmentStatus, TimingAssessment, TimingDecision
from trader.timing.central_directional_agent import DirectionalBias


def _make_timing_assessment(
    decision: TimingDecision = TimingDecision.ENTER_NOW,
    candle_bias: CandleBias = CandleBias.BULLISH,
    alignment: AlignmentStatus = AlignmentStatus.CONFIRMED,
    as_of: datetime | None = None,
) -> TimingAssessment:
    ts = as_of or datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)
    return TimingAssessment(
        symbol="TCS",
        as_of_ist=ts,
        decision=decision,
        alignment_status=alignment,
        central_direction=DirectionalBias.BULLISH,
        candle_bias=candle_bias,
        timing_quality_score=0.85 if decision == TimingDecision.ENTER_NOW else 0.20,
        price_action_score=0.80,
        entry_price=2500.0,
        invalidation_price=2462.50,
        target_price=2575.0,
        risk_reward_ratio=2.0,
        detected_patterns=(),
        timeframe_statuses={"1h": "VALID"},
        contradiction_reasons=() if alignment != AlignmentStatus.CONTRADICTED else ("Opposing momentum",),
        rationale=("Clean bullish setup",),
    )


def _make_instruction(
    *,
    authority: AuthoritySource,
    permission: PermissionType,
    scope: str = "TCS",
    valid_from: datetime,
    duration_hours: float = 6.0,
    instruction_id: str | None = None,
    reason: str = "Test instruction",
) -> AuthorityInstruction:
    import uuid

    return AuthorityInstruction(
        instruction_id=instruction_id or f"TEST-INSTR-{uuid.uuid4().hex[:8]}",
        authority=authority,
        scope=scope,
        permission=permission,
        valid_from=valid_from,
        expires_at=valid_from + timedelta(hours=duration_hours),
        reason=reason,
        version="1.0",
    )


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_central_buy_plus_candle_buy_approved() -> None:
    """Scenario 1: Human ALLOW_LONG + Central BUY + Candle BUY -> Approved when risk engine approves."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    gate.set_human_instruction(_make_instruction(
        authority=AuthoritySource.HUMAN,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=30),
    ))
    central_instr = _make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=30),
    )
    gate.set_central_instruction(central_instr)

    candle_eval = _make_timing_assessment(
        decision=TimingDecision.ENTER_NOW,
        candle_bias=CandleBias.BULLISH,
        alignment=AlignmentStatus.CONFIRMED,
        as_of=now,
    )

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
        stop_loss_price=2462.50,
        target_price=2575.0,
    )

    assert outcome.approved is True
    assert executed is True
    assert len(outcome.rejection_reasons) == 0
    assert len(portfolio.positions) == 1
    assert portfolio.positions["TCS"].quantity == 10


def test_central_buy_plus_candle_wait_blocked() -> None:
    """Scenario 2: Central BUY + Candle WAIT -> Blocked (Candle not confirmed)."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    central_instr = _make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=30),
    )
    gate.set_central_instruction(central_instr)

    candle_eval = _make_timing_assessment(
        decision=TimingDecision.WAIT,
        candle_bias=CandleBias.BULLISH,
        alignment=AlignmentStatus.UNCONFIRMED,
        as_of=now,
    )

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("CANDLE_TIMING_NOT_CONFIRMED" in r for r in outcome.rejection_reasons)
    assert len(portfolio.positions) == 0


def test_central_buy_plus_candle_short_disagreement_blocked() -> None:
    """Scenario 3: Central BUY + Candle SHORT -> Blocked (Disagreement / Contradiction)."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    central_instr = _make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=30),
    )
    gate.set_central_instruction(central_instr)

    candle_eval = _make_timing_assessment(
        decision=TimingDecision.CONTRADICT_CENTRAL,
        candle_bias=CandleBias.BEARISH,
        alignment=AlignmentStatus.CONTRADICTED,
        as_of=now,
    )

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("CENTRAL_CANDLE_DISAGREEMENT" in r for r in outcome.rejection_reasons)
    assert outcome.disagreement_explanation is not None
    assert "Disagreement" in outcome.disagreement_explanation


def test_central_no_new_entries_blocks_entry_permits_exit() -> None:
    """Scenario 4: Central NO_NEW_ENTRIES -> Blocks new entries, permits risk-reducing exits."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    central_instr = _make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.NO_NEW_ENTRIES,
        valid_from=now - timedelta(minutes=30),
    )
    gate.set_central_instruction(central_instr)

    candle_eval = _make_timing_assessment(as_of=now)

    # 1. New entry is blocked
    outcome_entry, executed_entry = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )
    assert outcome_entry.approved is False
    assert executed_entry is False
    assert any("NO_NEW_ENTRIES" in r for r in outcome_entry.rejection_reasons)

    # 2. Existing position exit is permitted
    portfolio.submit_order(symbol="TCS", side="BUY", quantity=10, price=2500.0, timestamp=now)
    assert len(portfolio.positions) == 1

    outcome_exit, executed_exit = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.EXIT_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2520.0,
    )
    assert outcome_exit.approved is True
    assert executed_exit is True
    assert len(portfolio.positions) == 0


def test_central_instruction_expiry_fails_closed() -> None:
    """Scenario 5: Expired Central instruction -> Fails closed for new entries."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 16, 0, tzinfo=IST_TZ)

    # Valid between 09:15 and 15:30; current time is 16:00 (expired)
    expired_instr = AuthorityInstruction(
        instruction_id="EXPIRED-1",
        authority=AuthoritySource.CENTRAL_AGENT,
        scope="TCS",
        permission=PermissionType.ALLOW_LONG,
        valid_from=datetime(2026, 11, 20, 9, 15, tzinfo=IST_TZ),
        expires_at=datetime(2026, 11, 20, 15, 30, tzinfo=IST_TZ),
        reason="Market hours only",
        version="1.0",
    )
    gate.set_central_instruction(expired_instr)

    candle_eval = _make_timing_assessment(as_of=now)

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("EXPIRED" in r for r in outcome.rejection_reasons)


def test_human_stop_during_pending_entry_blocks_immediately() -> None:
    """Scenario 6: Human STOP instruction blocks subsequent new entries."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)

    # Central permits BUY
    central_instr = _make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(hours=1),
    )
    gate.set_central_instruction(central_instr)

    # Human intervenes with STOP_ALL
    gate.emergency_human_stop(reason="Operator emergency stop", current_time=now, scope="*")

    candle_eval = _make_timing_assessment(as_of=now)

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("HUMAN_PERMISSION_DENIED" in r for r in outcome.rejection_reasons)
    assert any("STOP_ALL" in r for r in outcome.rejection_reasons)


def test_human_stop_immediately_before_execution_race_protected() -> None:
    """Scenario 7: Permission change immediately before execution is caught atomically."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)

    # Central permits BUY
    central_instr = _make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(hours=1),
    )
    gate.set_central_instruction(central_instr)

    candle_eval = _make_timing_assessment(as_of=now)

    # Simulate permission change right as execution is evaluated
    gate.set_human_instruction(_make_instruction(
        authority=AuthoritySource.HUMAN,
        permission=PermissionType.NO_NEW_ENTRIES,
        scope="*",
        valid_from=now - timedelta(minutes=1),
        reason="Risk committee pause",
    ))

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("NO_NEW_ENTRIES" in r for r in outcome.rejection_reasons)


def test_human_stop_with_open_position_invokes_safe_management() -> None:
    """Scenario 8: Human STOP with open paper position triggers safe position management."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)

    # Open a paper position
    portfolio.submit_order(symbol="TCS", side="BUY", quantity=20, price=2500.0, timestamp=now)
    assert len(portfolio.positions) == 1
    assert portfolio.positions["TCS"].quantity == 20

    # Human triggers emergency stop with portfolio passed
    gate.emergency_human_stop(
        reason="Critical market alert",
        current_time=now + timedelta(minutes=5),
        scope="*",
        portfolio=portfolio,
    )

    # Safe position management must have squared off the position
    assert len(portfolio.positions) == 0
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0]["quantity"] == 20


def test_central_agent_unavailable_fails_closed() -> None:
    """Scenario 9: Missing Central Agent instruction -> Fails closed."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    # No central instruction registered
    candle_eval = _make_timing_assessment(as_of=now)

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("CENTRAL_AGENT_PERMISSION_MISSING_OR_EXPIRED" in r for r in outcome.rejection_reasons)


def test_risk_engine_rejection_overrides_human_and_central_approval() -> None:
    """Scenario 10: Human approval CANNOT override mandatory deterministic risk limits."""
    # Sized with a tight daily loss limit
    risk_engine = DeterministicRiskEngine(max_daily_loss_pct=0.01)
    gate = MandatoryExecutionGate(risk_engine=risk_engine)
    portfolio = IntradayPaperPortfolio(initial_capital=1000.0)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    # Both Human and Central approve
    gate.set_human_instruction(_make_instruction(
        authority=AuthoritySource.HUMAN,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=10),
    ))
    gate.set_central_instruction(_make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=10),
    ))

    # Even after Risk Engine sizing, ₹1,000 capital cannot cover 1 share @ ₹2,500 -> Risk Engine MUST reject!
    candle_eval = _make_timing_assessment(as_of=now)

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=50,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("RISK_ENGINE_REJECTION" in r for r in outcome.rejection_reasons)


def test_conflicting_human_and_central_instructions_human_restriction_wins() -> None:
    """Scenario 11: Central ALLOW_LONG vs Human NO_NEW_ENTRIES -> Human restriction wins."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    # Central says BUY
    gate.set_central_instruction(_make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=10),
    ))
    # Human says NO_NEW_ENTRIES
    gate.set_human_instruction(_make_instruction(
        authority=AuthoritySource.HUMAN,
        permission=PermissionType.NO_NEW_ENTRIES,
        valid_from=now - timedelta(minutes=10),
    ))

    candle_eval = _make_timing_assessment(as_of=now)

    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("HUMAN_PERMISSION_DENIED" in r for r in outcome.rejection_reasons)


def test_repeated_delayed_instructions_checked_for_expiry() -> None:
    """Scenario 12: Delayed / old instructions are checked for valid_from <= now <= expires_at."""
    now = datetime(2026, 11, 20, 12, 0, tzinfo=IST_TZ)

    # Future instruction (not yet valid)
    future_instr = AuthorityInstruction(
        instruction_id="FUTURE-1",
        authority=AuthoritySource.CENTRAL_AGENT,
        scope="TCS",
        permission=PermissionType.ALLOW_LONG,
        valid_from=now + timedelta(hours=1),
        expires_at=now + timedelta(hours=5),
        reason="Scheduled afternoon breakout",
        version="1.0",
    )
    assert future_instr.is_valid_at(now) is False

    # Expired instruction
    past_instr = AuthorityInstruction(
        instruction_id="PAST-1",
        authority=AuthoritySource.CENTRAL_AGENT,
        scope="TCS",
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(hours=5),
        expires_at=now - timedelta(hours=1),
        reason="Old morning session",
        version="1.0",
    )
    assert past_instr.is_valid_at(now) is False


def test_exit_only_permission_blocks_entry_permits_exit() -> None:
    """Scenario 13: EXIT_ONLY prohibits new exposure while allowing approved position reductions."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    gate.set_central_instruction(_make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.EXIT_ONLY,
        valid_from=now - timedelta(minutes=10),
    ))

    candle_eval = _make_timing_assessment(as_of=now)

    # Entry is blocked
    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=candle_eval,
    )
    assert outcome.approved is False
    assert any("EXIT_ONLY prohibits new exposure" in r for r in outcome.rejection_reasons)

    # Exit is allowed
    portfolio.submit_order(symbol="TCS", side="BUY", quantity=10, price=2500.0, timestamp=now)
    outcome_exit, executed_exit = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.EXIT_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2510.0,
    )
    assert outcome_exit.approved is True
    assert executed_exit is True


def test_candle_agent_cannot_self_authorize() -> None:
    """Scenario 14: Candle Agent alone cannot authorize execution without Central & Human permissions."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    # Candle Agent is enthusiastic
    strong_candle = _make_timing_assessment(
        decision=TimingDecision.ENTER_NOW,
        candle_bias=CandleBias.BULLISH,
        alignment=AlignmentStatus.CONFIRMED,
        as_of=now,
    )

    # But no Central instruction exists
    outcome, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=strong_candle,
    )

    assert outcome.approved is False
    assert executed is False
    assert any("CENTRAL_AGENT_PERMISSION_MISSING" in r for r in outcome.rejection_reasons)


def test_sha256_audit_trail_immutable_and_verifiable() -> None:
    """Scenario 15: SHA-256 hash-chained immutable audit ledger records all details."""
    gate = MandatoryExecutionGate()
    portfolio = IntradayPaperPortfolio(initial_capital=DEFAULT_INITIAL_CAPITAL_INR)
    now = datetime(2026, 11, 20, 10, 15, tzinfo=IST_TZ)

    gate.set_human_instruction(_make_instruction(
        authority=AuthoritySource.HUMAN,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=10),
    ))
    gate.set_central_instruction(_make_instruction(
        authority=AuthoritySource.CENTRAL_AGENT,
        permission=PermissionType.ALLOW_LONG,
        valid_from=now - timedelta(minutes=10),
    ))

    candle_eval = _make_timing_assessment(as_of=now)

    gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=now,
        portfolio=portfolio,
        requested_quantity=5,
        price=2500.0,
        candle_assessment=candle_eval,
    )

    trail = gate.audit_trail
    assert len(trail) >= 1
    rec = trail[-1]

    # Verify cryptographic integrity
    assert rec.verify_hash() is True
    assert rec.symbol == "TCS"
    assert rec.final_execution_decision == "APPROVED"
    assert rec.prev_hash == "0" * 64
    assert len(rec.record_hash) == 64


def test_zero_broker_integration_or_real_money_execution_permitted() -> None:
    """Scenario 16: Zero live broker integration / order-routing methods exist on gate."""
    gate = MandatoryExecutionGate()
    assert not hasattr(gate, "place_order")
    assert not hasattr(gate, "route_order")
    assert not hasattr(gate, "broker_connect")
    assert not hasattr(gate, "live_trade")
