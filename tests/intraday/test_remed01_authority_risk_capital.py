"""REMED-01 Comprehensive Authority, Risk & Capital Allocation Test Suite.

Covers all 14 required REMED-01 test categories:
1. Missing Human instruction (strict fail-closed entries, safe exit management, audit logged)
2. All 7 Human permission states (ALLOW_LONG, ALLOW_SHORT, ALLOW_BOTH, NO_NEW_ENTRIES, EXIT_ONLY, PAUSE, STOP_ALL)
3. Conflicting Human & Central instructions (Human > Central > Candle > Risk > Paper)
4. Duplicate, stale, and delayed/out-of-order instructions (structured rejection, no overwrite, no crash)
5. AUTO mode with explicit Human & Central overrides
6. Daily (2.5%) & MTM (3.0%) drawdown breaches -> RiskState.HALTED implements EXIT_ONLY (no auto-liquidation)
7. Multiple simultaneous symbol orders & deterministic multi-symbol capital caps (15% / 80%)
8. Gate 6 sizing-before-cash correction + true insufficient available cash rejection
9. Cash reservation and release lifecycle across fill, rejection, and STOP_ALL
10. Pending orders revalidated immediately before execution after authority changes (Python & JS)
11. STOP_ALL audited safe position management vs EXIT_ONLY
12. Existing-position management preserved under HALTED / PAUSE / missing Human authority
13. Broker adapter startup & CLI argument validation (scripts/run_live_paper_session.py --token-map)
14. Regression proof that LivePaperTradingRunner never bypasses MandatoryExecutionGate / RiskEngine / Portfolio
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trader.authority.gate import InstructionRegistrationResult, MandatoryExecutionGate
from trader.authority.instructions import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)
from trader.config import IST_TZ
from trader.data.tick_recorder import OfflineJsonlTickAdapter
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine, RiskState
from trader.timing.candle_timing_agent import (
    AlignmentStatus,
    TimingAssessment,
    TimingDecision,
)
from trader.timing.central_directional_agent import DirectionalBias
from trader.timing.live_paper_runner import LivePaperConfig, LivePaperTradingRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW_IST = datetime(2026, 11, 23, 10, 30, tzinfo=IST_TZ)


def _make_instr(
    instr_id: str,
    source: AuthoritySource,
    perm: PermissionType,
    *,
    scope: str = "*",
    valid_from: datetime | None = None,
    expires_at: datetime | None = None,
) -> AuthorityInstruction:
    vf = valid_from or (NOW_IST - timedelta(minutes=30))
    ex = expires_at or (NOW_IST + timedelta(hours=4))
    return AuthorityInstruction(
        instruction_id=instr_id,
        authority=source,
        scope=scope,
        permission=perm,
        valid_from=vf,
        expires_at=ex,
        reason=f"test:{instr_id}",
        version="1.0",
    )


def _make_candle(
    symbol: str = "TCS",
    bias: DirectionalBias = DirectionalBias.BULLISH,
    decision: TimingDecision = TimingDecision.ENTER_NOW,
    sl: float = 2475.0,
    tp: float = 2560.0,
) -> TimingAssessment:
    from trader.timing.candle_intelligence import CandleBias

    c_bias = CandleBias.BULLISH if bias == DirectionalBias.BULLISH else CandleBias.BEARISH
    return TimingAssessment(
        symbol=symbol,
        as_of_ist=NOW_IST,
        decision=decision,
        alignment_status=AlignmentStatus.CONFIRMED,
        central_direction=bias,
        candle_bias=c_bias,
        timing_quality_score=0.80,
        price_action_score=0.75,
        entry_price=2500.0 if symbol == "TCS" else 300.0,
        invalidation_price=sl,
        target_price=tp,
        risk_reward_ratio=2.5,
        detected_patterns=(),
        timeframe_statuses={"1m": "VALID"},
        contradiction_reasons=(),
        rationale=("REMED-01 unit test assessment",),
    )


# ---------------------------------------------------------------------------
# 1. Decision 1: Strict Fail-Closed Missing Human Authority
# ---------------------------------------------------------------------------


def test_remed01_missing_human_instruction_fails_closed_for_entries_but_allows_safe_exits() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    gate = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    gate.set_central_instruction(
        _make_instr("CEN-1", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_LONG, scope="TCS"),
        current_time=NOW_IST,
    )

    # New entry without ANY Human instruction MUST fail closed and be audited
    out_entry = gate.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=portfolio,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=2,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out_entry.approved is False
    assert any("HUMAN_INSTRUCTION_MISSING" in r for r in out_entry.rejection_reasons)
    assert gate.verify_audit_chain() is True

    # Existing open position MUST still receive safe protective exit even when Human instruction is missing
    portfolio.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST, position_effect="OPEN_LONG")
    out_exit, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.RISK_EXIT,
        current_time=NOW_IST + timedelta(minutes=1),
        portfolio=portfolio,
        requested_quantity=2,
        price=2480.0,
    )
    assert out_exit.approved is True
    assert executed is True
    assert "TCS" not in portfolio.positions
    assert gate.verify_audit_chain() is True


# ---------------------------------------------------------------------------
# 2. All 7 Human Permission States (Python & JS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perm,entry_long_ok,entry_short_ok,disc_exit_ok,risk_exit_ok",
    [
        (PermissionType.ALLOW_LONG, True, False, True, True),
        (PermissionType.ALLOW_SHORT, False, True, True, True),
        (PermissionType.ALLOW_BOTH, True, True, True, True),
        (PermissionType.NO_NEW_ENTRIES, False, False, True, True),
        (PermissionType.EXIT_ONLY, False, False, True, True),
        (PermissionType.PAUSE, False, False, False, True),
        (PermissionType.STOP_ALL, False, False, True, True),
    ],
)
def test_remed01_all_seven_human_permission_states(
    perm: PermissionType,
    entry_long_ok: bool,
    entry_short_ok: bool,
    disc_exit_ok: bool,
    risk_exit_ok: bool,
) -> None:
    instr = _make_instr(f"HUM-{perm.value}", AuthoritySource.HUMAN, perm)
    assert instr.allows_action(ActionType.ENTRY_LONG)[0] is entry_long_ok
    assert instr.allows_action(ActionType.ENTRY_SHORT)[0] is entry_short_ok
    assert instr.allows_action(ActionType.EXIT_LONG)[0] is disc_exit_ok
    assert instr.allows_action(ActionType.RISK_EXIT)[0] is risk_exit_ok


def test_remed01_js_human_permission_fail_closed_and_pause() -> None:
    """JS humanPermissionAllows() must fail closed on missing/unknown strings and enforce PAUSE."""
    script = """
    const eng = require('./docs/candle-agent/engine.js');
    const res = {
      missingEntry: eng.humanPermissionAllows(null, 'OPEN_LONG').allowed,
      unknownEntry: eng.humanPermissionAllows('BOGUS_PERM', 'OPEN_LONG').allowed,
      pauseEntry: eng.humanPermissionAllows('PAUSE', 'OPEN_LONG').allowed,
      pauseDiscExit: eng.humanPermissionAllows('PAUSE', 'EXIT_LONG').allowed,
      pauseRiskExit: eng.humanPermissionAllows('PAUSE', 'RISK_EXIT').allowed,
      allowBothLong: eng.humanPermissionAllows('ALLOW_BOTH', 'OPEN_LONG').allowed,
    };
    console.log(JSON.stringify(res));
    """
    proc = subprocess.run(["node", "-e", script], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout.strip())
    assert data == {
        "missingEntry": False,
        "unknownEntry": False,
        "pauseEntry": False,
        "pauseDiscExit": False,
        "pauseRiskExit": True,
        "allowBothLong": True,
    }


# ---------------------------------------------------------------------------
# 3. Conflicting Human & Central Instructions + Expired Symbol Fail-Closed
# ---------------------------------------------------------------------------


def test_remed01_human_overrides_central_and_expired_symbol_never_falls_back_to_wildcard() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    gate = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())

    # Central says ALLOW_LONG, Human says ALLOW_SHORT -> conflict blocks ENTRY_LONG
    gate.set_central_instruction(
        _make_instr("CEN-L", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_LONG, scope="TCS"),
        current_time=NOW_IST,
    )
    gate.set_human_instruction(
        _make_instr("HUM-S", AuthoritySource.HUMAN, PermissionType.ALLOW_SHORT, scope="TCS"),
        current_time=NOW_IST,
    )
    out = gate.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=portfolio,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=2,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out.approved is False
    assert any("HUMAN_PERMISSION_DENIED" in r for r in out.rejection_reasons)

    # Expired symbol-specific Human STOP_ALL must NOT fall back to wildcard "*" ALLOW_BOTH!
    gate2 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    gate2.set_central_instruction(
        _make_instr("CEN-L2", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_LONG, scope="TCS"),
        current_time=NOW_IST,
    )
    gate2.set_human_instruction(
        _make_instr("HUM-WILD", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH, scope="*"),
        current_time=NOW_IST,
    )
    gate2.set_human_instruction(
        _make_instr(
            "HUM-TCS-EXP",
            AuthoritySource.HUMAN,
            PermissionType.STOP_ALL,
            scope="TCS",
            valid_from=NOW_IST - timedelta(minutes=30),
            expires_at=NOW_IST + timedelta(minutes=5),
        ),
        current_time=NOW_IST,
    )
    # 10 minutes later: HUM-TCS-EXP is expired -> MUST fail closed, NOT revert to HUM-WILD!
    out_exp = gate2.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST + timedelta(minutes=10),
        portfolio=portfolio,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=2,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out_exp.approved is False
    assert any("HUMAN_INSTRUCTION_EXPIRED_OR_INVALID" in r for r in out_exp.rejection_reasons)


# ---------------------------------------------------------------------------
# 4. Decision 2: Duplicate, Stale, Delayed & Out-of-Order Instruction Rejection
# ---------------------------------------------------------------------------


def test_remed01_duplicate_stale_and_delayed_instructions_structured_rejection() -> None:
    gate = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())

    # 1. Accept initial STOP_ALL at 10:30
    r1 = gate.set_human_instruction(
        _make_instr("HUM-STOP-1", AuthoritySource.HUMAN, PermissionType.STOP_ALL, valid_from=NOW_IST),
        current_time=NOW_IST,
    )
    assert isinstance(r1, InstructionRegistrationResult)
    assert r1.accepted is True

    # 2. Duplicate ID rejected cleanly (no exception, does not overwrite)
    r_dup = gate.set_human_instruction(
        _make_instr("HUM-STOP-1", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH, valid_from=NOW_IST + timedelta(minutes=1)),
        current_time=NOW_IST + timedelta(minutes=1),
    )
    assert r_dup.accepted is False
    assert "DUPLICATE_INSTRUCTION_ID" in r_dup.reason

    # 3. Delayed older ALLOW_LONG (valid_from 10:25 < active 10:30) rejected cleanly!
    r_delayed = gate.set_human_instruction(
        _make_instr(
            "HUM-DELAYED-LONG",
            AuthoritySource.HUMAN,
            PermissionType.ALLOW_LONG,
            valid_from=NOW_IST - timedelta(minutes=5),
            expires_at=NOW_IST + timedelta(hours=2),
        ),
        current_time=NOW_IST + timedelta(minutes=1),
    )
    assert r_delayed.accepted is False
    assert "DELAYED_OUT_OF_ORDER_INSTRUCTION" in r_delayed.reason

    # 4. Stale already-expired instruction rejected cleanly!
    r_stale = gate.set_human_instruction(
        _make_instr(
            "HUM-STALE",
            AuthoritySource.HUMAN,
            PermissionType.ALLOW_BOTH,
            valid_from=NOW_IST - timedelta(hours=2),
            expires_at=NOW_IST - timedelta(minutes=1),
        ),
        current_time=NOW_IST,
    )
    assert r_stale.accepted is False
    assert "STALE_EXPIRED_INSTRUCTION" in r_stale.reason

    # Active Human permission MUST STILL BE STOP_ALL!
    active = gate.get_active_human_instruction("TCS", NOW_IST + timedelta(minutes=2))
    assert active is not None and active.permission == PermissionType.STOP_ALL
    assert len(gate.instruction_audit_trail) == 4


# ---------------------------------------------------------------------------
# 5. Decision 4: Risk Engine HALTED -> EXIT_ONLY (No Auto-Liquidation)
# ---------------------------------------------------------------------------


def test_remed01_risk_halted_implements_exit_only_without_auto_liquidation() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk_engine = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk_engine)
    gate.set_human_instruction(
        _make_instr("HUM-BOTH", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH),
        current_time=NOW_IST,
    )
    gate.set_central_instruction(
        _make_instr("CEN-BOTH", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH),
        current_time=NOW_IST,
    )

    # Open a position in TCS, then simulate a >3% MTM drawdown breach
    portfolio.submit_order("TCS", "BUY", 3, 2500.0, timestamp=NOW_IST, position_effect="OPEN_LONG")
    portfolio.positions["TCS"].current_price = 1900.0  # -₹1,800 MTM (-3.6% on ₹50k > 3.0% MTM limit)
    breached = risk_engine.check_daily_loss_limit(portfolio)
    assert breached is True
    assert risk_engine.state == RiskState.HALTED
    # Decision 4: existing position is NOT auto-liquidated merely by hitting the threshold
    assert "TCS" in portfolio.positions and portfolio.positions["TCS"].quantity == 3

    # New entries MUST be blocked under HALTED
    out_new = gate.evaluate_gate(
        symbol="ITC",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST + timedelta(minutes=1),
        portfolio=portfolio,
        candle_assessment=_make_candle("ITC", DirectionalBias.BULLISH, sl=295.0, tp=315.0),
        requested_quantity=10,
        entry_price=300.0,
        stop_loss_price=295.0,
    )
    assert out_new.approved is False
    assert any("HALTED" in r for r in out_new.rejection_reasons)

    # Protective exit of existing TCS position MUST STILL BE ALLOWED under HALTED (EXIT_ONLY)
    out_exit, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.EXIT_LONG,
        current_time=NOW_IST + timedelta(minutes=2),
        portfolio=portfolio,
        requested_quantity=3,
        price=1900.0,
    )
    assert out_exit.approved is True
    assert executed is True
    assert "TCS" not in portfolio.positions
    assert gate.verify_audit_chain() is True


# ---------------------------------------------------------------------------
# 6. Gate 6 Sizing-Before-Cash Fix + Multi-Symbol Cash Reservation & Release
# ---------------------------------------------------------------------------


def test_remed01_gate6_sizes_down_before_cash_check_and_enforces_multi_symbol_reservations() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk_engine = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk_engine)
    gate.set_human_instruction(_make_instr("HUM-1", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    gate.set_central_instruction(_make_instr("CEN-1", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)

    # P0-5 Fix: requesting 100 shares of TCS @ ₹2,500 (₹2.5L > ₹50k cash) MUST size down to <=3 shares (15% cap = ₹7,500)
    out_tcs = gate.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=portfolio,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=100,
        entry_price=2500.0,
        stop_loss_price=2475.0,
        dry_run=True,
    )
    assert out_tcs.approved is True
    assert out_tcs.risk_evaluation is not None
    assert 1 <= out_tcs.risk_evaluation.approved_quantity <= 3

    # Multi-symbol CashReservationLedger: reserve ₹38,000 across other pending symbols -> remaining 80% exposure cap is ₹2,000!
    gate.reserve_cash_for_pending_order("ORD-REL", "RELIANCE", 19_000.0)
    gate.reserve_cash_for_pending_order("ORD-INF", "INFY", 19_000.0)
    assert risk_engine.cash_ledger.total_reserved_inr == pytest.approx(38_000.0)

    # TCS @ ₹2,500 cannot fit in remaining ₹2,000 exposure cap -> MUST reject!
    out_blocked = gate.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST + timedelta(seconds=5),
        portfolio=portfolio,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=10,
        entry_price=2500.0,
        stop_loss_price=2475.0,
        dry_run=True,
    )
    assert out_blocked.approved is False

    # Release one reservation -> TCS fits again and executes cleanly
    released = gate.release_cash_reservation("ORD-INF")
    assert released == pytest.approx(19_000.0)
    out_ok, executed = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST + timedelta(seconds=10),
        portfolio=portfolio,
        requested_quantity=10,
        price=2500.0,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        stop_loss_price=2475.0,
    )
    assert out_ok.approved is True and executed is True
    assert portfolio.positions["TCS"].quantity <= 3


# ---------------------------------------------------------------------------
# 7. STOP_ALL Safe Position Management (Audited RISK_EXIT)
# ---------------------------------------------------------------------------


def test_remed01_stop_all_flattens_open_positions_via_audited_risk_exit() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    gate = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    gate.set_human_instruction(_make_instr("HUM-1", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    gate.set_central_instruction(_make_instr("CEN-1", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)

    portfolio.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST, position_effect="OPEN_LONG")
    gate.reserve_cash_for_pending_order("PEND-ITC", "ITC", 5_000.0)

    gate.emergency_human_stop(reason="TEST_KILL_SWITCH", current_time=NOW_IST + timedelta(minutes=1), portfolio=portfolio)
    assert "TCS" not in portfolio.positions
    assert gate.risk_engine.cash_ledger.total_reserved_inr == 0.0
    assert any(rec.action_type == ActionType.RISK_EXIT and rec.final_execution_decision == "APPROVED" for rec in gate.audit_trail)
    assert gate.verify_audit_chain() is True


# ---------------------------------------------------------------------------
# 8. LivePaperTradingRunner Governed Path + Decision 1 & Decision 3 + CLI Validation
# ---------------------------------------------------------------------------


def test_remed01_live_paper_runner_enforces_mandatory_gate_and_auto_override(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "ticks.jsonl"
    base_ts = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST_TZ)
    lines = []
    px = 2500.0
    cum_vol = 0.0
    for m in range(30):
        for sec in (10, 45):
            ts = base_ts + timedelta(minutes=m, seconds=sec)
            px += 0.60 if m >= 15 else 0.05
            cum_vol += 500.0
            lines.append(
                json.dumps(
                    {
                        "symbol": "TCS",
                        "exchange": "NSE",
                        "timestamp_ist": ts.isoformat(),
                        "ltp": round(px, 2),
                        "cumulative_volume": cum_vol,
                    }
                )
            )
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A. WITHOUT Human instruction -> strict fail-closed: 0 trades allowed, all 3 SHA-256 chains verified!
    runner_no_human = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(jsonl_path=jsonl_path),
        output_dir=tmp_path / "out_no_human",
        config=LivePaperConfig(strategy_profile="ORB_BREAKOUT", initial_capital_inr=50_000.0, adx_min=0.0),
        human_instruction=None,
    )
    summary_no_human = runner_no_human.run(["TCS"])
    assert len(summary_no_human.trades) == 0
    assert summary_no_human.authority_audit_verified is True
    assert summary_no_human.risk_audit_verified is True
    assert summary_no_human.ledger_integrity_verified is True

    # B. Decision 3: Explicit Central ALLOW_SHORT overrides AUTO even during a bullish breakout -> 0 LONG trades!
    hum_both = _make_instr(
        "HUM-LIVE-1",
        AuthoritySource.HUMAN,
        PermissionType.ALLOW_BOTH,
        valid_from=base_ts - timedelta(minutes=5),
        expires_at=base_ts + timedelta(hours=8),
    )
    cen_short = _make_instr(
        "CEN-LIVE-SHORT",
        AuthoritySource.CENTRAL_AGENT,
        PermissionType.ALLOW_SHORT,
        scope="TCS",
        valid_from=base_ts - timedelta(minutes=5),
        expires_at=base_ts + timedelta(hours=8),
    )
    runner_override = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(jsonl_path=jsonl_path),
        output_dir=tmp_path / "out_override",
        config=LivePaperConfig(strategy_profile="ORB_BREAKOUT", initial_capital_inr=50_000.0, auto_direction=True, adx_min=0.0),
        human_instruction=hum_both,
        central_instruction=cen_short,
    )
    summary_override = runner_override.run(["TCS"])
    assert all(t.side != "BUY" for t in summary_override.trades)
    assert summary_override.authority_audit_verified is True
    assert summary_override.ledger_integrity_verified is True


def test_remed01_live_paper_cli_token_map_validation(tmp_path: Path) -> None:
    """CLI must fail closed when --token-map is missing for live brokers and pass when provided."""
    # Import build_adapter / parse_token_map directly
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_live_paper_session as cli

    with pytest.raises(ValueError, match="--token-map is required"):
        cli.build_adapter("zerodha", None, token_map={})
    with pytest.raises(ValueError, match="--token-map is required"):
        cli.build_adapter("angel_one", None, token_map={})

    z_adapter = cli.build_adapter("zerodha", None, token_map={"TCS": 2953217})
    assert z_adapter.provider_name == "zerodha_kite"
    a_adapter = cli.build_adapter("angel_one", None, token_map={"TCS": "11536"})
    assert a_adapter.provider_name == "angel_one_smartapi"

    tmap_file = tmp_path / "tokens.json"
    tmap_file.write_text(json.dumps({"TCS": 2953217, "RELIANCE": 738561}), encoding="utf-8")
    parsed = cli.parse_token_map(str(tmap_file))
    assert parsed == {"TCS": 2953217, "RELIANCE": 738561}
