"""BHARATALPHA-FORWARD-SOAK-AUTO-001: Comprehensive Autonomous Long-Only Soak Test Suite.

Covers all 18 required deterministic scenarios from Section 8:
 1. Automatic 09:30 IST transition after valid 15-min observation (NO_NEW_ENTRIES -> ALLOW_LONG)
 2. No transition when preflight fails
 3. No transition on NSE holiday (2026-01-26 & 2026-11-24)
 4. No transition with stale or incomplete market data
 5. No transition when Risk Engine is HALTED
 6. No transition when STOP_ALL is active
 7. Long entry allowed ONLY after authorised 09:30 IST transition (blocked 09:15-09:29, allowed >=09:30)
 8. Long exit permitted under EXIT_ONLY
 9. Short entry ALWAYS rejected (Gate, LivePaperTradingRunner long_only_standing_mode, CLI flags)
10. SELL cannot create a short position (EXIT_LONG qty > open_pos.qty rejected; CLOSE_LONG rejected)
11. Duplicate signals cannot create duplicate entries (_executed_signal_keys)
12. Concurrent entries cannot exceed capital or exposure limits (threading.Barrier(5))
13. Restart/checkpoint recovery cannot duplicate orders or lose paper positions (save/load_checkpoint)
14. Disconnection / stale-feed recovery handled safely (handle_disconnect -> NO_NEW_ENTRIES + RISK_EXIT)
15. End-of-day 15:15 square-off and reconciliation (0 open positions, cash == equity, 3 SHA-256 chains)
16. No real broker order API can be called (AST scan across trader/ & scripts/ + LIVE_TRADING_ENABLED=False)
17. Python/JavaScript parity for ALLOW_LONG, NO_NEW_ENTRIES -> ALLOW_LONG, short rejection, EXIT_ONLY
18. Audit-chain verification and scripts/evaluate_forward_paper_soak.py correctness
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts import evaluate_forward_paper_soak as soak_eval
from scripts import run_live_paper_session
from trader.authority.gate import MandatoryExecutionGate
from trader.authority.instructions import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)
from trader.config import IST_TZ, is_nse_trading_day
from trader.data.tick_recorder import (
    LIVE_TRADING_ENABLED as RECORDER_LIVE_TRADING_ENABLED,
    OfflineJsonlTickAdapter,
    RecordedBar,
)
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from trader.risk.engine import DeterministicRiskEngine, RiskState
from trader.timing.autonomous_session_controller import (
    LIVE_TRADING_ENABLED as CONTROLLER_LIVE_TRADING_ENABLED,
    AutonomousLongOnlySessionController,
    SessionStage,
)
from trader.timing.candle_intelligence import CandleBias
from trader.timing.candle_timing_agent import (
    AlignmentStatus,
    TimingAssessment,
    TimingDecision,
)
from trader.timing.central_directional_agent import DirectionalBias
from trader.timing.live_paper_runner import (
    LIVE_TRADING_ENABLED as RUNNER_LIVE_TRADING_ENABLED,
    READ_ONLY_MODE as RUNNER_READ_ONLY_MODE,
    LivePaperConfig,
    LivePaperTradingRunner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_TRADING_DAY = date(2026, 11, 20)  # Friday, non-holiday NSE trading day


# ---------------------------------------------------------------------------
# Deterministic Helpers
# ---------------------------------------------------------------------------


def _ist(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=IST_TZ)


def _make_advisory(
    symbol: str,
    ts: datetime,
    *,
    bullish: bool = True,
    entry: float = 2400.0,
) -> TimingAssessment:
    if bullish:
        sl = round(entry - 20.0, 2)
        tp = round(entry + 50.0, 2)
        cb = CandleBias.BULLISH
        cd = DirectionalBias.BULLISH
    else:
        sl = round(entry + 20.0, 2)
        tp = round(entry - 50.0, 2)
        cb = CandleBias.BEARISH
        cd = DirectionalBias.BEARISH
    return TimingAssessment(
        symbol=symbol,
        as_of_ist=ts,
        decision=TimingDecision.ENTER_NOW,
        alignment_status=AlignmentStatus.CONFIRMED,
        central_direction=cd,
        candle_bias=cb,
        timing_quality_score=0.80,
        price_action_score=0.80,
        entry_price=entry,
        invalidation_price=sl,
        target_price=tp,
        risk_reward_ratio=2.5,
        detected_patterns=(),
        timeframe_statuses={"1m": "VALID"},
        contradiction_reasons=(),
        rationale=("deterministic test advisory",),
    )


def _feed_clean_observation_window(
    controller: AutonomousLongOnlySessionController,
    gate: MandatoryExecutionGate,
    session_date: date,
    bars: int = 15,
) -> None:
    """Feed `bars` clean 1-minute ticks and bars starting at 09:15 IST."""
    for i in range(bars):
        t_tick = _ist(session_date, 9, 15) + timedelta(minutes=i, seconds=30)
        t_close = _ist(session_date, 9, 16) + timedelta(minutes=i)
        controller.record_tick_health(t_tick, execution_gate=gate)
        controller.record_completed_1m_bar(t_close)


def _write_trend_jsonl(
    jsonl_path: Path,
    *,
    symbol: str,
    session_date: date,
    direction: str = "UP",
    start_price: float = 2000.0,
    end_hour: int = 10,
    end_minute: int = 10,
) -> Path:
    """Write deterministic 1m-forming ticks (2 ticks/min) from 09:15 IST to end_hour:end_minute."""
    lines: list[str] = []
    cur = _ist(session_date, 9, 15, 5)
    end_ts = _ist(session_date, end_hour, end_minute, 55)
    price = start_price
    step = 1.5 if direction == "UP" else -1.5
    idx = 0
    while cur <= end_ts:
        # 3 ticks per minute so each 1m bar has clean OHLC and strong directional trend
        for sec_offset, delta in ((5, 0.0), (25, step * 0.4), (50, step)):
            ts = cur.replace(second=sec_offset)
            ltp = max(10.0, round(price + delta, 2))
            lines.append(
                json.dumps(
                    {
                        "symbol": symbol,
                        "exchange": "NSE",
                        "timestamp_ist": ts.isoformat(),
                        "ltp": ltp,
                        "volume_delta": 500.0 + (idx % 7) * 50.0,
                        "provider": "offline_replay",
                    }
                )
            )
        price = max(10.0, round(price + step, 2))
        cur += timedelta(minutes=1)
        idx += 1
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path


def _generate_valid_trading_dates(start_date: date, count: int) -> list[date]:
    days: list[date] = []
    cur = start_date
    while len(days) < count:
        if is_nse_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _write_soak_session(
    soak_dir: Path,
    folder_name: str,
    *,
    session_date: date,
    provider: str = "zerodha",
    source_mode: str = "LIVE_READ_ONLY",
    symbols: list[str] | None = None,
    trades: list[dict[str, Any]] | None = None,
    initial_capital: float = 50_000.0,
    authority_ok: bool = True,
    risk_ok: bool = True,
    ledger_ok: bool = True,
    interrupted: bool = False,
    session_status: str = "COMPLETED",
    live_trading_enabled: bool = False,
    read_only_mode: bool = True,
    open_positions: list[dict[str, Any]] | None = None,
    cash_equity_mismatch: float = 0.0,
) -> Path:
    sess_dir = soak_dir / folder_name
    sess_dir.mkdir(parents=True, exist_ok=True)
    syms = symbols or ["TCS", "RELIANCE", "ITC"]
    tr_list = trades or []
    total_gross = round(sum(float(t["gross_pnl_inr"]) for t in tr_list), 2)
    total_fric = round(sum(float(t["friction_inr"]) for t in tr_list), 2)
    total_net = round(sum(float(t["net_pnl_inr"]) for t in tr_list), 2)
    final_equity = round(initial_capital + total_net, 2)
    final_cash = round(final_equity + cash_equity_mismatch, 2)

    prov_files: list[str] = []
    for sym in syms:
        prov_dir = sess_dir / "ticks" / sym / session_date.isoformat()
        prov_dir.mkdir(parents=True, exist_ok=True)
        prov_path = prov_dir / "provenance.json"
        prov_path.write_text(
            json.dumps(
                {
                    "symbol": sym,
                    "trading_date": session_date.isoformat(),
                    "provider": provider,
                    "source_mode": source_mode,
                    "live_trading_enabled": live_trading_enabled,
                    "read_only": read_only_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        prov_files.append(str(prov_path))

    state = {
        "schema_version": "cycle021_remed01_live_paper_session_v2",
        "session_date": session_date.isoformat(),
        "session_status": session_status,
        "interrupted": interrupted,
        "provider": provider,
        "source_mode": source_mode,
        "symbols": syms,
        "strategy_profile": "ORB_BREAKOUT",
        "live_trading_enabled": live_trading_enabled,
        "read_only_mode": read_only_mode,
        "ticks_accepted": 720,
        "bars_1m_evaluated": 360,
        "trade_count": len(tr_list),
        "initial_capital_inr": initial_capital,
        "total_gross_pnl_inr": total_gross,
        "total_friction_inr": total_fric,
        "total_net_pnl_inr": total_net,
        "final_cash_inr": final_cash,
        "final_equity_inr": final_equity,
        "open_positions": open_positions if open_positions is not None else [],
        "final_risk_state": "REDUCING",
        "gate_rejections_count": 1,
        "authority_audit_verified": authority_ok,
        "risk_audit_verified": risk_ok,
        "ledger_integrity_verified": ledger_ok,
        "max_single_stock_exposure_pct": 14.4,
        "max_portfolio_exposure_pct": 43.2,
        "margin_call_triggered": False,
        "cap_breaches": 0,
        "trades": tr_list,
        "archived_provenance_files": prov_files,
    }
    state_path = sess_dir / "live_paper_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


# ===========================================================================
# Scenario 1: Automatic 09:30 IST transition after valid 15-min observation
# ===========================================================================


def test_01_automatic_0930_transition_after_valid_observation(tmp_path: Path) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)
    lock_file = tmp_path / "runner.lock"

    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS", "RELIANCE"),
        output_dir=tmp_path,
    )
    try:
        preflight = ctrl.run_preflight(risk_engine=risk, portfolio=portfolio, lock_file=lock_file)
        assert preflight.passed is True
        assert all(preflight.checks.values())

        obs_instr = ctrl.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))
        assert obs_instr.permission == PermissionType.NO_NEW_ENTRIES
        assert ctrl.stage == SessionStage.OBSERVING_NO_NEW_ENTRIES

        _feed_clean_observation_window(ctrl, gate, VALID_TRADING_DAY, bars=15)

        # At 09:29 IST -> still before 09:30 transition time -> must NOT transition
        early = ctrl.evaluate_0930_transition(
            execution_gate=gate,
            risk_engine=risk,
            portfolio=portfolio,
            current_time=_ist(VALID_TRADING_DAY, 9, 29, 59),
        )
        assert early.transitioned is False
        assert any("BEFORE_TRANSITION_TIME" in r for r in early.reasons)
        assert ctrl.stage == SessionStage.OBSERVING_NO_NEW_ENTRIES

        # At 09:30 IST -> all 8 preconditions pass -> transitions to ALLOW_LONG
        decision = ctrl.evaluate_0930_transition(
            execution_gate=gate,
            risk_engine=risk,
            portfolio=portfolio,
            current_time=_ist(VALID_TRADING_DAY, 9, 30, 0),
        )
        assert decision.transitioned is True
        assert decision.from_stage == SessionStage.OBSERVING_NO_NEW_ENTRIES.value
        assert decision.to_stage == SessionStage.AUTONOMOUS_ALLOW_LONG.value
        assert decision.instruction_id == f"AUTONOMOUS-LONG-{VALID_TRADING_DAY.isoformat()}"
        assert ctrl.stage == SessionStage.AUTONOMOUS_ALLOW_LONG

        active_human = gate.get_active_human_instruction("TCS", _ist(VALID_TRADING_DAY, 9, 31))
        assert active_human is not None
        assert active_human.permission == PermissionType.ALLOW_LONG

        # Re-evaluating at 09:31 is idempotent
        repeat = ctrl.evaluate_0930_transition(
            execution_gate=gate,
            risk_engine=risk,
            portfolio=portfolio,
            current_time=_ist(VALID_TRADING_DAY, 9, 31),
        )
        assert repeat.transitioned is False
        assert repeat.reasons == ("ALREADY_IN_AUTONOMOUS_ALLOW_LONG",)
    finally:
        ctrl.release_lock()
        assert not lock_file.exists()


# ===========================================================================
# Scenario 2: No transition when preflight fails
# ===========================================================================


def test_02_no_transition_when_preflight_fails(tmp_path: Path) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    # Case A: preflight fails due to connectivity_ok=False & clock_synced=False
    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    pf = ctrl.run_preflight(
        risk_engine=risk,
        portfolio=portfolio,
        connectivity_ok=False,
        clock_synced=False,
    )
    assert pf.passed is False
    assert "PREFLIGHT_FAILED:connectivity_ok" in pf.failure_reasons
    assert "PREFLIGHT_FAILED:clock_synced" in pf.failure_reasons
    assert ctrl.stage == SessionStage.HALTED_OR_INTERRUPTED

    decision = ctrl.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert decision.transitioned is False
    assert any("PRECONDITION_1_FAILED:PREFLIGHT_NOT_PASSED" in r for r in decision.reasons)
    assert gate.get_active_human_instruction("TCS", _ist(VALID_TRADING_DAY, 9, 30)) is None

    # Case B: preflight never run at all
    ctrl2 = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    ctrl2.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))
    _feed_clean_observation_window(ctrl2, gate, VALID_TRADING_DAY, bars=15)
    decision2 = ctrl2.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert decision2.transitioned is False
    assert "PRECONDITION_1_FAILED:PREFLIGHT_NOT_PASSED" in decision2.reasons
    active = gate.get_active_human_instruction("TCS", _ist(VALID_TRADING_DAY, 9, 30))
    assert active is not None and active.permission == PermissionType.NO_NEW_ENTRIES


# ===========================================================================
# Scenario 3: No transition on NSE holiday (2026-01-26 & 2026-11-24)
# ===========================================================================


@pytest.mark.parametrize(
    "holiday_date",
    [date(2026, 1, 26), date(2026, 11, 24)],
    ids=["republic_day_2026_01_26", "guru_nanak_jayanti_2026_11_24"],
)
def test_03_no_transition_on_nse_holiday(tmp_path: Path, holiday_date: date) -> None:
    assert is_nse_trading_day(holiday_date) is False

    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    ctrl = AutonomousLongOnlySessionController(
        session_date=holiday_date,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    pf = ctrl.run_preflight(risk_engine=risk, portfolio=portfolio)
    assert pf.passed is False
    assert "PREFLIGHT_FAILED:nse_trading_day" in pf.failure_reasons

    # Even if someone forces observation and feeds 15 bars, Stage C still blocks on calendar
    ctrl.start_observation(gate, _ist(holiday_date, 9, 15))
    _feed_clean_observation_window(ctrl, gate, holiday_date, bars=15)
    decision = ctrl.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(holiday_date, 9, 30),
    )
    assert decision.transitioned is False
    assert "PRECONDITION_2_FAILED:NOT_NSE_TRADING_DAY" in decision.reasons
    assert ctrl.stage != SessionStage.AUTONOMOUS_ALLOW_LONG
    active = gate.get_active_human_instruction("TCS", _ist(holiday_date, 9, 30))
    assert active is not None and active.permission == PermissionType.NO_NEW_ENTRIES


# ===========================================================================
# Scenario 4: No transition with stale or incomplete market data
# ===========================================================================


def test_04_no_transition_with_stale_or_incomplete_market_data(tmp_path: Path) -> None:
    # Subcase A: Incomplete market data (only 5 bars < min_observation_bars=10)
    portfolio_a = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk_a = DeterministicRiskEngine()
    gate_a = MandatoryExecutionGate(risk_engine=risk_a)
    ctrl_a = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path / "incomplete",
        min_observation_bars=10,
    )
    assert ctrl_a.run_preflight(risk_engine=risk_a, portfolio=portfolio_a).passed is True
    ctrl_a.start_observation(gate_a, _ist(VALID_TRADING_DAY, 9, 15))
    _feed_clean_observation_window(ctrl_a, gate_a, VALID_TRADING_DAY, bars=5)

    dec_a = ctrl_a.evaluate_0930_transition(
        execution_gate=gate_a,
        risk_engine=risk_a,
        portfolio=portfolio_a,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert dec_a.transitioned is False
    assert any("INSUFFICIENT_OBSERVATION_BARS" in r for r in dec_a.reasons)

    # Subcase B: Stale feed gap (> 180s) during observation
    portfolio_b = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk_b = DeterministicRiskEngine()
    gate_b = MandatoryExecutionGate(risk_engine=risk_b)
    ctrl_b = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path / "stale",
        max_stale_seconds=180.0,
    )
    assert ctrl_b.run_preflight(risk_engine=risk_b, portfolio=portfolio_b).passed is True
    ctrl_b.start_observation(gate_b, _ist(VALID_TRADING_DAY, 9, 15))
    ctrl_b.record_tick_health(_ist(VALID_TRADING_DAY, 9, 15, 10), execution_gate=gate_b)
    # 240-second gap (> 180s) triggers STALE_FEED_INTERRUPTION
    ctrl_b.record_tick_health(_ist(VALID_TRADING_DAY, 9, 19, 10), execution_gate=gate_b)
    assert ctrl_b.interrupted is True
    assert ctrl_b.stage == SessionStage.HALTED_OR_INTERRUPTED
    assert "STALE_FEED_INTERRUPTION" in ctrl_b.critical_errors

    for i in range(12):
        ctrl_b.record_completed_1m_bar(_ist(VALID_TRADING_DAY, 9, 16) + timedelta(minutes=i))

    dec_b = ctrl_b.evaluate_0930_transition(
        execution_gate=gate_b,
        risk_engine=risk_b,
        portfolio=portfolio_b,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert dec_b.transitioned is False
    assert "PRECONDITION_3_FAILED:FEED_INTERRUPTED_OR_STALE" in dec_b.reasons
    assert any("PRECONDITION_4_FAILED:CRITICAL_ERRORS" in r for r in dec_b.reasons)
    active_b = gate_b.get_active_human_instruction("TCS", _ist(VALID_TRADING_DAY, 9, 30))
    assert active_b is not None and active_b.permission == PermissionType.NO_NEW_ENTRIES


# ===========================================================================
# Scenario 5: No transition when Risk Engine is HALTED
# ===========================================================================


def test_05_no_transition_when_risk_engine_halted(tmp_path: Path) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    assert ctrl.run_preflight(risk_engine=risk, portfolio=portfolio).passed is True
    ctrl.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))
    _feed_clean_observation_window(ctrl, gate, VALID_TRADING_DAY, bars=15)

    # Risk engine enters HALTED before 09:30
    risk.state = RiskState.HALTED

    decision = ctrl.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert decision.transitioned is False
    assert "PRECONDITION_5_FAILED:RISK_ENGINE_NOT_ACTIVE:HALTED" in decision.reasons
    assert ctrl.stage == SessionStage.OBSERVING_NO_NEW_ENTRIES
    active = gate.get_active_human_instruction("TCS", _ist(VALID_TRADING_DAY, 9, 30))
    assert active is not None and active.permission == PermissionType.NO_NEW_ENTRIES


# ===========================================================================
# Scenario 6: No transition when STOP_ALL is active
# ===========================================================================


@pytest.mark.parametrize(
    "override_perm",
    [PermissionType.STOP_ALL, PermissionType.PAUSE, PermissionType.EXIT_ONLY],
)
def test_06_no_transition_when_stop_all_or_human_override_active(
    tmp_path: Path, override_perm: PermissionType
) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    assert ctrl.run_preflight(risk_engine=risk, portfolio=portfolio).passed is True
    ctrl.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))
    _feed_clean_observation_window(ctrl, gate, VALID_TRADING_DAY, bars=15)

    # Human issues STOP_ALL / PAUSE / EXIT_ONLY at 09:25 IST
    override = AuthorityInstruction(
        instruction_id=f"HUMAN-OVERRIDE-{override_perm.value}",
        authority=AuthoritySource.HUMAN,
        scope="*",
        permission=override_perm,
        valid_from=_ist(VALID_TRADING_DAY, 9, 25),
        expires_at=_ist(VALID_TRADING_DAY, 16, 0),
        reason=f"Human issued {override_perm.value}",
    )
    assert gate.set_human_instruction(override, current_time=_ist(VALID_TRADING_DAY, 9, 25)).accepted

    decision = ctrl.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert decision.transitioned is False
    assert f"PRECONDITION_7_FAILED:HUMAN_OVERRIDE_ACTIVE:{override_perm.value}" in decision.reasons
    active = gate.get_active_human_instruction("TCS", _ist(VALID_TRADING_DAY, 9, 30))
    assert active is not None and active.permission == override_perm


# ===========================================================================
# Scenario 7: Long entry allowed ONLY after authorised 09:30 IST transition
# ===========================================================================


def test_07_long_entry_blocked_during_observation_allowed_after_0930(tmp_path: Path) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    # Central agent permits LONG all day
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="CENTRAL-BULL-DAY",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 15),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Bullish daily bias",
        ),
        current_time=_ist(VALID_TRADING_DAY, 9, 15),
    )

    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    assert ctrl.run_preflight(risk_engine=risk, portfolio=portfolio).passed is True
    ctrl.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))
    _feed_clean_observation_window(ctrl, gate, VALID_TRADING_DAY, bars=15)

    # 09:25 IST (during observation window) -> ENTRY_LONG MUST BE BLOCKED
    ts_obs = _ist(VALID_TRADING_DAY, 9, 25)
    outcome_obs, exec_obs = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=ts_obs,
        portfolio=portfolio,
        requested_quantity=2,
        price=2400.0,
        candle_assessment=_make_advisory("TCS", ts_obs, bullish=True, entry=2400.0),
        stop_loss_price=2380.0,
        target_price=2450.0,
    )
    assert outcome_obs.approved is False
    assert exec_obs is False
    assert "TCS" not in portfolio.positions

    # 09:30 IST -> authorised transition
    tr = ctrl.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    )
    assert tr.transitioned is True

    # 09:31 IST -> ENTRY_LONG NOW ALLOWED AND EXECUTED
    ts_live = _ist(VALID_TRADING_DAY, 9, 31)
    outcome_live, exec_live = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=ts_live,
        portfolio=portfolio,
        requested_quantity=2,
        price=2400.0,
        candle_assessment=_make_advisory("TCS", ts_live, bullish=True, entry=2400.0),
        stop_loss_price=2380.0,
        target_price=2450.0,
    )
    assert outcome_live.approved is True
    assert exec_live is True
    assert "TCS" in portfolio.positions
    assert portfolio.positions["TCS"].side == "LONG"
    assert portfolio.positions["TCS"].quantity == 2


# ===========================================================================
# Scenario 8: Long exit permitted under EXIT_ONLY
# ===========================================================================


def test_08_long_exit_permitted_under_exit_only() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    t_entry = _ist(VALID_TRADING_DAY, 9, 31)
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-LONG",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Standing long-only",
        ),
        current_time=t_entry,
    )
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-LONG",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Bullish",
        ),
        current_time=t_entry,
    )

    _, opened = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=t_entry,
        portfolio=portfolio,
        requested_quantity=2,
        price=2400.0,
        candle_assessment=_make_advisory("TCS", t_entry, bullish=True, entry=2400.0),
        stop_loss_price=2380.0,
        target_price=2450.0,
    )
    assert opened is True
    assert portfolio.positions["TCS"].quantity == 2

    # Switch Human permission to EXIT_ONLY
    t_exit = _ist(VALID_TRADING_DAY, 9, 45)
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-EXIT-ONLY",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.EXIT_ONLY,
            valid_from=t_exit,
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Operator EXIT_ONLY",
        ),
        current_time=t_exit,
    )

    # New ENTRY_LONG must be blocked under EXIT_ONLY
    new_out, new_exec = gate.execute_with_gate_protection(
        symbol="RELIANCE",
        action_type=ActionType.ENTRY_LONG,
        current_time=t_exit,
        portfolio=portfolio,
        requested_quantity=2,
        price=1500.0,
        candle_assessment=_make_advisory("RELIANCE", t_exit, bullish=True, entry=1500.0),
        stop_loss_price=1480.0,
        target_price=1550.0,
    )
    assert new_out.approved is False
    assert new_exec is False

    # EXIT_LONG must be PERMITTED and close the open LONG position
    exit_out, exit_exec = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.EXIT_LONG,
        current_time=t_exit,
        portfolio=portfolio,
        requested_quantity=2,
        price=2420.0,
    )
    assert exit_out.approved is True
    assert exit_exec is True
    assert "TCS" not in portfolio.positions
    assert len(portfolio.closed_trades) == 1


# ===========================================================================
# Scenario 9: Short entry ALWAYS rejected (Gate, Runner, CLI)
# ===========================================================================


def test_09_short_entry_always_rejected_gate_runner_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Part A: MandatoryExecutionGate rejects ENTRY_SHORT under ALLOW_LONG
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)
    t_now = _ist(VALID_TRADING_DAY, 9, 35)
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-LONG-ONLY",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Standing long-only",
        ),
        current_time=t_now,
    )
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-SHORT",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_SHORT,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Bearish signal",
        ),
        current_time=t_now,
    )
    short_out, short_exec = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_SHORT,
        current_time=t_now,
        portfolio=portfolio,
        requested_quantity=2,
        price=2400.0,
        candle_assessment=_make_advisory("TCS", t_now, bullish=False, entry=2400.0),
        stop_loss_price=2420.0,
        target_price=2350.0,
    )
    assert short_out.approved is False
    assert short_exec is False
    assert len(portfolio.positions) == 0

    # Part B: LivePaperTradingRunner with long_only_standing_mode=True rejects shorts on bearish tape
    bear_jsonl = _write_trend_jsonl(
        tmp_path / "bear_ticks.jsonl",
        symbol="TCS",
        session_date=VALID_TRADING_DAY,
        direction="DOWN",
        start_price=2600.0,
        end_hour=10,
        end_minute=5,
    )
    runner = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(bear_jsonl),
        output_dir=tmp_path / "runner_out",
        config=LivePaperConfig(strategy_profile="BALANCED_ALL", auto_direction=False),
        human_instruction=AuthorityInstruction(
            instruction_id="H-BOTH-TEST",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.ALLOW_BOTH,
            valid_from=_ist(VALID_TRADING_DAY, 9, 0),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Stress test runner-level long-only guard",
        ),
        central_instruction=AuthorityInstruction(
            instruction_id="C-BEAR-TEST",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_SHORT,
            valid_from=_ist(VALID_TRADING_DAY, 9, 0),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Bearish central direction",
        ),
        long_only_standing_mode=True,
    )
    summary = runner.run(["TCS"])
    assert summary.bars_1m_evaluated >= 40
    assert all(t.side != "SELL" for t in summary.trades)
    assert len(summary.trades) == 0

    # Part C: CLI --autonomous-long-only rejects --human-permission ALLOW_SHORT and ALLOW_BOTH
    for forbidden_perm in ("ALLOW_SHORT", "ALLOW_BOTH"):
        rc = run_live_paper_session.main(
            [
                "--provider",
                "offline_jsonl",
                "--jsonl-file",
                str(bear_jsonl),
                "--symbols",
                "TCS",
                "--autonomous-long-only",
                "--human-permission",
                forbidden_perm,
                "--output-dir",
                str(tmp_path / f"cli_{forbidden_perm}"),
            ]
        )
        captured = capsys.readouterr()
        assert rc == 2
        assert "REJECTED" in captured.err
        assert forbidden_perm in captured.err


# ===========================================================================
# Scenario 10: SELL cannot create a short position
# ===========================================================================


def test_10_sell_cannot_create_short_position() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    t0 = _ist(VALID_TRADING_DAY, 9, 31)
    # Open a LONG position of 5 shares @ ₹1,000
    buy_order = portfolio.submit_order(
        symbol="ITC",
        side=OrderSide.BUY,
        quantity=5,
        price=1000.0,
        timestamp=t0,
        position_effect="OPEN_LONG",
    )
    assert buy_order.status == OrderStatus.FILLED
    assert portfolio.positions["ITC"].side == "LONG"
    assert portfolio.positions["ITC"].quantity == 5

    # 1. EXIT_LONG with qty > open_pos.qty (8 > 5) MUST BE REJECTED (cannot flip to short)
    t1 = _ist(VALID_TRADING_DAY, 9, 40)
    over_sell = portfolio.submit_order(
        symbol="ITC",
        side=OrderSide.SELL,
        quantity=8,
        price=1005.0,
        timestamp=t1,
        position_effect="EXIT_LONG",
    )
    assert over_sell.status == OrderStatus.REJECTED
    assert portfolio.positions["ITC"].side == "LONG"
    assert portfolio.positions["ITC"].quantity == 5

    # 2. MandatoryExecutionGate also rejects EXIT_LONG with qty > open_pos.qty
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-LONG",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Long only",
        ),
        current_time=t1,
    )
    gate_out, gate_exec = gate.execute_with_gate_protection(
        symbol="ITC",
        action_type=ActionType.EXIT_LONG,
        current_time=t1,
        portfolio=portfolio,
        requested_quantity=8,
        price=1005.0,
    )
    assert gate_out.approved is False
    assert gate_exec is False
    assert portfolio.positions["ITC"].quantity == 5

    # 3. Unrecognized position_effect="CLOSE_LONG" MUST BE REJECTED fail-closed
    close_long_order = portfolio.submit_order(
        symbol="ITC",
        side=OrderSide.SELL,
        quantity=5,
        price=1005.0,
        timestamp=t1,
        position_effect="CLOSE_LONG",  # type: ignore[arg-type]
    )
    assert close_long_order.status == OrderStatus.REJECTED
    assert "Unrecognized position_effect" in (close_long_order.rejection_reason or "")
    assert portfolio.positions["ITC"].side == "LONG"
    assert portfolio.positions["ITC"].quantity == 5

    # 4. SELL EXIT_LONG when flat cannot open a short position
    flat_sell = portfolio.submit_order(
        symbol="TCS",
        side=OrderSide.SELL,
        quantity=2,
        price=2400.0,
        timestamp=t1,
        position_effect="EXIT_LONG",
    )
    assert flat_sell.status == OrderStatus.REJECTED
    assert "TCS" not in portfolio.positions


# ===========================================================================
# Scenario 11: Duplicate signals cannot create duplicate entries
# ===========================================================================


def test_11_duplicate_signals_cannot_create_duplicate_entries(tmp_path: Path) -> None:
    bull_jsonl = _write_trend_jsonl(
        tmp_path / "bull_dup.jsonl",
        symbol="TCS",
        session_date=VALID_TRADING_DAY,
        direction="UP",
        start_price=2000.0,
        end_hour=10,
        end_minute=15,
    )
    cfg = LivePaperConfig(
        strategy_profile="BALANCED_ALL",
        auto_direction=False,
        cooldown_bars=0,
        max_entries_per_day=10,
        min_reward_to_friction=1.0,
    )
    human_long = AuthorityInstruction(
        instruction_id="H-DUP-LONG",
        authority=AuthoritySource.HUMAN,
        scope="*",
        permission=PermissionType.ALLOW_LONG,
        valid_from=_ist(VALID_TRADING_DAY, 9, 0),
        expires_at=_ist(VALID_TRADING_DAY, 16, 0),
        reason="Long only",
    )
    central_long = AuthorityInstruction(
        instruction_id="C-DUP-LONG",
        authority=AuthoritySource.CENTRAL_AGENT,
        scope="*",
        permission=PermissionType.ALLOW_LONG,
        valid_from=_ist(VALID_TRADING_DAY, 9, 0),
        expires_at=_ist(VALID_TRADING_DAY, 16, 0),
        reason="Bullish",
    )

    runner1 = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(bull_jsonl),
        output_dir=tmp_path / "run1",
        config=cfg,
        human_instruction=human_long,
        central_instruction=central_long,
        long_only_standing_mode=True,
    )
    summary1 = runner1.run(["TCS"])
    assert len(summary1.trades) >= 1, "Expected at least one genuine LONG trade on strong uptrend"
    assert len(runner1._executed_signal_keys) >= 1

    # Replay the exact same stream with a second runner that already has those signal keys
    runner2 = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(bull_jsonl),
        output_dir=tmp_path / "run2",
        config=cfg,
        human_instruction=human_long,
        central_instruction=central_long,
        long_only_standing_mode=True,
    )
    runner2._executed_signal_keys.update(runner1._executed_signal_keys)
    # Pre-seed ALL possible BUY signal keys from the first run -> zero duplicate entries for those keys
    first_signal_key = next(iter(runner1._executed_signal_keys))
    assert first_signal_key.startswith("TCS|") and first_signal_key.endswith("|BUY")

    # Verify re-evaluating the exact same bar timestamp with flat portfolio skips the duplicate key
    ts_iso = first_signal_key.split("|")[1]
    ts_open = datetime.fromisoformat(ts_iso)
    # Warm up runner2 with the exact bars before ts_open, then feed ts_open bar
    prior_bars = [b for b in runner1._bars_1m_history["TCS"] if b["timestamp"] < ts_open]
    target_bar_dict = next(b for b in runner1._bars_1m_history["TCS"] if b["timestamp"] == ts_open)
    runner2._bars_1m_history["TCS"] = list(prior_bars)
    runner2._active_session_date = VALID_TRADING_DAY
    runner2._evaluate_closed_1m_bar(
        RecordedBar(
            symbol="TCS",
            timeframe="1m",
            ts_open_ist=target_bar_dict["timestamp"],
            ts_close_ist=target_bar_dict["timestamp"] + timedelta(minutes=1),
            open=target_bar_dict["open"],
            high=target_bar_dict["high"],
            low=target_bar_dict["low"],
            close=target_bar_dict["close"],
            volume=target_bar_dict["volume"],
            tick_count=3,
            source_provider="offline_replay",
        )
    )
    assert len(runner2.portfolio.positions) == 0, (
        f"Duplicate signal {first_signal_key} must NOT create a duplicate entry"
    )


# ===========================================================================
# Scenario 12: Concurrent entries cannot exceed capital or exposure limits
# ===========================================================================


def test_12_concurrent_entries_cannot_exceed_capital_or_exposure_limits() -> None:
    initial_capital = 50_000.0
    portfolio = IntradayPaperPortfolio(initial_capital=initial_capital)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    t_now = _ist(VALID_TRADING_DAY, 9, 35)
    gate.set_human_instruction(
        AuthorityInstruction(
            instruction_id="H-CONC",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Standing long-only",
        ),
        current_time=t_now,
    )
    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-CONC",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 30),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Bullish",
        ),
        current_time=t_now,
    )

    symbols = ("TCS", "RELIANCE", "ITC", "INFY", "HDFCBANK")
    prices = {"TCS": 2400.0, "RELIANCE": 1500.0, "ITC": 450.0, "INFY": 1600.0, "HDFCBANK": 1700.0}
    barrier = threading.Barrier(len(symbols))
    errors: list[Exception] = []

    def _worker(sym: str) -> bool:
        try:
            px = prices[sym]
            sl = round(px * 0.99, 2)
            tp = round(px * 1.025, 2)
            adv = _make_advisory(sym, t_now, bullish=True, entry=px)
            barrier.wait(timeout=5.0)
            _, executed = gate.execute_with_gate_protection(
                symbol=sym,
                action_type=ActionType.ENTRY_LONG,
                current_time=t_now,
                portfolio=portfolio,
                requested_quantity=100,  # deliberately huge to stress 15% & 80% caps
                price=px,
                candle_assessment=adv,
                stop_loss_price=sl,
                target_price=tp,
            )
            return executed
        except Exception as exc:
            errors.append(exc)
            return False

    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        results = list(pool.map(_worker, symbols))

    assert errors == [], f"Unexpected thread errors: {errors}"
    assert any(results), "At least one concurrent LONG entry should succeed"

    total_exposure = 0.0
    for sym, pos in portfolio.positions.items():
        notional = pos.quantity * pos.entry_price
        total_exposure += notional
        assert notional <= initial_capital * 0.15 + 1e-6, (
            f"{sym} notional ₹{notional:.2f} breached 15% single-stock cap (₹{initial_capital * 0.15:.2f})"
        )

    assert total_exposure <= initial_capital * 0.80 + 1e-6, (
        f"Total exposure ₹{total_exposure:.2f} breached 80% portfolio cap"
    )
    assert portfolio.cash >= 0.0
    assert risk.cash_ledger.total_reserved_inr == 0.0
    ledger_ok, msg = portfolio.verify_ledger_integrity()
    assert ledger_ok is True, msg


# ===========================================================================
# Scenario 13: Restart/checkpoint recovery cannot duplicate orders or lose positions
# ===========================================================================


def test_13_restart_checkpoint_recovery_preserves_positions_and_dedup_keys(tmp_path: Path) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    t_entry = _ist(VALID_TRADING_DAY, 9, 35)
    order = portfolio.submit_order(
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=3,
        price=2400.0,
        timestamp=t_entry,
        stop_loss=2370.0,
        target=2475.0,
        position_effect="OPEN_LONG",
    )
    assert order.status == OrderStatus.FILLED

    sig_key = f"TCS|{_ist(VALID_TRADING_DAY, 9, 34).isoformat()}|BUY"
    ctrl1 = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    ctrl1.stage = SessionStage.AUTONOMOUS_ALLOW_LONG
    ckpt_path = ctrl1.save_checkpoint(portfolio, {sig_key}, trade_count=1)
    assert ckpt_path.exists()

    # Simulate process restart with a fresh controller instance
    ctrl2 = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    loaded = ctrl2.load_checkpoint()
    assert loaded is not None
    assert loaded["session_date"] == VALID_TRADING_DAY.isoformat()
    assert loaded["trade_count"] == 1
    assert sig_key in loaded["executed_signal_keys"]
    tcs_positions = [v for v in loaded["open_positions"].values() if v["symbol"] == "TCS"]
    assert len(tcs_positions) == 1
    assert tcs_positions[0]["quantity"] == 3
    assert tcs_positions[0]["entry_price"] == 2400.0
    assert tcs_positions[0]["side"] == "LONG"
    assert abs(loaded["cash_inr"] - round(portfolio.cash, 2)) < 1e-6

    # Fresh LivePaperTradingRunner automatically restores executed_signal_keys from checkpoint
    dummy_jsonl = tmp_path / "empty.jsonl"
    dummy_jsonl.write_text("", encoding="utf-8")
    runner_restarted = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(dummy_jsonl),
        output_dir=tmp_path,
        session_controller=ctrl2,
    )
    assert sig_key in runner_restarted._executed_signal_keys

    # A controller for a different trading date MUST ignore yesterday's checkpoint
    ctrl_other_day = AutonomousLongOnlySessionController(
        session_date=date(2026, 11, 23),
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    assert ctrl_other_day.load_checkpoint() is None


# ===========================================================================
# Scenario 14: Disconnection / stale-feed recovery handled safely
# ===========================================================================


def test_14_disconnect_safe_downgrade_preserves_risk_exit(tmp_path: Path) -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    gate.set_central_instruction(
        AuthorityInstruction(
            instruction_id="C-DISC",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 15),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Bullish",
        ),
        current_time=_ist(VALID_TRADING_DAY, 9, 15),
    )

    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path,
    )
    assert ctrl.run_preflight(risk_engine=risk, portfolio=portfolio).passed is True
    ctrl.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))
    _feed_clean_observation_window(ctrl, gate, VALID_TRADING_DAY, bars=15)
    assert ctrl.evaluate_0930_transition(
        execution_gate=gate,
        risk_engine=risk,
        portfolio=portfolio,
        current_time=_ist(VALID_TRADING_DAY, 9, 30),
    ).transitioned is True

    # Open LONG position after 09:30
    t_open = _ist(VALID_TRADING_DAY, 9, 32)
    _, opened = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=t_open,
        portfolio=portfolio,
        requested_quantity=2,
        price=2400.0,
        candle_assessment=_make_advisory("TCS", t_open, bullish=True, entry=2400.0),
        stop_loss_price=2380.0,
        target_price=2450.0,
    )
    assert opened is True

    # Feed disconnects at 09:45 IST -> handle_disconnect downgrades to NO_NEW_ENTRIES
    t_disc = _ist(VALID_TRADING_DAY, 9, 45)
    disc_instr = ctrl.handle_disconnect(gate, t_disc, "WEBSOCKET_DROPPED")
    assert disc_instr.permission == PermissionType.NO_NEW_ENTRIES
    assert ctrl.interrupted is True
    assert ctrl.stage == SessionStage.HALTED_OR_INTERRUPTED

    # New ENTRY_LONG at 09:46 MUST BE REJECTED
    t_after = _ist(VALID_TRADING_DAY, 9, 46)
    new_out, new_exec = gate.execute_with_gate_protection(
        symbol="RELIANCE",
        action_type=ActionType.ENTRY_LONG,
        current_time=t_after,
        portfolio=portfolio,
        requested_quantity=2,
        price=1500.0,
        candle_assessment=_make_advisory("RELIANCE", t_after, bullish=True, entry=1500.0),
        stop_loss_price=1480.0,
        target_price=1550.0,
    )
    assert new_out.approved is False
    assert new_exec is False

    # Safe RISK_EXIT MUST STILL BE PERMITTED to flatten the open position
    exit_out, exit_exec = gate.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.RISK_EXIT,
        current_time=t_after,
        portfolio=portfolio,
        requested_quantity=2,
        price=2395.0,
    )
    assert exit_out.approved is True
    assert exit_exec is True
    assert len(portfolio.positions) == 0


# ===========================================================================
# Scenario 15: End-of-day 15:15 square-off and reconciliation
# ===========================================================================


def test_15_eod_1515_squareoff_and_reconciliation(tmp_path: Path) -> None:
    jsonl_path = _write_trend_jsonl(
        tmp_path / "full_day_uptrend.jsonl",
        symbol="TCS",
        session_date=VALID_TRADING_DAY,
        direction="UP",
        start_price=2000.0,
        end_hour=15,
        end_minute=20,
    )
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    risk = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk)

    ctrl = AutonomousLongOnlySessionController(
        session_date=VALID_TRADING_DAY,
        provider="offline_jsonl",
        symbols=("TCS",),
        output_dir=tmp_path / "eod_out",
    )
    assert ctrl.run_preflight(risk_engine=risk, portfolio=portfolio).passed is True
    ctrl.start_observation(gate, _ist(VALID_TRADING_DAY, 9, 15))

    runner = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(jsonl_path),
        output_dir=tmp_path / "eod_out",
        config=LivePaperConfig(
            strategy_profile="BALANCED_ALL",
            auto_direction=False,
            min_reward_to_friction=1.0,
        ),
        portfolio=portfolio,
        execution_gate=gate,
        central_instruction=AuthorityInstruction(
            instruction_id="C-EOD-BULL",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=PermissionType.ALLOW_LONG,
            valid_from=_ist(VALID_TRADING_DAY, 9, 0),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="Full-day bullish bias",
        ),
        session_controller=ctrl,
        long_only_standing_mode=True,
    )
    summary = runner.run(["TCS"])

    assert len(runner.portfolio.positions) == 0
    assert abs(summary.final_cash_inr - summary.final_equity_inr) < 1e-6
    assert summary.authority_audit_verified is True
    assert summary.risk_audit_verified is True
    assert summary.ledger_integrity_verified is True
    assert summary.live_trading_enabled is False
    assert summary.read_only_mode is True
    assert summary.autonomous_controller_summary is not None
    assert summary.autonomous_controller_summary["stage"] == SessionStage.COMPLETED.value
    assert len(summary.trades) >= 1
    assert all(t.side == "BUY" for t in summary.trades)
    assert summary.state_json_path is not None and Path(summary.state_json_path).exists()


# ===========================================================================
# Scenario 16: No real broker order API can be called (AST scan + flags)
# ===========================================================================


def test_16_ast_zero_broker_order_calls_and_live_trading_disabled() -> None:
    forbidden = {"place_order", "modify_order", "cancel_order"}
    violations: list[str] = []

    for root_dir in (REPO_ROOT / "trader", REPO_ROOT / "scripts"):
        for py_file in sorted(root_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno} calls .{node.func.attr}()"
                        )
                    elif isinstance(node.func, ast.Name) and node.func.id in forbidden:
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno} calls {node.func.id}()"
                        )

    assert violations == [], f"Forbidden broker order API calls detected: {violations}"
    assert CONTROLLER_LIVE_TRADING_ENABLED is False
    assert RUNNER_LIVE_TRADING_ENABLED is False
    assert RECORDER_LIVE_TRADING_ENABLED is False
    assert RUNNER_READ_ONLY_MODE is True


# ===========================================================================
# Scenario 17: Python/JavaScript parity for long-only permissions
# ===========================================================================


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_17_python_js_parity_long_only_permissions() -> None:
    js_script = r"""
const E = require("./docs/candle-agent/engine.js");
const perms = ["NO_NEW_ENTRIES", "ALLOW_LONG", "EXIT_ONLY", "STOP_ALL"];
const actions = ["OPEN_LONG", "OPEN_SHORT", "EXIT_LONG", "RISK_EXIT"];
const matrix = {};
for (const p of perms) {
  matrix[p] = {};
  for (const a of actions) {
    matrix[p][a] = E.humanPermissionAllows(p, a).allowed;
  }
}
// Simulate 09:30 transition NO_NEW_ENTRIES -> ALLOW_LONG
const before0930 = E.humanPermissionAllows("NO_NEW_ENTRIES", "OPEN_LONG").allowed;
const after0930Long = E.humanPermissionAllows("ALLOW_LONG", "OPEN_LONG").allowed;
const after0930Short = E.humanPermissionAllows("ALLOW_LONG", "OPEN_SHORT").allowed;
const exitOnlyLongExit = E.humanPermissionAllows("EXIT_ONLY", "EXIT_LONG").allowed;
const exitOnlyLongEntry = E.humanPermissionAllows("EXIT_ONLY", "OPEN_LONG").allowed;
console.log(JSON.stringify({
  matrix,
  before0930,
  after0930Long,
  after0930Short,
  exitOnlyLongExit,
  exitOnlyLongEntry,
}));
"""
    proc = subprocess.run(
        ["node", "-e", js_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    js_res = json.loads(proc.stdout.strip())

    action_map = {
        "OPEN_LONG": ActionType.ENTRY_LONG,
        "OPEN_SHORT": ActionType.ENTRY_SHORT,
        "EXIT_LONG": ActionType.EXIT_LONG,
        "RISK_EXIT": ActionType.RISK_EXIT,
    }
    for perm_name, js_row in js_res["matrix"].items():
        py_instr = AuthorityInstruction(
            instruction_id=f"PARITY-{perm_name}",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType(perm_name),
            valid_from=_ist(VALID_TRADING_DAY, 9, 15),
            expires_at=_ist(VALID_TRADING_DAY, 16, 0),
            reason="JS/Py parity check",
        )
        for js_act, py_act in action_map.items():
            py_allowed, _ = py_instr.allows_action(py_act)
            assert js_row[js_act] is py_allowed, (
                f"Parity mismatch for ({perm_name}, {js_act}): JS={js_row[js_act]} vs Py={py_allowed}"
            )

    assert js_res["before0930"] is False
    assert js_res["after0930Long"] is True
    assert js_res["after0930Short"] is False
    assert js_res["exitOnlyLongExit"] is True
    assert js_res["exitOnlyLongEntry"] is False


# ===========================================================================
# Scenario 18: Audit-chain verification & evaluate_forward_paper_soak.py
# ===========================================================================


def test_18_evaluate_forward_paper_soak_passing_and_exclusions(tmp_path: Path) -> None:
    # Part A: 30 valid sessions, 60 completed trades, 3 distinct symbols -> QUALIFIED_FOR_FURTHER_REVIEW
    pass_soak_dir = tmp_path / "soak_pass"
    pass_out_dir = tmp_path / "soak_pass_out"
    valid_days = _generate_valid_trading_dates(date(2026, 9, 1), 30)
    symbols_cycle = ["TCS", "RELIANCE", "ITC"]

    for idx, d in enumerate(valid_days):
        s1 = symbols_cycle[idx % 3]
        s2 = symbols_cycle[(idx + 1) % 3]
        # 2 trades per session = 60 trades total; net positive with PF > 1.25 and low DD
        win_trade = {
            "symbol": s1,
            "side": "BUY",
            "entry_time_ist": _ist(d, 9, 35).isoformat(),
            "exit_time_ist": _ist(d, 10, 15).isoformat(),
            "entry_price": 2000.0,
            "exit_price": 2045.0,
            "quantity": 3,
            "gross_pnl_inr": 135.0,
            "friction_inr": 15.0,
            "net_pnl_inr": 120.0,
            "exit_reason": "TARGET_HIT",
            "strategy_profile": "ORB_BREAKOUT",
        }
        second_trade = (
            {
                "symbol": s2,
                "side": "BUY",
                "entry_time_ist": _ist(d, 11, 5).isoformat(),
                "exit_time_ist": _ist(d, 11, 45).isoformat(),
                "entry_price": 1500.0,
                "exit_price": 1485.0,
                "quantity": 4,
                "gross_pnl_inr": -60.0,
                "friction_inr": 10.0,
                "net_pnl_inr": -70.0,
                "exit_reason": "STOP_LOSS_HIT",
                "strategy_profile": "ORB_BREAKOUT",
            }
            if idx % 2 == 0
            else {
                "symbol": s2,
                "side": "BUY",
                "entry_time_ist": _ist(d, 11, 5).isoformat(),
                "exit_time_ist": _ist(d, 11, 45).isoformat(),
                "entry_price": 1500.0,
                "exit_price": 1525.0,
                "quantity": 4,
                "gross_pnl_inr": 100.0,
                "friction_inr": 10.0,
                "net_pnl_inr": 90.0,
                "exit_reason": "TARGET_HIT",
                "strategy_profile": "ORB_BREAKOUT",
            }
        )
        _write_soak_session(
            pass_soak_dir,
            f"session_{d.isoformat()}",
            session_date=d,
            trades=[win_trade, second_trade],
        )

    pass_report = soak_eval.evaluate_forward_paper_soak(
        pass_soak_dir,
        output_dir=pass_out_dir,
        write_outputs=True,
    )
    assert pass_report["qualification_status"] == "QUALIFIED_FOR_FURTHER_REVIEW"
    assert pass_report["real_money_authorized"] is False
    assert pass_report["valid_sessions"] == 30
    assert pass_report["invalid_sessions"] == 0
    assert pass_report["completed_trades"] == 60
    assert pass_report["distinct_symbol_count"] == 3
    assert pass_report["profit_factor"] >= 1.25
    assert all(g["passed"] for g in pass_report["gates"].values())
    assert (pass_out_dir / "session_index.json").exists()
    assert (pass_out_dir / "forward_evaluation_report.json").exists()
    assert (pass_out_dir / "forward_evaluation_report.md").exists()

    # Part B: Incomplete, NSE holiday, synthetic, and audit-failed sessions are excluded & fail qualification
    fail_soak_dir = tmp_path / "soak_fail"
    fail_out_dir = tmp_path / "soak_fail_out"
    sample_trade = {
        "symbol": "TCS",
        "side": "BUY",
        "entry_time_ist": _ist(VALID_TRADING_DAY, 9, 40).isoformat(),
        "exit_time_ist": _ist(VALID_TRADING_DAY, 10, 10).isoformat(),
        "entry_price": 2400.0,
        "exit_price": 2430.0,
        "quantity": 3,
        "gross_pnl_inr": 90.0,
        "friction_inr": 10.0,
        "net_pnl_inr": 80.0,
        "exit_reason": "TARGET_HIT",
        "strategy_profile": "ORB_BREAKOUT",
    }

    # 1 valid session
    _write_soak_session(
        fail_soak_dir,
        "01_valid",
        session_date=VALID_TRADING_DAY,
        trades=[sample_trade],
    )
    # 2 incomplete/interrupted session
    _write_soak_session(
        fail_soak_dir,
        "02_interrupted",
        session_date=date(2026, 11, 19),
        interrupted=True,
        session_status="INTERRUPTED",
        trades=[sample_trade],
    )
    # 3 NSE holiday session (2026-01-26)
    _write_soak_session(
        fail_soak_dir,
        "03_holiday",
        session_date=date(2026, 1, 26),
        trades=[sample_trade],
    )
    # 4 synthetic contaminated session
    _write_soak_session(
        fail_soak_dir,
        "04_synthetic",
        session_date=date(2026, 11, 18),
        provider="synthetic",
        source_mode="SYNTHETIC_TEST",
        trades=[sample_trade],
    )
    # 5 audit-failed session
    _write_soak_session(
        fail_soak_dir,
        "05_audit_fail",
        session_date=date(2026, 11, 17),
        ledger_ok=False,
        trades=[sample_trade],
    )

    fail_report = soak_eval.evaluate_forward_paper_soak(
        fail_soak_dir,
        output_dir=fail_out_dir,
        write_outputs=True,
    )
    assert fail_report["qualification_status"] == "NOT_QUALIFIED"
    assert fail_report["real_money_authorized"] is False
    assert fail_report["valid_sessions"] == 1
    assert fail_report["invalid_sessions"] == 4
    assert fail_report["synthetic_contamination_count"] == 1
    assert fail_report["gates"]["gate_1_data_provenance"]["passed"] is False
    assert fail_report["gates"]["gate_2_minimum_forward_sample"]["passed"] is False

    idx_data = json.loads((fail_out_dir / "session_index.json").read_text(encoding="utf-8"))
    by_id = {s["session_id"]: s for s in idx_data["sessions"]}
    assert by_id["01_valid"]["valid"] is True
    assert any("incomplete_or_interrupted_session" in r for r in by_id["02_interrupted"]["exclusion_reasons"])
    assert any("non_nse_trading_day_or_holiday" in r for r in by_id["03_holiday"]["exclusion_reasons"])
    assert "synthetic_data_contamination" in by_id["04_synthetic"]["exclusion_reasons"]
    assert any("audit_integrity_failed" in r for r in by_id["05_audit_fail"]["exclusion_reasons"])
