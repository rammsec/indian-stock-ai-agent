"""REMED-01-FIX-01 Comprehensive Adversarial Behavioral Test Suite.

Genuinely verifies every REMED-01 authority, risk, capital reservation, parity,
and non-bypass requirement under realistic failure modes:
1. auto_direction=True in IntradayCandleMonitoringSession (and yielding to explicit hypothesis)
2. Missing Human permission fail-closed across Gate, MonitoringSession, LivePaperRunner & JS engine.js
3. Invalid, expired, duplicate, stale, out-of-order, and delayed instructions (Human & Central + mid-session)
4. All 7 Human permission states end-to-end via execute_with_gate_protection (LONG & SHORT & RISK_EXIT)
5. Mandatory 15:15 IST square-off under Human PAUSE, EXIT_ONLY, STOP_ALL, and HALTED
6. STOP_ALL at bar index 0 (Python & JS)
7. LivePaperTradingRunner shutdown square-off with open LONG, open SHORT, and multi-symbol positions
8. Synchronized 5-thread concurrent multi-symbol reservations (ThreadPoolExecutor + Barrier)
9. Symbol-scoped STOP_ALL reservation release & selective flattening
10. Pending-order revalidation after mid-session authority changes (Python & JS Step A)
11. Isolated 2.5% daily drawdown breach AND isolated 3.0% MTM drawdown breach (HALTED -> EXIT_ONLY)
12. Fee-inclusive insufficient cash boundary & Gate 6 sizing-before-cash
13. CLI main() invocation and argument validation (scripts/run_live_paper_session.py)
14. Intrabar SL/TP audit trail in IntradayCandleMonitoringSession (zero unaudited mutations)
15. Explicit 5-scenario Python/JavaScript parity matrix
16. AST + dynamic spy regression proving zero execution-gate bypasses
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trader.authority.gate import InstructionRegistrationResult, MandatoryExecutionGate
from trader.authority.instructions import (
    ActionType,
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)
from trader.config import IST_TZ
from trader.data.market_feed import BaseMarketFeed, TimeframeLiteral
from trader.data.tick_recorder import OfflineJsonlTickAdapter
from trader.intelligence.data_contract import (
    OHLCVEvidenceFrame,
    ProvenanceTier,
    build_ohlcv_evidence_frame,
)
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine, RiskState
from trader.timing.candle_intelligence import CandleBias
from trader.timing.candle_timing_agent import (
    AlignmentStatus,
    TimingAssessment,
    TimingDecision,
)
from trader.timing.central_directional_agent import DirectionalBias, DirectionalHypothesis
from trader.timing.live_paper_runner import LivePaperConfig, LivePaperTradingRunner
from trader.timing.monitoring_session import IntradayCandleMonitoringSession

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW_IST = datetime(2026, 11, 23, 10, 30, tzinfo=IST_TZ)


class _StaticBarFeed(BaseMarketFeed):
    """Strictly causal in-memory feed for deterministic monitoring_session tests."""

    def __init__(self, symbol: str, timeframe: TimeframeLiteral, rows: list[dict[str, Any]]) -> None:
        self._symbol = symbol.upper()
        self._tf = timeframe
        self._df = pd.DataFrame(rows)

    @property
    def provider_name(self) -> str:
        return "remed01_test_feed"

    @property
    def is_authoritative(self) -> bool:
        return False

    @property
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        return (self._tf,)

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        as_of = as_of_ist.astimezone(IST_TZ)
        causal = self._df.loc[self._df["ts_close_ist"] <= as_of].tail(lookback_bars).reset_index(drop=True)
        return build_ohlcv_evidence_frame(
            symbol=symbol.strip().upper(),
            timeframe=timeframe,
            df=causal if not causal.empty else None,
            as_of_ist=as_of,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED if not causal.empty else ProvenanceTier.UNAVAILABLE,
            source_name=self.provider_name,
            exchange="NSE",
        )


def _make_5m_day_rows(
    session_date: date,
    *,
    base_px: float = 2500.0,
    step_px: float = 1.5,
    n_session_bars: int = 75,
    warmup_bars: int = 25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_day = session_date - timedelta(days=3 if session_date.weekday() == 0 else 1)
    t_warm = datetime(prev_day.year, prev_day.month, prev_day.day, 13, 0, tzinfo=IST_TZ)
    px = base_px - 15.0
    for k in range(warmup_bars):
        t0 = t_warm + timedelta(minutes=5 * k)
        t1 = t0 + timedelta(minutes=5)
        o = px
        c = o + 0.5
        rows.append(
            {
                "ts_open_ist": t0,
                "ts_close_ist": t1,
                "ts_init_ist": t1,
                "open": o,
                "high": max(o, c) + 1.0,
                "low": min(o, c) - 1.0,
                "close": c,
                "volume": 20_000.0,
            }
        )
        px = c

    t_open = datetime(session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST_TZ)
    for k in range(n_session_bars):
        t0 = t_open + timedelta(minutes=5 * k)
        t1 = t0 + timedelta(minutes=5)
        o = px
        c = o + step_px
        rows.append(
            {
                "ts_open_ist": t0,
                "ts_close_ist": t1,
                "ts_init_ist": t1,
                "open": o,
                "high": max(o, c) + 1.5,
                "low": min(o, c) - 0.8,
                "close": c,
                "volume": 25_000.0 + k * 200.0,
            }
        )
        px = c
    return rows


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
    price: float = 2500.0,
    sl: float = 2475.0,
    tp: float = 2560.0,
) -> TimingAssessment:
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
        entry_price=price,
        invalidation_price=sl,
        target_price=tp,
        risk_reward_ratio=2.5,
        detected_patterns=(),
        timeframe_statuses={"1m": "VALID"},
        contradiction_reasons=(),
        rationale=("REMED-01 behavioral test assessment",),
    )


# ---------------------------------------------------------------------------
# 1. auto_direction=True in monitoring_session.py (P0 NameError fix + override)
# ---------------------------------------------------------------------------


def test_remed01_monitoring_session_auto_direction_runs_and_yields_to_explicit_hypothesis() -> None:
    d = date(2026, 11, 23)
    rows = _make_5m_day_rows(d, step_px=2.0, n_session_bars=30)
    feed = _StaticBarFeed("TCS", "5m", rows)

    # A. auto_direction=True with hypothesis=None must execute cleanly (no NameError!)
    sess_auto = IntradayCandleMonitoringSession(feed=feed, initial_capital=50_000.0)
    res_auto = sess_auto.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        direction=DirectionalBias.NEUTRAL,
        hypothesis=None,
        human_permission=PermissionType.ALLOW_BOTH,
        auto_direction=True,
    )
    assert len(res_auto.decisions) == 30
    assert res_auto.authority_audit_chain_ok is True

    # B. auto_direction=True with explicit BEARISH hypothesis must YIELD (never switch to BULLISH)
    sess_exp = IntradayCandleMonitoringSession(feed=feed, initial_capital=50_000.0)
    explicit_hyp = DirectionalHypothesis(
        symbol="TCS",
        direction=DirectionalBias.BEARISH,
        confidence=0.8,
        prob_next_day_up=0.35,
        expected_next_day_return_pct=-1.0,
        news_sentiment_score=-0.15,
        market_regime="TEST",
        observation_date=d,
        as_of_ist=datetime(2026, 11, 23, 9, 15, tzinfo=IST_TZ),
        rationale=("Explicit Central BEARISH override",),
    )
    res_exp = sess_exp.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        hypothesis=explicit_hyp,
        human_permission=PermissionType.ALLOW_BOTH,
        auto_direction=True,
    )
    assert res_exp.central_direction == "BEARISH"
    assert all(dec.intended_direction == "SHORT" for dec in res_exp.decisions)


# ---------------------------------------------------------------------------
# 2. Missing Human Authority — Fail-Closed Across All 4 Surfaces
# ---------------------------------------------------------------------------


def test_remed01_missing_human_permission_fails_closed_in_monitoring_session_and_live_runner(tmp_path: Path) -> None:
    from dataclasses import replace as dc_replace
    from trader.timing.candle_timing_agent import AgentCall, CandleTimingAgent

    class _ForceEntryAgent:
        def __init__(self) -> None:
            self._inner = CandleTimingAgent()
            self.calls = 0

        def evaluate_bar_lifecycle(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            base = self._inner.evaluate_bar_lifecycle(*args, **kwargs)
            if self.calls == 1:
                ta = _make_candle(symbol="TCS", bias=DirectionalBias.BULLISH, decision=TimingDecision.ENTER_NOW)
                return dc_replace(base, call=AgentCall.OPEN_LONG, timing_assessment=ta)
            return dc_replace(base, call=AgentCall.WAIT)

    # A0. Direct MandatoryExecutionGate with NO Human instruction: blocks ENTRY_LONG, allows RISK_EXIT
    p_direct = IntradayPaperPortfolio(initial_capital=50_000.0)
    p_direct.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST - timedelta(minutes=5), position_effect="OPEN_LONG")
    g_direct = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g_direct.set_central_instruction(_make_instr("C-NO-H", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    res_block = g_direct.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=IntradayPaperPortfolio(initial_capital=50_000.0),
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=2,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert res_block.approved is False
    assert any("HUMAN_INSTRUCTION_MISSING" in r for r in res_block.rejection_reasons)
    _, rx_ok = g_direct.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.RISK_EXIT,
        current_time=NOW_IST,
        portfolio=p_direct,
        requested_quantity=2,
        price=2505.0,
    )
    assert rx_ok is True and len(p_direct.positions) == 0

    # A. MonitoringSession with human_permission=None & human_instruction=None -> forced OPEN_LONG MUST fail closed!
    d = date(2026, 11, 23)
    rows = _make_5m_day_rows(d, step_px=2.5, n_session_bars=10)
    feed = _StaticBarFeed("TCS", "5m", rows)
    sess = IntradayCandleMonitoringSession(feed=feed, timing_agent=_ForceEntryAgent(), initial_capital=50_000.0)  # type: ignore[arg-type]
    res = sess.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        direction=DirectionalBias.BULLISH,
        human_permission=None,
        human_instruction=None,
    )
    assert len(sess.portfolio.closed_trades) == 0
    assert len(sess.portfolio.positions) == 0
    assert len(sess.execution_gate.audit_trail) >= 1
    assert any(
        any("HUMAN_INSTRUCTION_MISSING" in r for r in rec.rejection_reasons)
        for rec in sess.execution_gate.audit_trail
    )
    assert res.authority_audit_chain_ok is True

    # B. LivePaperTradingRunner with human_instruction=None on a breakout stream -> MUST record rejection!
    jsonl_path = tmp_path / "breakout.jsonl"
    base_ts = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST_TZ)
    lines = []
    px = 2500.0
    cum_vol = 0.0
    for m in range(45):
        for sec in (10, 45):
            ts = base_ts + timedelta(minutes=m, seconds=sec)
            px += 1.25 if m >= 16 else 0.05
            cum_vol += 1000.0
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

    runner = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(jsonl_path=jsonl_path),
        output_dir=tmp_path / "out_no_human",
        config=LivePaperConfig(
            strategy_profile="SUPERTREND_ADX",
            initial_capital_inr=50_000.0,
            adx_min=0.0,
            min_reward_to_friction=0.1,
        ),
        human_instruction=None,
    )
    summary = runner.run(["TCS"])
    assert len(summary.trades) == 0
    assert summary.gate_rejections_count > 0
    assert any(
        any("HUMAN_INSTRUCTION_MISSING" in r for r in rec.rejection_reasons)
        for rec in runner.execution_gate.audit_trail
    )
    assert summary.authority_audit_verified is True


# ---------------------------------------------------------------------------
# 3. All 7 Human Permission States End-to-End via execute_with_gate_protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perm,entry_long_ok,entry_short_ok,exit_long_ok,exit_short_ok,risk_exit_ok",
    [
        (PermissionType.ALLOW_LONG, True, False, True, True, True),
        (PermissionType.ALLOW_SHORT, False, True, True, True, True),
        (PermissionType.ALLOW_BOTH, True, True, True, True, True),
        (PermissionType.NO_NEW_ENTRIES, False, False, True, True, True),
        (PermissionType.EXIT_ONLY, False, False, True, True, True),
        (PermissionType.PAUSE, False, False, False, False, True),
        (PermissionType.STOP_ALL, False, False, True, True, True),
    ],
)
def test_remed01_all_seven_human_permissions_end_to_end_gate_execution(
    perm: PermissionType,
    entry_long_ok: bool,
    entry_short_ok: bool,
    exit_long_ok: bool,
    exit_short_ok: bool,
    risk_exit_ok: bool,
) -> None:
    # Test ENTRY_LONG
    p1 = IntradayPaperPortfolio(initial_capital=50_000.0)
    g1 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g1.set_central_instruction(_make_instr("C1", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    g1.set_human_instruction(_make_instr("H1", AuthoritySource.HUMAN, perm), current_time=NOW_IST)
    out_l, exec_l = g1.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=p1,
        requested_quantity=2,
        price=2500.0,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        stop_loss_price=2475.0,
    )
    assert (out_l.approved and exec_l) is entry_long_ok

    # Test ENTRY_SHORT
    p2 = IntradayPaperPortfolio(initial_capital=50_000.0)
    g2 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g2.set_central_instruction(_make_instr("C2", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    g2.set_human_instruction(_make_instr("H2", AuthoritySource.HUMAN, perm), current_time=NOW_IST)
    out_s, exec_s = g2.execute_with_gate_protection(
        symbol="TCS",
        action_type=ActionType.ENTRY_SHORT,
        current_time=NOW_IST,
        portfolio=p2,
        requested_quantity=2,
        price=2500.0,
        candle_assessment=_make_candle("TCS", DirectionalBias.BEARISH, sl=2525.0, tp=2440.0),
        stop_loss_price=2525.0,
    )
    assert (out_s.approved and exec_s) is entry_short_ok

    # Test EXIT_LONG, EXIT_SHORT, RISK_EXIT on pre-existing positions
    for side_open, exit_act, expected_ok in (
        ("LONG", ActionType.EXIT_LONG, exit_long_ok),
        ("SHORT", ActionType.EXIT_SHORT, exit_short_ok),
        ("LONG", ActionType.RISK_EXIT, risk_exit_ok),
    ):
        px_p = IntradayPaperPortfolio(initial_capital=50_000.0)
        px_g = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
        px_g.set_central_instruction(_make_instr("C3", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
        px_g.set_human_instruction(_make_instr("H3", AuthoritySource.HUMAN, perm), current_time=NOW_IST)
        px_p.submit_order(
            "TCS",
            "BUY" if side_open == "LONG" else "SELL",
            2,
            2500.0,
            timestamp=NOW_IST - timedelta(minutes=10),
            position_effect="OPEN_LONG" if side_open == "LONG" else "OPEN_SHORT",
        )
        out_x, exec_x = px_g.execute_with_gate_protection(
            symbol="TCS",
            action_type=exit_act,
            current_time=NOW_IST,
            portfolio=px_p,
            requested_quantity=2,
            price=2505.0,
        )
        assert (out_x.approved and exec_x) is expected_ok
        assert px_g.verify_audit_chain() is True


# ---------------------------------------------------------------------------
# 4. Mandatory 15:15 IST Square-Off Under Human PAUSE (P1 Fix) & Intrabar Audit (P2 Fix)
# ---------------------------------------------------------------------------


def test_remed01_mandatory_1515_squareoff_succeeds_under_pause_and_intrabar_sl_is_audited() -> None:
    d = date(2026, 11, 23)
    rows = _make_5m_day_rows(d, base_px=2500.0, step_px=0.2, n_session_bars=75)
    feed = _StaticBarFeed("TCS", "5m", rows)

    from dataclasses import replace as dc_replace
    from trader.timing.candle_timing_agent import AgentCall, CandleTimingAgent

    class _HoldTimingAgent:
        def __init__(self) -> None:
            self._inner = CandleTimingAgent()

        def evaluate_bar_lifecycle(self, *args: Any, **kwargs: Any) -> Any:
            return dc_replace(self._inner.evaluate_bar_lifecycle(*args, **kwargs), call=AgentCall.HOLD)

    # A. Open LONG & SHORT positions held into 15:15 IST under PAUSE, EXIT_ONLY, STOP_ALL, and RiskState.HALTED
    for perm in (PermissionType.PAUSE, PermissionType.EXIT_ONLY, PermissionType.STOP_ALL):
        for side in ("LONG", "SHORT"):
            sess = IntradayCandleMonitoringSession(feed=feed, timing_agent=_HoldTimingAgent(), initial_capital=50_000.0)  # type: ignore[arg-type]
            sess.portfolio.submit_order(
                "TCS",
                "BUY" if side == "LONG" else "SELL",
                2,
                2500.0,
                stop_loss=2000.0 if side == "LONG" else 3000.0,
                target=3200.0 if side == "LONG" else 1800.0,
                timestamp=datetime(2026, 11, 23, 9, 10, tzinfo=IST_TZ),
                position_effect="OPEN_LONG" if side == "LONG" else "OPEN_SHORT",
            )
            res = sess.run_session(
                symbol="TCS",
                session_date=d,
                timeframe="5m",
                direction=DirectionalBias.BULLISH if side == "LONG" else DirectionalBias.BEARISH,
                human_permission=perm,
            )
            assert "TCS" not in sess.portfolio.positions
            assert len(sess.portfolio.positions) == 0
            assert any(
                rec.action_type == ActionType.RISK_EXIT and rec.final_execution_decision == "APPROVED"
                for rec in sess.execution_gate.audit_trail
            )
            assert res.authority_audit_chain_ok is True

    # A2. RiskState.HALTED (EXIT_ONLY) must also square off open position at 15:15 IST via RISK_EXIT
    sess_halted = IntradayCandleMonitoringSession(feed=feed, timing_agent=_HoldTimingAgent(), initial_capital=50_000.0)  # type: ignore[arg-type]
    sess_halted.portfolio.submit_order(
        "TCS",
        "BUY",
        2,
        2500.0,
        stop_loss=2000.0,
        target=3200.0,
        timestamp=datetime(2026, 11, 23, 9, 10, tzinfo=IST_TZ),
        position_effect="OPEN_LONG",
    )
    sess_halted.risk_engine.state = RiskState.HALTED
    res_halted = sess_halted.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        human_permission=PermissionType.ALLOW_BOTH,
    )
    assert len(sess_halted.portfolio.positions) == 0
    assert any(
        rec.action_type == ActionType.RISK_EXIT and rec.final_execution_decision == "APPROVED"
        for rec in sess_halted.execution_gate.audit_trail
    )
    assert res_halted.authority_audit_chain_ok is True

    # A3. Python MonitoringSession STOP_ALL at bar index 0 (09:15 IST) -> immediately flattens open position
    bar0_open = datetime(2026, 11, 23, 9, 15, tzinfo=IST_TZ)
    rows_short = _make_5m_day_rows(d, base_px=2500.0, step_px=0.0, n_session_bars=3)
    sess_b0 = IntradayCandleMonitoringSession(feed=_StaticBarFeed("TCS", "5m", rows_short), initial_capital=50_000.0)
    sess_b0.portfolio.submit_order(
        "TCS",
        "BUY",
        2,
        2500.0,
        stop_loss=2400.0,
        target=2700.0,
        timestamp=datetime(2026, 11, 23, 9, 10, tzinfo=IST_TZ),
        position_effect="OPEN_LONG",
    )
    stop_b0 = _make_instr(
        "H-STOP-BAR0",
        AuthoritySource.HUMAN,
        PermissionType.STOP_ALL,
        scope="TCS",
        valid_from=bar0_open,
        expires_at=datetime(2026, 11, 23, 16, 0, tzinfo=IST_TZ),
    )
    res_b0 = sess_b0.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        human_permission=PermissionType.ALLOW_BOTH,
        mid_session_human_instructions={bar0_open: stop_b0},
    )
    assert len(sess_b0.portfolio.positions) == 0
    assert any("HUMAN_STOP_ALL_SAFE_MANAGEMENT:H-STOP-BAR0" in n for n in res_b0.decisions[0].rationale)
    assert any(
        rec.action_type == ActionType.RISK_EXIT and rec.final_execution_decision == "APPROVED"
        for rec in sess_b0.execution_gate.audit_trail
    )

    # B. Intrabar SL hit in monitoring_session -> MUST be audited in BOTH gate & risk engine,
    #    and surviving bars MUST update highest_price/lowest_price (MFE tracking for Chandelier stops).
    rows_sl = _make_5m_day_rows(d, base_px=2500.0, step_px=0.0, n_session_bars=10)
    rows_sl[25 + 0]["high"] = 2525.0
    rows_sl[25 + 0]["low"] = 2492.0
    # Bar 2 crashes through SL=2480
    rows_sl[25 + 2]["low"] = 2470.0
    rows_sl[25 + 2]["close"] = 2475.0
    sess_sl = IntradayCandleMonitoringSession(feed=_StaticBarFeed("TCS", "5m", rows_sl), initial_capital=50_000.0)
    sess_sl.portfolio.submit_order(
        "TCS",
        "BUY",
        2,
        2500.0,
        stop_loss=2480.0,
        target=2600.0,
        timestamp=datetime(2026, 11, 23, 9, 10, tzinfo=IST_TZ),
        position_effect="OPEN_LONG",
    )
    res_sl = sess_sl.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        human_permission=PermissionType.PAUSE,
    )
    assert len(sess_sl.portfolio.positions) == 0
    assert any(
        rec.action_type == ActionType.RISK_EXIT and rec.final_execution_decision == "APPROVED"
        for rec in sess_sl.execution_gate.audit_trail
    )
    assert len(sess_sl.risk_engine.audit_log) >= 1
    assert res_sl.authority_audit_chain_ok is True
    assert res_sl.risk_audit_chain_ok is True

    # B2. Verify surviving bar updates highest_price & lowest_price in Step B (prevents Chandelier MFE regression)
    rows_mfe = _make_5m_day_rows(d, base_px=2500.0, step_px=0.0, n_session_bars=2)
    rows_mfe[25 + 0]["high"] = 2535.0
    rows_mfe[25 + 0]["low"] = 2491.0
    sess_mfe = IntradayCandleMonitoringSession(feed=_StaticBarFeed("TCS", "5m", rows_mfe), initial_capital=50_000.0)
    sess_mfe.portfolio.submit_order(
        "TCS",
        "BUY",
        2,
        2500.0,
        stop_loss=2400.0,
        target=2700.0,
        timestamp=datetime(2026, 11, 23, 9, 10, tzinfo=IST_TZ),
        position_effect="OPEN_LONG",
    )
    sess_mfe.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        human_permission=PermissionType.PAUSE,
    )
    assert "TCS" in sess_mfe.portfolio.positions
    assert sess_mfe.portfolio.positions["TCS"].highest_price >= 2535.0
    assert sess_mfe.portfolio.positions["TCS"].lowest_price <= 2491.0


# ---------------------------------------------------------------------------
# 5. LivePaperTradingRunner Shutdown Flattens Open LONG, SHORT & Multi-Symbol Positions (P1 Fix)
# ---------------------------------------------------------------------------


def test_remed01_live_paper_runner_shutdown_flattens_open_long_short_and_multi_symbol(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "two_ticks.jsonl"
    t0 = datetime(2026, 11, 23, 10, 0, 10, tzinfo=IST_TZ)
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"symbol": "TCS", "exchange": "NSE", "timestamp_ist": t0.isoformat(), "ltp": 2510.0, "cumulative_volume": 100.0}),
                json.dumps({"symbol": "RELIANCE", "exchange": "NSE", "timestamp_ist": (t0 + timedelta(seconds=5)).isoformat(), "ltp": 1390.0, "cumulative_volume": 100.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    portfolio = IntradayPaperPortfolio(initial_capital=100_000.0)
    # Seed simultaneous LONG in TCS and SHORT in RELIANCE (exchange-qualified keys NSE:TCS, NSE:RELIANCE)
    portfolio.submit_order("TCS", "BUY", 2, 2500.0, stop_loss=2400.0, target=2700.0, timestamp=t0 - timedelta(minutes=5), position_effect="OPEN_LONG")
    portfolio.submit_order("RELIANCE", "SELL", 4, 1400.0, stop_loss=1500.0, target=1200.0, timestamp=t0 - timedelta(minutes=5), position_effect="OPEN_SHORT")
    assert len(portfolio.positions) == 2

    runner = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(jsonl_path=jsonl_path),
        output_dir=tmp_path / "shutdown_out",
        config=LivePaperConfig(initial_capital_inr=100_000.0),
        portfolio=portfolio,
        human_instruction=_make_instr(
            "HUM-NO-NEW",
            AuthoritySource.HUMAN,
            PermissionType.NO_NEW_ENTRIES,
            valid_from=t0 - timedelta(hours=1),
            expires_at=t0 + timedelta(hours=6),
        ),
    )
    summary = runner.run(["TCS", "RELIANCE"])
    # BOTH exchange-qualified positions MUST be flattened via audited RISK_EXIT on shutdown!
    assert len(portfolio.positions) == 0
    assert len(summary.trades) == 2
    assert {t.symbol for t in summary.trades} == {"TCS", "RELIANCE"}
    assert all(t.exit_reason == "SESSION_SHUTDOWN_SQUARE_OFF" for t in summary.trades)
    assert summary.authority_audit_verified is True
    assert summary.risk_audit_verified is True
    assert summary.ledger_integrity_verified is True


# ---------------------------------------------------------------------------
# 6. Synchronized 5-Thread Concurrent Multi-Symbol Cash Reservations + Symbol-Scoped STOP_ALL
# ---------------------------------------------------------------------------


def test_remed01_synchronized_5_thread_concurrent_multi_symbol_reservations() -> None:
    """5 threads race simultaneously via Barrier(5) against ₹50k capital -> 0 cap breaches, 0 leaked holds."""
    symbols = [
        ("TCS", 2500.0, 2475.0),
        ("RELIANCE", 1400.0, 1386.0),
        ("ITC", 300.0, 297.0),
        ("TATAELXSI", 3800.0, 3762.0),
        ("BIRLACABLE", 280.0, 277.0),
    ]
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    # 25% per-stock cap so 5 concurrent symbols demand 125% vs 80% portfolio exposure cap (₹40k)!
    risk_engine = DeterministicRiskEngine(max_position_weight_pct=0.25, max_portfolio_exposure_pct=0.80, max_concurrent_positions=5)
    gate = MandatoryExecutionGate(risk_engine=risk_engine)
    gate.set_human_instruction(_make_instr("H-CONC", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    gate.set_central_instruction(_make_instr("C-CONC", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)

    barrier = threading.Barrier(len(symbols))

    def _worker(sym: str, px: float, sl: float) -> bool:
        res_id = f"CONC-{sym}"
        gate.reserve_cash_for_pending_order(res_id, sym, px * 2)
        barrier.wait(timeout=5.0)
        _, executed = gate.execute_with_gate_protection(
            symbol=sym,
            action_type=ActionType.ENTRY_LONG,
            current_time=NOW_IST,
            portfolio=portfolio,
            requested_quantity=50,
            price=px,
            candle_assessment=_make_candle(sym, DirectionalBias.BULLISH, price=px, sl=sl, tp=px * 1.03),
            stop_loss_price=sl,
            reservation_order_id=res_id,
        )
        return executed

    with ThreadPoolExecutor(max_workers=len(symbols)) as ex:
        futs = [ex.submit(_worker, s, p, sl) for s, p, sl in symbols]
        results = [f.result(timeout=10.0) for f in futs]

    assert any(results)
    assert portfolio.cash >= 0.0
    assert portfolio.invested_capital <= 50_000.0 * 0.80 + 1e-6
    assert risk_engine.cash_ledger.total_reserved_inr == 0.0
    assert gate.verify_audit_chain() is True
    assert risk_engine.verify_audit_chain() is True
    ok_led, _ = portfolio.verify_ledger_integrity()
    assert ok_led is True


def test_remed01_symbol_scoped_stop_all_releases_only_target_symbol_and_flattens_only_target() -> None:
    portfolio = IntradayPaperPortfolio(initial_capital=100_000.0)
    risk_engine = DeterministicRiskEngine()
    gate = MandatoryExecutionGate(risk_engine=risk_engine)
    gate.set_human_instruction(_make_instr("H-ALL", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    gate.set_central_instruction(_make_instr("C-ALL", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)

    portfolio.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST, position_effect="OPEN_LONG")
    portfolio.submit_order("RELIANCE", "SELL", 4, 1400.0, timestamp=NOW_IST, position_effect="OPEN_SHORT")
    gate.reserve_cash_for_pending_order("RES-TCS", "TCS", 5_000.0)
    gate.reserve_cash_for_pending_order("RES-REL", "RELIANCE", 6_000.0)

    # Symbol-scoped STOP_ALL on TCS ONLY
    gate.emergency_human_stop(reason="STOP_TCS_ONLY", current_time=NOW_IST + timedelta(minutes=1), scope="TCS", portfolio=portfolio)

    # TCS flattened & TCS reservation released; RELIANCE position & reservation intact!
    assert all(p.symbol != "TCS" for p in portfolio.positions.values())
    assert any(p.symbol == "RELIANCE" for p in portfolio.positions.values())
    assert risk_engine.cash_ledger.reserved_for_symbol("TCS") == 0.0
    assert risk_engine.cash_ledger.reserved_for_symbol("RELIANCE") == pytest.approx(6_000.0)
    assert gate.verify_audit_chain() is True


# ---------------------------------------------------------------------------
# 7. Isolated 2.5% Daily Loss Breach vs Isolated 3.0% MTM Breach (Decision 4)
# ---------------------------------------------------------------------------


def test_remed01_isolated_daily_2_5_pct_and_isolated_mtm_3_0_pct_drawdown_exit_only() -> None:
    def _assert_halted_exit_only(re_obj: DeterministicRiskEngine, port: IntradayPaperPortfolio) -> None:
        g = MandatoryExecutionGate(risk_engine=re_obj)
        g.set_human_instruction(_make_instr("H-HALT", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
        g.set_central_instruction(_make_instr("C-HALT", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
        # New entry MUST be blocked when RiskState.HALTED
        ent = g.evaluate_gate(
            symbol="INFY",
            action_type=ActionType.ENTRY_LONG,
            current_time=NOW_IST,
            portfolio=port,
            candle_assessment=_make_candle("INFY", DirectionalBias.BULLISH),
            requested_quantity=1,
            entry_price=1500.0,
            stop_loss_price=1480.0,
        )
        assert ent.approved is False
        # Existing position exit (EXIT_LONG & RISK_EXIT) MUST still be allowed
        ex = g.evaluate_gate(
            symbol="TCS",
            action_type=ActionType.EXIT_LONG,
            current_time=NOW_IST,
            portfolio=port,
            candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
            requested_quantity=2,
            entry_price=port.positions["TCS"].current_price,
        )
        assert ex.approved is True
        rx = g.evaluate_gate(
            symbol="TCS",
            action_type=ActionType.RISK_EXIT,
            current_time=NOW_IST,
            portfolio=port,
            requested_quantity=2,
            entry_price=port.positions["TCS"].current_price,
        )
        assert rx.approved is True

    # A. Isolated 2.5% daily loss breach (equity drops -2.6% from day start; MTM DD is also 2.6% < 3.0%)
    p_day = IntradayPaperPortfolio(initial_capital=50_000.0)
    re_day = DeterministicRiskEngine(max_daily_loss_pct=0.025, max_intraday_mtm_drawdown_pct=0.030)
    p_day.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST, position_effect="OPEN_LONG")
    p_day.positions["TCS"].current_price = 1850.0  # -₹1,300 = -2.6% (< 3.0% MTM, > 2.5% daily)
    assert re_day.check_daily_loss_limit(p_day) is True
    assert re_day.state == RiskState.HALTED
    # Decision 4: NOT auto-liquidated, and gate enforces EXIT_ONLY!
    assert len(p_day.positions) == 1
    _assert_halted_exit_only(re_day, p_day)

    # B. Isolated 3.0% MTM drawdown breach while day P&L is still POSITIVE (+2.0% from day start, -3.41% from peak)
    p_mtm = IntradayPaperPortfolio(initial_capital=50_000.0)
    re_mtm = DeterministicRiskEngine(max_daily_loss_pct=0.025, max_intraday_mtm_drawdown_pct=0.030)
    p_mtm.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST, position_effect="OPEN_LONG")
    p_mtm.positions["TCS"].current_price = 3900.0  # Peak equity = ₹52,800 (+5.6%)
    assert re_mtm.check_daily_loss_limit(p_mtm) is False
    p_mtm.positions["TCS"].current_price = 3000.0  # Equity = ₹51,000 (+2.0% daily, but -3.41% from peak!)
    assert re_mtm.check_daily_loss_limit(p_mtm) is True
    assert re_mtm.state == RiskState.HALTED
    assert len(p_mtm.positions) == 1
    _assert_halted_exit_only(re_mtm, p_mtm)


# ---------------------------------------------------------------------------
# 8. Fee-Inclusive Cash Boundary, Gate 6 Sizing & Full Human/Central Validation (Cat 3 & 12)
# ---------------------------------------------------------------------------


def test_remed01_fee_inclusive_cash_boundary_and_central_staleness_rejection() -> None:
    # A. Exact cash = ₹2,500.00 for 1 share @ ₹2,500.00 -> notional fits, but notional + NSE entry friction > ₹2,500 -> REJECT!
    p = IntradayPaperPortfolio(initial_capital=2_500.0)
    re = DeterministicRiskEngine(max_position_weight_pct=1.0, max_portfolio_exposure_pct=1.0, max_risk_per_trade_pct=1.0)
    g = MandatoryExecutionGate(risk_engine=re, max_central_staleness_minutes=30)
    g.set_human_instruction(_make_instr("H-FEE", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    g.set_central_instruction(_make_instr("C-FEE", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH, valid_from=NOW_IST - timedelta(minutes=5)), current_time=NOW_IST)

    out_fee = g.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=p,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=1,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out_fee.approved is False
    assert any("RISK_ENGINE_REJECTION" in r for r in out_fee.rejection_reasons)

    # B. Gate 6 sizing-before-cash: requested_quantity=100 @ ₹2,500 on ₹50k capital -> sizes down to 1..3 shares & APPROVES!
    p_size = IntradayPaperPortfolio(initial_capital=50_000.0)
    g_size = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g_size.set_human_instruction(_make_instr("H-SZ", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    g_size.set_central_instruction(_make_instr("C-SZ", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    out_size = g_size.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST,
        portfolio=p_size,
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=100,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out_size.approved is True
    assert out_size.risk_evaluation is not None
    assert 1 <= out_size.risk_evaluation.approved_quantity <= 3

    # C. Central staleness (> 30 min) + duplicate & delayed Central registration
    out_stale = g.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST + timedelta(minutes=45),
        portfolio=IntradayPaperPortfolio(initial_capital=50_000.0),
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=1,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out_stale.approved is False
    assert any("CENTRAL_INSTRUCTION_STALE" in r for r in out_stale.rejection_reasons)

    dup_c = g.set_central_instruction(_make_instr("C-FEE", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_SHORT), current_time=NOW_IST)
    assert isinstance(dup_c, InstructionRegistrationResult) and dup_c.accepted is False and "DUPLICATE_INSTRUCTION_ID" in dup_c.reason
    del_c = g.set_central_instruction(
        _make_instr("C-OLD", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_SHORT, valid_from=NOW_IST - timedelta(hours=1)),
        current_time=NOW_IST,
    )
    assert isinstance(del_c, InstructionRegistrationResult) and del_c.accepted is False and "DELAYED_OUT_OF_ORDER_INSTRUCTION" in del_c.reason

    # D. Human instruction validation: invalid window, duplicate, delayed/out-of-order,
    #    expired symbol-scoped NOT falling back to wildcard "*", and mid-session rejected Human update!
    with pytest.raises(ValueError):
        _make_instr("H-INV", AuthoritySource.HUMAN, PermissionType.ALLOW_LONG, valid_from=NOW_IST, expires_at=NOW_IST - timedelta(minutes=1))

    g_h = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g_h.set_central_instruction(_make_instr("C-HVAL", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    ok_wild = g_h.set_human_instruction(_make_instr("H-WILD", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH, scope="*"), current_time=NOW_IST)
    assert ok_wild.accepted is True
    # Duplicate Human ID -> structured rejection (no exception, no overwrite)
    dup_h = g_h.set_human_instruction(_make_instr("H-WILD", AuthoritySource.HUMAN, PermissionType.STOP_ALL, scope="*"), current_time=NOW_IST)
    assert dup_h.accepted is False and "DUPLICATE_INSTRUCTION_ID" in dup_h.reason
    # Delayed/out-of-order Human instruction -> structured rejection
    del_h = g_h.set_human_instruction(
        _make_instr("H-OLD", AuthoritySource.HUMAN, PermissionType.STOP_ALL, scope="*", valid_from=NOW_IST - timedelta(hours=2)),
        current_time=NOW_IST,
    )
    assert del_h.accepted is False and "DELAYED_OUT_OF_ORDER_INSTRUCTION" in del_h.reason
    # Expired symbol-scoped Human instruction MUST fail closed (NOT fall back to active wildcard "*")
    g_h.set_human_instruction(
        _make_instr("H-TCS-EXP", AuthoritySource.HUMAN, PermissionType.ALLOW_LONG, scope="TCS", valid_from=NOW_IST - timedelta(minutes=20), expires_at=NOW_IST + timedelta(minutes=5)),
        current_time=NOW_IST,
    )
    out_exp = g_h.evaluate_gate(
        symbol="TCS",
        action_type=ActionType.ENTRY_LONG,
        current_time=NOW_IST + timedelta(minutes=10),
        portfolio=IntradayPaperPortfolio(initial_capital=50_000.0),
        candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH),
        requested_quantity=1,
        entry_price=2500.0,
        stop_loss_price=2475.0,
    )
    assert out_exp.approved is False
    assert any("HUMAN_INSTRUCTION_EXPIRED" in r or "HUMAN_INSTRUCTION_MISSING" in r for r in out_exp.rejection_reasons)

    # Mid-session delayed/duplicate Human instruction injected into MonitoringSession -> logged as HUMAN_INSTRUCTION_REJECTED
    d = date(2026, 11, 23)
    rows_m = _make_5m_day_rows(d, base_px=2500.0, step_px=0.5, n_session_bars=4)
    sess_m = IntradayCandleMonitoringSession(feed=_StaticBarFeed("TCS", "5m", rows_m), initial_capital=50_000.0)
    bar1_open = datetime(2026, 11, 23, 9, 20, tzinfo=IST_TZ)
    stale_mid = _make_instr(
        "H-STALE-MID",
        AuthoritySource.HUMAN,
        PermissionType.STOP_ALL,
        scope="TCS",
        valid_from=datetime(2026, 11, 23, 8, 0, tzinfo=IST_TZ),  # older than 09:15 initial instruction!
        expires_at=datetime(2026, 11, 23, 16, 0, tzinfo=IST_TZ),
    )
    res_m = sess_m.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        human_permission=PermissionType.ALLOW_LONG,
        mid_session_human_instructions={bar1_open: stale_mid},
    )
    assert any("HUMAN_INSTRUCTION_REJECTED:" in n for dec in res_m.decisions for n in dec.rationale)


# ---------------------------------------------------------------------------
# 9. Explicit 5-Scenario Python / JavaScript Parity Matrix
# ---------------------------------------------------------------------------


def test_remed01_python_js_five_scenario_parity_matrix() -> None:
    """Verify all 5 REMED-01-FIX-01 parity scenarios match between Python and JS engine.js."""
    node_script = r"""
    const eng = require('./docs/candle-agent/engine.js');
    const mkBar = (t, tc, o, h, l, c, v = 10000) => ({ t, tc, o, h, l, c, v });
    const warmup = [];
    for (let i = 0; i < 20; i++) {
      warmup.push(mkBar(`2026-11-20T10:${String(i).padStart(2,'0')}:00+05:30`, `2026-11-20T10:${String(i+1).padStart(2,'0')}:00+05:30`, 2500+i, 2502+i, 2499+i, 2501+i));
    }

    // Scenario 1: Missing Human permission -> constructor MUST fail closed (null) AND reject queued OPEN_LONG
    const s1 = new eng.IntradaySessionSimulator({
      symbol: 'TCS', sessionDate: '2026-11-23', timeframe: '5m',
      warmupBars: warmup,
      sessionBars: [
        mkBar('2026-11-23T09:15:00+05:30', '2026-11-23T09:20:00+05:30', 2520, 2530, 2519, 2528),
      ],
      centralDirection: 'BULLISH',
      humanPermission: undefined,
    });
    s1.pendingOrder = { actionType: 'OPEN_LONG', qty: 2, stopLoss: 2490, target: 2580, signalCloseTs: '2026-11-23T09:15:00+05:30' };
    const r1 = s1.stepNextBar();

    // Scenario 2: 14:45 cutoff boundary -> pending entry queued at 14:44 MUST be blocked at 14:45 open
    const s2 = new eng.IntradaySessionSimulator({
      symbol: 'TCS', sessionDate: '2026-11-23', timeframe: '5m',
      warmupBars: warmup,
      sessionBars: [
        mkBar('2026-11-23T14:45:00+05:30', '2026-11-23T14:50:00+05:30', 2520, 2530, 2519, 2528),
      ],
      centralDirection: 'BULLISH',
      humanPermission: 'ALLOW_BOTH',
    });
    s2.pendingOrder = { actionType: 'OPEN_LONG', qty: 2, stopLoss: 2490, target: 2580, signalCloseTs: '2026-11-23T14:45:00+05:30' };
    const r2 = s2.stepNextBar();

    // Scenario 3: PAUSE exit timing -> discretionary EXIT_LONG MUST be rejected immediately at bar close
    const s3 = new eng.IntradaySessionSimulator({
      symbol: 'TCS', sessionDate: '2026-11-23', timeframe: '5m',
      warmupBars: warmup,
      sessionBars: [
        mkBar('2026-11-23T10:00:00+05:30', '2026-11-23T10:05:00+05:30', 2520, 2521, 2440, 2442, 80000),
      ],
      centralDirection: 'BEARISH',
      humanPermission: 'PAUSE',
    });
    s3.position = { side: 'LONG', qty: 2, entryPrice: 2515, currentPrice: 2515, stopLoss: 2300, target: 2700, entryCost: 5, highestPrice: 2520, lowestPrice: 2515, barsHeld: 5 };
    const r3 = s3.stepNextBar();

    // Scenario 4: STOP_ALL at barIdx === 0 -> immediately flattens open position
    const s4 = new eng.IntradaySessionSimulator({
      symbol: 'TCS', sessionDate: '2026-11-23', timeframe: '5m',
      warmupBars: warmup,
      sessionBars: [mkBar('2026-11-23T09:15:00+05:30', '2026-11-23T09:20:00+05:30', 2520, 2525, 2518, 2522)],
      centralDirection: 'BULLISH',
      humanPermission: 'ALLOW_BOTH',
    });
    s4.position = { side: 'LONG', qty: 2, entryPrice: 2520, currentPrice: 2520, stopLoss: 2480, target: 2600, entryCost: 5, highestPrice: 2520, lowestPrice: 2520, barsHeld: 0 };
    s4.setHumanPermission('STOP_ALL');

    // Scenario 5: AUTO Central revalidation at Step A -> NEUTRAL/opposing effectiveDirection blocks pending OPEN_LONG
    const s5 = new eng.IntradaySessionSimulator({
      symbol: 'TCS', sessionDate: '2026-11-23', timeframe: '5m',
      warmupBars: warmup,
      sessionBars: [mkBar('2026-11-23T10:00:00+05:30', '2026-11-23T10:05:00+05:30', 2520, 2525, 2518, 2522)],
      centralDirection: 'AUTO',
      humanPermission: 'ALLOW_BOTH',
    });
    s5.lastEffectiveDirection = 'BEARISH';
    s5.pendingOrder = { actionType: 'OPEN_LONG', qty: 2, stopLoss: 2490, target: 2580, signalCloseTs: '2026-11-23T10:00:00+05:30' };
    const r5 = s5.stepNextBar();

    console.log(JSON.stringify({
      s1_humanPerm: s1.humanPermission,
      s1_pos: s1.position,
      s1_blocked: r1.notes.some(x => x.includes('HUMAN_INSTRUCTION_MISSING')),
      s2_pos: s2.position,
      s2_blocked: r2.notes.some(x => x.includes('PAST_1445_OR_REDUCING')),
      s3_pending: s3.pendingOrder,
      s3_call: r3.call,
      s3_gateStatus: r3.gateStatus,
      s4_pos: s4.position,
      s4_closed: s4.closedTrades.length,
      s5_pos: s5.position,
      s5_blocked: r5.notes.some(x => x.includes('CENTRAL_DIRECTION_CHANGED:BEARISH')),
    }));
    """
    proc = subprocess.run(["node", "-e", node_script], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())

    # Unconditional assertions for all 5 JS parity scenarios (zero vacuous if-guards!)
    assert res["s1_humanPerm"] is None and res["s1_pos"] is None and res["s1_blocked"] is True
    assert res["s2_pos"] is None and res["s2_blocked"] is True
    assert res["s3_pending"] is None and res["s3_call"] == "REJECT" and res["s3_gateStatus"] == "REJECTED"
    assert res["s4_pos"] is None and res["s4_closed"] == 1
    assert res["s5_pos"] is None and res["s5_blocked"] is True

    # Side-by-side Python parity verification for ALL 5 scenarios:
    # Py-1: Missing Human permission -> fails closed with HUMAN_INSTRUCTION_MISSING
    g1 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g1.set_central_instruction(_make_instr("C-P1", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    o1 = g1.evaluate_gate(symbol="TCS", action_type=ActionType.ENTRY_LONG, current_time=NOW_IST, portfolio=IntradayPaperPortfolio(initial_capital=50_000.0), candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH), requested_quantity=2, entry_price=2500.0, stop_loss_price=2475.0)
    assert o1.approved is False and any("HUMAN_INSTRUCTION_MISSING" in r for r in o1.rejection_reasons)

    # Py-2: 14:45 cutoff boundary -> 14:45:00 MUST reject
    g2 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    t1445 = datetime(2026, 11, 23, 14, 45, tzinfo=IST_TZ)
    g2.set_human_instruction(_make_instr("H1445", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH, valid_from=t1445 - timedelta(hours=5), expires_at=t1445 + timedelta(hours=2)), current_time=t1445)
    g2.set_central_instruction(_make_instr("C1445", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH, valid_from=t1445 - timedelta(hours=5), expires_at=t1445 + timedelta(hours=2)), current_time=t1445)
    o2 = g2.evaluate_gate(symbol="TCS", action_type=ActionType.ENTRY_LONG, current_time=t1445, portfolio=IntradayPaperPortfolio(initial_capital=50_000.0), candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH), requested_quantity=2, entry_price=2500.0, stop_loss_price=2475.0)
    assert o2.approved is False

    # Py-3: PAUSE rejects discretionary EXIT_LONG immediately
    p3 = IntradayPaperPortfolio(initial_capital=50_000.0)
    p3.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST - timedelta(minutes=5), position_effect="OPEN_LONG")
    g3 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g3.set_human_instruction(_make_instr("H-P3", AuthoritySource.HUMAN, PermissionType.PAUSE), current_time=NOW_IST)
    g3.set_central_instruction(_make_instr("C-P3", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    o3 = g3.evaluate_gate(symbol="TCS", action_type=ActionType.EXIT_LONG, current_time=NOW_IST, portfolio=p3, candle_assessment=_make_candle("TCS", DirectionalBias.BEARISH), requested_quantity=2, entry_price=2480.0)
    assert o3.approved is False and any("HUMAN_PERMISSION_DENIED" in r for r in o3.rejection_reasons)

    # Py-4: STOP_ALL immediately flattens open position
    p4 = IntradayPaperPortfolio(initial_capital=50_000.0)
    p4.submit_order("TCS", "BUY", 2, 2500.0, timestamp=NOW_IST - timedelta(minutes=5), position_effect="OPEN_LONG")
    g4 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g4.emergency_human_stop(reason="parity_stop", current_time=NOW_IST, portfolio=p4, scope="TCS")
    assert len(p4.positions) == 0

    # Py-5: Opposing Central BEARISH (ALLOW_SHORT) blocks ENTRY_LONG
    g5 = MandatoryExecutionGate(risk_engine=DeterministicRiskEngine())
    g5.set_human_instruction(_make_instr("H-P5", AuthoritySource.HUMAN, PermissionType.ALLOW_BOTH), current_time=NOW_IST)
    g5.set_central_instruction(_make_instr("C-P5", AuthoritySource.CENTRAL_AGENT, PermissionType.ALLOW_SHORT), current_time=NOW_IST)
    o5 = g5.evaluate_gate(symbol="TCS", action_type=ActionType.ENTRY_LONG, current_time=NOW_IST, portfolio=IntradayPaperPortfolio(initial_capital=50_000.0), candle_assessment=_make_candle("TCS", DirectionalBias.BULLISH), requested_quantity=2, entry_price=2500.0, stop_loss_price=2475.0)
    assert o5.approved is False and any("CENTRAL_PERMISSION_DENIED" in r for r in o5.rejection_reasons)


# ---------------------------------------------------------------------------
# 10. CLI main() Invocation & Argument Validation
# ---------------------------------------------------------------------------


def test_remed01_live_paper_cli_main_and_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_live_paper_session as cli

    # Missing --token-map for live brokers fails closed
    with pytest.raises(ValueError, match="--token-map is required"):
        cli.build_adapter("zerodha", None, token_map={})
    with pytest.raises(ValueError, match="--token-map is required"):
        cli.build_adapter("angel_one", None, token_map={})
    with pytest.raises(ValueError, match="--jsonl-file is required"):
        cli.build_adapter("offline_jsonl", None)
    with pytest.raises(ValueError, match="JSON object"):
        cli.parse_token_map("[1, 2, 3]")

    # Missing requested symbol in --token-map fails closed inside main()
    with pytest.raises(ValueError, match="missing instrument tokens"):
        cli.main(["--provider", "zerodha", "--symbols", "TCS", "RELIANCE", "--token-map", '{"TCS": 2953217}', "--dry-run"])

    # Valid offline_jsonl --dry-run invocation through main() succeeds with exit 0
    jsonl_path = tmp_path / "dry.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    rc = cli.main(
        [
            "--provider",
            "offline_jsonl",
            "--jsonl-file",
            str(jsonl_path),
            "--symbols",
            "TCS",
            "--human-permission",
            "NO_NEW_ENTRIES",
            "--dry-run",
        ]
    )
    assert rc == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["status"] == "DRY_RUN_OK"
    assert out_json["live_trading_enabled"] is False
    assert out_json["read_only_mode"] is True


# ---------------------------------------------------------------------------
# 11. AST + Dynamic Spy Regression: Zero Execution-Gate Bypasses
# ---------------------------------------------------------------------------


def test_remed01_ast_and_spy_zero_execution_gate_bypass(tmp_path: Path) -> None:
    """Prove neither monitoring_session.py nor live_paper_runner.py ever calls portfolio.submit_order() directly."""
    for rel_path in ("trader/timing/monitoring_session.py", "trader/timing/live_paper_runner.py"):
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(src)
        direct_calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("submit_order", "update_candle_bar")
        ]
        assert direct_calls == [], f"Unauthorized direct portfolio mutation in {rel_path} at lines {direct_calls}"

    # Dynamic spy: verify every portfolio.submit_order call during LivePaperTradingRunner occurs inside MandatoryExecutionGate
    jsonl_path = tmp_path / "spy.jsonl"
    t0 = datetime(2026, 11, 23, 10, 0, 10, tzinfo=IST_TZ)
    jsonl_path.write_text(
        json.dumps({"symbol": "TCS", "exchange": "NSE", "timestamp_ist": t0.isoformat(), "ltp": 2510.0, "cumulative_volume": 100.0}) + "\n",
        encoding="utf-8",
    )
    portfolio = IntradayPaperPortfolio(initial_capital=50_000.0)
    portfolio.submit_order("TCS", "BUY", 2, 2500.0, timestamp=t0 - timedelta(minutes=5), position_effect="OPEN_LONG")

    orig_submit = portfolio.submit_order
    call_frames: list[bool] = []

    def _spy_submit(*args: Any, **kwargs: Any) -> Any:
        import inspect

        in_gate = any(f.function == "execute_with_gate_protection" for f in inspect.stack())
        call_frames.append(in_gate)
        return orig_submit(*args, **kwargs)

    portfolio.submit_order = _spy_submit  # type: ignore[method-assign]
    runner = LivePaperTradingRunner(
        adapter=OfflineJsonlTickAdapter(jsonl_path=jsonl_path),
        output_dir=tmp_path / "spy_out",
        config=LivePaperConfig(initial_capital_inr=50_000.0),
        portfolio=portfolio,
        human_instruction=_make_instr("H-SPY", AuthoritySource.HUMAN, PermissionType.NO_NEW_ENTRIES, valid_from=t0 - timedelta(hours=1), expires_at=t0 + timedelta(hours=6)),
    )
    runner.run(["TCS"])
    assert len(call_frames) >= 1
    assert all(call_frames), "Detected portfolio.submit_order call outside MandatoryExecutionGate.execute_with_gate_protection!"


# ---------------------------------------------------------------------------
# 16. Mid-Session PendingOrder Revalidation & Cash Reservation Release (Python)
# ---------------------------------------------------------------------------


def test_remed01_mid_session_pending_order_revalidation_and_reservation_release() -> None:
    """Bar 0 queues a PendingOrder under ALLOW_LONG (reserving cash); mid-session
    NO_NEW_ENTRIES arrives at bar 1 open -> Step A revalidation MUST reject the
    queued PendingOrder, open 0 positions, and release the cash reservation."""

    from dataclasses import replace as dc_replace
    from trader.timing.candle_timing_agent import AgentCall, CandleTimingAgent

    class _EnterOnceTimingAgent:
        def __init__(self) -> None:
            self._inner = CandleTimingAgent()
            self.calls = 0

        def evaluate_bar_lifecycle(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            base = self._inner.evaluate_bar_lifecycle(*args, **kwargs)
            if self.calls == 1:
                ta = _make_candle(
                    symbol="TCS",
                    bias=DirectionalBias.BULLISH,
                    decision=TimingDecision.ENTER_NOW,
                    price=2500.0,
                    sl=2475.0,
                    tp=2560.0,
                )
                return dc_replace(base, call=AgentCall.OPEN_LONG, timing_assessment=ta)
            return dc_replace(base, call=AgentCall.WAIT)

    d = date(2026, 11, 23)
    rows = _make_5m_day_rows(d, base_px=2500.0, step_px=1.0, n_session_bars=6)
    feed = _StaticBarFeed("TCS", "5m", rows)
    bar1_open = datetime(2026, 11, 23, 9, 20, tzinfo=IST_TZ)

    sess = IntradayCandleMonitoringSession(
        feed=feed,
        timing_agent=_EnterOnceTimingAgent(),  # type: ignore[arg-type]
        initial_capital=50_000.0,
    )
    mid_instr = _make_instr(
        "H-MID-BLOCK",
        AuthoritySource.HUMAN,
        PermissionType.NO_NEW_ENTRIES,
        scope="TCS",
        valid_from=bar1_open,
        expires_at=datetime(2026, 11, 23, 16, 0, tzinfo=IST_TZ),
    )
    res = sess.run_session(
        symbol="TCS",
        session_date=d,
        timeframe="5m",
        direction=DirectionalBias.BULLISH,
        human_permission=PermissionType.ALLOW_LONG,
        mid_session_human_instructions={bar1_open: mid_instr},
    )
    assert len(sess.portfolio.orders) == 0
    assert len(sess.portfolio.closed_trades) == 0
    assert len(sess.portfolio.positions) == 0
    assert sess.risk_engine.cash_ledger.total_reserved_inr == 0.0
    assert any(
        rec.action_type == ActionType.ENTRY_LONG and rec.final_execution_decision == "REJECTED"
        for rec in sess.execution_gate.audit_trail
    )
    assert res.authority_audit_chain_ok is True
    assert res.risk_audit_chain_ok is True

