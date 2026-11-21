"""Formal 7-Gate Strategy Qualification Framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ..config import IST_TZ

QualificationStatus = Literal[
    "REJECTED",
    "EXPERIMENTAL",
    "IMPROVING",
    "VALIDATION PENDING",
    "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT EVIDENCE",
    "INVALIDATED",
    "BLOCKED_INTEGRITY",
    "QUALIFIED",
    "SUSPENDED",
]


@dataclass(frozen=True)
class GateResult:
    gate_number: int
    gate_name: str
    passed: bool
    status_label: Literal["PASS", "FAIL", "INSUFFICIENT_EVIDENCE"]
    reason: str


@dataclass(frozen=True)
class QualificationEvaluation:
    strategy_id: str
    strategy_name: str
    final_status: QualificationStatus
    gates: list[GateResult]
    passed_gates_count: int
    total_gates: int
    summary: str
    provenance_status: str = "UNKNOWN"
    holdout_status: str = "UNKNOWN"
    validation_timestamp: str = ""
    failure_reasons: list[str] = field(default_factory=list)


def evaluate_seven_gates(
    strategy_id: str,
    strategy_name: str,
    multi_day_metrics: dict[str, Any],
    unseen_metrics: dict[str, Any],
    conservative_slippage_metrics: dict[str, Any],
    data_provenance_verified: bool | None = None,
    stat_validation_verdict: Literal["PASS", "FAIL", "INSUFFICIENT EVIDENCE"] = "INSUFFICIENT EVIDENCE",
    red_team_approved: bool = False,
    red_team_reason: str = "",
    min_sessions_required: int = 60,
    min_trades_required: int = 30,
    independent_holdout_valid: bool | None = None,
    provenance_status: str | None = None,
    holdout_status: str | None = None,
    integrity_guard_result: dict[str, Any] | None = None,
) -> QualificationEvaluation:
    """Evaluate a strategy candidate against all 7 mandatory institutional qualification gates (fail-closed)."""
    gates: list[GateResult] = []
    val_ts = datetime.now(tz=IST_TZ).isoformat()
    failure_reasons: list[str] = []

    if integrity_guard_result is not None:
        prov_status_eff = str(integrity_guard_result.get("provenance_status", "FAILED"))
        hold_status_eff = str(integrity_guard_result.get("holdout_status", "FAILED"))
        prov_ok = (prov_status_eff == "VERIFIED") and bool(integrity_guard_result.get("qualification_allowed", False))
        hold_ok = bool(integrity_guard_result.get("qualification_allowed", False)) and bool(
            integrity_guard_result.get("ownership_valid", False)
        ) and bool(integrity_guard_result.get("split_valid", False))
        failure_reasons.extend(integrity_guard_result.get("failure_reasons", []))
    else:
        prov_status_eff = provenance_status or ("VERIFIED" if data_provenance_verified is True else "MISSING_OR_UNVERIFIED")
        hold_status_eff = holdout_status or ("VALID_UNSEEN" if independent_holdout_valid is True else "MISSING_OR_INVALID")
        prov_ok = (data_provenance_verified is True) and (prov_status_eff == "VERIFIED")
        hold_ok = independent_holdout_valid is True

    # Gate 1 — Data & Holdout Integrity (fail closed if missing or partial)
    g1_pass = prov_ok and hold_ok
    if data_provenance_verified is None and integrity_guard_result is None:
        g1_reason = "BLOCKED: Missing data provenance verification result"
        failure_reasons.append(g1_reason)
    elif not prov_ok:
        g1_reason = f"BLOCKED: Provenance status is {prov_status_eff} (requires VERIFIED)"
        failure_reasons.append(g1_reason)
    elif independent_holdout_valid is None and integrity_guard_result is None:
        g1_reason = "BLOCKED: Missing independent holdout validation result"
        failure_reasons.append(g1_reason)
    elif not hold_ok:
        g1_reason = f"INVALIDATED: Holdout integrity failed ({hold_status_eff})"
        failure_reasons.append(g1_reason)
    else:
        g1_reason = "Verified historical provenance, zero lookahead, untouched owned holdout"
    gates.append(GateResult(
        1, "Data & Holdout Integrity", g1_pass, "PASS" if g1_pass else "FAIL", g1_reason,
    ))

    # Gate 2 — Net Profitability (Overall + Unseen)
    net_ret = float(multi_day_metrics.get("total_return_pct", 0.0))
    net_pnl = float(multi_day_metrics.get("net_pnl_inr", 0.0))
    unseen_ret = float(unseen_metrics.get("total_return_pct", 0.0))
    g2_pass = net_ret > 0.0 and net_pnl > 0.0 and unseen_ret > 0.0
    gates.append(GateResult(
        2, "Net Profitability", g2_pass, "PASS" if g2_pass else "FAIL",
        f"Overall Net={net_ret:+.2f}% (₹{net_pnl:+.2f}), Unseen={unseen_ret:+.2f}% (requires > 0% on both)",
    ))

    # Gate 3 — Benchmark Comparison (Positive alpha vs Buy & Hold and > 0% vs Cash)
    alpha_bh = float(multi_day_metrics.get("alpha_vs_buy_and_hold_pct", 0.0))
    g3_pass = alpha_bh > 0.0 and net_ret > 0.0
    gates.append(GateResult(
        3, "Benchmark Comparison", g3_pass, "PASS" if g3_pass else "FAIL",
        f"Alpha vs B&H={alpha_bh:+.2f}%, vs Cash={net_ret:+.2f}% (requires positive alpha & positive net)",
    ))

    # Gate 4 — Risk Targets (DD <= 10%, PF > 1.20, Sharpe > 1.0, Expectancy > 0)
    max_dd = float(multi_day_metrics.get("max_drawdown_pct", 0.0))
    pf = float(multi_day_metrics.get("profit_factor", 0.0))
    sharpe = float(multi_day_metrics.get("sharpe_ratio", 0.0))
    exp = float(multi_day_metrics.get("expectancy_inr", 0.0))
    g4_pass = max_dd <= 10.0 and pf > 1.20 and sharpe > 1.0 and exp > 0.0
    gates.append(GateResult(
        4, "Risk Controls", g4_pass, "PASS" if g4_pass else "FAIL",
        f"MaxDD={max_dd:.2f}% (<=10%), PF={pf:.2f} (>1.20), Sharpe={sharpe:.2f} (>1.0), Expectancy=₹{exp:+.2f}",
    ))

    # Gate 5 — Statistical Reliability (>= 60 sessions, >= 30 trades)
    sessions = int(multi_day_metrics.get("sessions_evaluated", 0))
    trades = int(multi_day_metrics.get("total_trades", 0))
    g5_pass = sessions >= min_sessions_required and trades >= min_trades_required
    gates.append(GateResult(
        5, "Statistical Reliability", g5_pass, "PASS" if g5_pass else "INSUFFICIENT_EVIDENCE",
        f"Sessions={sessions}/{min_sessions_required}, Trades={trades}/{min_trades_required}",
    ))

    # Gate 6 — Execution Robustness (Net positive under conservative 10 bps slippage & realistic brokerage)
    cons_ret = float(conservative_slippage_metrics.get("net_return_pct", 0.0))
    g6_pass = cons_ret > 0.0
    gates.append(GateResult(
        6, "Execution Robustness", g6_pass, "PASS" if g6_pass else "FAIL",
        f"10 bps slippage Net Return={cons_ret:+.2f}% (must remain > 0% under realistic brokerage)",
    ))

    # Gate 7 — Independent Approval (Statistical Validation + Red Team Challenger)
    g7_pass = (stat_validation_verdict == "PASS") and red_team_approved
    gates.append(GateResult(
        7, "Independent Approval", g7_pass, "PASS" if g7_pass else "FAIL",
        f"StatVal={stat_validation_verdict}, RedTeam={'APPROVED' if red_team_approved else 'REJECTED'} ({red_team_reason})",
    ))

    passed_count = sum(1 for g in gates if g.passed)

    # Determine final qualification status (integrity gates take strict precedence)
    if not g1_pass:
        if (not hold_ok) and prov_ok:
            final_status: QualificationStatus = "INVALIDATED"
            summary = f"Invalidated: {g1_reason} ({passed_count}/7 gates passed)."
        else:
            final_status = "BLOCKED_INTEGRITY"
            summary = f"Blocked Integrity: {g1_reason} ({passed_count}/7 gates passed)."
    elif not g2_pass or not g6_pass or not red_team_approved:
        final_status = "REJECTED"
        summary = f"Rejected: Failed profitability/robustness/red-team ({passed_count}/7 gates passed)."
    elif not g5_pass or stat_validation_verdict == "INSUFFICIENT EVIDENCE":
        final_status = "INSUFFICIENT_EVIDENCE"
        summary = f"Insufficient statistical evidence ({sessions}/{min_sessions_required} sessions, {trades}/{min_trades_required} trades)."
    elif passed_count == 7:
        final_status = "QUALIFIED"
        summary = "Passed all 7 independent qualification gates."
    else:
        final_status = "REJECTED"
        summary = f"Did not meet all qualification criteria ({passed_count}/7 gates passed)."

    return QualificationEvaluation(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        final_status=final_status,
        gates=gates,
        passed_gates_count=passed_count,
        total_gates=7,
        summary=summary,
        provenance_status=prov_status_eff,
        holdout_status=hold_status_eff,
        validation_timestamp=val_ts,
        failure_reasons=failure_reasons,
    )


def separate_strategies_and_benchmarks(
    qualification_board: list[dict[str, Any]],
    period_label: str = "",
) -> dict[str, Any]:
    """Strictly separate executable trading strategies from Cash/Buy-and-Hold reference benchmarks."""
    active_strategies: list[dict[str, Any]] = []
    reference_benchmarks: list[dict[str, Any]] = []

    for row in qualification_board:
        sid = str(row.get("strategy_id", ""))
        if sid.startswith("BENCHMARK-"):
            net_ret = float(row.get("net_return_pct", 0.0))
            alpha_bh = float(row.get("alpha_vs_bh_pct", 0.0))
            reference_benchmarks.append({
                "benchmark_id": sid,
                "benchmark_name": row.get("strategy_name", "Cash / No-Trade Baseline"),
                "category": "CAPITAL_PRESERVATION_BENCHMARK",
                "period": period_label,
                "sessions": row.get("sessions", 0),
                "net_return_pct": net_ret,
                "relative_outperformance_vs_bh_pct": alpha_bh,
                "note": "Capital preservation baseline (0% return); not an active trading strategy.",
            })
        else:
            active_strategies.append(row)

    # Derive implied Buy & Hold benchmark from the first row's net_return_pct - alpha_vs_bh_pct
    if qualification_board:
        sample = qualification_board[0]
        bh_ret = round(float(sample.get("net_return_pct", 0.0)) - float(sample.get("alpha_vs_bh_pct", 0.0)), 2)
        reference_benchmarks.append({
            "benchmark_id": "BENCHMARK-BUY-AND-HOLD",
            "benchmark_name": "5-Stock Equal-Weight Buy & Hold",
            "category": "PASSIVE_MARKET_BENCHMARK",
            "period": period_label,
            "sessions": sample.get("sessions", 0),
            "net_return_pct": bh_ret,
            "relative_outperformance_vs_bh_pct": 0.0,
            "note": "Equal-weight passive benchmark across RELIANCE, TCS, BIRLACABLE, ITC, TATAELXSI.",
        })

    active_sorted = sorted(active_strategies, key=lambda x: float(x.get("net_return_pct", -999.0)), reverse=True)
    qualified_active = sum(1 for s in active_sorted if s.get("final_status") == "QUALIFIED")

    return {
        "active_strategy_leaderboard": active_sorted,
        "reference_benchmark_leaderboard": reference_benchmarks,
        "qualified_active_strategies": qualified_active,
        "best_active_strategy": active_sorted[0] if active_sorted else None,
    }
