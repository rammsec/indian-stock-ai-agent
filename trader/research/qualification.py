"""Formal 7-Gate Strategy Qualification Framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QualificationStatus = Literal[
    "REJECTED",
    "EXPERIMENTAL",
    "IMPROVING",
    "VALIDATION PENDING",
    "INSUFFICIENT EVIDENCE",
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


def evaluate_seven_gates(
    strategy_id: str,
    strategy_name: str,
    multi_day_metrics: dict[str, Any],
    unseen_metrics: dict[str, Any],
    conservative_slippage_metrics: dict[str, Any],
    data_provenance_verified: bool,
    stat_validation_verdict: Literal["PASS", "FAIL", "INSUFFICIENT EVIDENCE"],
    red_team_approved: bool,
    red_team_reason: str,
    min_sessions_required: int = 60,
    min_trades_required: int = 30,
) -> QualificationEvaluation:
    """Evaluate a strategy candidate against all 7 mandatory institutional qualification gates."""
    gates: list[GateResult] = []

    # Gate 1 — Data Integrity
    g1_pass = bool(data_provenance_verified)
    gates.append(GateResult(
        1, "Data Integrity", g1_pass, "PASS" if g1_pass else "FAIL",
        "Verified historical provenance, zero lookahead, explicit eligibility" if g1_pass else "Unverified data provenance",
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

    # Determine final qualification status
    if not g2_pass or not g6_pass or not red_team_approved:
        final_status: QualificationStatus = "REJECTED"
        summary = f"Rejected: Failed profitability/robustness/red-team ({passed_count}/7 gates passed)."
    elif not g5_pass or stat_validation_verdict == "INSUFFICIENT EVIDENCE":
        final_status = "INSUFFICIENT EVIDENCE"
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
    )
