"""Phase 11 & 11.1: Chief Research Orchestrator, Integrity Gates & Multi-Timeframe Research."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import IST_TZ, LAB_DEFAULT_UNIVERSE
from ..data.provenance_1h import validate_1h_provenance
from ..data.schema import DataSourceMode
from ..lab.data_service import (
    compute_data_quality_analytics,
    filter_research_eligible_dates,
    get_available_historical_dates,
)
from ..lab.database import HistoricalLabDatabase
from ..lab.replay_engine import STRATEGY_REGISTRY, HistoricalLabReplayEngine
from ..strategies.cost_aware_wrapper import CostAwareFilterWrapper
from ..strategies.orb_breakout import ORBBreakoutStrategy
from ..strategies.vwap_pullback import VWAPPullbackStrategy
from .agents import (
    AgentActivityReport,
    AIModelDecisionAgent,
    ExecutionCostAgent,
    MarketDataIntelligenceAgent,
    QuantitativeStrategyResearchAgent,
    RiskManagementAgent,
    SoftwareQualityAgent,
    StatisticalValidationAgent,
    StrategyChallengerRedTeamAgent,
)
from .holdout import SEALED_HOLDOUT_REGISTRY_PATH, SealedHoldoutGuard, _get_git_commit_sha
from .integrity_guard import ResearchIntegrityGuard
from .qualification import evaluate_seven_gates, separate_strategies_and_benchmarks

RESEARCH_REPORT_PATH = Path("data/lab/research_cycle_report.json")
CYCLE_002_REPORT_PATH = Path("data/lab/cycle_002_report.json")
CYCLE_003_REPORT_PATH = Path("data/lab/cycle_003_report.json")
CYCLE_003_REASSESSMENT_PATH = Path("data/lab/cycle_003_integrity_reassessment.json")
CYCLE_003_PHASE11_3_REASSESSMENT_PATH = Path("data/lab/cycle_003_phase11_3_reassessment.json")
CYCLE_003_PHASE11_3_1_REASSESSMENT_PATH = Path("data/lab/cycle_003_phase11_3_1_reassessment.json")
CYCLE_004_REPORT_PATH = Path("data/lab/cycle_004_report.json")
CYCLE_005_REPORT_PATH = Path("data/lab/cycle_005_report.json")
PROVENANCE_REPORT_PATH = Path("data/lab/provenance_report.json")
ARCHIVE_HEALTH_PATH = Path("data/lab/daily_archive_health.json")
LAB_INDEX_PATH = Path("docs/lab/data/lab_index.json")


def _get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def verify_dataset_provenance(
    ordered_dates: list[date],
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    timeframe_minutes: int = 5,
    resampling_methodology: str = "NONE_RAW_5M",
) -> tuple[bool, dict[str, Any]]:
    """Independently verify provider, timestamps, completeness, duplicates, and SHA-256 fingerprints."""
    if not ordered_dates:
        return False, {"verified": False, "reason": "No eligible dates provided.", "symbols": {}}

    import pandas as pd
    from ..data.loader import CACHE_DIR

    avail_by_date = {
        d.date: d for d in get_available_historical_dates(tuple(symbols), max_dates=max(150, len(ordered_dates) + 30))
    }

    per_symbol: dict[str, Any] = {}
    all_ok = True
    total_missing = 0
    total_duplicates = 0

    for sym in symbols:
        providers: set[str] = set()
        retrieval_timestamps: list[str] = []
        sha_hashes: set[str] = set()
        cache_path = CACHE_DIR / f"{sym}_5m.parquet"
        sym_dups = 0
        if cache_path.exists():
            df_sym = pd.read_parquet(cache_path, columns=["timestamp"])
            sym_dups = int(df_sym["timestamp"].duplicated().sum())

        sym_missing = 0
        sym_candles = 0
        bad_sessions: list[str] = []

        for d in ordered_dates:
            day_obj = avail_by_date.get(d.isoformat())
            q = next((s for s in day_obj.stocks if s.symbol == sym), None) if day_obj else None
            if q is None:
                bad_sessions.append(d.isoformat())
                continue
            if q.provider:
                providers.add(q.provider)
            if q.retrieved_at_ist:
                retrieval_timestamps.append(q.retrieved_at_ist)
            if q.sha256_hash and q.sha256_hash not in ("UNAVAILABLE", "EXCHANGE_HOLIDAY"):
                sha_hashes.add(q.sha256_hash[:16])
            sym_missing += q.missing_intervals_count
            sym_candles += q.candles_count
            if q.candles_count < 70 or q.provider == "UNAVAILABLE":
                bad_sessions.append(d.isoformat())

        if bad_sessions or not sha_hashes or sym_dups > 0:
            all_ok = False
        total_missing += sym_missing
        total_duplicates += sym_dups

        per_symbol[sym] = {
            "symbol": sym,
            "exchange": "NSE",
            "providers": sorted(providers),
            "candle_interval": f"{timeframe_minutes}m",
            "resampling_methodology": resampling_methodology,
            "sessions_checked": len(ordered_dates),
            "total_5m_candles": sym_candles,
            "missing_candles": sym_missing,
            "duplicate_candles": sym_dups,
            "latest_retrieval_timestamp": max(retrieval_timestamps) if retrieval_timestamps else "N/A",
            "exchange_date_range": f"{ordered_dates[0].isoformat()} -> {ordered_dates[-1].isoformat()}",
            "source_files": [str(cache_path)],
            "dataset_sha256_samples": sorted(sha_hashes)[:5],
            "bad_sessions": bad_sessions,
        }

    report = {
        "verified": all_ok,
        "verified_at_ist": datetime.now(tz=IST_TZ).isoformat(),
        "sessions_verified": len(ordered_dates),
        "total_missing_candles": total_missing,
        "total_duplicate_candles": total_duplicates,
        "resampling_methodology": resampling_methodology,
        "symbols": per_symbol,
    }
    return all_ok, report


def collect_dynamic_test_metrics(run_pytest: bool = True) -> dict[str, Any]:
    """Run pytest dynamically and return structured execution metrics (never hardcoded)."""
    import sys

    ts = datetime.now(tz=IST_TZ).isoformat()
    commit_sha = _get_git_commit_sha()
    cmd = [sys.executable, "-m", "pytest", "tests/intraday/", "tests/phase6/", "-q"]
    cmd_str = "pytest tests/intraday/ tests/phase6/ -q"
    if not run_pytest or os.environ.get("PYTEST_CURRENT_TEST"):
        return {
            "executed": False,
            "status": "NOT_RUN",
            "independently_verified": False,
            "command": None,
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "execution_timestamp": ts,
            "commit_sha": commit_sha,
            "mode": "SKIPPED_NESTED_PYTEST" if os.environ.get("PYTEST_CURRENT_TEST") else "NOT_RUN",
        }

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+)\s+passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+)\s+failed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+)\s+error", out)) else 0
    skipped = int(m.group(1)) if (m := re.search(r"(\d+)\s+skipped", out)) else 0
    executed = passed + failed + errors + skipped
    return {
        "executed": True,
        "status": "EXECUTED",
        "independently_verified": True,
        "command": cmd_str,
        "tests_executed": executed,
        "tests_passed": passed,
        "tests_failed": failed + errors,
        "tests_skipped": skipped,
        "execution_timestamp": ts,
        "commit_sha": commit_sha,
        "returncode": proc.returncode,
        "mode": "LIVE_PYTEST",
    }


def compute_regime_breakdown(daily_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate strategy performance across market regimes."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for d in daily_runs:
        reg = d.get("regime", "UNKNOWN")
        buckets.setdefault(reg, []).append(d)

    out: dict[str, dict[str, Any]] = {}
    for reg, rows in buckets.items():
        net_pnl = round(sum(r["net_pnl"] for r in rows), 2)
        trades = sum(r["trades"] for r in rows)
        fees = round(sum(r["fees"] for r in rows), 2)
        wins = sum(1 for r in rows if r["net_pnl"] > 0)
        out[reg] = {
            "regime": reg,
            "sessions": len(rows),
            "trades": trades,
            "net_pnl_inr": net_pnl,
            "fees_inr": fees,
            "positive_sessions": wins,
            "session_win_rate_pct": round((wins / len(rows)) * 100.0, 1) if rows else 0.0,
        }
    return out


class ChiefResearchOrchestrator:
    """Agent 1: Coordinates bounded multi-agent research cycles without self-approval authority."""

    MAX_CANDIDATES_PER_CYCLE = 10
    MAX_TUNING_ROUNDS_PER_HYPOTHESIS = 3

    def __init__(self) -> None:
        self.data_agent = MarketDataIntelligenceAgent()
        self.quant_agent = QuantitativeStrategyResearchAgent()
        self.risk_agent = RiskManagementAgent()
        self.exec_agent = ExecutionCostAgent()
        self.stat_agent = StatisticalValidationAgent()
        self.ai_agent = AIModelDecisionAgent()
        self.sq_agent = SoftwareQualityAgent()
        self.red_team = StrategyChallengerRedTeamAgent()

    def run_research_cycle(
        self,
        cycle_id: str = "CYCLE-001",
        min_bars_for_research: int = 70,
        candidate_ids: list[str] | None = None,
        test_report: dict[str, Any] | None = None,
        run_live_pytest: bool = True,
        slippage_bps_list: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0),
        record_sealed_holdout: bool = True,
        output_path: Path = RESEARCH_REPORT_PATH,
    ) -> dict[str, Any]:
        """Execute a complete bounded research cycle across explicitly reconciled eligible dates."""
        avail_dates = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=150)
        eligible_objs = filter_research_eligible_dates(avail_dates, min_bars_per_stock=min_bars_for_research)
        elig_set = {d.date for d in eligible_objs}
        ordered_dates = sorted(date.fromisoformat(d.date) for d in eligible_objs)

        excluded_dates = [
            {
                "date": d.date,
                "date_status": d.date_status,
                "reason": "; ".join(
                    f"{s.symbol}: {s.candles_count}/{s.expected_candles} bars"
                    for s in d.stocks if s.candles_count < min_bars_for_research
                ),
            }
            for d in avail_dates if d.date not in elig_set
        ]

        dq_analytics = compute_data_quality_analytics(LAB_DEFAULT_UNIVERSE, min_bars_for_research=min_bars_for_research)
        data_report = self.data_agent.audit_data(dq_analytics, excluded_dates)

        # Phase 11.1: Genuine dataset provenance verification (never hardcoded True)
        prov_verified, prov_report = verify_dataset_provenance(
            ordered_dates=ordered_dates,
            symbols=LAB_DEFAULT_UNIVERSE,
            timeframe_minutes=5,
            resampling_methodology=(
                "OHLCV_0915_IST_STRICT_NO_INTERPOLATION" if cycle_id == "CYCLE-002" else "NONE_RAW_5M"
            ),
        )
        PROVENANCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE_REPORT_PATH.write_text(json.dumps(prov_report, indent=2), encoding="utf-8")

        # Register bounded Phase 11 candidate strategies if not already registered
        if "STRAT-001-VWAP-V2" not in STRATEGY_REGISTRY:
            STRATEGY_REGISTRY["STRAT-001-VWAP-V2"] = {
                "name": "VWAP Pullback (v2.0 Cost-Aware)",
                "cls": lambda: CostAwareFilterWrapper(VWAPPullbackStrategy(), min_expected_move_pct=0.012),
                "kind": "STRATEGY",
                "timeframe_minutes": 5,
            }
        if "STRAT-002-ORB-V2-08" not in STRATEGY_REGISTRY:
            STRATEGY_REGISTRY["STRAT-002-ORB-V2-08"] = {
                "name": "ORB Breakout (v2.0 0.8% Hurdle)",
                "cls": lambda: CostAwareFilterWrapper(ORBBreakoutStrategy(), version_tag="v2.0-08", min_expected_move_pct=0.008),
                "kind": "STRATEGY",
                "timeframe_minutes": 5,
            }

        if candidate_ids is None:
            selected_ids = list(STRATEGY_REGISTRY.keys())[: self.MAX_CANDIDATES_PER_CYCLE]
            if "STRAT-001-VWAP-V2" not in selected_ids:
                selected_ids[-1] = "STRAT-001-VWAP-V2"
        else:
            selected_ids = list(candidate_ids)[: self.MAX_CANDIDATES_PER_CYCLE]

        engine = HistoricalLabReplayEngine()
        db = HistoricalLabDatabase()
        start_d, end_d = ordered_dates[0], ordered_dates[-1]

        # Sealed final-test holdout split protection
        n_dates = len(ordered_dates)
        t_end_idx = max(1, int(n_dates * 0.60))
        v_end_idx = max(t_end_idx + 1, int(n_dates * 0.80))
        unseen_dates_list = ordered_dates[v_end_idx:] or [ordered_dates[-1]]
        holdout_guard = SealedHoldoutGuard()
        holdout_status = (
            holdout_guard.record_exposure(cycle_id, unseen_dates_list, notes=f"{cycle_id} unseen_test evaluation")
            if record_sealed_holdout
            else {"total_sealed_dates": len(holdout_guard.load_registry().get("sealed_dates", [])), "leakage_detected": False}
        )

        leaderboard: list[dict[str, Any]] = []
        qualification_board: list[dict[str, Any]] = []
        regime_summary_by_strategy: dict[str, Any] = {}
        orders_by_strategy: dict[str, list[dict[str, Any]]] = {}
        all_slippage_rows: list[dict[str, Any]] = []
        slippage_matrix_by_strategy: dict[str, list[dict[str, Any]]] = {}

        for sid in selected_ids:
            exec_mode = "AI_AGENT" if sid.startswith("AGENT-") else "DETERMINISTIC"
            res = engine.run_date_range(
                start_date=start_d,
                end_date=end_d,
                strategy_id=sid,
                symbols=LAB_DEFAULT_UNIVERSE,
                data_mode=DataSourceMode.HISTORICAL_REAL,
                execution_mode=exec_mode,  # type: ignore[arg-type]
                db=db,
                eligible_dates=ordered_dates,
            )
            assert res["sessions_evaluated"] == len(ordered_dates), (
                f"Session mismatch for {sid}: {res['sessions_evaluated']} != {len(ordered_dates)}"
            )
            orders_by_strategy[sid] = res.pop("orders", [])
            res.pop("trades", [])

            # Phase 11.1: Genuine slippage sensitivity from actual fill prices (5, 10, 20, 30 bps)
            if res["total_trades"] > 0:
                slip_rows = engine.run_slippage_sensitivity(
                    start_date=start_d,
                    end_date=end_d,
                    strategy_id=sid,
                    slippage_bps_list=slippage_bps_list,
                    symbols=LAB_DEFAULT_UNIVERSE,
                    data_mode=DataSourceMode.HISTORICAL_REAL,
                    eligible_dates=ordered_dates,
                )
            else:
                slip_rows = [
                    {
                        "slippage_bps": float(bps),
                        "gross_return_pct": res["gross_return_pct"],
                        "net_return_pct": res["total_return_pct"],
                        "net_pnl_inr": res["net_pnl_inr"],
                        "total_fees_inr": res["total_fees_inr"],
                        "total_brokerage_inr": res.get("total_brokerage_inr", 0.0),
                        "total_statutory_inr": res.get("total_statutory_inr", 0.0),
                        "total_slippage_inr": 0.0,
                        "total_trades": 0,
                        "win_rate_pct": 0.0,
                    }
                    for bps in slippage_bps_list
                ]
            slippage_matrix_by_strategy[sid] = slip_rows
            all_slippage_rows.extend({"strategy_id": sid, **sr} for sr in slip_rows)
            cons_10 = next((sr for sr in slip_rows if float(sr["slippage_bps"]) == 10.0), slip_rows[0])

            leaderboard.append(res)
            regime_summary_by_strategy[sid] = compute_regime_breakdown(res["daily_runs"])

            # Derive chronological Train (60%) / Val (20%) / Unseen (20%) directly from res["daily_runs"]
            dr = res["daily_runs"]
            n_dr = len(dr)
            t_end = max(1, int(n_dr * 0.60))
            v_end = max(t_end + 1, int(n_dr * 0.80))
            unseen_slice = dr[v_end:] or [dr[-1]]
            unseen_pnl = sum(d["net_pnl"] for d in unseen_slice)
            unseen_ret_pct = round((unseen_pnl / engine.risk_config.initial_capital_inr) * 100.0, 2)
            wf = {
                "train": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[:t_end]) / engine.risk_config.initial_capital_inr) * 100.0, 2)},
                "validation": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[t_end:v_end]) / engine.risk_config.initial_capital_inr) * 100.0, 2)},
                "unseen_test": {"total_return_pct": unseen_ret_pct, "total_trades": sum(d["trades"] for d in unseen_slice)},
            }

            stat_verdict, stat_reason = self.stat_agent.validate(res, wf)
            rt_approved, rt_reason = self.red_team.challenge(sid, res, wf)

            qual = evaluate_seven_gates(
                strategy_id=sid,
                strategy_name=res["strategy_name"],
                multi_day_metrics=res,
                unseen_metrics=wf["unseen_test"],
                conservative_slippage_metrics=cons_10,
                data_provenance_verified=prov_verified,
                stat_validation_verdict=stat_verdict,
                red_team_approved=rt_approved,
                red_team_reason=rt_reason,
                min_sessions_required=60,
                min_trades_required=30,
                independent_holdout_valid=holdout_status.get("independent_holdout_valid", False),
                provenance_status="VERIFIED" if prov_verified else "FAILED",
                holdout_status=holdout_status.get("status", "UNKNOWN"),
            )

            qualification_board.append({
                "strategy_id": sid,
                "strategy_name": res["strategy_name"],
                "timeframe_minutes": res.get("timeframe_minutes", 5),
                "final_status": qual.final_status,
                "passed_gates": f"{qual.passed_gates_count}/{qual.total_gates}",
                "sessions": res["sessions_evaluated"],
                "trades": res["total_trades"],
                "gross_return_pct": res["gross_return_pct"],
                "net_return_pct": res["total_return_pct"],
                "conservative_10bps_net_pct": cons_10["net_return_pct"],
                "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "sharpe_ratio": res["sharpe_ratio"],
                "sortino_ratio": res["sortino_ratio"],
                "profit_factor": res["profit_factor"],
                "expectancy_inr": res["expectancy_inr"],
                "total_fees_inr": res["total_fees_inr"],
                "total_brokerage_inr": res.get("total_brokerage_inr", 0.0),
                "total_statutory_inr": res.get("total_statutory_inr", 0.0),
                "total_slippage_inr": res.get("total_slippage_inr", 0.0),
                "unseen_return_pct": wf["unseen_test"]["total_return_pct"],
                "statistical_verdict": stat_verdict,
                "red_team_verdict": "APPROVED" if rt_approved else "REJECTED",
                "summary": qual.summary,
                "gates": [asdict(g) for g in qual.gates],
            })

        proposals, quant_report = self.quant_agent.propose_improvements(leaderboard)
        risk_report = self.risk_agent.review_risk(leaderboard)
        exec_report = self.exec_agent.audit_costs(leaderboard, all_slippage_rows, orders_by_strategy=orders_by_strategy)
        ai_report = self.ai_agent.audit_ai_vs_deterministic(leaderboard)

        live_test_metrics = test_report if test_report is not None else collect_dynamic_test_metrics(run_pytest=run_live_pytest)
        sq_report = self.sq_agent.review_system(test_report=live_test_metrics)

        sep_boards = separate_strategies_and_benchmarks(qualification_board, period_label=f"{start_d.isoformat()} -> {end_d.isoformat()}")
        qualified_cnt = sep_boards["qualified_active_strategies"]
        rejected_cnt = sum(1 for q in sep_boards["active_strategy_leaderboard"] if q["final_status"] in ("REJECTED", "INVALIDATED"))
        best_strat = sep_boards["best_active_strategy"]

        orch_report = AgentActivityReport(
            agent_name="Agent 1 — Chief Research Orchestrator",
            role="Research Coordinator & Backlog Manager",
            execution_type="DETERMINISTIC_LOCAL",
            assigned_task=f"Coordinate {cycle_id} across {len(ordered_dates)} reconciled sessions",
            latest_finding=(
                f"Evaluated {len(qualification_board)} candidates: {qualified_cnt} QUALIFIED, {rejected_cnt} REJECTED. "
                f"Best candidate: {best_strat['strategy_id'] if best_strat else 'None'} ({best_strat['net_return_pct']:+.2f}%)."
            ),
            status="ACTIVE",
            proposals_count=len(proposals),
            rejections_count=rejected_cnt,
        )

        stat_report = AgentActivityReport(
            agent_name="Agent 6 — Statistical Validation",
            role="Independent Scientific Reviewer",
            execution_type="DETERMINISTIC_LOCAL",
            assigned_task="Audit 60-session threshold & unseen out-of-sample splits",
            latest_finding=(
                f"Evaluated {len(ordered_dates)}/60 sessions; {qualified_cnt} strategies passed positive unseen profitability."
            ),
            status="PASS" if qualified_cnt > 0 else "WARNING",
        )
        rt_report = AgentActivityReport(
            agent_name="Agent 9 — Strategy Challenger (Red Team)",
            role="Independent Adversarial Challenger",
            execution_type="DETERMINISTIC_LOCAL",
            assigned_task="Challenge concentration, unseen performance & cost survival",
            latest_finding=f"Rejected {rejected_cnt}/{len(qualification_board)} candidates failing net cost survival or unseen splits.",
            status="PASS",
            rejections_count=rejected_cnt,
        )

        agent_reports = [
            asdict(orch_report),
            asdict(data_report),
            asdict(quant_report),
            asdict(risk_report),
            asdict(exec_report),
            asdict(stat_report),
            asdict(ai_report),
            asdict(sq_report),
            asdict(rt_report),
        ]

        archive_health = {}
        if ARCHIVE_HEALTH_PATH.exists():
            archive_health = json.loads(ARCHIVE_HEALTH_PATH.read_text(encoding="utf-8"))

        # Multi-timeframe comparison summary
        tf_groups: dict[str, list[dict[str, Any]]] = {}
        for q in qualification_board:
            tf_key = f"{q['timeframe_minutes']}m" if not q["strategy_id"].startswith("BENCHMARK-") else "CASH"
            tf_groups.setdefault(tf_key, []).append(q)
        timeframe_comparison = [
            {
                "timeframe": tf_k,
                "strategies_count": len(rows),
                "avg_trades": round(sum(r["trades"] for r in rows) / len(rows), 1),
                "avg_net_return_pct": round(sum(r["net_return_pct"] for r in rows) / len(rows), 2),
                "best_strategy_id": max(rows, key=lambda x: x["net_return_pct"])["strategy_id"],
                "best_net_return_pct": max(rows, key=lambda x: x["net_return_pct"])["net_return_pct"],
                "avg_total_fees_inr": round(sum(r["total_fees_inr"] for r in rows) / len(rows), 2),
            }
            for tf_k, rows in tf_groups.items()
        ]

        cycle_output = {
            "cycle_id": cycle_id,
            "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
            "reconciled_eligible_sessions": len(ordered_dates),
            "date_range": f"{start_d.isoformat()} -> {end_d.isoformat()}",
            "excluded_dates": excluded_dates,
            "provenance_verified": prov_verified,
            "provenance_summary": {
                "sessions_verified": prov_report["sessions_verified"],
                "total_missing_candles": prov_report["total_missing_candles"],
                "total_duplicate_candles": prov_report["total_duplicate_candles"],
                "resampling_methodology": prov_report["resampling_methodology"],
            },
            "sealed_holdout_status": holdout_status,
            "dynamic_test_metrics": live_test_metrics,
            "command_centre": {
                "current_cycle": cycle_id,
                "active_agents": len(agent_reports),
                "experiments_completed": len(qualification_board),
                "experiments_rejected": rejected_cnt,
                "qualified_strategies": qualified_cnt,
                "best_experimental_strategy": best_strat["strategy_id"] if best_strat else None,
                "best_experimental_net_pct": best_strat["net_return_pct"] if best_strat else 0.0,
                "last_archive_ist": archive_health.get("last_successful_run_ist", "N/A"),
                "archive_status": archive_health.get("archive_status", "UNKNOWN"),
                "max_candidates_per_cycle": self.MAX_CANDIDATES_PER_CYCLE,
                "max_tuning_rounds": self.MAX_TUNING_ROUNDS_PER_HYPOTHESIS,
            },
            "timeframe_comparison": timeframe_comparison,
            "slippage_sensitivity_matrix": slippage_matrix_by_strategy,
            "agent_performance": agent_reports,
            "qualification_board": sorted(qualification_board, key=lambda x: x["net_return_pct"], reverse=True),
            "proposals": [asdict(p) for p in proposals],
            "regime_breakdown": regime_summary_by_strategy,
            "archive_health": archive_health,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cycle_output, indent=2), encoding="utf-8")

        if LAB_INDEX_PATH.exists():
            idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
            idx["phase11_research_cycle"] = cycle_output
            if cycle_id == "CYCLE-002":
                idx["cycle_002_report"] = cycle_output
            idx["multi_day_leaderboard"] = sorted(
                leaderboard, key=lambda r: (r["total_return_pct"], r["sharpe_ratio"]), reverse=True
            )
            LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

        return cycle_output

    def run_cycle_002(
        self,
        min_bars_for_research: int = 70,
        test_report: dict[str, Any] | None = None,
        run_live_pytest: bool = True,
    ) -> dict[str, Any]:
        """Execute CYCLE-002 comparing 5m baselines vs 15m/30m lower-frequency strategies and Cash."""
        cycle_002_candidates = [
            "STRAT-002-ORB-BREAKOUT",
            "STRAT-001-VWAP-PULLBACK",
            "STRAT-003-EMA-CROSS",
            "STRAT-008-15M-RS-LEADER",
            "STRAT-009-15M-VWAP-TREND",
            "STRAT-010-30M-EMA-TREND",
            "STRAT-011-30M-BREAKOUT-VOL",
            "STRAT-012-LOW-FREQ-TREND",
            "BENCHMARK-CASH-NO-TRADE",
        ]
        return self.run_research_cycle(
            cycle_id="CYCLE-002",
            min_bars_for_research=min_bars_for_research,
            candidate_ids=cycle_002_candidates,
            test_report=test_report,
            run_live_pytest=run_live_pytest,
            slippage_bps_list=(5.0, 10.0, 20.0, 30.0),
            record_sealed_holdout=True,
            output_path=CYCLE_002_REPORT_PATH,
        )

    def run_cycle_003(
        self,
        test_report: dict[str, Any] | None = None,
        run_live_pytest: bool = True,
    ) -> dict[str, Any]:
        """Execute CYCLE-003: Track A (30m Concentrated, 56 sessions) + Track B (60m Extended History, 240+ sessions)."""
        import hashlib
        import pandas as pd
        from ..config import LabRiskConfig
        from ..data.loader import CACHE_DIR
        from ..data.loader_1h import get_available_1h_dates

        live_test_metrics = test_report if test_report is not None else collect_dynamic_test_metrics(run_pytest=run_live_pytest)

        # Track A: 30m Concentrated on 56 reconciled 5m->30m sessions
        track_a = self.run_research_cycle(
            cycle_id="CYCLE-003-30M",
            min_bars_for_research=70,
            candidate_ids=[
                "STRAT-011-30M-BREAKOUT-VOL",
                "STRAT-013-30M-CONC-BREAKOUT",
                "STRAT-014-30M-VWAP-SQUEEZE",
                "BENCHMARK-CASH-NO-TRADE",
            ],
            test_report=live_test_metrics,
            run_live_pytest=False,
            slippage_bps_list=(5.0, 10.0, 20.0, 30.0),
            record_sealed_holdout=False,
            output_path=Path("data/lab/cycle_003_30m_track.json"),
        )

        # Track B: 60m (1H) Extended 1-Year History (240+ NSE sessions)
        elig_1h, excl_1h = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
        assert len(elig_1h) >= 60, f"Expected >= 60 1h sessions, found {len(elig_1h)}"
        start_1h, end_1h = elig_1h[0], elig_1h[-1]

        # Independent 1h provenance check
        prov_1h_ok = True
        prov_1h_syms: dict[str, Any] = {}
        for sym in LAB_DEFAULT_UNIVERSE:
            p_file = CACHE_DIR / f"{sym}_1h.parquet"
            if not p_file.exists():
                prov_1h_ok = False
                continue
            df_1h = pd.read_parquet(p_file, columns=["timestamp"])
            dups = int(df_1h["timestamp"].duplicated().sum())
            if dups > 0:
                prov_1h_ok = False
            sha = hashlib.sha256(p_file.read_bytes()).hexdigest()[:16]
            prov_1h_syms[sym] = {"file": str(p_file), "bars": len(df_1h), "duplicates": dups, "sha256": sha}

        # Concentrated risk config: max 2 concurrent positions up to 50% each (~₹25k per position)
        conc_risk = LabRiskConfig(max_open_positions=2, risk_per_trade_pct=0.015)
        engine_1h = HistoricalLabReplayEngine(risk_config=conc_risk)
        db = HistoricalLabDatabase()

        track_b_ids = [
            "STRAT-015-60M-TREND-BREAKOUT",
            "STRAT-016-60M-VWAP-MOMENTUM",
            "STRAT-017-60M-ORB-FILTERED",
            "BENCHMARK-CASH-NO-TRADE",
        ]

        n_1h = len(elig_1h)
        t_end_1h = max(1, int(n_1h * 0.60))
        v_end_1h = max(t_end_1h + 1, int(n_1h * 0.80))
        unseen_1h_dates = elig_1h[v_end_1h:] or [elig_1h[-1]]
        holdout_guard = SealedHoldoutGuard()
        holdout_1h = holdout_guard.record_exposure("CYCLE-003-60M", unseen_1h_dates, notes="1Y 1h unseen_test evaluation")

        lb_1h: list[dict[str, Any]] = []
        qual_1h: list[dict[str, Any]] = []
        orders_1h: dict[str, list[dict[str, Any]]] = {}
        slip_1h_all: list[dict[str, Any]] = []
        slip_1h_matrix: dict[str, list[dict[str, Any]]] = {}

        for sid in track_b_ids:
            res = engine_1h.run_date_range(
                start_date=start_1h,
                end_date=end_1h,
                strategy_id=sid,
                symbols=LAB_DEFAULT_UNIVERSE,
                data_mode=DataSourceMode.HISTORICAL_REAL,
                execution_mode="DETERMINISTIC",
                db=db,
                eligible_dates=elig_1h,
                timeframe_minutes=60,  # type: ignore[arg-type]
            )
            orders_1h[sid] = res.pop("orders", [])
            res.pop("trades", [])

            if res["total_trades"] > 0:
                slip_rows = engine_1h.run_slippage_sensitivity(
                    start_date=start_1h,
                    end_date=end_1h,
                    strategy_id=sid,
                    slippage_bps_list=(5.0, 10.0, 20.0, 30.0),
                    symbols=LAB_DEFAULT_UNIVERSE,
                    data_mode=DataSourceMode.HISTORICAL_REAL,
                    eligible_dates=elig_1h,
                    timeframe_minutes=60,  # type: ignore[arg-type]
                )
            else:
                slip_rows = [
                    {
                        "slippage_bps": float(b),
                        "gross_return_pct": 0.0,
                        "net_return_pct": 0.0,
                        "net_pnl_inr": 0.0,
                        "total_fees_inr": 0.0,
                        "total_brokerage_inr": 0.0,
                        "total_statutory_inr": 0.0,
                        "total_slippage_inr": 0.0,
                        "total_trades": 0,
                        "win_rate_pct": 0.0,
                    }
                    for b in (5.0, 10.0, 20.0, 30.0)
                ]
            slip_1h_matrix[sid] = slip_rows
            slip_1h_all.extend({"strategy_id": sid, **sr} for sr in slip_rows)
            cons_10 = next((sr for sr in slip_rows if float(sr["slippage_bps"]) == 10.0), slip_rows[0])

            lb_1h.append(res)
            dr = res["daily_runs"]
            unseen_slice = dr[v_end_1h:] or [dr[-1]]
            unseen_pnl = sum(d["net_pnl"] for d in unseen_slice)
            unseen_ret_pct = round((unseen_pnl / engine_1h.risk_config.initial_capital_inr) * 100.0, 2)
            wf = {
                "train": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[:t_end_1h]) / engine_1h.risk_config.initial_capital_inr) * 100.0, 2)},
                "validation": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[t_end_1h:v_end_1h]) / engine_1h.risk_config.initial_capital_inr) * 100.0, 2)},
                "unseen_test": {"total_return_pct": unseen_ret_pct, "total_trades": sum(d["trades"] for d in unseen_slice)},
            }

            stat_verdict, _ = self.stat_agent.validate(res, wf)
            rt_approved, rt_reason = self.red_team.challenge(sid, res, wf)

            qual = evaluate_seven_gates(
                strategy_id=sid,
                strategy_name=res["strategy_name"],
                multi_day_metrics=res,
                unseen_metrics=wf["unseen_test"],
                conservative_slippage_metrics=cons_10,
                data_provenance_verified=prov_1h_ok,
                stat_validation_verdict=stat_verdict,
                red_team_approved=rt_approved,
                red_team_reason=rt_reason,
                min_sessions_required=60,
                min_trades_required=30,
            )

            qual_1h.append({
                "strategy_id": sid,
                "strategy_name": res["strategy_name"],
                "timeframe_minutes": 60,
                "final_status": qual.final_status,
                "passed_gates": f"{qual.passed_gates_count}/{qual.total_gates}",
                "sessions": res["sessions_evaluated"],
                "trades": res["total_trades"],
                "gross_return_pct": res["gross_return_pct"],
                "net_return_pct": res["total_return_pct"],
                "conservative_10bps_net_pct": cons_10["net_return_pct"],
                "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "sharpe_ratio": res["sharpe_ratio"],
                "sortino_ratio": res["sortino_ratio"],
                "profit_factor": res["profit_factor"],
                "expectancy_inr": res["expectancy_inr"],
                "total_fees_inr": res["total_fees_inr"],
                "total_brokerage_inr": res.get("total_brokerage_inr", 0.0),
                "total_statutory_inr": res.get("total_statutory_inr", 0.0),
                "total_slippage_inr": res.get("total_slippage_inr", 0.0),
                "unseen_return_pct": wf["unseen_test"]["total_return_pct"],
                "statistical_verdict": stat_verdict,
                "red_team_verdict": "APPROVED" if rt_approved else "REJECTED",
                "summary": qual.summary,
                "gates": [asdict(g) for g in qual.gates],
            })

        exec_1h_report = self.exec_agent.audit_costs(lb_1h, slip_1h_all, orders_by_strategy=orders_1h)
        sq_report = self.sq_agent.review_system(test_report=live_test_metrics)
        combined_board = sorted(qual_1h + track_a["qualification_board"], key=lambda x: x["net_return_pct"], reverse=True)
        qualified_cnt = sum(1 for q in combined_board if q["final_status"] == "QUALIFIED")
        rejected_cnt = sum(1 for q in combined_board if q["final_status"] == "REJECTED")
        best_strat = max(combined_board, key=lambda x: x["net_return_pct"]) if combined_board else None

        prov_1h_strict, _ = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
        cycle_003_out = {
            "cycle_id": "CYCLE-003",
            "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
            "reconciled_eligible_sessions": len(elig_1h),
            "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
            "excluded_dates": excl_1h,
            "provenance_verified": bool(prov_1h_strict and track_a["provenance_verified"]),
            "provenance_1h_symbols": prov_1h_syms,
            "sealed_holdout_status": holdout_1h,
            "dynamic_test_metrics": live_test_metrics,
            "command_centre": {
                "current_cycle": "CYCLE-003",
                "active_agents": 9,
                "experiments_completed": len(combined_board),
                "experiments_rejected": rejected_cnt,
                "qualified_strategies": qualified_cnt,
                "best_experimental_strategy": best_strat["strategy_id"] if best_strat else None,
                "best_experimental_net_pct": best_strat["net_return_pct"] if best_strat else 0.0,
                "track_a_30m_sessions": track_a["reconciled_eligible_sessions"],
                "track_b_60m_sessions": len(elig_1h),
                "max_candidates_per_cycle": self.MAX_CANDIDATES_PER_CYCLE,
                "max_tuning_rounds": self.MAX_TUNING_ROUNDS_PER_HYPOTHESIS,
            },
            "track_a_30m": track_a,
            "track_b_60m_qualification_board": sorted(qual_1h, key=lambda x: x["net_return_pct"], reverse=True),
            "qualification_board": combined_board,
            "slippage_sensitivity_matrix": {**track_a.get("slippage_sensitivity_matrix", {}), **slip_1h_matrix},
            "agent_performance": [asdict(exec_1h_report), asdict(sq_report)],
            "archive_health": track_a.get("archive_health", {}),
        }

        CYCLE_003_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CYCLE_003_REPORT_PATH.write_text(json.dumps(cycle_003_out, indent=2), encoding="utf-8")

        if LAB_INDEX_PATH.exists():
            idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
            idx["cycle_003_report"] = cycle_003_out
            idx["phase11_research_cycle"] = cycle_003_out
            LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

        return cycle_003_out

    def build_cycle_progression(self) -> list[dict[str, Any]]:
        """Build chronological progression metrics across CYCLE-001 through CYCLE-004 for dashboard charting."""
        cycle_specs = [
            ("CYCLE-001", RESEARCH_REPORT_PATH, "5m Intraday Baseline (56 sessions)", "Over-trading & ₹20/order fees crushed 5m strategies (-2.20% best active net)."),
            ("CYCLE-002", CYCLE_002_REPORT_PATH, "15m/30m Discovery (56 sessions)", "30m cut fees by 80% & turned STRAT-011 gross positive (+0.40%), -1.59% net."),
            ("CYCLE-003", CYCLE_003_REPORT_PATH, "1Y 60m Extended History (240 sessions)", "STRAT-017 60m ORB proved +3.19% gross & +0.65% unseen edge, dragged to -4.65% net by 109 intraday trades."),
            ("CYCLE-004", CYCLE_004_REPORT_PATH, "60m Selective, Scaling & CNC Swing (240 sessions)", "Tested ultra-selective 60m, ₹2L/₹5L scaling, and ₹0-brokerage multi-day CNC swing."),
        ]
        progression: list[dict[str, Any]] = []
        for cid, path, title, default_insight in cycle_specs:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            board = data.get("qualification_board", [])
            active = [q for q in board if not str(q.get("strategy_id", "")).startswith("BENCHMARK-")]
            best_active = max(active, key=lambda x: x.get("net_return_pct", -999.0)) if active else (board[0] if board else None)
            if not best_active:
                continue
            cc = data.get("command_centre", {})
            progression.append({
                "cycle_id": cid,
                "title": title,
                "sessions": int(best_active.get("sessions", data.get("reconciled_eligible_sessions", 0))),
                "best_active_strategy": best_active.get("strategy_id", "--"),
                "timeframe": f"{best_active.get('timeframe_minutes', 5)}m",
                "trades": int(best_active.get("trades", 0)),
                "gross_return_pct": float(best_active.get("gross_return_pct", best_active.get("net_return_pct", 0.0))),
                "net_return_pct": float(best_active.get("net_return_pct", 0.0)),
                "unseen_return_pct": float(best_active.get("unseen_return_pct", 0.0)),
                "total_fees_inr": float(best_active.get("total_fees_inr", 0.0)),
                "qualified_count": int(cc.get("qualified_strategies", 0)),
                "candidates_tested": len(board),
                "key_insight": default_insight,
            })
        return progression

    def run_cycle_004(
        self,
        test_report: dict[str, Any] | None = None,
        run_live_pytest: bool = True,
        allow_exploratory_on_blocked: bool = True,
        include_unseen_in_exploratory: bool = True,
        registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH,
        output_path: Path = CYCLE_004_REPORT_PATH,
    ) -> dict[str, Any]:
        """Execute CYCLE-004 with strict holdout consumption, ownership, and crash recovery."""
        import uuid
        from ..config import LabRiskConfig
        from ..data.loader_1h import get_available_1h_dates
        from .cycle004 import run_cnc_swing_backtest

        live_test_metrics = test_report if test_report is not None else collect_dynamic_test_metrics(run_pytest=run_live_pytest)
        evaluation_id = f"eval_c4_{uuid.uuid4().hex[:8]}"
        evaluation_started_at = datetime.now(tz=IST_TZ).isoformat()

        holdout_guard = SealedHoldoutGuard(registry_path=registry_path)
        # Recover any stale EVALUATION_IN_PROGRESS reservations conservatively as EXPOSED
        stale_recovery = holdout_guard.recover_stale_reservations("CYCLE-004")

        # Step 1: Identify eligible dates & splits (no duplicates, no overlaps)
        elig_1h, excl_1h = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
        start_1h, end_1h = elig_1h[0], elig_1h[-1]
        n_1h = len(elig_1h)
        t_end_1h = max(1, int(n_1h * 0.60))
        v_end_1h = max(t_end_1h + 1, int(n_1h * 0.80))
        train_1h_dates = elig_1h[:t_end_1h]
        val_1h_dates = elig_1h[t_end_1h:v_end_1h]
        unseen_1h_dates = elig_1h[v_end_1h:] or [elig_1h[-1]]

        # Step 2: Validate 1h provenance & reserve unseen dates as SEALED owned by CYCLE-004
        prov_ok_004, prov_rep_004 = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
        dataset_fp = str(prov_rep_004.get("dataset_fingerprint", ""))

        seal_res = holdout_guard.seal_dates("CYCLE-004", unseen_1h_dates, dataset_fingerprint=dataset_fp)

        # Step 3: Validate integrity centrally via ResearchIntegrityGuard
        integrity_guard = ResearchIntegrityGuard(registry_path=registry_path)
        guard_res = integrity_guard.validate(
            cycle_id="CYCLE-004",
            training_dates=train_1h_dates,
            validation_dates=val_1h_dates,
            unseen_dates=unseen_1h_dates,
            dataset_fingerprint=dataset_fp,
            provenance_result=prov_rep_004,
        )

        qualification_allowed = bool(guard_res["integrity_passed"] and guard_res["qualification_allowed"])
        exploratory_allowed = bool(guard_res.get("exploratory_research_allowed", False))

        # Determine whether any evaluation will start & which dates it will read
        if qualification_allowed:
            eval_type = "FORMAL_QUALIFICATION"
            eval_dates = elig_1h
            consumed_dates = unseen_1h_dates
        elif exploratory_allowed and allow_exploratory_on_blocked:
            if include_unseen_in_exploratory:
                eval_type = "EXPLORATORY_WITH_UNSEEN"
                eval_dates = elig_1h
                consumed_dates = unseen_1h_dates
            else:
                eval_type = "EXPLORATORY_TRAIN_VAL_ONLY"
                eval_dates = train_1h_dates + val_1h_dates
                consumed_dates = []
        else:
            # Evaluation never starts: safely release unused reservation ONLY if we sealed it & never accessed it
            release_res = (
                holdout_guard.release_reservation("CYCLE-004", unseen_1h_dates, "Evaluation aborted before data access")
                if seal_res.get("sealed_count", 0) > 0
                else {"released": [], "denied": [], "ok": True}
            )
            blocked_out = {
                "cycle_id": "CYCLE-004",
                "evaluation_id": evaluation_id,
                "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
                "reconciled_eligible_sessions": len(elig_1h),
                "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
                "excluded_dates": excl_1h,
                "integrity_passed": False,
                "qualification_allowed": False,
                "exploratory_research_allowed": exploratory_allowed,
                "exploratory_only": True,
                "holdout_consumed": False,
                "holdout_state": "AVAILABLE" if release_res.get("released") else guard_res["holdout_status"],
                "evaluation_status": "ABORTED_NO_EVALUATION",
                "evaluation_type": "NONE",
                "evaluation_started_at": evaluation_started_at,
                "evaluation_completed_at": datetime.now(tz=IST_TZ).isoformat(),
                "consumed_holdout_dates": [],
                "dataset_fingerprint": dataset_fp,
                "strategy_ids_evaluated": [],
                "failure_reasons": guard_res["failure_reasons"],
                "provenance_verified": bool(prov_rep_004.get("verified", False)),
                "provenance_status": prov_rep_004.get("final_provenance_status", "FAILED"),
                "qualification_blocked": True,
                "seal_reservation": seal_res,
                "release_result": release_res,
                "stale_recovery": stale_recovery,
                "integrity_guard_result": guard_res,
                "dynamic_test_metrics": live_test_metrics,
                "qualification_board": [],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(blocked_out, indent=2), encoding="utf-8")
            return blocked_out

        # Step 4: Enforce mark_evaluation_in_progress() BEFORE reading any consumed holdout dates
        if consumed_dates:
            prog_res = holdout_guard.mark_evaluation_in_progress(
                "CYCLE-004",
                consumed_dates,
                dataset_fingerprint=dataset_fp,
                evaluation_type=eval_type,
            )
            if not prog_res.get("ok"):
                # Evaluation aborted immediately before any holdout data access.
                # Attempt to safely release the unused reservation back to AVAILABLE (restricted to owning cycle).
                release_res = holdout_guard.release_reservation(
                    "CYCLE-004",
                    consumed_dates,
                    reason=f"Evaluation aborted before data access: mark_evaluation_in_progress failed ({prog_res.get('reason')})",
                )
                cleanup_success = bool(release_res.get("ok") and len(release_res.get("released", [])) == len(consumed_dates))
                cleanup_status = "RESERVATION_RELEASED" if cleanup_success else "RELEASE_FAILED_RESERVATION_PRESERVED"
                final_holdout_state = "AVAILABLE" if cleanup_success else "SEALED"

                cleanup_failure_msgs: list[str] = []
                if not cleanup_success and release_res.get("denied"):
                    cleanup_failure_msgs.append(f"release_reservation denied: {release_res.get('denied')}")

                aborted_out = {
                    "cycle_id": "CYCLE-004",
                    "evaluation_id": evaluation_id,
                    "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
                    "reconciled_eligible_sessions": len(elig_1h),
                    "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
                    "excluded_dates": excl_1h,
                    "integrity_passed": False,
                    "qualification_allowed": False,
                    "exploratory_research_allowed": exploratory_allowed,
                    "exploratory_only": not qualification_allowed,
                    "holdout_consumed": False,
                    "holdout_state": final_holdout_state,
                    "evaluation_status": "ABORTED_TRANSITION_FAILED",
                    "evaluation_type": eval_type,
                    "evaluation_started_at": evaluation_started_at,
                    "evaluation_completed_at": datetime.now(tz=IST_TZ).isoformat(),
                    "consumed_holdout_dates": [],
                    "dataset_fingerprint": dataset_fp,
                    "strategy_ids_evaluated": [],
                    "failure_reasons": guard_res["failure_reasons"] + [
                        f"mark_evaluation_in_progress failed: {prog_res.get('reason')}"
                    ] + cleanup_failure_msgs,
                    "provenance_verified": bool(prov_rep_004.get("verified", False)),
                    "provenance_status": prov_rep_004.get("final_provenance_status", "FAILED"),
                    "qualification_blocked": True,
                    "seal_reservation": seal_res,
                    "transition_error": prog_res,
                    "cleanup_result": release_res,
                    "cleanup_status": cleanup_status,
                    "stale_recovery": stale_recovery,
                    "integrity_guard_result": guard_res,
                    "dynamic_test_metrics": live_test_metrics,
                    "qualification_board": [],
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(aborted_out, indent=2), encoding="utf-8")
                return aborted_out

        eval_start_date, eval_end_date = eval_dates[0], eval_dates[-1]
        db = HistoricalLabDatabase()
        leaderboard: list[dict[str, Any]] = []
        qualification_board: list[dict[str, Any]] = []
        orders_by_strat: dict[str, list[dict[str, Any]]] = {}
        slip_matrix: dict[str, list[dict[str, Any]]] = {}
        slip_all: list[dict[str, Any]] = []

        try:
            intraday_configs = [
                ("STRAT-018-60M-ORB-SELECTIVE", "STRAT-018-60M-ORB-SELECTIVE", 50_000.0),
                ("STRAT-017-60M-ORB@200K", "STRAT-017-60M-ORB-FILTERED", 200_000.0),
                ("STRAT-017-60M-ORB@500K", "STRAT-017-60M-ORB-FILTERED", 500_000.0),
                ("BENCHMARK-CASH-NO-TRADE", "BENCHMARK-CASH-NO-TRADE", 50_000.0),
            ]

            for disp_id, base_sid, cap in intraday_configs:
                eng = HistoricalLabReplayEngine(
                    risk_config=LabRiskConfig(initial_capital_inr=cap, max_open_positions=2, risk_per_trade_pct=0.015)
                )
                res = eng.run_date_range(
                    start_date=eval_start_date,
                    end_date=eval_end_date,
                    strategy_id=base_sid,
                    symbols=LAB_DEFAULT_UNIVERSE,
                    data_mode=DataSourceMode.HISTORICAL_REAL,
                    execution_mode="DETERMINISTIC",
                    db=db,
                    eligible_dates=eval_dates,
                    timeframe_minutes=60,  # type: ignore[arg-type]
                )
                res["strategy_id"] = disp_id
                if cap > 50_000.0:
                    res["strategy_name"] = f"{res['strategy_name']} (₹{int(cap/1000)}K Capital)"
                orders_by_strat[disp_id] = res.pop("orders", [])
                res.pop("trades", [])

                if res["total_trades"] > 0:
                    slip_rows = eng.run_slippage_sensitivity(
                        start_date=eval_start_date,
                        end_date=eval_end_date,
                        strategy_id=base_sid,
                        slippage_bps_list=(5.0, 10.0, 20.0, 30.0),
                        symbols=LAB_DEFAULT_UNIVERSE,
                        data_mode=DataSourceMode.HISTORICAL_REAL,
                        eligible_dates=eval_dates,
                        timeframe_minutes=60,  # type: ignore[arg-type]
                    )
                else:
                    slip_rows = [
                        {
                            "slippage_bps": float(b), "gross_return_pct": 0.0, "net_return_pct": 0.0,
                            "net_pnl_inr": 0.0, "total_fees_inr": 0.0, "total_brokerage_inr": 0.0,
                            "total_statutory_inr": 0.0, "total_slippage_inr": 0.0, "total_trades": 0, "win_rate_pct": 0.0,
                        }
                        for b in (5.0, 10.0, 20.0, 30.0)
                    ]
                slip_matrix[disp_id] = slip_rows
                slip_all.extend({"strategy_id": disp_id, **sr} for sr in slip_rows)
                cons_10 = next((sr for sr in slip_rows if float(sr["slippage_bps"]) == 10.0), slip_rows[0])

                dr = res["daily_runs"]
                unseen_slice = dr[v_end_1h:] or [dr[-1]]
                unseen_ret_pct = round((sum(d["net_pnl"] for d in unseen_slice) / cap) * 100.0, 2)
                wf = {
                    "train": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[:t_end_1h]) / cap) * 100.0, 2)},
                    "validation": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[t_end_1h:v_end_1h]) / cap) * 100.0, 2)},
                    "unseen_test": {"total_return_pct": unseen_ret_pct, "total_trades": sum(d["trades"] for d in unseen_slice)},
                }
                stat_v, _ = self.stat_agent.validate(res, wf)
                rt_ok, rt_reason = self.red_team.challenge(disp_id, res, wf)
                qual = evaluate_seven_gates(
                    strategy_id=disp_id, strategy_name=res["strategy_name"], multi_day_metrics=res,
                    unseen_metrics=wf["unseen_test"], conservative_slippage_metrics=cons_10,
                    data_provenance_verified=prov_ok_004, stat_validation_verdict=stat_v,
                    red_team_approved=rt_ok, red_team_reason=rt_reason,
                    min_sessions_required=60, min_trades_required=30,
                    independent_holdout_valid=guard_res["integrity_passed"],
                    provenance_status=prov_rep_004.get("final_provenance_status", "FAILED"),
                    holdout_status=guard_res.get("holdout_status", "FAILED"),
                    integrity_guard_result=guard_res,
                )
                leaderboard.append(res)
                qualification_board.append({
                    "strategy_id": disp_id, "strategy_name": res["strategy_name"], "timeframe_minutes": 60,
                    "initial_capital_inr": cap, "final_status": qual.final_status,
                    "passed_gates": f"{qual.passed_gates_count}/{qual.total_gates}",
                    "sessions": res["sessions_evaluated"], "trades": res["total_trades"],
                    "gross_return_pct": res["gross_return_pct"], "net_return_pct": res["total_return_pct"],
                    "conservative_10bps_net_pct": cons_10["net_return_pct"],
                    "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                    "max_drawdown_pct": res["max_drawdown_pct"], "sharpe_ratio": res["sharpe_ratio"],
                    "sortino_ratio": res["sortino_ratio"], "profit_factor": res["profit_factor"],
                    "expectancy_inr": res["expectancy_inr"], "total_fees_inr": res["total_fees_inr"],
                    "total_brokerage_inr": res.get("total_brokerage_inr", 0.0),
                    "total_statutory_inr": res.get("total_statutory_inr", 0.0),
                    "total_slippage_inr": res.get("total_slippage_inr", 0.0),
                    "unseen_return_pct": unseen_ret_pct, "statistical_verdict": stat_v,
                    "red_team_verdict": "APPROVED" if rt_ok else "REJECTED",
                    "exploratory_only": not qualification_allowed,
                    "summary": qual.summary, "gates": [asdict(g) for g in qual.gates],
                })

            cnc_configs = [
                ("STRAT-019-60M-CNC-ORB-SWING", "60m CNC Delivery ORB Swing (₹50K)", "ORB_SWING"),
                ("STRAT-020-60M-CNC-TREND-SWING", "60m CNC Delivery Trend Swing (₹50K)", "TREND_SWING"),
            ]
            for sid, sname, mode in cnc_configs:
                res = run_cnc_swing_backtest(eval_dates, sid, sname, mode=mode, initial_capital_inr=50_000.0, slippage_bps=5.0)
                orders_by_strat[sid] = res.pop("orders", [])
                slip_rows = []
                for bps in (5.0, 10.0, 20.0, 30.0):
                    sr_res = run_cnc_swing_backtest(eval_dates, sid, sname, mode=mode, initial_capital_inr=50_000.0, slippage_bps=bps)
                    slip_rows.append({
                        "slippage_bps": bps, "gross_return_pct": sr_res["gross_return_pct"],
                        "net_return_pct": sr_res["total_return_pct"], "net_pnl_inr": sr_res["net_pnl_inr"],
                        "total_fees_inr": sr_res["total_fees_inr"], "total_brokerage_inr": sr_res["total_brokerage_inr"],
                        "total_statutory_inr": sr_res["total_statutory_inr"], "total_slippage_inr": sr_res["total_slippage_inr"],
                        "total_trades": sr_res["total_trades"], "win_rate_pct": sr_res["win_rate_pct"],
                    })
                slip_matrix[sid] = slip_rows
                slip_all.extend({"strategy_id": sid, **sr} for sr in slip_rows)
                cons_10 = next((sr for sr in slip_rows if float(sr["slippage_bps"]) == 10.0), slip_rows[0])

                dr = res["daily_runs"]
                unseen_slice = dr[v_end_1h:] or [dr[-1]]
                unseen_ret_pct = round((sum(d["net_pnl"] for d in unseen_slice) / 50_000.0) * 100.0, 2)
                wf = {
                    "train": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[:t_end_1h]) / 50_000.0) * 100.0, 2)},
                    "validation": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[t_end_1h:v_end_1h]) / 50_000.0) * 100.0, 2)},
                    "unseen_test": {"total_return_pct": unseen_ret_pct, "total_trades": sum(d["trades"] for d in unseen_slice)},
                }
                stat_v, _ = self.stat_agent.validate(res, wf)
                rt_ok, rt_reason = self.red_team.challenge(sid, res, wf)
                qual = evaluate_seven_gates(
                    strategy_id=sid, strategy_name=sname, multi_day_metrics=res,
                    unseen_metrics=wf["unseen_test"], conservative_slippage_metrics=cons_10,
                    data_provenance_verified=prov_ok_004, stat_validation_verdict=stat_v,
                    red_team_approved=rt_ok, red_team_reason=rt_reason,
                    min_sessions_required=60, min_trades_required=30,
                    independent_holdout_valid=guard_res["integrity_passed"],
                    provenance_status=prov_rep_004.get("final_provenance_status", "FAILED"),
                    holdout_status=guard_res.get("holdout_status", "FAILED"),
                    integrity_guard_result=guard_res,
                )
                leaderboard.append(res)
                qualification_board.append({
                    "strategy_id": sid, "strategy_name": sname, "timeframe_minutes": 60,
                    "initial_capital_inr": 50_000.0, "final_status": qual.final_status,
                    "passed_gates": f"{qual.passed_gates_count}/{qual.total_gates}",
                    "sessions": res["sessions_evaluated"], "trades": res["total_trades"],
                    "gross_return_pct": res["gross_return_pct"], "net_return_pct": res["total_return_pct"],
                    "conservative_10bps_net_pct": cons_10["net_return_pct"],
                    "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                    "max_drawdown_pct": res["max_drawdown_pct"], "sharpe_ratio": res["sharpe_ratio"],
                    "sortino_ratio": res["sortino_ratio"], "profit_factor": res["profit_factor"],
                    "expectancy_inr": res["expectancy_inr"], "total_fees_inr": res["total_fees_inr"],
                    "total_brokerage_inr": res["total_brokerage_inr"], "total_statutory_inr": res["total_statutory_inr"],
                    "total_slippage_inr": res["total_slippage_inr"],
                    "unseen_return_pct": unseen_ret_pct, "statistical_verdict": stat_v,
                    "red_team_verdict": "APPROVED" if rt_ok else "REJECTED",
                    "exploratory_only": not qualification_allowed,
                    "summary": qual.summary, "gates": [asdict(g) for g in qual.gates],
                })
        except Exception as exc:
            if consumed_dates:
                holdout_guard.mark_evaluation_failed(
                    "CYCLE-004",
                    consumed_dates,
                    reason=str(exc),
                    data_accessed=True,
                )
            raise

        # Step 5: Transition SEALED -> EXPOSED for ANY evaluation (formal or exploratory) that consumed unseen dates
        evaluation_completed_at = datetime.now(tz=IST_TZ).isoformat()
        strategy_ids_evaluated = [q["strategy_id"] for q in qualification_board]
        recovery_failed = False
        recovery_failure_reasons: list[str] = []
        if consumed_dates:
            exposure_status = holdout_guard.complete_evaluation(
                "CYCLE-004",
                evaluation_id=evaluation_id,
                evaluation_type=eval_type,
                consumed_dates=consumed_dates,
                strategy_ids=strategy_ids_evaluated,
                dataset_fingerprint=dataset_fp,
                evaluation_ref=str(output_path),
            )
            completion_ok = bool(exposure_status.get("ok"))
            if completion_ok:
                holdout_consumed = True
                holdout_state = exposure_status.get("holdout_state", "EXPOSED")
                eval_status = "COMPLETED_FORMAL" if qualification_allowed else "COMPLETED_EXPLORATORY_WITH_UNSEEN"
            else:
                # Inspect actual persisted registry state; only invoke fallback recovery if owned accessed dates remain SEALED.
                try:
                    reg_after = holdout_guard.load_registry()
                    reg_records = reg_after.get("records", {})
                    unrecovered_accessed = [
                        d
                        for d in consumed_dates
                        if reg_records.get(d.isoformat(), {}).get("state") == "SEALED"
                        and reg_records.get(d.isoformat(), {}).get("owning_cycle_id") == "CYCLE-004"
                        and (
                            bool(reg_records.get(d.isoformat(), {}).get("data_accessed"))
                            or reg_records.get(d.isoformat(), {}).get("reservation_status") == "EVALUATION_IN_PROGRESS"
                        )
                    ]
                    if unrecovered_accessed:
                        holdout_guard.mark_evaluation_failed(
                            "CYCLE-004",
                            unrecovered_accessed,
                            reason="; ".join(exposure_status.get("failure_reasons", ["complete_evaluation failed"])),
                            data_accessed=True,
                        )
                        reg_after = holdout_guard.load_registry()
                        reg_records = reg_after.get("records", {})

                    still_sealed_accessed = [
                        d.isoformat()
                        for d in consumed_dates
                        if reg_records.get(d.isoformat(), {}).get("state") == "SEALED"
                        and reg_records.get(d.isoformat(), {}).get("owning_cycle_id") == "CYCLE-004"
                        and (
                            bool(reg_records.get(d.isoformat(), {}).get("data_accessed"))
                            or reg_records.get(d.isoformat(), {}).get("reservation_status") == "EVALUATION_IN_PROGRESS"
                        )
                    ]
                    any_owned_exposed = any(
                        reg_records.get(d.isoformat(), {}).get("state") == "EXPOSED"
                        and reg_records.get(d.isoformat(), {}).get("owning_cycle_id") == "CYCLE-004"
                        for d in consumed_dates
                    )
                    if still_sealed_accessed:
                        recovery_failed = True
                        holdout_consumed = any_owned_exposed
                        holdout_state = "SEALED"
                        eval_status = "COMPLETION_FAILED_RECOVERY_FAILED"
                        recovery_failure_reasons.append(
                            f"Recovery failed: accessed dates remained SEALED: {still_sealed_accessed}"
                        )
                    elif any_owned_exposed or bool(exposure_status.get("holdout_consumed")):
                        holdout_consumed = True
                        holdout_state = "EXPOSED"
                        eval_status = "COMPLETION_FAILED"
                    else:
                        holdout_consumed = False
                        holdout_state = exposure_status.get("holdout_state", "SEALED")
                        eval_status = "COMPLETION_FAILED"
                except Exception as rec_exc:
                    recovery_failed = True
                    holdout_consumed = bool(exposure_status.get("holdout_consumed", False))
                    holdout_state = "RECOVERY_FAILED"
                    eval_status = "COMPLETION_FAILED_RECOVERY_FAILED"
                    recovery_failure_reasons.append(f"Recovery failed: {rec_exc}")
        else:
            exposure_status = {
                "status": guard_res["holdout_status"],
                "independent_holdout_valid": False,
                "exposed": False,
                "holdout_consumed": False,
                "reason": "Exploratory evaluation ran on train+validation only; SEALED holdout preserved.",
            }
            holdout_consumed = False
            holdout_state = "SEALED" if seal_res.get("sealed_count", 0) > 0 else guard_res["holdout_status"]
            eval_status = "COMPLETED_EXPLORATORY_TRAIN_VAL_ONLY"

        exec_rep = self.exec_agent.audit_costs(leaderboard, slip_all, orders_by_strategy=orders_by_strat)
        sq_rep = self.sq_agent.review_system(test_report=live_test_metrics)

        sep_004 = separate_strategies_and_benchmarks(qualification_board, period_label=f"{start_1h.isoformat()} -> {end_1h.isoformat()}")
        qualified_cnt = sep_004["qualified_active_strategies"]
        rejected_cnt = sum(1 for q in sep_004["active_strategy_leaderboard"] if q["final_status"] in ("REJECTED", "INVALIDATED", "BLOCKED_INTEGRITY"))
        best_strat = sep_004["best_active_strategy"]

        cycle_failure_reasons = list(guard_res.get("failure_reasons", []))
        if consumed_dates and not exposure_status.get("ok"):
            cycle_failure_reasons.extend(exposure_status.get("failure_reasons", ["complete_evaluation failed"]))
        cycle_failure_reasons.extend(recovery_failure_reasons)

        cycle_004_out = {
            "cycle_id": "CYCLE-004",
            "evaluation_id": evaluation_id,
            "generated_at_ist": evaluation_completed_at,
            "reconciled_eligible_sessions": len(elig_1h),
            "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
            "excluded_dates": excl_1h,
            "integrity_passed": bool(guard_res.get("integrity_passed", False)),
            "qualification_allowed": qualification_allowed,
            "exploratory_research_allowed": exploratory_allowed,
            "exploratory_only": not qualification_allowed,
            "holdout_consumed": holdout_consumed,
            "holdout_state": holdout_state,
            "recovery_failed": recovery_failed,
            "evaluation_status": eval_status,
            "evaluation_type": eval_type,
            "evaluation_started_at": evaluation_started_at,
            "evaluation_completed_at": evaluation_completed_at,
            "consumed_holdout_dates": [d.isoformat() for d in consumed_dates],
            "dataset_fingerprint": dataset_fp,
            "strategy_ids_evaluated": strategy_ids_evaluated,
            "failure_reasons": cycle_failure_reasons,
            "provenance_verified": bool(prov_rep_004.get("verified", False)),
            "provenance_status": prov_rep_004.get("final_provenance_status", "FAILED"),
            "seal_reservation": seal_res,
            "stale_recovery": stale_recovery,
            "integrity_guard_result": guard_res,
            "sealed_holdout_status": exposure_status,
            "dynamic_test_metrics": live_test_metrics,
            "command_centre": {
                "current_cycle": "CYCLE-004",
                "active_agents": 9,
                "experiments_completed": len(qualification_board),
                "experiments_rejected": rejected_cnt,
                "qualified_strategies": qualified_cnt,
                "best_experimental_strategy": best_strat["strategy_id"] if best_strat else None,
                "best_experimental_net_pct": best_strat["net_return_pct"] if best_strat else 0.0,
                "max_candidates_per_cycle": self.MAX_CANDIDATES_PER_CYCLE,
                "max_tuning_rounds": self.MAX_TUNING_ROUNDS_PER_HYPOTHESIS,
            },
            "qualification_board": sorted(qualification_board, key=lambda x: x["net_return_pct"], reverse=True),
            "reference_benchmark_leaderboard": sep_004["reference_benchmark_leaderboard"],
            "slippage_sensitivity_matrix": slip_matrix,
            "agent_performance": [asdict(exec_rep), asdict(sq_rep)],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cycle_004_out, indent=2), encoding="utf-8")

        if output_path == CYCLE_004_REPORT_PATH and LAB_INDEX_PATH.exists():
            progression = self.build_cycle_progression()
            idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
            idx["cycle_004_report"] = cycle_004_out
            idx["phase11_research_cycle"] = cycle_004_out
            idx["cycle_progression"] = progression
            LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

        return cycle_004_out

    def reassess_cycle_003(self) -> dict[str, Any]:
        """Phase 11.2 Mandatory Repair E: Reassess CYCLE-003 under strict provenance, holdout & benchmark rules."""
        if not CYCLE_003_REPORT_PATH.exists():
            raise FileNotFoundError(f"Missing original CYCLE-003 report: {CYCLE_003_REPORT_PATH}")

        orig = json.loads(CYCLE_003_REPORT_PATH.read_text(encoding="utf-8"))
        prov_ok, prov_1h_report = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
        PROVENANCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE_REPORT_PATH.write_text(json.dumps(prov_1h_report, indent=2), encoding="utf-8")

        # Inspect historical exposures in sealed_holdout_registry.json
        guard = SealedHoldoutGuard()
        reg = guard.load_registry()
        guard.save_registry(reg)  # Ensures legacy exposures are migrated cleanly into per-date records

        cycle2_dates: set[str] = set()
        cycle3_unseen_dates: list[str] = []
        for exp in reg.get("sealed_exposures", []):
            if exp.get("cycle_id") == "CYCLE-002":
                cycle2_dates.update(exp.get("unseen_dates", []))
            elif exp.get("cycle_id") == "CYCLE-003-60M" and not cycle3_unseen_dates:
                cycle3_unseen_dates = list(exp.get("unseen_dates", []))

        reused_in_c3 = sorted(d for d in cycle3_unseen_dates if d in cycle2_dates)
        untouched_in_c3 = sorted(d for d in cycle3_unseen_dates if d not in cycle2_dates)
        holdout_valid = len(reused_in_c3) == 0 and len(untouched_in_c3) > 0

        # Separate benchmarks from active strategies
        sep = separate_strategies_and_benchmarks(
            orig.get("qualification_board", []),
            period_label=orig.get("date_range", ""),
        )

        recalculated_strategies: list[dict[str, Any]] = []
        downgrade_explanations: list[dict[str, Any]] = []

        for row in sep["active_strategy_leaderboard"]:
            sid = row["strategy_id"]
            orig_status = row.get("final_status", "REJECTED")
            sessions = int(row.get("sessions", 0))
            trades = int(row.get("trades", 0))
            net_ret = float(row.get("net_return_pct", 0.0))
            unseen_ret = float(row.get("unseen_return_pct", 0.0))

            if not prov_ok:
                new_status = "INVALIDATED"
                reason = "1-hour dataset provenance verification failed."
            elif int(row.get("timeframe_minutes", 5)) == 60 and not holdout_valid:
                new_status = "INVALIDATED"
                reason = (
                    f"EXPOSED_HOLDOUT_REUSED: {len(reused_in_c3)}/48 unseen dates were previously exposed in CYCLE-002; "
                    f"independent out-of-sample claim ({unseen_ret:+.2f}%) is invalidated."
                )
            elif sessions < 60 or trades < 30:
                new_status = "INSUFFICIENT_EVIDENCE" if net_ret >= 0 else "REJECTED"
                reason = f"Sample size ({sessions}/60 sessions, {trades}/30 trades) below statistical threshold."
            else:
                new_status = "REJECTED"
                reason = f"Negative net return ({net_ret:+.2f}%) after realistic transaction costs."

            recalculated_strategies.append({
                **row,
                "original_status": orig_status,
                "reassessed_status": new_status,
                "independent_holdout_valid": holdout_valid if int(row.get("timeframe_minutes", 5)) == 60 else False,
                "reassessment_reason": reason,
            })
            downgrade_explanations.append({
                "strategy_id": sid,
                "original_status": orig_status,
                "reassessed_status": new_status,
                "reason": reason,
            })

        archive_health = {}
        if ARCHIVE_HEALTH_PATH.exists():
            archive_health = json.loads(ARCHIVE_HEALTH_PATH.read_text(encoding="utf-8"))

        reassessment = {
            "reassessment_id": "CYCLE-003-INTEGRITY-REASSESSMENT",
            "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
            "original_cycle_id": orig.get("cycle_id", "CYCLE-003"),
            "original_generated_at_ist": orig.get("generated_at_ist"),
            "original_best_strategy_reported": orig.get("command_centre", {}).get("best_experimental_strategy"),
            "corrected_best_active_strategy": (
                sep["best_active_strategy"]["strategy_id"] if sep["best_active_strategy"] else None
            ),
            "corrected_best_active_net_pct": (
                sep["best_active_strategy"]["net_return_pct"] if sep["best_active_strategy"] else 0.0
            ),
            "qualified_active_strategies": sep["qualified_active_strategies"],
            "provenance_verification": {
                "verified": prov_ok,
                "final_provenance_status": prov_1h_report["final_provenance_status"],
                "validator_version": prov_1h_report["validator_version"],
                "total_missing_intervals": prov_1h_report["total_missing_intervals"],
                "total_duplicates": prov_1h_report["total_duplicates"],
                "total_invalid_ohlc": prov_1h_report["total_invalid_ohlc"],
            },
            "holdout_integrity": {
                "independent_holdout_valid": holdout_valid,
                "holdout_status": "EXPOSED_HOLDOUT_REUSED" if reused_in_c3 else "VALID_UNSEEN",
                "total_unseen_dates_in_cycle_003": len(cycle3_unseen_dates),
                "previously_exposed_dates_count": len(reused_in_c3),
                "previously_exposed_dates": reused_in_c3,
                "total_exposed_dates_in_registry": len(reg.get("records", {})),
            },
            "data_quality_warnings": [
                f"EXPOSED_HOLDOUT_REUSED: {len(reused_in_c3)} of {len(cycle3_unseen_dates)} unseen dates in CYCLE-003 overlapped with CYCLE-002.",
                f"1h archive status is {prov_1h_report['final_provenance_status']} ({prov_1h_report['total_missing_intervals']} missing intervals across 5 stocks).",
                "BENCHMARK-CASH-NO-TRADE removed from active strategy rankings (capital preservation baseline only).",
            ],
            "active_strategy_leaderboard": recalculated_strategies,
            "reference_benchmark_leaderboard": sep["reference_benchmark_leaderboard"],
            "downgrade_explanations": downgrade_explanations,
            "archive_health": archive_health,
        }

        CYCLE_003_REASSESSMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CYCLE_003_REASSESSMENT_PATH.write_text(json.dumps(reassessment, indent=2), encoding="utf-8")

        if LAB_INDEX_PATH.exists():
            idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
            idx["cycle_003_integrity_reassessment"] = reassessment
            idx["provenance_report_1h"] = prov_1h_report
            idx["sealed_holdout_summary"] = reassessment["holdout_integrity"]
            LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

        return reassessment

    def reassess_cycle_003_phase11_3(self) -> dict[str, Any]:
        """Phase 11.3: Correct Gate 1 & qualification statuses for CYCLE-003 using ResearchIntegrityGuard."""
        if not CYCLE_003_REPORT_PATH.exists():
            raise FileNotFoundError(f"Missing original CYCLE-003 report: {CYCLE_003_REPORT_PATH}")

        orig = json.loads(CYCLE_003_REPORT_PATH.read_text(encoding="utf-8"))
        prov_ok_strict, prov_1h_report = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
        PROVENANCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE_REPORT_PATH.write_text(json.dumps(prov_1h_report, indent=2), encoding="utf-8")

        guard = SealedHoldoutGuard()
        reg = guard.load_registry()
        guard.save_registry(reg)

        cycle2_dates: set[str] = set()
        cycle3_unseen_dates: list[str] = []
        cycle3_commit_sha = "unknown"
        for exp in reg.get("sealed_exposures", []):
            if exp.get("cycle_id") == "CYCLE-002":
                cycle2_dates.update(exp.get("unseen_dates", []))
            elif exp.get("cycle_id") == "CYCLE-003-60M" and not cycle3_unseen_dates:
                cycle3_unseen_dates = list(exp.get("unseen_dates", []))
                cycle3_commit_sha = str(exp.get("commit_sha", "unknown"))

        reused_in_c3 = sorted(d for d in cycle3_unseen_dates if d in cycle2_dates)
        holdout_valid = len(reused_in_c3) == 0 and len(cycle3_unseen_dates) > 0
        holdout_status_str = "EXPOSED_HOLDOUT_REUSED" if reused_in_c3 else "VALID_UNSEEN"
        prov_status_str = str(prov_1h_report.get("final_provenance_status", "FAILED"))

        sep = separate_strategies_and_benchmarks(
            orig.get("qualification_board", []),
            period_label=orig.get("date_range", ""),
        )

        recalculated_strategies: list[dict[str, Any]] = []
        downgrade_explanations: list[dict[str, Any]] = []

        for row in sep["active_strategy_leaderboard"]:
            sid = row["strategy_id"]
            orig_status = row.get("final_status", "REJECTED")
            is_60m = int(row.get("timeframe_minutes", 5)) == 60
            orig_gates = [dict(g) for g in row.get("gates", [])]

            if not prov_ok_strict and is_60m:
                g1_passed = False
                g1_reason = f"BLOCKED: 1H provenance is {prov_status_str} (requires VERIFIED); holdout={holdout_status_str}."
                new_status = "BLOCKED_INTEGRITY"
            elif is_60m and not holdout_valid:
                g1_passed = False
                g1_reason = f"INVALIDATED: {holdout_status_str} ({len(reused_in_c3)}/48 unseen dates reused from CYCLE-002)."
                new_status = "INVALIDATED"
            elif not is_60m:
                g1_passed = False
                g1_reason = "BLOCKED: 5m->30m archive has partial sessions and < 60 sessions."
                new_status = "REJECTED" if float(row.get("net_return_pct", 0.0)) < 0 else "INSUFFICIENT_EVIDENCE"
            else:
                g1_passed = True
                g1_reason = "Verified provenance and untouched holdout."
                new_status = orig_status

            # Update Gate 1 in the strategy's gates list so Gate 1 never falsely shows PASS
            for g in orig_gates:
                if int(g.get("gate_number", 0)) == 1:
                    g["passed"] = g1_passed
                    g["status_label"] = "PASS" if g1_passed else "FAIL"
                    g["reason"] = g1_reason

            passed_cnt = sum(1 for g in orig_gates if g.get("passed"))
            recalculated_strategies.append({
                **row,
                "original_status": orig_status,
                "reassessed_status": new_status,
                "passed_gates": f"{passed_cnt}/{len(orig_gates) or 7}",
                "gate_1_passed": g1_passed,
                "gate_1_reason": g1_reason,
                "provenance_status": prov_status_str if is_60m else "PARTIALLY_VERIFIED",
                "holdout_status": holdout_status_str if is_60m else "INSUFFICIENT_UNSEEN_DATA",
                "gates": orig_gates,
            })
            downgrade_explanations.append({
                "strategy_id": sid,
                "original_status": orig_status,
                "reassessed_status": new_status,
                "gate_1_passed": g1_passed,
                "reason": g1_reason,
            })

        out = {
            "reassessment_id": "CYCLE-003-PHASE-11-3-REASSESSMENT",
            "reassessment_timestamp": datetime.now(tz=IST_TZ).isoformat(),
            "original_cycle_reference": orig.get("cycle_id", "CYCLE-003"),
            "original_commit_sha": cycle3_commit_sha,
            "original_generated_at_ist": orig.get("generated_at_ist"),
            "provenance_status": prov_status_str,
            "provenance_verified": prov_ok_strict,
            "holdout_status": holdout_status_str,
            "independent_holdout_valid": holdout_valid,
            "previously_exposed_dates_count": len(reused_in_c3),
            "previously_exposed_dates": reused_in_c3,
            "qualification_allowed": False,
            "active_strategy_leaderboard": recalculated_strategies,
            "reference_benchmark_leaderboard": sep["reference_benchmark_leaderboard"],
            "downgrade_explanations": downgrade_explanations,
        }

        CYCLE_003_PHASE11_3_REASSESSMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CYCLE_003_PHASE11_3_REASSESSMENT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

        if LAB_INDEX_PATH.exists():
            idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
            idx["cycle_003_phase11_3_reassessment"] = out
            idx["provenance_report_1h"] = prov_1h_report
            LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

        return out

    def reassess_cycle_003_phase11_3_1(
        self,
        registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH,
    ) -> dict[str, Any]:
        """Phase 11.3.1: Reassess CYCLE-003 exclusively via the central ResearchIntegrityGuard."""
        from ..data.loader_1h import get_available_1h_dates

        if not CYCLE_003_REPORT_PATH.exists():
            raise FileNotFoundError(f"Missing original CYCLE-003 report: {CYCLE_003_REPORT_PATH}")

        orig = json.loads(CYCLE_003_REPORT_PATH.read_text(encoding="utf-8"))
        _, prov_1h_report = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
        PROVENANCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE_REPORT_PATH.write_text(json.dumps(prov_1h_report, indent=2), encoding="utf-8")

        elig_1h, _ = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
        n_1h = len(elig_1h)
        t_end_1h = max(1, int(n_1h * 0.60))
        v_end_1h = max(t_end_1h + 1, int(n_1h * 0.80))
        train_1h = elig_1h[:t_end_1h]
        val_1h = elig_1h[t_end_1h:v_end_1h]
        unseen_1h = elig_1h[v_end_1h:] or [elig_1h[-1]]

        # Call central ResearchIntegrityGuard (never manually override its output)
        guard_res = ResearchIntegrityGuard(registry_path=registry_path).validate(
            cycle_id="CYCLE-003-60M",
            training_dates=train_1h,
            validation_dates=val_1h,
            unseen_dates=unseen_1h,
            dataset_fingerprint=str(prov_1h_report.get("dataset_fingerprint", "")),
            provenance_result=prov_1h_report,
        )

        # Count exact CYCLE-002 / CYCLE-003 overlap from holdout registry
        holdout_guard = SealedHoldoutGuard(registry_path=registry_path)
        reg = holdout_guard.load_registry()
        cycle2_dates: set[str] = set()
        for exp in reg.get("sealed_exposures", []):
            if exp.get("cycle_id") == "CYCLE-002":
                cycle2_dates.update(exp.get("unseen_dates", []))
        reused_dates = sorted(d.isoformat() for d in unseen_1h if d.isoformat() in cycle2_dates)

        sep = separate_strategies_and_benchmarks(
            orig.get("qualification_board", []),
            period_label=orig.get("date_range", ""),
        )

        recalculated_strategies: list[dict[str, Any]] = []
        downgrade_explanations: list[dict[str, Any]] = []

        for row in sep["active_strategy_leaderboard"]:
            sid = row["strategy_id"]
            orig_status = row.get("final_status", "REJECTED")
            orig_gates = [dict(g) for g in row.get("gates", [])]
            g1_passed = bool(guard_res["integrity_passed"] and guard_res["qualification_allowed"])
            g1_reason = (
                "Verified provenance and untouched owned holdout."
                if g1_passed
                else "; ".join(guard_res["failure_reasons"]) or f"Blocked by guard ({guard_res['holdout_status']})"
            )
            new_status = (
                orig_status
                if g1_passed
                else ("INVALIDATED" if guard_res["provenance_verified"] else "BLOCKED_INTEGRITY")
            )

            for g in orig_gates:
                if int(g.get("gate_number", 0)) == 1:
                    g["passed"] = g1_passed
                    g["status_label"] = "PASS" if g1_passed else "FAIL"
                    g["reason"] = g1_reason

            passed_cnt = sum(1 for g in orig_gates if g.get("passed"))
            recalculated_strategies.append({
                **row,
                "original_status": orig_status,
                "reassessed_status": new_status,
                "passed_gates": f"{passed_cnt}/{len(orig_gates) or 7}",
                "gate_1_passed": g1_passed,
                "gate_1_reason": g1_reason,
                "exploratory_only": not guard_res["qualification_allowed"],
                "gates": orig_gates,
            })
            downgrade_explanations.append({
                "strategy_id": sid,
                "original_status": orig_status,
                "reassessed_status": new_status,
                "gate_1_passed": g1_passed,
                "reason": g1_reason,
            })

        out = {
            "reassessment_id": "CYCLE-003-PHASE-11-3-1-REASSESSMENT",
            "original_reassessment_reference": "CYCLE-003-PHASE-11-3-REASSESSMENT",
            "original_cycle_reference": orig.get("cycle_id", "CYCLE-003"),
            "guard_version": guard_res["guard_version"],
            "validation_timestamp": guard_res["validation_timestamp"],
            "provenance_status": guard_res["provenance_status"],
            "provenance_verified": guard_res["provenance_verified"],
            "holdout_status": guard_res["holdout_status"],
            "holdout_metadata": guard_res["holdout_metadata"],
            "previously_exposed_dates_count": len(reused_dates),
            "previously_exposed_dates": reused_dates,
            "integrity_passed": guard_res["integrity_passed"],
            "qualification_allowed": guard_res["qualification_allowed"],
            "exploratory_research_allowed": guard_res["exploratory_research_allowed"],
            "failure_reasons": guard_res["failure_reasons"],
            "integrity_guard_result": guard_res,
            "active_strategy_leaderboard": recalculated_strategies,
            "reference_benchmark_leaderboard": sep["reference_benchmark_leaderboard"],
            "downgrade_explanations": downgrade_explanations,
        }

        CYCLE_003_PHASE11_3_1_REASSESSMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CYCLE_003_PHASE11_3_1_REASSESSMENT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

        if LAB_INDEX_PATH.exists():
            idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
            idx["cycle_003_phase11_3_1_reassessment"] = out
            idx["integrity_guard_status"] = guard_res
            idx["provenance_report_1h"] = prov_1h_report
            LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

        return out

    def run_cycle_005(
        self,
        test_report: dict[str, Any] | None = None,
        run_live_pytest: bool = True,
        allow_exploratory_on_blocked: bool = True,
        include_unseen_in_exploratory: bool = False,
        registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH,
        output_path: Path = CYCLE_005_REPORT_PATH,
    ) -> dict[str, Any]:
        """Execute CYCLE-005 with strict Phase 11.3 holdout lifecycle, provenance, and recovery integrity."""
        import uuid
        from ..config import LabRiskConfig
        from ..data.loader_1h import get_available_1h_dates
        from .cycle004 import run_cnc_swing_backtest

        if test_report is None:
            live_test_metrics = collect_dynamic_test_metrics(run_pytest=run_live_pytest)
        else:
            live_test_metrics = {
                **test_report,
                "executed": bool(test_report.get("executed", False)),
                "status": "NOT_RUN" if not test_report.get("executed", False) else "SUPPLIED_UNVERIFIED",
                "independently_verified": False,
                "source": "CALLER_SUPPLIED",
            }
        evaluation_id = f"eval_c5_{uuid.uuid4().hex[:8]}"
        evaluation_started_at = datetime.now(tz=IST_TZ).isoformat()

        holdout_guard = SealedHoldoutGuard(registry_path=registry_path)
        stale_recovery = holdout_guard.recover_stale_reservations("CYCLE-005")

        # Step 1: Identify eligible dates & splits
        elig_1h, excl_1h = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
        start_1h, end_1h = elig_1h[0], elig_1h[-1]
        n_1h = len(elig_1h)
        t_end_1h = max(1, int(n_1h * 0.60))
        v_end_1h = max(t_end_1h + 1, int(n_1h * 0.80))
        train_1h_dates = elig_1h[:t_end_1h]
        val_1h_dates = elig_1h[t_end_1h:v_end_1h]
        unseen_1h_dates = elig_1h[v_end_1h:] or [elig_1h[-1]]

        # Step 2: Validate 1h provenance & attempt reservation for unseen dates owned by CYCLE-005
        prov_ok_005, prov_rep_005 = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
        dataset_fp = str(prov_rep_005.get("dataset_fingerprint", ""))

        seal_res = holdout_guard.seal_dates("CYCLE-005", unseen_1h_dates, dataset_fingerprint=dataset_fp)

        # Step 3: Central integrity guard validation
        integrity_guard = ResearchIntegrityGuard(registry_path=registry_path)
        guard_res = integrity_guard.validate(
            cycle_id="CYCLE-005",
            training_dates=train_1h_dates,
            validation_dates=val_1h_dates,
            unseen_dates=unseen_1h_dates,
            dataset_fingerprint=dataset_fp,
            provenance_result=prov_rep_005,
        )

        qualification_allowed = bool(guard_res["integrity_passed"] and guard_res["qualification_allowed"])
        exploratory_allowed = bool(guard_res.get("exploratory_research_allowed", False))
        global_holdout_integrity_status = str(guard_res.get("holdout_status", "UNKNOWN"))
        independent_holdout_available = bool(
            global_holdout_integrity_status == "VALID_UNSEEN"
            and not guard_res.get("holdout_metadata", {}).get("unseen_reused_dates")
        )

        if qualification_allowed:
            eval_type = "FORMAL_QUALIFICATION"
            eval_dates = elig_1h
            consumed_dates = unseen_1h_dates
        elif exploratory_allowed and allow_exploratory_on_blocked:
            if include_unseen_in_exploratory:
                eval_type = "EXPLORATORY_WITH_UNSEEN"
                eval_dates = elig_1h
                consumed_dates = unseen_1h_dates
            else:
                eval_type = "EXPLORATORY_TRAIN_VAL_ONLY"
                eval_dates = train_1h_dates + val_1h_dates
                consumed_dates = []
                # Release any freshly sealed unseen reservation so AVAILABLE dates remain untouched
                if seal_res.get("sealed_count", 0) > 0:
                    holdout_guard.release_reservation(
                        "CYCLE-005",
                        unseen_1h_dates,
                        "Exploratory train+val only run; preserving unseen holdout dates",
                    )
        else:
            release_res = (
                holdout_guard.release_reservation("CYCLE-005", unseen_1h_dates, "Evaluation aborted before data access")
                if seal_res.get("sealed_count", 0) > 0
                else {"released": [], "denied": [], "ok": True}
            )
            cur_state = "AVAILABLE" if release_res.get("released") else "UNCONSUMED"
            blocked_out = {
                "cycle_id": "CYCLE-005",
                "evaluation_id": evaluation_id,
                "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
                "reconciled_eligible_sessions": len(elig_1h),
                "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
                "excluded_dates": excl_1h,
                "integrity_passed": False,
                "qualification_allowed": False,
                "exploratory_research_allowed": exploratory_allowed,
                "exploratory_only": True,
                "current_evaluation_holdout_consumed": False,
                "current_evaluation_holdout_state": cur_state,
                "global_holdout_integrity_status": global_holdout_integrity_status,
                "independent_holdout_available": independent_holdout_available,
                "holdout_consumed": False,
                "holdout_state": cur_state,
                "evaluation_status": "ABORTED_NO_EVALUATION",
                "evaluation_type": "NONE",
                "evaluation_started_at": evaluation_started_at,
                "evaluation_completed_at": datetime.now(tz=IST_TZ).isoformat(),
                "consumed_holdout_dates": [],
                "dataset_fingerprint": dataset_fp,
                "strategy_ids_evaluated": [],
                "failure_reasons": guard_res["failure_reasons"],
                "provenance_verified": bool(prov_rep_005.get("verified", False)),
                "provenance_status": prov_rep_005.get("final_provenance_status", "FAILED"),
                "qualification_blocked": True,
                "seal_reservation": seal_res,
                "release_result": release_res,
                "stale_recovery": stale_recovery,
                "integrity_guard_result": guard_res,
                "dynamic_test_metrics": live_test_metrics,
                "qualification_board": [],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(blocked_out, indent=2), encoding="utf-8")
            return blocked_out

        # Step 4: Enforce mark_evaluation_in_progress() before reading consumed holdout dates
        if consumed_dates:
            prog_res = holdout_guard.mark_evaluation_in_progress(
                "CYCLE-005",
                consumed_dates,
                dataset_fingerprint=dataset_fp,
                evaluation_type=eval_type,
            )
            if not prog_res.get("ok"):
                release_res = holdout_guard.release_reservation(
                    "CYCLE-005",
                    consumed_dates,
                    reason=f"Evaluation aborted before data access: mark_evaluation_in_progress failed ({prog_res.get('reason')})",
                )
                cleanup_success = bool(release_res.get("ok") and len(release_res.get("released", [])) == len(consumed_dates))
                cleanup_status = "RESERVATION_RELEASED" if cleanup_success else "RELEASE_FAILED_RESERVATION_PRESERVED"
                final_holdout_state = "AVAILABLE" if cleanup_success else "SEALED"
                aborted_out = {
                    "cycle_id": "CYCLE-005",
                    "evaluation_id": evaluation_id,
                    "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
                    "reconciled_eligible_sessions": len(elig_1h),
                    "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
                    "excluded_dates": excl_1h,
                    "integrity_passed": False,
                    "qualification_allowed": False,
                    "exploratory_research_allowed": exploratory_allowed,
                    "exploratory_only": not qualification_allowed,
                    "current_evaluation_holdout_consumed": False,
                    "current_evaluation_holdout_state": final_holdout_state,
                    "global_holdout_integrity_status": global_holdout_integrity_status,
                    "independent_holdout_available": independent_holdout_available,
                    "holdout_consumed": False,
                    "holdout_state": final_holdout_state,
                    "evaluation_status": "ABORTED_TRANSITION_FAILED",
                    "evaluation_type": eval_type,
                    "evaluation_started_at": evaluation_started_at,
                    "evaluation_completed_at": datetime.now(tz=IST_TZ).isoformat(),
                    "consumed_holdout_dates": [],
                    "dataset_fingerprint": dataset_fp,
                    "strategy_ids_evaluated": [],
                    "failure_reasons": guard_res["failure_reasons"] + [
                        f"mark_evaluation_in_progress failed: {prog_res.get('reason')}"
                    ],
                    "provenance_verified": bool(prov_rep_005.get("verified", False)),
                    "provenance_status": prov_rep_005.get("final_provenance_status", "FAILED"),
                    "qualification_blocked": True,
                    "seal_reservation": seal_res,
                    "transition_error": prog_res,
                    "cleanup_result": release_res,
                    "cleanup_status": cleanup_status,
                    "stale_recovery": stale_recovery,
                    "integrity_guard_result": guard_res,
                    "dynamic_test_metrics": live_test_metrics,
                    "qualification_board": [],
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(aborted_out, indent=2), encoding="utf-8")
                return aborted_out

        eval_start_date, eval_end_date = eval_dates[0], eval_dates[-1]
        db = HistoricalLabDatabase()
        leaderboard: list[dict[str, Any]] = []
        qualification_board: list[dict[str, Any]] = []
        orders_by_strat: dict[str, list[dict[str, Any]]] = {}
        slip_matrix: dict[str, list[dict[str, Any]]] = {}
        slip_all: list[dict[str, Any]] = []

        try:
            intraday_configs = [
                ("STRAT-018-60M-ORB-SELECTIVE", "STRAT-018-60M-ORB-SELECTIVE", 50_000.0),
                ("STRAT-017-60M-ORB@200K", "STRAT-017-60M-ORB-FILTERED", 200_000.0),
                ("STRAT-017-60M-ORB@500K", "STRAT-017-60M-ORB-FILTERED", 500_000.0),
                ("BENCHMARK-CASH-NO-TRADE", "BENCHMARK-CASH-NO-TRADE", 50_000.0),
            ]

            for disp_id, base_sid, cap in intraday_configs:
                eng = HistoricalLabReplayEngine(
                    risk_config=LabRiskConfig(initial_capital_inr=cap, max_open_positions=2, risk_per_trade_pct=0.015)
                )
                res = eng.run_date_range(
                    start_date=eval_start_date,
                    end_date=eval_end_date,
                    strategy_id=base_sid,
                    symbols=LAB_DEFAULT_UNIVERSE,
                    data_mode=DataSourceMode.HISTORICAL_REAL,
                    execution_mode="DETERMINISTIC",
                    db=db,
                    eligible_dates=eval_dates,
                    timeframe_minutes=60,  # type: ignore[arg-type]
                )
                res["strategy_id"] = disp_id
                if cap > 50_000.0:
                    res["strategy_name"] = f"{res['strategy_name']} (₹{int(cap/1000)}K Capital)"
                orders_by_strat[disp_id] = res.pop("orders", [])
                res.pop("trades", [])

                if res["total_trades"] > 0:
                    slip_rows = eng.run_slippage_sensitivity(
                        start_date=eval_start_date,
                        end_date=eval_end_date,
                        strategy_id=base_sid,
                        slippage_bps_list=(5.0, 10.0, 20.0, 30.0),
                        symbols=LAB_DEFAULT_UNIVERSE,
                        data_mode=DataSourceMode.HISTORICAL_REAL,
                        eligible_dates=eval_dates,
                        timeframe_minutes=60,  # type: ignore[arg-type]
                    )
                else:
                    slip_rows = [
                        {
                            "slippage_bps": float(b), "gross_return_pct": 0.0, "net_return_pct": 0.0,
                            "net_pnl_inr": 0.0, "total_fees_inr": 0.0, "total_brokerage_inr": 0.0,
                            "total_statutory_inr": 0.0, "total_slippage_inr": 0.0, "total_trades": 0, "win_rate_pct": 0.0,
                        }
                        for b in (5.0, 10.0, 20.0, 30.0)
                    ]
                slip_matrix[disp_id] = slip_rows
                slip_all.extend({"strategy_id": disp_id, **sr} for sr in slip_rows)
                cons_10 = next((sr for sr in slip_rows if float(sr["slippage_bps"]) == 10.0), slip_rows[0])

                dr = res["daily_runs"]
                unseen_slice = dr[v_end_1h:] or [dr[-1]]
                unseen_ret_pct = round((sum(d["net_pnl"] for d in unseen_slice) / cap) * 100.0, 2)
                wf = {
                    "train": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[:t_end_1h]) / cap) * 100.0, 2)},
                    "validation": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[t_end_1h:v_end_1h]) / cap) * 100.0, 2)},
                    "unseen_test": {"total_return_pct": unseen_ret_pct, "total_trades": sum(d["trades"] for d in unseen_slice)},
                }
                stat_v, _ = self.stat_agent.validate(res, wf)
                rt_ok, rt_reason = self.red_team.challenge(disp_id, res, wf)
                qual = evaluate_seven_gates(
                    strategy_id=disp_id, strategy_name=res["strategy_name"], multi_day_metrics=res,
                    unseen_metrics=wf["unseen_test"], conservative_slippage_metrics=cons_10,
                    data_provenance_verified=prov_ok_005, stat_validation_verdict=stat_v,
                    red_team_approved=rt_ok, red_team_reason=rt_reason,
                    min_sessions_required=60, min_trades_required=30,
                    independent_holdout_valid=guard_res["integrity_passed"],
                    provenance_status=prov_rep_005.get("final_provenance_status", "FAILED"),
                    holdout_status=guard_res.get("holdout_status", "FAILED"),
                    integrity_guard_result=guard_res,
                )
                leaderboard.append(res)
                qualification_board.append({
                    "strategy_id": disp_id, "strategy_name": res["strategy_name"], "timeframe_minutes": 60,
                    "initial_capital_inr": cap, "final_status": qual.final_status,
                    "passed_gates": f"{qual.passed_gates_count}/{qual.total_gates}",
                    "sessions": res["sessions_evaluated"], "trades": res["total_trades"],
                    "gross_return_pct": res["gross_return_pct"], "net_return_pct": res["total_return_pct"],
                    "conservative_10bps_net_pct": cons_10["net_return_pct"],
                    "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                    "max_drawdown_pct": res["max_drawdown_pct"], "sharpe_ratio": res["sharpe_ratio"],
                    "sortino_ratio": res["sortino_ratio"], "profit_factor": res["profit_factor"],
                    "expectancy_inr": res["expectancy_inr"], "total_fees_inr": res["total_fees_inr"],
                    "total_brokerage_inr": res.get("total_brokerage_inr", 0.0),
                    "total_statutory_inr": res.get("total_statutory_inr", 0.0),
                    "total_slippage_inr": res.get("total_slippage_inr", 0.0),
                    "unseen_return_pct": unseen_ret_pct, "statistical_verdict": stat_v,
                    "red_team_verdict": "APPROVED" if rt_ok else "REJECTED",
                    "exploratory_only": not qualification_allowed,
                    "summary": qual.summary, "gates": [asdict(g) for g in qual.gates],
                })

            cnc_configs = [
                ("STRAT-019-60M-CNC-ORB-SWING", "60m CNC Delivery ORB Swing (₹50K)", "ORB_SWING"),
                ("STRAT-020-60M-CNC-TREND-SWING", "60m CNC Delivery Trend Swing (₹50K)", "TREND_SWING"),
            ]
            for sid, sname, mode in cnc_configs:
                res = run_cnc_swing_backtest(eval_dates, sid, sname, mode=mode, initial_capital_inr=50_000.0, slippage_bps=5.0)
                orders_by_strat[sid] = res.pop("orders", [])
                slip_rows = []
                for bps in (5.0, 10.0, 20.0, 30.0):
                    sr_res = run_cnc_swing_backtest(eval_dates, sid, sname, mode=mode, initial_capital_inr=50_000.0, slippage_bps=bps)
                    slip_rows.append({
                        "slippage_bps": bps, "gross_return_pct": sr_res["gross_return_pct"],
                        "net_return_pct": sr_res["total_return_pct"], "net_pnl_inr": sr_res["net_pnl_inr"],
                        "total_fees_inr": sr_res["total_fees_inr"], "total_brokerage_inr": sr_res["total_brokerage_inr"],
                        "total_statutory_inr": sr_res["total_statutory_inr"], "total_slippage_inr": sr_res["total_slippage_inr"],
                        "total_trades": sr_res["total_trades"], "win_rate_pct": sr_res["win_rate_pct"],
                    })
                slip_matrix[sid] = slip_rows
                slip_all.extend({"strategy_id": sid, **sr} for sr in slip_rows)
                cons_10 = next((sr for sr in slip_rows if float(sr["slippage_bps"]) == 10.0), slip_rows[0])

                dr = res["daily_runs"]
                unseen_slice = dr[v_end_1h:] or [dr[-1]]
                unseen_ret_pct = round((sum(d["net_pnl"] for d in unseen_slice) / 50_000.0) * 100.0, 2)
                wf = {
                    "train": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[:t_end_1h]) / 50_000.0) * 100.0, 2)},
                    "validation": {"total_return_pct": round((sum(d["net_pnl"] for d in dr[t_end_1h:v_end_1h]) / 50_000.0) * 100.0, 2)},
                    "unseen_test": {"total_return_pct": unseen_ret_pct, "total_trades": sum(d["trades"] for d in unseen_slice)},
                }
                stat_v, _ = self.stat_agent.validate(res, wf)
                rt_ok, rt_reason = self.red_team.challenge(sid, res, wf)
                qual = evaluate_seven_gates(
                    strategy_id=sid, strategy_name=sname, multi_day_metrics=res,
                    unseen_metrics=wf["unseen_test"], conservative_slippage_metrics=cons_10,
                    data_provenance_verified=prov_ok_005, stat_validation_verdict=stat_v,
                    red_team_approved=rt_ok, red_team_reason=rt_reason,
                    min_sessions_required=60, min_trades_required=30,
                    independent_holdout_valid=guard_res["integrity_passed"],
                    provenance_status=prov_rep_005.get("final_provenance_status", "FAILED"),
                    holdout_status=guard_res.get("holdout_status", "FAILED"),
                    integrity_guard_result=guard_res,
                )
                leaderboard.append(res)
                qualification_board.append({
                    "strategy_id": sid, "strategy_name": sname, "timeframe_minutes": 60,
                    "initial_capital_inr": 50_000.0, "final_status": qual.final_status,
                    "passed_gates": f"{qual.passed_gates_count}/{qual.total_gates}",
                    "sessions": res["sessions_evaluated"], "trades": res["total_trades"],
                    "gross_return_pct": res["gross_return_pct"], "net_return_pct": res["total_return_pct"],
                    "conservative_10bps_net_pct": cons_10["net_return_pct"],
                    "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                    "max_drawdown_pct": res["max_drawdown_pct"], "sharpe_ratio": res["sharpe_ratio"],
                    "sortino_ratio": res["sortino_ratio"], "profit_factor": res["profit_factor"],
                    "expectancy_inr": res["expectancy_inr"], "total_fees_inr": res["total_fees_inr"],
                    "total_brokerage_inr": res["total_brokerage_inr"], "total_statutory_inr": res["total_statutory_inr"],
                    "total_slippage_inr": res["total_slippage_inr"],
                    "unseen_return_pct": unseen_ret_pct, "statistical_verdict": stat_v,
                    "red_team_verdict": "APPROVED" if rt_ok else "REJECTED",
                    "exploratory_only": not qualification_allowed,
                    "summary": qual.summary, "gates": [asdict(g) for g in qual.gates],
                })
        except Exception as exc:
            if consumed_dates:
                holdout_guard.mark_evaluation_failed(
                    "CYCLE-005",
                    consumed_dates,
                    reason=str(exc),
                    data_accessed=True,
                )
            raise

        # Step 5: Complete evaluation or record train+val-only completion
        evaluation_completed_at = datetime.now(tz=IST_TZ).isoformat()
        strategy_ids_evaluated = [q["strategy_id"] for q in qualification_board]
        recovery_failed = False
        recovery_failure_reasons: list[str] = []
        if consumed_dates:
            exposure_status = holdout_guard.complete_evaluation(
                "CYCLE-005",
                evaluation_id=evaluation_id,
                evaluation_type=eval_type,
                consumed_dates=consumed_dates,
                strategy_ids=strategy_ids_evaluated,
                dataset_fingerprint=dataset_fp,
                evaluation_ref=str(output_path),
            )
            completion_ok = bool(exposure_status.get("ok"))
            if completion_ok:
                holdout_consumed = True
                holdout_state = exposure_status.get("holdout_state", "EXPOSED")
                eval_status = "COMPLETED_FORMAL" if qualification_allowed else "COMPLETED_EXPLORATORY_WITH_UNSEEN"
            else:
                try:
                    reg_after = holdout_guard.load_registry()
                    reg_records = reg_after.get("records", {})
                    unrecovered_accessed = [
                        d
                        for d in consumed_dates
                        if reg_records.get(d.isoformat(), {}).get("state") == "SEALED"
                        and reg_records.get(d.isoformat(), {}).get("owning_cycle_id") == "CYCLE-005"
                        and (
                            bool(reg_records.get(d.isoformat(), {}).get("data_accessed"))
                            or reg_records.get(d.isoformat(), {}).get("reservation_status") == "EVALUATION_IN_PROGRESS"
                        )
                    ]
                    if unrecovered_accessed:
                        holdout_guard.mark_evaluation_failed(
                            "CYCLE-005",
                            unrecovered_accessed,
                            reason="; ".join(exposure_status.get("failure_reasons", ["complete_evaluation failed"])),
                            data_accessed=True,
                        )
                        reg_after = holdout_guard.load_registry()
                        reg_records = reg_after.get("records", {})
                    any_owned_exposed = any(
                        reg_records.get(d.isoformat(), {}).get("state") == "EXPOSED"
                        and reg_records.get(d.isoformat(), {}).get("owning_cycle_id") == "CYCLE-005"
                        for d in consumed_dates
                    )
                    holdout_consumed = any_owned_exposed or bool(exposure_status.get("holdout_consumed"))
                    holdout_state = "EXPOSED" if holdout_consumed else exposure_status.get("holdout_state", "SEALED")
                    eval_status = "COMPLETION_FAILED"
                except Exception as rec_exc:
                    recovery_failed = True
                    holdout_consumed = bool(exposure_status.get("holdout_consumed", False))
                    holdout_state = "RECOVERY_FAILED"
                    eval_status = "COMPLETION_FAILED_RECOVERY_FAILED"
                    recovery_failure_reasons.append(f"Recovery failed: {rec_exc}")
        else:
            exposure_status = {
                "status": "UNCONSUMED",
                "global_holdout_integrity_status": global_holdout_integrity_status,
                "independent_holdout_valid": False,
                "exposed": False,
                "holdout_consumed": False,
                "reason": "Exploratory evaluation ran on train+validation only; unseen holdout preserved.",
            }
            holdout_consumed = False
            holdout_state = "UNCONSUMED"
            eval_status = "COMPLETED_EXPLORATORY_TRAIN_VAL_ONLY"

        exec_rep = self.exec_agent.audit_costs(leaderboard, slip_all, orders_by_strategy=orders_by_strat)
        sq_rep = self.sq_agent.review_system(test_report=live_test_metrics)

        sep_005 = separate_strategies_and_benchmarks(qualification_board, period_label=f"{start_1h.isoformat()} -> {end_1h.isoformat()}")
        qualified_cnt = sep_005["qualified_active_strategies"]
        rejected_cnt = sum(1 for q in sep_005["active_strategy_leaderboard"] if q["final_status"] in ("REJECTED", "INVALIDATED", "BLOCKED_INTEGRITY"))
        best_strat = sep_005["best_active_strategy"]

        cycle_failure_reasons = list(guard_res.get("failure_reasons", []))
        if consumed_dates and not exposure_status.get("ok"):
            cycle_failure_reasons.extend(exposure_status.get("failure_reasons", ["complete_evaluation failed"]))
        cycle_failure_reasons.extend(recovery_failure_reasons)

        cycle_005_out = {
            "cycle_id": "CYCLE-005",
            "evaluation_id": evaluation_id,
            "generated_at_ist": evaluation_completed_at,
            "reconciled_eligible_sessions": len(elig_1h),
            "date_range": f"{start_1h.isoformat()} -> {end_1h.isoformat()}",
            "excluded_dates": excl_1h,
            "integrity_passed": bool(guard_res.get("integrity_passed", False)),
            "qualification_allowed": qualification_allowed,
            "exploratory_research_allowed": exploratory_allowed,
            "exploratory_only": not qualification_allowed,
            "current_evaluation_holdout_consumed": holdout_consumed,
            "current_evaluation_holdout_state": holdout_state,
            "global_holdout_integrity_status": global_holdout_integrity_status,
            "independent_holdout_available": independent_holdout_available,
            "holdout_consumed": holdout_consumed,
            "holdout_state": holdout_state,
            "recovery_failed": recovery_failed,
            "evaluation_status": eval_status,
            "evaluation_type": eval_type,
            "evaluation_started_at": evaluation_started_at,
            "evaluation_completed_at": evaluation_completed_at,
            "consumed_holdout_dates": [d.isoformat() for d in consumed_dates],
            "dataset_fingerprint": dataset_fp,
            "strategy_ids_evaluated": strategy_ids_evaluated,
            "failure_reasons": cycle_failure_reasons,
            "provenance_verified": bool(prov_rep_005.get("verified", False)),
            "provenance_status": prov_rep_005.get("final_provenance_status", "FAILED"),
            "seal_reservation": seal_res,
            "stale_recovery": stale_recovery,
            "integrity_guard_result": guard_res,
            "sealed_holdout_status": exposure_status,
            "dynamic_test_metrics": live_test_metrics,
            "command_centre": {
                "current_cycle": "CYCLE-005",
                "active_agents": 9,
                "experiments_completed": len(qualification_board),
                "experiments_rejected": rejected_cnt,
                "qualified_strategies": qualified_cnt,
                "best_experimental_strategy": best_strat["strategy_id"] if best_strat else None,
                "best_experimental_net_pct": best_strat["net_return_pct"] if best_strat else 0.0,
                "max_candidates_per_cycle": self.MAX_CANDIDATES_PER_CYCLE,
                "max_tuning_rounds": self.MAX_TUNING_ROUNDS_PER_HYPOTHESIS,
            },
            "qualification_board": sorted(qualification_board, key=lambda x: x["net_return_pct"], reverse=True),
            "reference_benchmark_leaderboard": sep_005["reference_benchmark_leaderboard"],
            "slippage_sensitivity_matrix": slip_matrix,
            "agent_performance": [asdict(exec_rep), asdict(sq_rep)],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cycle_005_out, indent=2), encoding="utf-8")

        return cycle_005_out
