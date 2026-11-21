"""Phase 11.3.3 CYCLE-005 Audit, Test Evidence & Holdout Reporting Tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from trader.research.holdout import SealedHoldoutGuard
from trader.research.orchestrator import ChiefResearchOrchestrator, collect_dynamic_test_metrics


def _seed_1h_dates(n: int = 80) -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _dummy_backtest_result(sessions: int) -> dict:
    return {
        "strategy_id": "S",
        "strategy_name": "S",
        "sessions_evaluated": sessions,
        "total_trades": 0,
        "gross_return_pct": 0.0,
        "total_return_pct": 0.0,
        "net_pnl_inr": 0.0,
        "alpha_vs_buy_and_hold_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "profit_factor": 0.0,
        "expectancy_inr": 0.0,
        "total_fees_inr": 0.0,
        "total_brokerage_inr": 0.0,
        "total_statutory_inr": 0.0,
        "total_slippage_inr": 0.0,
        "win_rate_pct": 0.0,
        "daily_runs": [{"net_pnl": 0.0, "trades": 0} for _ in range(sessions)],
        "orders": [],
        "trades": [],
    }


def test_01_train_val_only_with_previously_exposed_global_dates(tmp_path: Path) -> None:
    """1-3 & 10. Train/val-only evaluation preserves current holdout, reports global reuse separately, and leaves registry unchanged."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_005_test01.json"
    guard = SealedHoldoutGuard(registry_path=reg_path)
    dates = _seed_1h_dates(80)
    # Pre-expose unseen dates from an older cycle
    guard.record_exposure("CYCLE-002", dates[64:], notes="Legacy exposure", dataset_fingerprint="fp-old")
    before_reg = json.loads(reg_path.read_text(encoding="utf-8"))

    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c5-01",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    orch = ChiefResearchOrchestrator()
    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine.run_date_range", return_value=_dummy_backtest_result(64)),
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(64)),
    ):
        report = orch.run_cycle_005(
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_type"] == "EXPLORATORY_TRAIN_VAL_ONLY"
    assert report["evaluation_status"] == "COMPLETED_EXPLORATORY_TRAIN_VAL_ONLY"
    assert report["current_evaluation_holdout_consumed"] is False
    assert report["current_evaluation_holdout_state"] == "UNCONSUMED"
    assert report["global_holdout_integrity_status"] == "EXPOSED_HOLDOUT_REUSED"
    assert report["independent_holdout_available"] is False
    assert report["qualification_allowed"] is False

    after_reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert after_reg == before_reg


def test_02_missing_test_execution_is_recorded_as_not_run() -> None:
    """5. Missing test execution is recorded as NOT_RUN with executed=False and independently_verified=False."""
    metrics = collect_dynamic_test_metrics(run_pytest=False)
    assert metrics["executed"] is False
    assert metrics["status"] == "NOT_RUN"
    assert metrics["independently_verified"] is False
    assert metrics["command"] is None
    assert metrics["tests_executed"] == 0


def test_03_caller_supplied_test_report_never_marked_independently_verified(tmp_path: Path) -> None:
    """4. Caller-supplied test metadata is never represented as independently verified."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_005_test03.json"
    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c5-03",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    orch = ChiefResearchOrchestrator()
    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine.run_date_range", return_value=_dummy_backtest_result(64)),
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(64)),
    ):
        report = orch.run_cycle_005(
            test_report={"tests_executed": 999, "tests_passed": 999, "executed": True},
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    dtm = report["dynamic_test_metrics"]
    assert dtm["independently_verified"] is False
    assert dtm["source"] == "CALLER_SUPPLIED"
    assert dtm["status"] == "SUPPLIED_UNVERIFIED"


def test_04_completion_without_in_progress_and_wrong_owner_rejected_no_false_exposed(tmp_path: Path) -> None:
    """6-8. Completion without EVALUATION_IN_PROGRESS or by wrong owner is rejected without false EXPOSED history."""
    guard = SealedHoldoutGuard(registry_path=tmp_path / "sealed_holdout_registry.json")
    d = [date(2026, 10, 1)]
    guard.seal_dates("CYCLE-005", d, dataset_fingerprint="fp-c5-04")

    # Without EVALUATION_IN_PROGRESS
    c1 = guard.complete_evaluation("CYCLE-005", "eval-c5-no-prog", "FORMAL_QUALIFICATION", d, ["S1"], "fp-c5-04")
    assert c1["ok"] is False
    assert c1["holdout_consumed"] is False
    assert c1["holdout_state"] == "SEALED"

    # Start in progress as CYCLE-005, then attempt completion as CYCLE-006
    guard.mark_evaluation_in_progress("CYCLE-005", d, dataset_fingerprint="fp-c5-04")
    c2 = guard.complete_evaluation("CYCLE-006", "eval-c5-wrong-owner", "FORMAL_QUALIFICATION", d, ["S1"], "fp-c5-04")
    assert c2["ok"] is False
    assert c2["holdout_consumed"] is False
    assert c2["holdout_state"] == "SEALED"

    reg = guard.load_registry()
    assert reg["records"]["2026-10-01"]["state"] == "SEALED"
    assert all(h["final_lifecycle_state"] == "SEALED" for h in reg["evaluation_history"])
    assert all(h["completion_success"] is False for h in reg["evaluation_history"])


def test_05_failed_mark_evaluation_in_progress_aborts_and_releases_unused_reservation(tmp_path: Path) -> None:
    """9. Failed lifecycle transition in run_cycle_005 aborts before backtest and releases unused reservation."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_005_test05.json"
    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c5-05",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    mock_replay = MagicMock()
    orch = ChiefResearchOrchestrator()
    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch.object(SealedHoldoutGuard, "mark_evaluation_in_progress", return_value={"ok": False, "reason": "Forced abort"}),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine", mock_replay),
    ):
        report = orch.run_cycle_005(
            run_live_pytest=False,
            include_unseen_in_exploratory=True,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_status"] == "ABORTED_TRANSITION_FAILED"
    assert report["current_evaluation_holdout_consumed"] is False
    assert report["current_evaluation_holdout_state"] == "AVAILABLE"
    assert report["cleanup_status"] == "RESERVATION_RELEASED"
    mock_replay.assert_not_called()

    reg = SealedHoldoutGuard(registry_path=reg_path).load_registry()
    for d in dates[64:]:
        assert reg["records"][d.isoformat()]["state"] == "AVAILABLE"
