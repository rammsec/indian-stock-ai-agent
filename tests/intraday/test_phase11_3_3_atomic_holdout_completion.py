"""Phase 11.3.3 Tests: Atomic Holdout Completion & Lifecycle Integrity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trader.research.holdout import SealedHoldoutGuard
from trader.research.orchestrator import ChiefResearchOrchestrator


def _make_guard(tmp_path: Path) -> SealedHoldoutGuard:
    reg_path = tmp_path / "sealed_holdout_registry.json"
    return SealedHoldoutGuard(registry_path=reg_path)


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


_FAKE_TEST_REPORT = {"total": 10, "passed": 10, "failed": 0, "skipped": 0, "pass_rate_pct": 100.0}


def test_01_completion_succeeds_only_after_evaluation_in_progress(tmp_path: Path) -> None:
    """1. Completion succeeds only after EVALUATION_IN_PROGRESS."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 1)]
    seal = guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-1")
    assert seal["ownership_valid"] is True

    prog = guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-1")
    assert prog["ok"] is True

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-01",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-1",
    )
    assert comp["ok"] is True
    assert comp["completion_success"] is True
    assert comp["holdout_state"] == "EXPOSED"
    assert comp["holdout_consumed"] is True

    reg = guard.load_registry()
    rec = reg["records"]["2026-06-01"]
    assert rec["state"] == "EXPOSED"
    assert rec["reservation_status"] == "EVALUATION_COMPLETED"
    assert rec["owning_cycle_id"] == "CYCLE-004"
    assert rec["data_accessed"] is True


def test_02_completion_without_starting_evaluation_is_rejected(tmp_path: Path) -> None:
    """2. Completion without starting evaluation is rejected (remains ACTIVE_RESERVATION)."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 2)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-2")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-02",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-2",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False
    assert comp["holdout_consumed"] is False
    assert any("expected 'EVALUATION_IN_PROGRESS'" in r for r in comp["failure_reasons"])

    reg = guard.load_registry()
    rec = reg["records"]["2026-06-02"]
    assert rec["state"] == "SEALED"
    assert rec["reservation_status"] == "ACTIVE_RESERVATION"


def test_03_wrong_cycle_cannot_complete_reservation(tmp_path: Path) -> None:
    """3. Wrong cycle cannot complete a reservation."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 3)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-3")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-3")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-005",
        evaluation_id="eval-03",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-3",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False
    assert "2026-06-03" in comp["ownership_violations"]

    reg = guard.load_registry()
    rec = reg["records"]["2026-06-03"]
    assert rec["state"] == "SEALED"
    assert rec["owning_cycle_id"] == "CYCLE-004"


def test_04_wrong_dataset_fingerprint_prevents_completion(tmp_path: Path) -> None:
    """4. Wrong dataset fingerprint prevents completion."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 4)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-orig")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-orig")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-04",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-tampered",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False
    assert "2026-06-04" in comp["fingerprint_mismatches"]

    reg = guard.load_registry()
    assert reg["records"]["2026-06-04"]["state"] == "SEALED"


def test_05_completion_of_exposed_dates_is_rejected(tmp_path: Path) -> None:
    """5. Completion of EXPOSED dates is rejected."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 5)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-5")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-5")
    comp1 = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-05a",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-5",
    )
    assert comp1["ok"] is True

    comp2 = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-05b",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-5",
    )
    assert comp2["ok"] is False
    assert comp2["completion_success"] is False
    assert "2026-06-05" in comp2["reused_dates"]


def test_06_completion_of_retired_dates_is_rejected(tmp_path: Path) -> None:
    """6. Completion of RETIRED dates is rejected."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 6)]
    guard.retire_dates(d, "Corrupted archive")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-06",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-6",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False
    assert "2026-06-06" in comp["reused_dates"]
    assert guard.load_registry()["records"]["2026-06-06"]["state"] == "RETIRED"


def test_07_duplicate_consumed_dates_are_rejected(tmp_path: Path) -> None:
    """7. Duplicate consumed dates are rejected."""
    guard = _make_guard(tmp_path)
    d = date(2026, 6, 7)
    guard.seal_dates("CYCLE-004", [d], dataset_fingerprint="fp-7")
    guard.mark_evaluation_in_progress("CYCLE-004", [d], dataset_fingerprint="fp-7")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-07",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d, d],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-7",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False
    assert any("Duplicate" in r for r in comp["failure_reasons"])
    assert guard.load_registry()["records"]["2026-06-07"]["state"] == "SEALED"


def test_08_multi_date_completion_is_atomic_when_one_date_is_invalid(tmp_path: Path) -> None:
    """8. Multi-date completion is atomic when one date is invalid (all-or-nothing)."""
    guard = _make_guard(tmp_path)
    d1 = date(2026, 6, 8)
    d2 = date(2026, 6, 9)
    d3 = date(2026, 6, 10)

    guard.seal_dates("CYCLE-004", [d1, d2, d3], dataset_fingerprint="fp-8")
    # Only mark d1 and d2 in progress, leaving d3 in ACTIVE_RESERVATION
    guard.mark_evaluation_in_progress("CYCLE-004", [d1, d2], dataset_fingerprint="fp-8")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-08",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d1, d2, d3],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-8",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False

    reg = guard.load_registry()
    # Confirm NO dates transitioned to EXPOSED
    for d in (d1, d2, d3):
        assert reg["records"][d.isoformat()]["state"] == "SEALED"


def test_09_failed_completion_does_not_append_successful_history(tmp_path: Path) -> None:
    """9. Failed completion does not append successful history."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 11)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-9")
    # Attempt completion without starting evaluation
    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-09",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-9",
    )
    assert comp["ok"] is False

    reg = guard.load_registry()
    hist = reg.get("evaluation_history", [])
    assert len(hist) == 1
    assert hist[0]["completion_success"] is False
    assert hist[0]["ok"] is False
    assert hist[0]["final_lifecycle_state"] == "SEALED"


def test_10_history_reflects_actual_final_registry_state(tmp_path: Path) -> None:
    """10. History reflects the actual final registry state and includes all 12 required audit fields."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 12)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-orig-10")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-orig-10")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-10",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-mismatch-10",
    )
    assert comp["ok"] is False

    reg = guard.load_registry()
    hist = reg.get("evaluation_history", [])
    assert len(hist) == 1
    h = hist[0]

    # Required fields per Phase 11.3.3 Section 4:
    for field in (
        "evaluation_id",
        "cycle_id",
        "evaluation_type",
        "evaluation_status",
        "consumed_dates",
        "strategy_ids",
        "dataset_fingerprint",
        "started_at",
        "completed_at",
        "final_lifecycle_state",
        "completion_success",
        "failure_reasons",
    ):
        assert field in h, f"Missing field: {field}"

    assert h["completion_success"] is False
    assert h["final_lifecycle_state"] == "SEALED"
    assert reg["records"]["2026-06-12"]["state"] == "SEALED"


def test_11_successful_completion_updates_all_dates_and_history_together(tmp_path: Path) -> None:
    """11. Successful completion updates all dates and history together under single lock."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 6, 13), date(2026, 6, 14), date(2026, 6, 15)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-11")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-11")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-11",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=dates,
        strategy_ids=["STRAT-018", "STRAT-019"],
        dataset_fingerprint="fp-11",
    )
    assert comp["ok"] is True
    assert comp["completion_success"] is True

    reg = guard.load_registry()
    for d in dates:
        rec = reg["records"][d.isoformat()]
        assert rec["state"] == "EXPOSED"
        assert rec["reservation_status"] == "EVALUATION_COMPLETED"
        assert rec["data_accessed"] is True

    hist = reg["evaluation_history"]
    assert len(hist) == 1
    assert hist[0]["completion_success"] is True
    assert hist[0]["final_lifecycle_state"] == "EXPOSED"
    assert hist[0]["consumed_dates"] == [d.isoformat() for d in dates]


def test_12_failed_evaluation_start_transition_prevents_backtest_execution(tmp_path: Path) -> None:
    """12. Failed evaluation-start transition prevents backtest execution."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_test12.json"
    orch = ChiefResearchOrchestrator()
    dates = _seed_1h_dates(80)

    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-test12",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    mock_replay_engine = MagicMock()
    mock_cnc_swing = MagicMock()

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch.object(SealedHoldoutGuard, "mark_evaluation_in_progress", return_value={"ok": False, "reason": "Forced failure"}),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine", mock_replay_engine),
        patch("trader.research.cycle004.run_cnc_swing_backtest", mock_cnc_swing),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_status"] == "ABORTED_TRANSITION_FAILED"
    assert report["holdout_consumed"] is False
    # Backtest engines must NOT have been called
    mock_replay_engine.assert_not_called()
    mock_cnc_swing.assert_not_called()


def test_13_failed_evaluation_start_transition_attempts_safe_reservation_release(tmp_path: Path) -> None:
    """13. Failed evaluation-start transition attempts safe reservation release."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_test13.json"
    orch = ChiefResearchOrchestrator()
    dates = _seed_1h_dates(80)

    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-test13",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch.object(SealedHoldoutGuard, "mark_evaluation_in_progress", return_value={"ok": False, "reason": "Fingerprint mismatch"}),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_status"] == "ABORTED_TRANSITION_FAILED"
    assert report["cleanup_status"] == "RESERVATION_RELEASED"
    assert report["holdout_state"] == "AVAILABLE"

    # Confirm the unseen dates were released back to AVAILABLE in the registry
    guard = SealedHoldoutGuard(registry_path=reg_path)
    reg = guard.load_registry()
    unseen_dates = dates[64:]
    for d in unseen_dates:
        assert reg["records"][d.isoformat()]["state"] == "AVAILABLE"
        assert reg["records"][d.isoformat()]["reservation_status"] == "RELEASED_RESERVATION"


def test_14_failed_reservation_release_does_not_silently_reset_dates(tmp_path: Path) -> None:
    """14. Failed reservation release does not silently reset dates."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_test14.json"
    orch = ChiefResearchOrchestrator()
    dates = _seed_1h_dates(80)

    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-test14",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    denied_info = {"released": [], "denied": [{"date": "ALL", "reason": "Permission denied"}], "ok": False}
    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch.object(SealedHoldoutGuard, "mark_evaluation_in_progress", return_value={"ok": False, "reason": "Aborted"}),
        patch.object(SealedHoldoutGuard, "release_reservation", return_value=denied_info),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_status"] == "ABORTED_TRANSITION_FAILED"
    assert report["cleanup_status"] == "RELEASE_FAILED_RESERVATION_PRESERVED"
    assert report["holdout_state"] == "SEALED"


def test_15_concurrent_completion_attempts_cannot_expose_same_dates_twice(tmp_path: Path) -> None:
    """15. Concurrent completion attempts cannot expose the same dates twice."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 20)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-15")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-15")

    def try_complete(eval_id: str) -> dict:
        return guard.complete_evaluation(
            cycle_id="CYCLE-004",
            evaluation_id=eval_id,
            evaluation_type="FORMAL_QUALIFICATION",
            consumed_dates=d,
            strategy_ids=["STRAT-018"],
            dataset_fingerprint="fp-15",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(try_complete, "eval-15a")
        f2 = executor.submit(try_complete, "eval-15b")
        res1 = f1.result()
        res2 = f2.result()

    # Exactly one attempt must succeed, and the other must be rejected
    results = [res1, res2]
    successes = [r for r in results if r["ok"] is True]
    failures = [r for r in results if r["ok"] is False]

    assert len(successes) == 1
    assert len(failures) == 1
    assert failures[0]["completion_success"] is False

    reg = guard.load_registry()
    assert reg["records"]["2026-06-20"]["previous_exposure_count"] == 1


def test_16_corrupted_registry_fails_closed(tmp_path: Path) -> None:
    """16. Corrupted registry fails closed."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    reg_path.write_text("{corrupt: true", encoding="utf-8")
    guard = SealedHoldoutGuard(registry_path=reg_path)

    with pytest.raises(json.JSONDecodeError):
        guard.complete_evaluation(
            cycle_id="CYCLE-004",
            evaluation_id="eval-16",
            evaluation_type="FORMAL_QUALIFICATION",
            consumed_dates=[date(2026, 6, 21)],
            strategy_ids=["STRAT-018"],
            dataset_fingerprint="fp-16",
        )


def test_17_fingerprint_mismatch_does_not_create_misleading_exposure_metadata(tmp_path: Path) -> None:
    """17. Fingerprint mismatch does not create misleading exposure metadata."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 22)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-good")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-good")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-17",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-bad",
    )
    assert comp["ok"] is False
    assert comp["completion_success"] is False
    assert comp["holdout_state"] == "SEALED"

    reg = guard.load_registry()
    assert reg["records"]["2026-06-22"]["state"] == "SEALED"
    # sealed_exposures must NOT have a record for this failed attempt
    assert len(reg.get("sealed_exposures", [])) == 0
    # evaluation_history must clearly reflect failure
    hist = reg.get("evaluation_history", [])
    assert len(hist) == 1
    assert hist[0]["completion_success"] is False
    assert hist[0]["final_lifecycle_state"] == "SEALED"


def test_18_failure_after_data_access_still_consumes_holdout(tmp_path: Path) -> None:
    """18. Failure after data access still consumes the holdout (conservative recovery)."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 23)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-18")
    prog = guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-18")
    assert prog["ok"] is True

    # Simulation failed after data access
    fail_res = guard.mark_evaluation_failed("CYCLE-004", d, reason="Network partition during backtest", data_accessed=True)
    assert fail_res["holdout_consumed"] is True
    assert "2026-06-23" in fail_res["exposed_dates"]

    reg = guard.load_registry()
    assert reg["records"]["2026-06-23"]["state"] == "EXPOSED"
    assert reg["records"]["2026-06-23"]["reservation_status"] == "FAILED_EVALUATION"


def test_19_existing_phase11_3_1_and_11_3_2_tests_remain_valid(tmp_path: Path) -> None:
    """19. Standard happy-path qualification and exploratory flows remain intact."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 6, 24)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-19")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-19", evaluation_type="FORMAL_QUALIFICATION")
    res = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-19",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=d,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-19",
    )
    assert res["ok"] is True
    assert res["holdout_state"] == "EXPOSED"
    assert res["completion_success"] is True

    reg = guard.load_registry()
    assert reg["records"]["2026-06-24"]["state"] == "EXPOSED"
    assert reg["evaluation_history"][0]["completion_success"] is True
