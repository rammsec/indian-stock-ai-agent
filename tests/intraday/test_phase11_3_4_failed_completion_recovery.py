"""Phase 11.3.4 Tests: Failed Holdout Completion Recovery after Data Access."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from trader.research.holdout import SealedHoldoutGuard
from trader.research.orchestrator import ChiefResearchOrchestrator


def _make_guard(tmp_path: Path) -> SealedHoldoutGuard:
    return SealedHoldoutGuard(registry_path=tmp_path / "sealed_holdout_registry.json")


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


def test_01_fingerprint_mismatch_after_data_access_exposes_owned_dates(tmp_path: Path) -> None:
    """1. Fingerprint mismatch after data access exposes the owned dates."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 7, 1), date(2026, 7, 2)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-expected")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-expected")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-01",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=dates,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-wrong",
    )
    assert comp["ok"] is False
    assert comp["holdout_consumed"] is True
    assert comp["holdout_state"] == "EXPOSED"
    assert set(comp["exposed_dates"]) == {"2026-07-01", "2026-07-02"}

    reg = guard.load_registry()
    for d in dates:
        rec = reg["records"][d.isoformat()]
        assert rec["state"] == "EXPOSED"
        assert rec["reservation_status"] == "FAILED_EVALUATION"
        assert rec["data_accessed"] is True
        assert rec["first_exposed_cycle"] == "CYCLE-004"
        assert rec["exposure_timestamp"] is not None
        assert rec["previous_exposure_count"] == 1


def test_02_lifecycle_validation_failure_after_data_access_exposes_owned_dates(tmp_path: Path) -> None:
    """2. Lifecycle validation failure after data access exposes the owned dates."""
    guard = _make_guard(tmp_path)
    d1 = date(2026, 7, 3)
    d2 = date(2026, 7, 4)  # Missing from registry -> triggers lifecycle validation failure
    guard.seal_dates("CYCLE-004", [d1], dataset_fingerprint="fp-02")
    guard.mark_evaluation_in_progress("CYCLE-004", [d1], dataset_fingerprint="fp-02")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-02",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d1, d2],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-02",
    )
    assert comp["ok"] is False
    assert comp["holdout_consumed"] is True
    assert comp["exposed_dates"] == ["2026-07-03"]

    reg = guard.load_registry()
    assert reg["records"]["2026-07-03"]["state"] == "EXPOSED"
    assert reg["records"]["2026-07-03"]["reservation_status"] == "FAILED_EVALUATION"
    assert "2026-07-04" not in reg["records"]


def test_03_failed_completion_records_completion_success_false(tmp_path: Path) -> None:
    """3. Failed completion records completion_success=False."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 7, 5)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-03")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-03")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-03",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=dates,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-mismatch",
    )
    assert comp["completion_success"] is False

    reg = guard.load_registry()
    hist = reg["evaluation_history"]
    assert len(hist) == 1
    assert hist[0]["completion_success"] is False
    assert hist[0]["evaluation_status"] == "FAILED_EVALUATION"


def test_04_failed_completion_history_matches_actual_registry_state(tmp_path: Path) -> None:
    """4. Failed completion history matches the actual registry state."""
    guard = _make_guard(tmp_path)
    d_accessed = date(2026, 7, 6)
    d_unaccessed = date(2026, 7, 7)

    # Case A: accessed -> EXPOSED
    guard.seal_dates("CYCLE-004", [d_accessed], dataset_fingerprint="fp-04")
    guard.mark_evaluation_in_progress("CYCLE-004", [d_accessed], dataset_fingerprint="fp-04")
    guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-04a",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d_accessed],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-bad",
    )

    # Case B: not accessed -> SEALED
    guard.seal_dates("CYCLE-004", [d_unaccessed], dataset_fingerprint="fp-04")
    guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-04b",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d_unaccessed],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-04",
    )

    reg = guard.load_registry()
    assert reg["records"][d_accessed.isoformat()]["state"] == "EXPOSED"
    assert reg["records"][d_unaccessed.isoformat()]["state"] == "SEALED"
    assert reg["evaluation_history"][0]["final_lifecycle_state"] == "EXPOSED"
    assert reg["evaluation_history"][1]["final_lifecycle_state"] == "SEALED"


def test_05_failed_completion_never_exposes_dates_owned_by_another_cycle(tmp_path: Path) -> None:
    """5. Failed completion never exposes dates owned by another cycle."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 7, 8)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-05")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-05")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-005",
        evaluation_id="eval-34-05",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=dates,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-05",
    )
    assert comp["ok"] is False
    assert comp["holdout_consumed"] is False
    assert comp["exposed_dates"] == []
    assert "2026-07-08" in comp["ownership_violations"]

    reg = guard.load_registry()
    assert reg["records"]["2026-07-08"]["state"] == "SEALED"
    assert reg["records"]["2026-07-08"]["owning_cycle_id"] == "CYCLE-004"


def test_06_mixture_of_owned_and_foreign_dates_never_exposes_foreign_dates(tmp_path: Path) -> None:
    """6. A mixture of owned and foreign dates does not cause foreign dates to be exposed."""
    guard = _make_guard(tmp_path)
    d_owned = date(2026, 7, 9)
    d_foreign = date(2026, 7, 10)

    guard.seal_dates("CYCLE-004", [d_owned], dataset_fingerprint="fp-06")
    guard.mark_evaluation_in_progress("CYCLE-004", [d_owned], dataset_fingerprint="fp-06")

    guard.seal_dates("CYCLE-999", [d_foreign], dataset_fingerprint="fp-foreign")
    guard.mark_evaluation_in_progress("CYCLE-999", [d_foreign], dataset_fingerprint="fp-foreign")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-06",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d_owned, d_foreign],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-06",
    )
    assert comp["ok"] is False
    assert comp["holdout_consumed"] is True
    assert comp["exposed_dates"] == ["2026-07-09"]
    assert "2026-07-10" in comp["ownership_violations"]

    reg = guard.load_registry()
    assert reg["records"]["2026-07-09"]["state"] == "EXPOSED"
    assert reg["records"]["2026-07-09"]["reservation_status"] == "FAILED_EVALUATION"
    # Foreign date must remain untouched in SEALED/EVALUATION_IN_PROGRESS
    assert reg["records"]["2026-07-10"]["state"] == "SEALED"
    assert reg["records"]["2026-07-10"]["owning_cycle_id"] == "CYCLE-999"
    assert reg["records"]["2026-07-10"]["reservation_status"] == "EVALUATION_IN_PROGRESS"


def test_07_already_exposed_dates_not_counted_as_newly_exposed_again(tmp_path: Path) -> None:
    """7. Already EXPOSED dates are not counted as newly exposed again."""
    guard = _make_guard(tmp_path)
    d_exposed = date(2026, 7, 11)
    d_new = date(2026, 7, 12)

    # First expose d_exposed cleanly
    guard.seal_dates("CYCLE-004", [d_exposed], dataset_fingerprint="fp-07")
    guard.mark_evaluation_in_progress("CYCLE-004", [d_exposed], dataset_fingerprint="fp-07")
    c1 = guard.complete_evaluation("CYCLE-004", "eval-34-07a", "FORMAL_QUALIFICATION", [d_exposed], ["S"], "fp-07")
    assert c1["ok"] is True

    # Now seal & access d_new, then try completing [d_exposed, d_new]
    guard.seal_dates("CYCLE-004", [d_new], dataset_fingerprint="fp-07")
    guard.mark_evaluation_in_progress("CYCLE-004", [d_new], dataset_fingerprint="fp-07")
    c2 = guard.complete_evaluation("CYCLE-004", "eval-34-07b", "FORMAL_QUALIFICATION", [d_exposed, d_new], ["S"], "fp-07")
    assert c2["ok"] is False
    assert c2["exposed_dates"] == ["2026-07-12"]
    assert c2["newly_exposed_dates"] == 1

    reg = guard.load_registry()
    assert reg["records"]["2026-07-11"]["previous_exposure_count"] == 1
    assert reg["records"]["2026-07-12"]["previous_exposure_count"] == 1
    assert reg["records"]["2026-07-12"]["state"] == "EXPOSED"


def test_08_retired_dates_remain_retired(tmp_path: Path) -> None:
    """8. RETIRED dates remain RETIRED."""
    guard = _make_guard(tmp_path)
    d_retired = date(2026, 7, 13)
    d_owned = date(2026, 7, 14)

    guard.retire_dates([d_retired], "Corrupted session")
    guard.seal_dates("CYCLE-004", [d_owned], dataset_fingerprint="fp-08")
    guard.mark_evaluation_in_progress("CYCLE-004", [d_owned], dataset_fingerprint="fp-08")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-08",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=[d_retired, d_owned],
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-08",
    )
    assert comp["ok"] is False
    assert comp["exposed_dates"] == ["2026-07-14"]

    reg = guard.load_registry()
    assert reg["records"]["2026-07-13"]["state"] == "RETIRED"
    assert reg["records"]["2026-07-14"]["state"] == "EXPOSED"


def test_09_cycle_004_reports_holdout_consumed_true_when_recovery_exposed_dates(tmp_path: Path) -> None:
    """9. CYCLE-004 reports holdout_consumed=True when data was accessed and recovery exposed the dates."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_test09.json"
    orch = ChiefResearchOrchestrator()
    dates = _seed_1h_dates(80)

    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c004-09",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    # Wrap complete_evaluation so it runs with a mismatched fingerprint, simulating validation failure after data access
    orig_complete = SealedHoldoutGuard.complete_evaluation

    def tampered_complete(self, *args, **kwargs):
        kwargs["dataset_fingerprint"] = "fp-tampered-after-access"
        return orig_complete(self, *args, **kwargs)

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine.run_date_range", return_value=_dummy_backtest_result(80)),
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(80)),
        patch.object(SealedHoldoutGuard, "complete_evaluation", tampered_complete),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_status"] == "COMPLETION_FAILED"
    assert report["holdout_consumed"] is True
    assert report["holdout_state"] == "EXPOSED"
    assert report["recovery_failed"] is False

    reg = SealedHoldoutGuard(registry_path=reg_path).load_registry()
    unseen_dates = dates[64:]
    for d in unseen_dates:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"
        assert reg["records"][d.isoformat()]["reservation_status"] == "FAILED_EVALUATION"
    # Exactly 1 evaluation_history entry (no duplicates from redundant mark_evaluation_failed)
    assert len(reg["evaluation_history"]) == 1


def test_10_cycle_004_reports_recovery_failure_clearly_if_registry_transition_fails(tmp_path: Path) -> None:
    """10. CYCLE-004 reports recovery failure clearly if the registry transition cannot be completed."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_test10.json"
    orch = ChiefResearchOrchestrator()
    dates = _seed_1h_dates(80)

    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c004-10",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    fake_failed_complete = {
        "ok": False,
        "completion_success": False,
        "holdout_consumed": False,
        "holdout_state": "SEALED",
        "failure_reasons": ["Simulated completion failure"],
    }

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine.run_date_range", return_value=_dummy_backtest_result(80)),
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(80)),
        patch.object(SealedHoldoutGuard, "complete_evaluation", return_value=fake_failed_complete),
        patch.object(SealedHoldoutGuard, "mark_evaluation_failed", side_effect=OSError("Disk I/O error during recovery")),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["recovery_failed"] is True
    assert report["evaluation_status"] == "COMPLETION_FAILED_RECOVERY_FAILED"
    assert report["holdout_state"] == "RECOVERY_FAILED"
    assert any("Recovery failed" in r for r in report["failure_reasons"])


def test_11_no_duplicate_evaluation_history_records_created(tmp_path: Path) -> None:
    """11. No duplicate evaluation-history records are created during failed completion + recovery."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 7, 15), date(2026, 7, 16)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-11")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-11")

    comp = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-34-11",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=dates,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-bad-11",
    )
    assert comp["ok"] is False

    reg = guard.load_registry()
    hist = reg.get("evaluation_history", [])
    assert len(hist) == 1
    assert hist[0]["evaluation_id"] == "eval-34-11"
    assert hist[0]["final_lifecycle_state"] == "EXPOSED"
    assert hist[0]["completion_success"] is False


def test_12_stale_recovery_and_unaccessed_release_remain_intact(tmp_path: Path) -> None:
    """12. Stale crash recovery exposes accessed dates and unaccessed release remains restricted to owner."""
    guard = _make_guard(tmp_path)
    d_crash = [date(2026, 7, 17)]
    d_unaccessed = [date(2026, 7, 18)]

    guard.seal_dates("CYCLE-004", d_crash, dataset_fingerprint="fp-12")
    guard.mark_evaluation_in_progress("CYCLE-004", d_crash, dataset_fingerprint="fp-12")
    rec_res = guard.recover_stale_reservations("CYCLE-004")
    assert rec_res["recovered_dates"] == ["2026-07-17"]

    guard.seal_dates("CYCLE-004", d_unaccessed, dataset_fingerprint="fp-12")
    denied = guard.release_reservation("CYCLE-999", d_unaccessed, "Unauthorized release")
    assert denied["ok"] is False
    allowed = guard.release_reservation("CYCLE-004", d_unaccessed, "Aborted before access")
    assert allowed["ok"] is True

    reg = guard.load_registry()
    assert reg["records"]["2026-07-17"]["state"] == "EXPOSED"
    assert reg["records"]["2026-07-18"]["state"] == "AVAILABLE"
