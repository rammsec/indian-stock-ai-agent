"""Phase 11.3.6 Tests: Final Holdout Idempotency Edge Cases."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from trader.research.holdout import SealedHoldoutGuard


def _make_guard(tmp_path: Path) -> SealedHoldoutGuard:
    return SealedHoldoutGuard(registry_path=tmp_path / "sealed_holdout_registry.json")


def test_01_existing_successful_evaluation_retried_with_empty_dates(tmp_path: Path) -> None:
    """1. Existing successful evaluation retried with empty dates returns EVALUATION_ID_CONFLICT, not INSUFFICIENT_UNSEEN_DATA."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 1)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-01")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-01")
    first = guard.complete_evaluation("CYCLE-004", "eval-36-01", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-01")
    assert first["ok"] is True

    retry = guard.complete_evaluation("CYCLE-004", "eval-36-01", "FORMAL_QUALIFICATION", [], ["S1"], "fp-36-01")
    assert retry["ok"] is False
    assert retry["status"] == "EVALUATION_ID_CONFLICT"
    assert retry["request_applied"] is False
    assert "consumed_dates" in retry["conflict_fields"]
    assert retry["holdout_consumed"] is True
    assert retry["holdout_state"] == "EXPOSED"


def test_02_existing_successful_evaluation_retried_with_duplicate_dates(tmp_path: Path) -> None:
    """2. Existing successful evaluation retried with duplicate dates returns EVALUATION_ID_CONFLICT, not DUPLICATE_CONSUMED_DATES."""
    guard = _make_guard(tmp_path)
    d = date(2026, 9, 2)
    guard.seal_dates("CYCLE-004", [d], dataset_fingerprint="fp-36-02")
    guard.mark_evaluation_in_progress("CYCLE-004", [d], dataset_fingerprint="fp-36-02")
    first = guard.complete_evaluation("CYCLE-004", "eval-36-02", "FORMAL_QUALIFICATION", [d], ["S1"], "fp-36-02")
    assert first["ok"] is True

    retry = guard.complete_evaluation("CYCLE-004", "eval-36-02", "FORMAL_QUALIFICATION", [d, d], ["S1"], "fp-36-02")
    assert retry["ok"] is False
    assert retry["status"] == "EVALUATION_ID_CONFLICT"
    assert retry["request_applied"] is False
    assert "consumed_dates" in retry["conflict_fields"]
    assert retry["holdout_consumed"] is True
    assert retry["holdout_state"] == "EXPOSED"


def test_03_existing_failed_evaluation_retried_with_empty_dates(tmp_path: Path) -> None:
    """3. Existing failed evaluation retried with empty dates returns EVALUATION_ID_CONFLICT."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 3)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-03")
    # Failed because not marked in progress -> holdout_consumed=False, state=SEALED
    first = guard.complete_evaluation("CYCLE-004", "eval-36-03", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-03")
    assert first["ok"] is False
    assert first["holdout_consumed"] is False

    retry = guard.complete_evaluation("CYCLE-004", "eval-36-03", "FORMAL_QUALIFICATION", [], ["S1"], "fp-36-03")
    assert retry["ok"] is False
    assert retry["status"] == "EVALUATION_ID_CONFLICT"
    assert retry["request_applied"] is False
    assert retry["holdout_consumed"] is False
    assert retry["holdout_state"] == "SEALED"


def test_04_existing_id_with_malformed_dates_returns_conflict_when_params_differ(tmp_path: Path) -> None:
    """4. Existing ID with malformed dates returns an identity conflict when request parameters differ."""
    guard = _make_guard(tmp_path)
    d = date(2026, 9, 4)
    guard.seal_dates("CYCLE-004", [d], dataset_fingerprint="fp-36-04")
    guard.mark_evaluation_in_progress("CYCLE-004", [d], dataset_fingerprint="fp-36-04")
    guard.complete_evaluation("CYCLE-004", "eval-36-04", "FORMAL_QUALIFICATION", [d], ["S1"], "fp-36-04")

    conflict = guard.complete_evaluation("CYCLE-005", "eval-36-04", "EXPLORATORY_WITH_UNSEEN", [d, d], ["S2"], "fp-other")
    assert conflict["ok"] is False
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert set(conflict["conflict_fields"]) == {
        "cycle_id",
        "evaluation_type",
        "consumed_dates",
        "dataset_fingerprint",
        "strategy_ids",
    }


def test_05_conflict_response_accurately_reflects_original_holdout_consumption(tmp_path: Path) -> None:
    """5. Conflict response accurately reflects original holdout consumption (True when originally exposed)."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 5), date(2026, 9, 6)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-05")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-05")
    guard.complete_evaluation("CYCLE-004", "eval-36-05", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-05")

    conflict = guard.complete_evaluation("CYCLE-004", "eval-36-05", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-wrong")
    assert conflict["ok"] is False
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert conflict["holdout_consumed"] is True
    assert conflict["holdout_state"] == "EXPOSED"
    assert conflict["exposed_dates"] == ["2026-09-05", "2026-09-06"]
    assert conflict["newly_exposed_dates"] == 0


def test_06_conflict_response_has_request_applied_false(tmp_path: Path) -> None:
    """6. Conflict response has request_applied=False."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 7)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-06")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-06")
    guard.complete_evaluation("CYCLE-004", "eval-36-06", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-06")

    conflict = guard.complete_evaluation("CYCLE-005", "eval-36-06", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-06")
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert conflict["request_applied"] is False


def test_07_conflict_does_not_mutate_registry(tmp_path: Path) -> None:
    """7. Conflict does not mutate the registry or increment exposure counters."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 8)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-07")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-07")
    guard.complete_evaluation("CYCLE-004", "eval-36-07", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-07")

    before_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))
    conflict = guard.complete_evaluation("CYCLE-004", "eval-36-07", "FORMAL_QUALIFICATION", [], ["S1"], "fp-36-07")
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    after_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))

    assert after_reg == before_reg
    assert after_reg["records"]["2026-09-08"]["previous_exposure_count"] == 1


def test_08_conflict_does_not_append_history(tmp_path: Path) -> None:
    """8. Conflict does not append history or sealed_exposures."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 9)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-08")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-08")
    guard.complete_evaluation("CYCLE-004", "eval-36-08", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-08")

    guard.complete_evaluation("CYCLE-004", "eval-36-08", "FORMAL_QUALIFICATION", [date(2026, 9, 10)], ["S1"], "fp-36-08")
    reg = guard.load_registry()
    assert len(reg["evaluation_history"]) == 1
    assert len(reg["sealed_exposures"]) == 1


def test_09_duplicate_historical_evaluation_ids_fail_closed(tmp_path: Path) -> None:
    """9. Duplicate historical evaluation IDs fail closed with DUPLICATE_EVALUATION_ID_HISTORY without modifying registry."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 11)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-09")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-09")
    guard.complete_evaluation("CYCLE-004", "eval-dup-hist", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-09")

    # Inject a duplicate evaluation_history entry with the same evaluation_id
    reg = guard.load_registry()
    reg["evaluation_history"].append(dict(reg["evaluation_history"][0]))
    guard.save_registry(reg)
    before_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))

    res = guard.complete_evaluation("CYCLE-004", "eval-dup-hist", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-09")
    assert res["ok"] is False
    assert res["completion_success"] is False
    assert res["status"] == "DUPLICATE_EVALUATION_ID_HISTORY"
    assert res["request_applied"] is False
    assert "DUPLICATE_EVALUATION_ID_HISTORY" in res["lifecycle_violations"]

    after_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))
    assert after_reg == before_reg


def test_10_new_evaluation_id_with_empty_dates_retains_validation_behaviour(tmp_path: Path) -> None:
    """10. New evaluation IDs with empty dates retain existing INSUFFICIENT_UNSEEN_DATA validation behaviour."""
    guard = _make_guard(tmp_path)
    res = guard.complete_evaluation("CYCLE-004", "eval-new-empty", "FORMAL_QUALIFICATION", [], ["S1"], "fp-36-10")
    assert res["ok"] is False
    assert res["status"] == "INSUFFICIENT_UNSEEN_DATA"
    assert "EMPTY_DATES" in res["lifecycle_violations"]


def test_11_new_evaluation_id_with_duplicate_dates_retains_validation_behaviour(tmp_path: Path) -> None:
    """11. New evaluation IDs with duplicate dates retain existing DUPLICATE_CONSUMED_DATES validation behaviour."""
    guard = _make_guard(tmp_path)
    d = date(2026, 9, 12)
    guard.seal_dates("CYCLE-004", [d], dataset_fingerprint="fp-36-11")
    guard.mark_evaluation_in_progress("CYCLE-004", [d], dataset_fingerprint="fp-36-11")

    res = guard.complete_evaluation("CYCLE-004", "eval-new-dup", "FORMAL_QUALIFICATION", [d, d], ["S1"], "fp-36-11")
    assert res["ok"] is False
    assert res["status"] == "DUPLICATE_CONSUMED_DATES"
    assert "DUPLICATE_DATES" in res["lifecycle_violations"]
    assert guard.load_registry()["records"]["2026-09-12"]["state"] == "SEALED"


def test_12_successful_replay_reports_zero_newly_exposed_dates(tmp_path: Path) -> None:
    """12. Phase 11.3.6.1: Successful replay returns newly_exposed_dates=0 while preserving exposed_dates and registry."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 13), date(2026, 9, 14)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-36-12")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-36-12")

    first = guard.complete_evaluation("CYCLE-004", "eval-36-12", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-12")
    assert first["ok"] is True
    assert first["newly_exposed_dates"] == 2

    before_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))

    replay = guard.complete_evaluation("CYCLE-004", "eval-36-12", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-36-12")
    assert replay["ok"] is True
    assert replay["completion_success"] is True
    assert replay["evaluation_status"] == "EVALUATION_COMPLETED"
    assert replay["holdout_consumed"] is True
    assert replay["holdout_state"] == "EXPOSED"
    assert replay["exposed_dates"] == ["2026-09-13", "2026-09-14"]
    assert replay["newly_exposed_dates"] == 0
    assert replay["completed_at"] == first["completed_at"]
    assert replay["idempotent_replay"] is True

    after_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))
    assert after_reg == before_reg
    assert len(after_reg["evaluation_history"]) == 1
    assert len(after_reg["sealed_exposures"]) == 1
    for d in dates:
        assert after_reg["records"][d.isoformat()]["previous_exposure_count"] == 1


def test_13_failed_replay_reports_zero_newly_exposed_dates(tmp_path: Path) -> None:
    """13. Phase 11.3.6.1: Failed replay after data access returns newly_exposed_dates=0 while preserving original outcome."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 9, 15)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-good-13")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-good-13")

    first = guard.complete_evaluation("CYCLE-004", "eval-36-13", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-bad-13")
    assert first["ok"] is False
    assert first["holdout_consumed"] is True
    assert first["newly_exposed_dates"] == 1
    assert first["exposed_dates"] == ["2026-09-15"]

    before_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))

    replay = guard.complete_evaluation("CYCLE-004", "eval-36-13", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-bad-13")
    assert replay["ok"] is False
    assert replay["completion_success"] is False
    assert replay["evaluation_status"] == "FAILED_EVALUATION"
    assert replay["holdout_consumed"] is True
    assert replay["holdout_state"] == "EXPOSED"
    assert replay["exposed_dates"] == ["2026-09-15"]
    assert replay["newly_exposed_dates"] == 0
    assert replay["completed_at"] == first["completed_at"]
    assert replay["idempotent_replay"] is True

    after_reg = json.loads(guard.registry_path.read_text(encoding="utf-8"))
    assert after_reg == before_reg
    assert len(after_reg["evaluation_history"]) == 1
    assert after_reg["records"]["2026-09-15"]["previous_exposure_count"] == 1

