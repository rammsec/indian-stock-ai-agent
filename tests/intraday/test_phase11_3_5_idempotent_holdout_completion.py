"""Phase 11.3.5 Tests: Idempotent Holdout Evaluation Completion."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from trader.research.holdout import SealedHoldoutGuard


def _make_guard(tmp_path: Path) -> SealedHoldoutGuard:
    return SealedHoldoutGuard(registry_path=tmp_path / "sealed_holdout_registry.json")


def test_01_successful_completion_retried_returns_original_result(tmp_path: Path) -> None:
    """1. Successful completion retried with the same evaluation ID returns the original successful result."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 1), date(2026, 8, 2)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-01")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-01")

    first = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-35-01",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=dates,
        strategy_ids=["STRAT-018", "STRAT-019"],
        dataset_fingerprint="fp-35-01",
    )
    assert first["ok"] is True
    assert first["completion_success"] is True
    assert first["holdout_consumed"] is True
    assert first["holdout_state"] == "EXPOSED"

    # Simulated caller retry with identical parameters (even if strategy_ids/dates order differs)
    retry = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-35-01",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=list(reversed(dates)),
        strategy_ids=["STRAT-019", "STRAT-018"],
        dataset_fingerprint="fp-35-01",
    )
    assert retry["ok"] is True
    assert retry["completion_success"] is True
    assert retry["holdout_consumed"] is True
    assert retry["holdout_state"] == "EXPOSED"
    assert retry["completed_at"] == first["completed_at"]
    assert retry["idempotent_replay"] is True


def test_02_failed_completion_retried_returns_original_failed_result(tmp_path: Path) -> None:
    """2. Failed completion retried with the same evaluation ID returns the original failed result."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 3)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-good")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-good")

    first = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-35-02",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=dates,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-bad",
    )
    assert first["ok"] is False
    assert first["completion_success"] is False
    assert first["holdout_consumed"] is True
    assert first["holdout_state"] == "EXPOSED"
    assert first["status"] == "DATASET_FINGERPRINT_MISMATCH"

    retry = guard.complete_evaluation(
        cycle_id="CYCLE-004",
        evaluation_id="eval-35-02",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=dates,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-bad",
    )
    assert retry["ok"] is False
    assert retry["completion_success"] is False
    assert retry["holdout_consumed"] is True
    assert retry["holdout_state"] == "EXPOSED"
    assert retry["status"] == "DATASET_FINGERPRINT_MISMATCH"
    assert retry["failure_reasons"] == first["failure_reasons"]
    assert retry["completed_at"] == first["completed_at"]
    assert retry["idempotent_replay"] is True


def test_03_retrying_successful_evaluation_does_not_append_failed_history(tmp_path: Path) -> None:
    """3. Retrying a successful evaluation does not append a failed history record."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 4)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-03")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-03")

    guard.complete_evaluation("CYCLE-004", "eval-35-03", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-03")
    guard.complete_evaluation("CYCLE-004", "eval-35-03", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-03")

    reg = guard.load_registry()
    hist = reg["evaluation_history"]
    assert len(hist) == 1
    assert hist[0]["completion_success"] is True
    assert hist[0]["ok"] is True


def test_04_retrying_failed_evaluation_does_not_append_another_history_record(tmp_path: Path) -> None:
    """4. Retrying a failed evaluation does not append another history record."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 5)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-04")
    # Not marked in progress -> fails lifecycle validation
    guard.complete_evaluation("CYCLE-004", "eval-35-04", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-04")
    guard.complete_evaluation("CYCLE-004", "eval-35-04", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-04")

    reg = guard.load_registry()
    hist = reg["evaluation_history"]
    assert len(hist) == 1
    assert hist[0]["completion_success"] is False
    assert reg["records"]["2026-08-05"]["state"] == "SEALED"


def test_05_retrying_does_not_increment_exposure_counts(tmp_path: Path) -> None:
    """5. Retrying does not increment previous_exposure_count."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 6)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-05")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-05")

    guard.complete_evaluation("CYCLE-004", "eval-35-05", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-05")
    guard.complete_evaluation("CYCLE-004", "eval-35-05", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-05")
    guard.complete_evaluation("CYCLE-004", "eval-35-05", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-05")

    reg = guard.load_registry()
    assert reg["records"]["2026-08-06"]["previous_exposure_count"] == 1


def test_06_retrying_does_not_append_duplicate_sealed_exposures(tmp_path: Path) -> None:
    """6. Retrying does not append duplicate sealed_exposures."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 7)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-06")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-06")

    guard.complete_evaluation("CYCLE-004", "eval-35-06", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-06")
    guard.complete_evaluation("CYCLE-004", "eval-35-06", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-06")

    reg = guard.load_registry()
    assert len(reg["sealed_exposures"]) == 1


def test_07_reusing_evaluation_id_with_different_dates_is_rejected(tmp_path: Path) -> None:
    """7. Reusing an evaluation ID with different dates is rejected."""
    guard = _make_guard(tmp_path)
    d1 = [date(2026, 8, 8)]
    d2 = [date(2026, 8, 9)]
    guard.seal_dates("CYCLE-004", d1 + d2, dataset_fingerprint="fp-35-07")
    guard.mark_evaluation_in_progress("CYCLE-004", d1 + d2, dataset_fingerprint="fp-35-07")

    first = guard.complete_evaluation("CYCLE-004", "eval-35-07", "FORMAL_QUALIFICATION", d1, ["S1"], "fp-35-07")
    assert first["ok"] is True

    conflict = guard.complete_evaluation("CYCLE-004", "eval-35-07", "FORMAL_QUALIFICATION", d2, ["S1"], "fp-35-07")
    assert conflict["ok"] is False
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert "consumed_dates" in conflict["conflict_fields"]


def test_08_reusing_evaluation_id_with_different_cycle_id_is_rejected(tmp_path: Path) -> None:
    """8. Reusing an evaluation ID with a different cycle ID is rejected."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 10)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-08")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-08")

    first = guard.complete_evaluation("CYCLE-004", "eval-35-08", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-08")
    assert first["ok"] is True

    conflict = guard.complete_evaluation("CYCLE-005", "eval-35-08", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-08")
    assert conflict["ok"] is False
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert "cycle_id" in conflict["conflict_fields"]


def test_09_reusing_evaluation_id_with_different_fingerprint_is_rejected(tmp_path: Path) -> None:
    """9. Reusing an evaluation ID with a different fingerprint is rejected."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 11)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-09")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-09")

    first = guard.complete_evaluation("CYCLE-004", "eval-35-09", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-09")
    assert first["ok"] is True

    conflict = guard.complete_evaluation("CYCLE-004", "eval-35-09", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-different")
    assert conflict["ok"] is False
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert "dataset_fingerprint" in conflict["conflict_fields"]


def test_10_reusing_evaluation_id_with_different_evaluation_type_is_rejected(tmp_path: Path) -> None:
    """10. Reusing an evaluation ID with a different evaluation type is rejected."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 8, 12)]
    guard.seal_dates("CYCLE-004", dates, dataset_fingerprint="fp-35-10")
    guard.mark_evaluation_in_progress("CYCLE-004", dates, dataset_fingerprint="fp-35-10")

    first = guard.complete_evaluation("CYCLE-004", "eval-35-10", "FORMAL_QUALIFICATION", dates, ["S1"], "fp-35-10")
    assert first["ok"] is True

    conflict = guard.complete_evaluation("CYCLE-004", "eval-35-10", "EXPLORATORY_WITH_UNSEEN", dates, ["S1"], "fp-35-10")
    assert conflict["ok"] is False
    assert conflict["status"] == "EVALUATION_ID_CONFLICT"
    assert "evaluation_type" in conflict["conflict_fields"]


def test_11_conflicting_retry_does_not_mutate_registry(tmp_path: Path) -> None:
    """11. A conflicting retry (including different strategy_ids) does not mutate the registry."""
    guard = _make_guard(tmp_path)
    d1 = [date(2026, 8, 13)]
    d2 = [date(2026, 8, 14)]
    guard.seal_dates("CYCLE-004", d1, dataset_fingerprint="fp-35-11")
    guard.mark_evaluation_in_progress("CYCLE-004", d1, dataset_fingerprint="fp-35-11")
    guard.seal_dates("CYCLE-004", d2, dataset_fingerprint="fp-35-11")
    guard.mark_evaluation_in_progress("CYCLE-004", d2, dataset_fingerprint="fp-35-11")

    guard.complete_evaluation("CYCLE-004", "eval-35-11", "FORMAL_QUALIFICATION", d1, ["S1"], "fp-35-11")
    before_bytes = guard.registry_path.read_text(encoding="utf-8")

    # Conflict on dates
    c1 = guard.complete_evaluation("CYCLE-004", "eval-35-11", "FORMAL_QUALIFICATION", d2, ["S1"], "fp-35-11")
    assert c1["status"] == "EVALUATION_ID_CONFLICT"
    # Conflict on strategy_ids
    c2 = guard.complete_evaluation("CYCLE-004", "eval-35-11", "FORMAL_QUALIFICATION", d1, ["S2"], "fp-35-11")
    assert c2["status"] == "EVALUATION_ID_CONFLICT"
    assert "strategy_ids" in c2["conflict_fields"]

    after_bytes = guard.registry_path.read_text(encoding="utf-8")
    assert json.loads(after_bytes) == json.loads(before_bytes)
    assert guard.load_registry()["records"]["2026-08-14"]["state"] == "SEALED"


def test_12_legacy_history_record_replays_cleanly(tmp_path: Path) -> None:
    """12. Existing historical evaluation_history records without new replay fields still replay cleanly."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 8, 15)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-35-12")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-35-12")
    guard.complete_evaluation("CYCLE-004", "eval-legacy-12", "FORMAL_QUALIFICATION", d, ["S1"], "fp-35-12")

    # Strip new replay fields from persisted history to simulate a legacy Phase 11.3.3 record
    reg = guard.load_registry()
    hist = reg["evaluation_history"][0]
    for k in ("holdout_state", "holdout_consumed", "exposed_dates", "status"):
        hist.pop(k, None)
    guard.save_registry(reg)

    replay = guard.complete_evaluation("CYCLE-004", "eval-legacy-12", "FORMAL_QUALIFICATION", d, ["S1"], "fp-35-12")
    assert replay["ok"] is True
    assert replay["completion_success"] is True
    assert replay["holdout_consumed"] is True
    assert replay["holdout_state"] == "EXPOSED"
    assert replay["exposed_dates"] == ["2026-08-15"]
    assert replay["idempotent_replay"] is True
