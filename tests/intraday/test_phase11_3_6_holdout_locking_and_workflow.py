"""Phase 11.3.6 Tests: Holdout Registry Concurrency, Locking & Legacy Weekly Workflow Audit."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_weekly_research import run_weekly_audit
from trader.research.holdout import SealedHoldoutGuard


def _make_guard(tmp_path: Path) -> SealedHoldoutGuard:
    return SealedHoldoutGuard(registry_path=tmp_path / "sealed_holdout_registry.json")


def test_01_concurrent_record_exposure_preserves_all_entries_and_counts(tmp_path: Path) -> None:
    """1. Concurrent record_exposure() calls under _locked_registry() never lose sealed_exposures or counts."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 10, 1) + timedelta(days=i) for i in range(6)]

    def _expose(idx: int) -> dict:
        return guard.record_exposure(
            cycle_id=f"CYCLE-CONC-{idx}",
            unseen_dates=[dates[idx]],
            notes=f"Concurrent exposure {idx}",
            dataset_fingerprint=f"fp-{idx}",
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_expose, range(6)))

    assert all(r["independent_holdout_valid"] is True for r in results)
    reg = guard.load_registry()
    assert len(reg["sealed_exposures"]) == 6
    assert len(reg["records"]) == 6
    for d in dates:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"
        assert reg["records"][d.isoformat()]["previous_exposure_count"] == 1


def test_02_concurrent_record_exposure_same_date_increments_atomically(tmp_path: Path) -> None:
    """2. Concurrent record_exposure() calls on the same date never lose exposure count increments."""
    guard = _make_guard(tmp_path)
    target = [date(2026, 10, 10)]

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda i: guard.record_exposure(f"C-{i}", target, notes=f"n-{i}"), range(5)))

    reg = guard.load_registry()
    assert len(reg["sealed_exposures"]) == 5
    assert reg["records"]["2026-10-10"]["previous_exposure_count"] == 5


def test_03_concurrent_retire_dates_preserves_all_retired_dates(tmp_path: Path) -> None:
    """3. Concurrent retire_dates() calls under _locked_registry() never lose retired dates."""
    guard = _make_guard(tmp_path)
    dates = [date(2026, 10, 15) + timedelta(days=i) for i in range(6)]

    with ThreadPoolExecutor(max_workers=6) as pool:
        counts = list(pool.map(lambda i: guard.retire_dates([dates[i]], f"Reason-{i}"), range(6)))

    assert all(c == 1 for c in counts)
    reg = guard.load_registry()
    for d in dates:
        assert reg["records"][d.isoformat()]["state"] == "RETIRED"

    # Retired dates cannot be sealed or reused
    seal = guard.seal_dates("CYCLE-006", dates)
    assert seal["sealed_count"] == 0
    assert len(seal["blocked_dates"]) == 6


def test_04_concurrent_failure_recovery_and_completion_stay_consistent(tmp_path: Path) -> None:
    """4. Concurrent mark_evaluation_failed() and complete_evaluation() on disjoint cycles never corrupt registry."""
    guard = _make_guard(tmp_path)
    d_ok = [date(2026, 11, 1)]
    d_fail = [date(2026, 11, 2)]

    guard.seal_dates("CYCLE-A", d_ok, dataset_fingerprint="fp-a")
    guard.seal_dates("CYCLE-B", d_fail, dataset_fingerprint="fp-b")
    guard.mark_evaluation_in_progress("CYCLE-A", d_ok, dataset_fingerprint="fp-a")
    guard.mark_evaluation_in_progress("CYCLE-B", d_fail, dataset_fingerprint="fp-b")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_ok = pool.submit(
            guard.complete_evaluation, "CYCLE-A", "eval-a", "FORMAL_QUALIFICATION", d_ok, ["S1"], "fp-a"
        )
        f_fail = pool.submit(
            guard.mark_evaluation_failed, "CYCLE-B", d_fail, "Simulated crash", True
        )
        res_ok = f_ok.result()
        res_fail = f_fail.result()

    assert res_ok["ok"] is True
    assert res_fail["holdout_consumed"] is True

    reg = guard.load_registry()
    assert reg["records"]["2026-11-01"]["state"] == "EXPOSED"
    assert reg["records"]["2026-11-01"]["reservation_status"] == "EVALUATION_COMPLETED"
    assert reg["records"]["2026-11-02"]["state"] == "EXPOSED"
    assert reg["records"]["2026-11-02"]["reservation_status"] == "FAILED_EVALUATION"


def test_05_save_registry_cleans_temp_file_on_failure(tmp_path: Path) -> None:
    """5. Atomic save_registry() cleans up temp files and preserves prior registry on write/replace error."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 11, 5)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-5")
    before = guard.registry_path.read_text(encoding="utf-8")

    with patch("os.replace", side_effect=OSError("Simulated replace failure")):
        with pytest.raises(OSError):
            guard.save_registry(guard.load_registry())

    assert guard.registry_path.read_text(encoding="utf-8") == before
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_06_weekly_audit_is_read_only_and_preserves_registry(tmp_path: Path) -> None:
    """6. Weekly readiness audit never mutates sealed_holdout_registry.json or calls record_exposure."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    guard = SealedHoldoutGuard(registry_path=reg_path)
    guard.save_registry(guard.load_registry())
    before_bytes = reg_path.read_text(encoding="utf-8")

    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(80)]
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-weekly",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    with (
        patch("scripts.run_weekly_research.get_available_1h_dates", return_value=(dates, [])),
        patch("scripts.run_weekly_research.validate_1h_provenance", return_value=(False, fake_prov)),
    ):
        code, status = run_weekly_audit(
            run_live_pytest=False,
            registry_path=reg_path,
            status_path=tmp_path / "weekly_research_status.json",
            history_dir=tmp_path / "history",
        )

    assert code == 0
    assert status["mode"] == "READ_ONLY_PHASE11_3_AUDIT"
    assert status["qualification_ready"] is False
    assert status["status"] == "AUDIT_BLOCKED_INTEGRITY"
    assert reg_path.read_text(encoding="utf-8") == before_bytes


def test_07_weekly_workflow_yaml_has_no_push_to_main() -> None:
    """7. weekly-research-cycle.yml uses contents: read and does not commit/push to main."""
    wf_path = Path(".github/workflows/weekly-research-cycle.yml")
    content = wf_path.read_text(encoding="utf-8")
    assert "contents: read" in content
    assert "contents: write" not in content
    assert "git push" not in content
    assert "git commit" not in content
