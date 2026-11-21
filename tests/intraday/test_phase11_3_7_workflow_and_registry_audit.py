"""Phase 11.3.7 Tests: Workflow Push-to-Main Audit, Concurrency, Provenance & Holdout Safety."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from trader.research.holdout import SealedHoldoutGuard
from trader.research.integrity_guard import ResearchIntegrityGuard


def test_01_audited_workflows_have_no_push_to_main_and_enforce_read_only() -> None:
    """1. Audited workflows enforce contents: read, have no git push/commit, and isolate concurrent runs."""
    for wf_name in ("daily_nse_research.yml", "daily-5m-archive.yml", "intraday-lab-tests.yml"):
        content = Path(".github/workflows", wf_name).read_text(encoding="utf-8")
        assert "contents: read" in content, f"{wf_name} missing contents: read"
        assert "contents: write" not in content, f"{wf_name} must not grant contents: write"
        assert "git push" not in content, f"{wf_name} must not push directly to main"
        assert "git commit" not in content, f"{wf_name} must not commit directly to main"
        if wf_name != "intraday-lab-tests.yml":
            assert "concurrency:" in content, f"{wf_name} missing concurrency group"


def test_02_sealed_exposed_retired_dates_blocked_from_training_tuning_validation(tmp_path: Path) -> None:
    """2. SEALED, EXPOSED, and RETIRED dates can never be used in training, tuning, or validation."""
    guard = SealedHoldoutGuard(registry_path=tmp_path / "reg.json")
    d_sealed = date(2026, 11, 1)
    d_exposed = date(2026, 11, 2)
    d_retired = date(2026, 11, 3)
    d_avail_1 = date(2026, 11, 4)
    d_avail_2 = date(2026, 11, 5)
    d_avail_3 = date(2026, 11, 6)

    guard.seal_dates("CYCLE-X", [d_sealed], dataset_fingerprint="fp")
    guard.seal_dates("CYCLE-Y", [d_exposed], dataset_fingerprint="fp")
    guard.mark_evaluation_in_progress("CYCLE-Y", [d_exposed], dataset_fingerprint="fp")
    guard.complete_evaluation("CYCLE-Y", "eval-y", "FORMAL_QUALIFICATION", [d_exposed], ["S1"], "fp")
    guard.retire_dates([d_retired], "Corrupted")

    leakage = guard.check_tuning_leakage([d_sealed, d_exposed, d_retired, d_avail_1])
    assert set(leakage) == {d_sealed.isoformat(), d_exposed.isoformat(), d_retired.isoformat()}

    for bad in (d_sealed, d_exposed, d_retired):
        res_train = guard.validate_cycle_splits([bad], [d_avail_1], [d_avail_2], cycle_id="CYCLE-Z")
        assert res_train["independent_holdout_valid"] is False
        assert res_train["leakage_detected"] is True

        res_val = guard.validate_cycle_splits([d_avail_1], [bad], [d_avail_3], cycle_id="CYCLE-Z")
        assert res_val["independent_holdout_valid"] is False
        assert res_val["leakage_detected"] is True


def test_03_provenance_guard_fails_closed_for_all_non_verified_statuses(tmp_path: Path) -> None:
    """3. ResearchIntegrityGuard fails closed unless provenance_status == VERIFIED and verified is True."""
    ig = ResearchIntegrityGuard(registry_path=tmp_path / "reg.json")
    d_t = [date(2026, 11, 10)]
    d_v = [date(2026, 11, 11)]
    d_u = [date(2026, 11, 12)]

    for prov in (
        None,
        {},
        {"final_provenance_status": "PARTIALLY_VERIFIED", "verified": False},
        {"final_provenance_status": "FAILED", "verified": False},
        {"final_provenance_status": "VERIFIED", "verified": False},
        {"final_provenance_status": "PARTIALLY_VERIFIED", "verified": True},
    ):
        res = ig.validate("CYCLE-007", d_t, d_v, d_u, "fp", provenance_result=prov)
        assert res["integrity_passed"] is False
        assert res["qualification_allowed"] is False


def test_04_failed_evaluations_never_leave_accessed_dates_available_or_stuck_in_progress(tmp_path: Path) -> None:
    """4. Failed evaluations conservatively expose accessed dates and stale recovery clears crashed runs."""
    guard = SealedHoldoutGuard(registry_path=tmp_path / "reg.json")
    d1 = [date(2026, 11, 20)]
    d2 = [date(2026, 11, 21)]

    # Case A: failure after data access via mark_evaluation_failed
    guard.seal_dates("CYCLE-A", d1, dataset_fingerprint="fp-a")
    guard.mark_evaluation_in_progress("CYCLE-A", d1, dataset_fingerprint="fp-a")
    guard.mark_evaluation_failed("CYCLE-A", d1, reason="Crash", data_accessed=True)
    assert guard.get_date_state(d1[0]) == "EXPOSED"

    # Case B: process crash during EVALUATION_IN_PROGRESS recovered on next startup
    guard.seal_dates("CYCLE-B", d2, dataset_fingerprint="fp-b")
    guard.mark_evaluation_in_progress("CYCLE-B", d2, dataset_fingerprint="fp-b")
    rec = guard.recover_stale_reservations("CYCLE-B")
    assert rec["recovered_dates"] == ["2026-11-21"]
    assert guard.get_date_state(d2[0]) == "EXPOSED"


def test_05_concurrent_reads_during_writes_never_see_corrupt_registry(tmp_path: Path) -> None:
    """5. Concurrent seal/complete writes and validate_cycle_splits reads never observe partial JSON."""
    guard = SealedHoldoutGuard(registry_path=tmp_path / "reg.json")
    dates = [date(2026, 12, 1) + timedelta(days=i) for i in range(10)]

    def _writer(i: int) -> None:
        d = [dates[i]]
        guard.seal_dates(f"C-{i}", d, dataset_fingerprint="fp")
        guard.mark_evaluation_in_progress(f"C-{i}", d, dataset_fingerprint="fp")
        guard.complete_evaluation(f"C-{i}", f"eval-{i}", "FORMAL_QUALIFICATION", d, ["S1"], "fp")

    def _reader(i: int) -> dict:
        return guard.validate_cycle_splits(
            [date(2026, 1, 1)], [date(2026, 1, 2)], [dates[i % 10]], cycle_id="READER", dataset_fingerprint="fp"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        write_futs = [pool.submit(_writer, i) for i in range(10)]
        read_futs = [pool.submit(_reader, i) for i in range(20)]
        for f in write_futs:
            f.result()
        for f in read_futs:
            r = f.result()
            assert r["status"] != "REGISTRY_CORRUPTED"

    reg = guard.load_registry()
    for d in dates:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"
