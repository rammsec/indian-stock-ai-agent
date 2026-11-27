"""Phase 11.3.1 Mandatory Regression Tests (28 Scenarios): Holdout Sequencing, Recovery, Provenance & Central Guard."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, is_nse_trading_day
from trader.data.provenance_1h import validate_1h_provenance
from trader.research.holdout import SealedHoldoutGuard
from trader.research.integrity_guard import ResearchIntegrityGuard
from trader.research.orchestrator import (
    CYCLE_003_PHASE11_3_REASSESSMENT_PATH,
    CYCLE_003_REASSESSMENT_PATH,
    CYCLE_003_REPORT_PATH,
    ChiefResearchOrchestrator,
)
from trader.research.qualification import evaluate_seven_gates


def _write_1h(cache_dir: Path, days: list[date], bars_per_day: int = 7) -> None:
    for sym in LAB_DEFAULT_UNIVERSE:
        rows = []
        for idx_d, d in enumerate(days):
            start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST_TZ)
            cnt = bars_per_day if isinstance(bars_per_day, int) else bars_per_day(sym, idx_d)
            for i in range(cnt):
                rows.append({
                    "timestamp": start + timedelta(hours=i),
                    "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000,
                })
        pd.DataFrame(rows).to_parquet(cache_dir / f"{sym}_1h.parquet", index=False)


# ---------------------------------------------------------------------------
# Holdout Lifecycle Tests (Scenarios 1 - 14)
# ---------------------------------------------------------------------------

def test_01_02_03_cycle004_seals_and_validates_own_dates_but_blocks_other_cycle(tmp_path: Path) -> None:
    reg = tmp_path / "holdout.json"
    hg = SealedHoldoutGuard(registry_path=reg)
    guard = ResearchIntegrityGuard(registry_path=reg)
    d_train = [date(2026, 11, 9), date(2026, 11, 10)]
    d_val = [date(2026, 11, 11)]
    d_unseen = [date(2026, 11, 12), date(2026, 11, 13)]

    # 1. New dates can be sealed by CYCLE-004
    seal = hg.seal_dates("CYCLE-004", d_unseen, dataset_fingerprint="fp_ok")
    assert seal["ownership_valid"] is True
    assert hg.get_date_state(d_unseen[0]) == "SEALED"

    # 2. CYCLE-004 can validate its own sealed dates
    res_own = guard.validate("CYCLE-004", d_train, d_val, d_unseen, "fp_ok", {"final_provenance_status": "VERIFIED", "verified": True})
    assert res_own["integrity_passed"] is True
    assert res_own["qualification_allowed"] is True

    # 3. CYCLE-005 cannot validate dates owned by CYCLE-004
    res_other = guard.validate("CYCLE-005", d_train, d_val, d_unseen, "fp_ok", {"final_provenance_status": "VERIFIED", "verified": True})
    assert res_other["integrity_passed"] is False
    assert res_other["ownership_valid"] is False
    assert res_other["holdout_status"] == "HOLDOUT_OWNERSHIP_VIOLATION"


def test_04_05_06_07_exposed_retired_duplicate_and_overlap_rejected(tmp_path: Path) -> None:
    reg = tmp_path / "holdout.json"
    hg = SealedHoldoutGuard(registry_path=reg)
    guard = ResearchIntegrityGuard(registry_path=reg)
    d_train = [date(2026, 11, 9)]
    d_val = [date(2026, 11, 10)]
    d_unseen = [date(2026, 11, 12)]
    d_ret = [date(2026, 11, 13)]

    # 4. Previously exposed dates are rejected
    hg.record_exposure("CYCLE-003", d_unseen, dataset_fingerprint="fp_ok")
    res_exp = guard.validate("CYCLE-004", d_train, d_val, d_unseen, "fp_ok", {"final_provenance_status": "VERIFIED", "verified": True})
    assert res_exp["integrity_passed"] is False
    assert res_exp["holdout_status"] == "EXPOSED_HOLDOUT_REUSED"

    # 5. Retired dates are rejected
    hg.retire_dates(d_ret, "Retired window")
    res_ret = guard.validate("CYCLE-004", d_train, d_val, d_ret, "fp_ok", {"final_provenance_status": "VERIFIED", "verified": True})
    assert res_ret["integrity_passed"] is False

    # 6. Duplicate reservations are rejected
    d_fresh = [date(2026, 11, 16)]
    assert hg.seal_dates("CYCLE-004", d_fresh, "fp_ok")["ownership_valid"] is True
    dup_seal = hg.seal_dates("CYCLE-004", d_fresh, "fp_ok")
    assert dup_seal["ownership_valid"] is False
    assert dup_seal["sealed_count"] == 0

    # 7. Split overlap is rejected
    res_ov = guard.validate("CYCLE-004", d_train, [d_train[0]], d_fresh, "fp_ok", {"final_provenance_status": "VERIFIED", "verified": True})
    assert res_ov["split_valid"] is False
    assert res_ov["qualification_allowed"] is False


def test_08_09_10_11_12_13_14_sequencing_recovery_concurrency_and_corruption(tmp_path: Path) -> None:
    reg = tmp_path / "holdout.json"
    hg = SealedHoldoutGuard(registry_path=reg)
    guard = ResearchIntegrityGuard(registry_path=reg)
    d_fresh = [date(2026, 11, 16), date(2026, 11, 17)]
    d_train = [date(2026, 11, 10)]
    d_val = [date(2026, 11, 11)]

    # 8 & 9. Failed integrity validation does not expose dates; exposure happens only after evaluation
    hg.seal_dates("CYCLE-004", d_fresh, "fp1")
    res_bad = guard.validate("CYCLE-004", d_train, d_val, d_fresh, "fp1", {"final_provenance_status": "PARTIALLY_VERIFIED", "verified": False})
    assert res_bad["qualification_allowed"] is False
    assert hg.get_date_state(d_fresh[0]) == "SEALED"  # Still SEALED, not EXPOSED!

    # 10. Failed evaluation before data access preserves SEALED
    hg.mark_evaluation_failed("CYCLE-004", d_fresh, "Pre-access failure", data_accessed=False)
    loaded = hg.load_registry()
    assert loaded["records"]["2026-11-16"]["reservation_status"] == "FAILED_EVALUATION"
    assert loaded["records"]["2026-11-16"]["state"] == "SEALED"

    # 11. Safe release is restricted to the owning cycle
    denied = hg.release_reservation("CYCLE-OTHER", d_fresh, "Unauthorized")
    assert denied["ok"] is False and len(denied["denied"]) == 2
    assert hg.get_date_state(d_fresh[0]) == "SEALED"

    released = hg.release_reservation("CYCLE-004", d_fresh, "Safe recovery after failed run")
    assert released["ok"] is True and len(released["released"]) == 2
    assert hg.get_date_state(d_fresh[0]) == "AVAILABLE"

    # 12. Exposed dates cannot be released
    hg.seal_dates("CYCLE-004", d_fresh, "fp1")
    hg.record_exposure("CYCLE-004", d_fresh, dataset_fingerprint="fp1")
    rel_exp = hg.release_reservation("CYCLE-004", d_fresh, "Attempt release after exposure")
    assert rel_exp["ok"] is False and len(rel_exp["denied"]) == 2
    assert hg.get_date_state(d_fresh[0]) == "EXPOSED"

    # 13. Concurrent reservation attempts cannot claim the same dates
    d_conc = [date(2026, 11, 18)]
    assert hg.seal_dates("CYCLE-A", d_conc, "fp1")["sealed_count"] == 1
    assert hg.seal_dates("CYCLE-B", d_conc, "fp1")["sealed_count"] == 0

    # 14. Corrupted registry fails closed
    reg.write_text("{invalid_json", encoding="utf-8")
    res_corr = guard.validate("CYCLE-004", d_train, d_val, d_fresh, "fp1", {"final_provenance_status": "VERIFIED", "verified": True})
    assert res_corr["integrity_passed"] is False
    assert res_corr["qualification_allowed"] is False


# ---------------------------------------------------------------------------
# Provenance Tests (Scenarios 15 - 20)
# ---------------------------------------------------------------------------

def test_15_to_20_provenance_statuses_and_no_override(tmp_path: Path) -> None:
    days = [d for d in pd.date_range("2026-08-01", "2026-11-20", freq="D").date if is_nse_trading_day(d)][:60]

    # 15. VERIFIED produces valid provenance
    d_ver = tmp_path / "ver"
    d_ver.mkdir()
    _write_1h(d_ver, days, bars_per_day=7)
    ok_v, rep_v = validate_1h_provenance(LAB_DEFAULT_UNIVERSE, cache_dir=d_ver, min_sessions_required=55)
    assert ok_v is True and rep_v["final_provenance_status"] == "VERIFIED" and rep_v["verified"] is True

    # 16. PARTIALLY_VERIFIED is rejected for qualification
    d_part = tmp_path / "part"
    d_part.mkdir()
    _write_1h(d_part, days, bars_per_day=lambda sym, idx: 4 if (sym == "RELIANCE" and idx == 0) else 7)
    ok_p, rep_p = validate_1h_provenance(LAB_DEFAULT_UNIVERSE, cache_dir=d_part, min_sessions_required=55)
    assert ok_p is False and rep_p["final_provenance_status"] == "PARTIALLY_VERIFIED" and rep_p["verified"] is False

    # 17. INSUFFICIENT_DATA is rejected
    d_ins = tmp_path / "ins"
    d_ins.mkdir()
    _write_1h(d_ins, days[:10], bars_per_day=7)
    ok_i, rep_i = validate_1h_provenance(LAB_DEFAULT_UNIVERSE, cache_dir=d_ins, min_sessions_required=60)
    assert ok_i is False and rep_i["final_provenance_status"] == "INSUFFICIENT_DATA"

    # 18. FAILED is rejected (missing files)
    d_fail = tmp_path / "empty"
    d_fail.mkdir()
    ok_f, rep_f = validate_1h_provenance(LAB_DEFAULT_UNIVERSE, cache_dir=d_fail)
    assert ok_f is False and rep_f["final_provenance_status"] == "FAILED"

    # 19. Missing provenance is rejected by ResearchIntegrityGuard & evaluate_seven_gates
    guard = ResearchIntegrityGuard(registry_path=tmp_path / "h.json")
    g_miss = guard.validate("CYCLE-X", [days[0]], [days[1]], [days[2]], provenance_result=None)
    assert g_miss["qualification_allowed"] is False

    # 20. Hardcoded success cannot override actual validator results
    ev = evaluate_seven_gates(
        "S", "S", {"total_return_pct": 10.0, "net_pnl_inr": 5000.0, "alpha_vs_buy_and_hold_pct": 5.0,
                   "max_drawdown_pct": 2.0, "profit_factor": 2.0, "sharpe_ratio": 1.5, "expectancy_inr": 100.0,
                   "sessions_evaluated": 100, "total_trades": 40},
        {"total_return_pct": 3.0}, {"net_return_pct": 7.0},
        data_provenance_verified=True,
        independent_holdout_valid=True,
        integrity_guard_result=g_miss,
        stat_validation_verdict="PASS", red_team_approved=True, red_team_reason="OK",
    )
    assert ev.final_status == "BLOCKED_INTEGRITY"


# ---------------------------------------------------------------------------
# Central Guard Integration & Immutability Tests (Scenarios 21 - 28)
# ---------------------------------------------------------------------------

def test_21_to_28_central_guard_reassessment_fingerprints_and_immutability(tmp_path: Path) -> None:
    # 27. Preserve original CYCLE-003 reports untouched
    c3_before = CYCLE_003_REPORT_PATH.read_bytes() if CYCLE_003_REPORT_PATH.exists() else None
    c3_11_2_before = CYCLE_003_REASSESSMENT_PATH.read_bytes() if CYCLE_003_REASSESSMENT_PATH.exists() else None
    c3_11_3_before = CYCLE_003_PHASE11_3_REASSESSMENT_PATH.read_bytes() if CYCLE_003_PHASE11_3_REASSESSMENT_PATH.exists() else None

    orch = ChiefResearchOrchestrator()
    # 21. CYCLE-003 reassessment calls central guard (isolated to tmp_path so tracked JSONs stay clean)
    lab_idx_tmp = tmp_path / "lab_index.json"
    tracked_lab_idx = Path("docs/lab/data/lab_index.json")
    if tracked_lab_idx.exists():
        lab_idx_tmp.write_bytes(tracked_lab_idx.read_bytes())
    reass = orch.reassess_cycle_003_phase11_3_1(
        output_path=tmp_path / "cycle_003_phase11_3_1_reassessment.json",
        provenance_report_path=tmp_path / "provenance_report_1h.json",
        lab_index_path=lab_idx_tmp,
    )
    assert reass["guard_version"] == "11.3.1"
    assert reass["provenance_verified"] is False
    assert reass["qualification_allowed"] is False
    assert reass["integrity_guard_result"]["qualification_allowed"] is False

    # Verify immutability of historical reports
    if c3_before is not None:
        assert CYCLE_003_REPORT_PATH.read_bytes() == c3_before
    if c3_11_2_before is not None:
        assert CYCLE_003_REASSESSMENT_PATH.read_bytes() == c3_11_2_before
    if c3_11_3_before is not None:
        assert CYCLE_003_PHASE11_3_REASSESSMENT_PATH.read_bytes() == c3_11_3_before

    # 22 & 23. Central guard blocks blocked CYCLE-004 without exposing dates when allow_exploratory_on_blocked=False
    reg_tmp = tmp_path / "c4_holdout.json"
    blocked_c4 = orch.run_cycle_004(
        test_report={"tests_executed": 1, "tests_passed": 1, "tests_failed": 0, "tests_skipped": 0, "commit_sha": "test"},
        run_live_pytest=False,
        allow_exploratory_on_blocked=False,
        registry_path=reg_tmp,
        output_path=tmp_path / "c4_blocked.json",
    )
    assert blocked_c4["qualification_blocked"] is True
    assert blocked_c4["provenance_verified"] is False
    assert blocked_c4["qualification_allowed"] is False
    assert blocked_c4["integrity_guard_result"]["qualification_allowed"] is False
    # Sealed dates must NOT be EXPOSED when qualification was blocked; aborted pre-access releases to AVAILABLE
    reg_loaded = SealedHoldoutGuard(registry_path=reg_tmp).load_registry()
    assert all(rec["state"] == "AVAILABLE" for rec in reg_loaded["records"].values())

    # 24. Invalid fingerprints block qualification
    hg = SealedHoldoutGuard(registry_path=reg_tmp)
    first_sealed = date.fromisoformat(next(iter(reg_loaded["records"].keys())))
    guard = ResearchIntegrityGuard(registry_path=reg_tmp)
    fp_bad = guard.validate(
        "CYCLE-004", [date(2025, 11, 21)], [date(2025, 11, 24)], [first_sealed],
        dataset_fingerprint="WRONG_FP",
        provenance_result={"final_provenance_status": "VERIFIED", "verified": True},
    )
    assert fp_bad["dataset_fingerprint_valid"] is False
    assert fp_bad["qualification_allowed"] is False

    # 25 & 28. No unsafe defaults permit qualification
    ev_default = evaluate_seven_gates("S", "S", {}, {}, {})
    assert ev_default.final_status in ("BLOCKED_INTEGRITY", "INVALIDATED")

    # 26. Dashboard index reflects actual guard output
    idx = json.loads(lab_idx_tmp.read_text(encoding="utf-8"))
    assert idx["integrity_guard_status"]["qualification_allowed"] is False
    assert idx["integrity_guard_status"]["provenance_verified"] is False
