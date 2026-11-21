"""Weekly Automated Phase 11.3 Readiness & Integrity Audit (Non-Mutating)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.loader_1h import get_available_1h_dates
from trader.data.provenance_1h import validate_1h_provenance
from trader.research.holdout import SEALED_HOLDOUT_REGISTRY_PATH
from trader.research.integrity_guard import ResearchIntegrityGuard
from trader.research.orchestrator import collect_dynamic_test_metrics

WEEKLY_STATUS_PATH = Path("data/lab/weekly_research_status.json")
HISTORY_DIR = Path("data/lab/history")


def run_weekly_audit(
    run_live_pytest: bool = True,
    registry_path: Path = SEALED_HOLDOUT_REGISTRY_PATH,
    status_path: Path = WEEKLY_STATUS_PATH,
    history_dir: Path = HISTORY_DIR,
) -> tuple[int, dict]:
    """Run read-only test, provenance, and holdout readiness checks without mutating historical artifacts."""
    now_ist = datetime.now(tz=IST_TZ)

    # 1. Run dynamic pytest suite first
    test_metrics = collect_dynamic_test_metrics(run_pytest=run_live_pytest)
    if run_live_pytest and (test_metrics["tests_executed"] == 0 or test_metrics["tests_failed"] > 0):
        status_payload = {
            "last_run_ist": now_ist.isoformat(),
            "status": "ABORTED_TEST_FAILURE",
            "qualification_ready": False,
            "test_metrics": test_metrics,
        }
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        return 1, status_payload

    # 2. Validate 1h provenance & holdout readiness (strictly read-only; never calls record_exposure or run_cycle_002)
    elig_1h, excl_1h = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
    n_1h = len(elig_1h)
    t_end = max(1, int(n_1h * 0.60))
    v_end = max(t_end + 1, int(n_1h * 0.80))
    train_dates = elig_1h[:t_end]
    val_dates = elig_1h[t_end:v_end]
    unseen_dates = elig_1h[v_end:] or ([elig_1h[-1]] if elig_1h else [])

    prov_ok, prov_report = validate_1h_provenance(LAB_DEFAULT_UNIVERSE)
    dataset_fp = str(prov_report.get("dataset_fingerprint", ""))

    guard_res = ResearchIntegrityGuard(registry_path=registry_path).validate(
        cycle_id="WEEKLY-READINESS-AUDIT",
        training_dates=train_dates,
        validation_dates=val_dates,
        unseen_dates=unseen_dates,
        dataset_fingerprint=dataset_fp,
        provenance_result=prov_report,
    )

    history_dir.mkdir(parents=True, exist_ok=True)
    hist_file = history_dir / f"weekly_audit_{now_ist.strftime('%Y%m%d_%H%M%S')}.json"

    weekly_status = {
        "last_run_ist": now_ist.isoformat(),
        "status": "AUDIT_READY" if guard_res["qualification_allowed"] else "AUDIT_BLOCKED_INTEGRITY",
        "mode": "READ_ONLY_PHASE11_3_AUDIT",
        "reconciled_eligible_sessions": n_1h,
        "excluded_sessions_count": len(excl_1h),
        "provenance_verified": prov_ok,
        "provenance_status": guard_res["provenance_status"],
        "holdout_status": guard_res["holdout_status"],
        "qualification_ready": bool(guard_res["qualification_allowed"]),
        "failure_reasons": guard_res["failure_reasons"],
        "test_metrics": test_metrics,
        "history_snapshot": str(hist_file),
    }

    hist_file.write_text(json.dumps(weekly_status, indent=2), encoding="utf-8")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(weekly_status, indent=2), encoding="utf-8")
    return 0, weekly_status


def main() -> int:
    print(f"=== Weekly Phase 11.3 Readiness & Integrity Audit ({datetime.now(tz=IST_TZ).isoformat()}) ===")
    code, status = run_weekly_audit(run_live_pytest=True)
    print(
        f"Weekly audit complete: status={status['status']} | "
        f"provenance={status.get('provenance_status')} | "
        f"holdout={status.get('holdout_status')} | "
        f"qualification_ready={status.get('qualification_ready')}"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
