"""Weekly Automated Research & Qualification Runner (Phase 11.1 & CYCLE-002)."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.lab.data_service import filter_research_eligible_dates, get_available_historical_dates
from trader.research.orchestrator import (
    LAB_INDEX_PATH,
    ChiefResearchOrchestrator,
    collect_dynamic_test_metrics,
    verify_dataset_provenance,
)

WEEKLY_STATUS_PATH = Path("data/lab/weekly_research_status.json")
HISTORY_DIR = Path("data/lab/history")


def main() -> int:
    now_ist = datetime.now(tz=IST_TZ)
    print(f"=== Weekly Automated Research Runner ({now_ist.isoformat()}) ===")

    # 1. Run dynamic pytest suite first
    test_metrics = collect_dynamic_test_metrics(run_pytest=True)
    print(
        f"Pytest check: executed={test_metrics['tests_executed']}, "
        f"passed={test_metrics['tests_passed']}, failed={test_metrics['tests_failed']}"
    )
    if test_metrics["tests_executed"] == 0 or test_metrics["tests_failed"] > 0:
        status_payload = {
            "last_run_ist": now_ist.isoformat(),
            "status": "ABORTED_TEST_FAILURE",
            "test_metrics": test_metrics,
        }
        WEEKLY_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        WEEKLY_STATUS_PATH.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        print("ABORTING: Critical test check failed.", file=sys.stderr)
        return 1

    # 2. Verify data quality & provenance before executing research cycle
    avail_dates = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=150)
    eligible_objs = filter_research_eligible_dates(avail_dates, min_bars_per_stock=70)
    ordered_dates = sorted(date.fromisoformat(d.date) for d in eligible_objs)
    prov_ok, prov_report = verify_dataset_provenance(ordered_dates, LAB_DEFAULT_UNIVERSE)
    if not prov_ok:
        status_payload = {
            "last_run_ist": now_ist.isoformat(),
            "status": "ABORTED_PROVENANCE_FAILURE",
            "provenance_report": prov_report,
        }
        WEEKLY_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        WEEKLY_STATUS_PATH.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        print("ABORTING: Dataset provenance verification failed.", file=sys.stderr)
        return 1

    # 3. Execute CYCLE-002
    orch = ChiefResearchOrchestrator()
    report = orch.run_cycle_002(min_bars_for_research=70, test_report=test_metrics, run_live_pytest=False)

    # 4. Preserve historical archive of this weekly run
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_file = HISTORY_DIR / f"{report['cycle_id']}_{now_ist.strftime('%Y%m%d_%H%M%S')}.json"
    hist_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    weekly_status = {
        "last_run_ist": now_ist.isoformat(),
        "status": "COMPLETED",
        "cycle_id": report["cycle_id"],
        "reconciled_eligible_sessions": report["reconciled_eligible_sessions"],
        "qualified_strategies": report["command_centre"]["qualified_strategies"],
        "best_strategy": report["command_centre"]["best_experimental_strategy"],
        "best_net_return_pct": report["command_centre"]["best_experimental_net_pct"],
        "provenance_verified": report["provenance_verified"],
        "test_metrics": test_metrics,
        "history_snapshot": str(hist_file),
    }
    WEEKLY_STATUS_PATH.write_text(json.dumps(weekly_status, indent=2), encoding="utf-8")

    if LAB_INDEX_PATH.exists():
        idx = json.loads(LAB_INDEX_PATH.read_text(encoding="utf-8"))
        idx["weekly_research_status"] = weekly_status
        LAB_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")

    print(
        f"Weekly research complete: {report['cycle_id']} | "
        f"sessions={report['reconciled_eligible_sessions']} | "
        f"qualified={report['command_centre']['qualified_strategies']} | "
        f"best={report['command_centre']['best_experimental_strategy']} ({report['command_centre']['best_experimental_net_pct']:+.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
