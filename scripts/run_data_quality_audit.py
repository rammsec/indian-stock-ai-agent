"""Run the independent Data Quality Audit (read-only against production data)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.audit.data_quality_audit import run_full_data_quality_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="BharatAlpha AI Independent Data Quality Audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write structured JSON audit report (never overwrites production data).",
    )
    args = parser.parse_args()

    report = run_full_data_quality_audit()
    payload_json = report.model_dump_json(indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload_json, encoding="utf-8")
        print(f"Audit report written to {args.output}")
    else:
        print(payload_json)

    print(
        f"\n[SUMMARY] status={report.overall_status} | inspected={report.total_records_inspected} | "
        f"valid={report.valid_records} | invalid={report.invalid_records} | "
        f"duplicates={report.duplicate_records} | missing={report.missing_data} | "
        f"quarantined={report.quarantined_records} | warnings={len(report.warnings)} | "
        f"critical_failures={len(report.critical_failures)}",
        file=sys.stderr,
    )
    return 1 if report.critical_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
