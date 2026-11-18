"""CLI runner for the BharatAlpha AI Data Integrity Audit Engine (read-only)."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.audit.data_integrity import run_data_integrity_audit
from backend.storage.duckdb_store import DuckDBStore


def _load_reference_outcomes(rev: str | None) -> pd.DataFrame | None:
    if not rev:
        return None
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{rev}:data/predictions/forward_outcomes.parquet"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        return pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:
        print(f"[WARN] Could not load reference outcomes from git rev {rev!r}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BharatAlpha AI read-only data integrity audit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "DATA_QUALITY_REPORT.json",
        help="Primary output JSON report path (must not target production data directories).",
    )
    parser.add_argument(
        "--legacy-output",
        type=Path,
        default=ROOT / "reports" / "data_integrity_audit.json",
        help="Secondary output JSON path for backward compatibility.",
    )
    parser.add_argument(
        "--reference-rev",
        type=str,
        default="efa8f3c",
        help="Optional git revision to compare finalized MATURED outcomes against (pass '' to skip).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with code 1 if overall_status is FAIL.",
    )
    args = parser.parse_args()

    ref_df = _load_reference_outcomes(args.reference_rev.strip() or None)
    report = run_data_integrity_audit(
        store=DuckDBStore(),
        docs_data_dir=ROOT / "docs" / "data",
        reference_outcomes_df=ref_df,
    )

    payload_txt = json.dumps(report.model_dump(), indent=2, default=str) + "\n"
    for target in (args.output, args.legacy_output):
        if target is None:
            continue
        out_path = target if target.is_absolute() else (ROOT / target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload_txt, encoding="utf-8")

    print(
        f"Overall status: {report.overall_status} | "
        f"CRITICAL={report.critical_findings_count} HIGH={report.high_findings_count} "
        f"MEDIUM={report.medium_findings_count} LOW={report.low_findings_count} -> {args.output}"
    )
    for ds_name, ds in report.datasets.items():
        print(
            f"  - {ds_name}: status={ds.status} inspected={ds.records_inspected} "
            f"passed={ds.passed_checks} failed={ds.failed_checks} "
            f"errors={len(ds.errors)} warnings={len(ds.warnings)}"
        )
    for h, s in report.classification.horizon_summaries.items():
        m = report.classification.prospective_metrics[h]
        print(
            f"  [Class {h}] total={s.total_stored} canonical={s.unique_canonical} "
            f"prospective={s.prospective} (matured={s.matured}, pending={s.pending}) "
            f"historical={s.historical} invalid={s.invalid} duplicate={s.duplicate} "
            f"conflicts={s.conflicting_records} | "
            f"prospective_acc={m.directional_accuracy_pct}% (n={m.n_matured_prospective})"
        )
    if report.classification.news_summary:
        ns = report.classification.news_summary
        print(
            f"  [News] total={ns.total_stored} prospective={ns.prospective} "
            f"historical={ns.historical} duplicate={ns.duplicate} invalid={ns.invalid} "
            f"leak_conflicts={ns.collection_leakage_conflicts}"
        )

    if args.fail_on_error and report.overall_status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
