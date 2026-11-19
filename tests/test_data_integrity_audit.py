"""Regression tests for the read-only Data Integrity & Audit module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.audit.data_integrity import (
    DEFAULT_PROSPECTIVE_CUTOFF,
    audit_equities_and_indices,
    audit_forward_predictions_and_outcomes,
    audit_metadata_version_drift,
    audit_paper_trades,
    audit_prediction_log,
    audit_structured_events,
    classify_forward_snapshot_record,
    run_full_data_integrity_audit,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_classify_forward_snapshot_record_lookahead_and_cohorts() -> None:
    # 1. Valid prospective matured (pre-market 03:00 IST before 09:15 IST open)
    valid_matured = classify_forward_snapshot_record(
        observation_date="2026-11-17",
        prediction_timestamp="2026-11-18T03:00:20+05:30",
        is_duplicate_run=False,
        outcome_status="MATURED",
        entry_date="2026-11-18",
    )
    assert valid_matured["cohort"] == "PROSPECTIVE"
    assert valid_matured["identity_status"] == "CANONICAL"
    assert valid_matured["audit_status"] == "MATURED"
    assert valid_matured["lookahead_violation"] is False

    # 2. Look-ahead matured (intraday 10:25 IST after 09:15 IST open on same entry_date)
    lookahead_matured = classify_forward_snapshot_record(
        observation_date="2026-11-16",
        prediction_timestamp="2026-11-17T10:25:29+05:30",
        is_duplicate_run=False,
        outcome_status="MATURED",
        entry_date="2026-11-17",
    )
    assert lookahead_matured["cohort"] == "HISTORICAL"
    assert lookahead_matured["audit_status"] == "INVALID_LOOKAHEAD"
    assert lookahead_matured["lookahead_violation"] is True
    assert lookahead_matured["lookahead_delta_minutes"] == 70.48

    # 3. Look-ahead pending (post-market 17:28 IST on 2026-11-19 with stale obs_date 2026-11-18)
    lookahead_pending = classify_forward_snapshot_record(
        observation_date="2026-11-18",
        prediction_timestamp="2026-11-19T17:28:47+05:30",
        is_duplicate_run=False,
        outcome_status="PENDING",
        entry_date=None,
    )
    assert lookahead_pending["cohort"] == "PROSPECTIVE"
    assert lookahead_pending["audit_status"] == "INVALID_LOOKAHEAD"
    assert lookahead_pending["lookahead_violation"] is True

    # 4. Duplicate + invalid price outcome
    dup_invalid = classify_forward_snapshot_record(
        observation_date="2026-11-17",
        prediction_timestamp="2026-11-18T03:00:00+05:30",
        is_duplicate_run=True,
        outcome_status="INVALID",
        entry_date="2026-11-18",
    )
    assert dup_invalid["identity_status"] == "DUPLICATE"
    assert dup_invalid["audit_status"] == "INVALID"


def test_metadata_version_drift_live_and_clean_fixture(tmp_path: Path) -> None:
    live = audit_metadata_version_drift(REPO_ROOT)
    assert live["has_version_drift"] is True
    assert live["unique_versions_detected"] == ["5.0.0-pilot", "5.3.0-pilot"]
    assert live["severity"] == "HIGH"

    # Synthetic clean repo where all versions align to 5.3.0-pilot
    (tmp_path / "data" / "predictions").mkdir(parents=True)
    (tmp_path / "docs" / "data").mkdir(parents=True)
    for rel in (
        "data/predictions/phase5_intelligence.json",
        "docs/data/phase5_intelligence.json",
        "data/predictions/forward_validation.json",
        "docs/data/forward_validation.json",
    ):
        (tmp_path / rel).write_text(json.dumps({"model_version": "5.3.0-pilot", "generated_at": "2026-11-19T12:00:00+05:30"}))
    pd.DataFrame({"model_version": ["5.3.0-pilot"]}).to_parquet(tmp_path / "data" / "predictions" / "forward_predictions.parquet")

    clean = audit_metadata_version_drift(tmp_path)
    assert clean["has_version_drift"] is False
    assert clean["unique_versions_detected"] == ["5.3.0-pilot"]
    assert clean["severity"] == "OK"


def test_forward_predictions_and_outcomes_live_audit() -> None:
    res = audit_forward_predictions_and_outcomes(REPO_ROOT, prospective_cutoff=DEFAULT_PROSPECTIVE_CUTOFF)
    assert res["total_snapshots"] == 65
    assert res["canonical_snapshots"] == 30
    assert res["duplicate_run_snapshots"] == 35
    assert res["prospective_canonical_snapshots"] == 20
    assert res["historical_canonical_snapshots"] == 10
    assert res["total_outcomes"] == 195
    assert res["derived_audit_status_counts_canonical_1d"] == {
        "INVALID_LOOKAHEAD": 15,
        "MATURED": 10,
        "PENDING": 5,
    }
    assert len(res["canonical_lookahead_runs"]) == 3
    assert res["benchmark_return_pct_non_null"] == 0
    assert res["severity"] == "CRITICAL"


def test_prediction_log_duplicate_contamination_and_clean_fixture(tmp_path: Path) -> None:
    live = audit_prediction_log(REPO_ROOT)
    assert live["total_rows"] == 90
    assert live["duplicate_symbol_observation_date"] == 45
    assert live["evaluated_1d_count"] == 75
    assert live["severity"] == "HIGH"

    (tmp_path / "data" / "predictions").mkdir(parents=True)
    pd.DataFrame(
        {
            "prediction_id": ["p1", "p2"],
            "symbol": ["TCS", "ITC"],
            "observation_date": ["2026-11-18", "2026-11-18"],
            "generated_at": ["2026-11-18T18:00:00+05:30", "2026-11-18T18:00:00+05:30"],
            "realized_next_day_return_pct": [None, None],
            "realized_5d_return_pct": [None, None],
            "realized_10d_return_pct": [None, None],
        }
    ).to_parquet(tmp_path / "data" / "predictions" / "prediction_log.parquet")
    clean = audit_prediction_log(tmp_path)
    assert clean["duplicate_symbol_observation_date"] == 0
    assert clean["severity"] == "OK"


def test_equities_and_indices_live_audit() -> None:
    res = audit_equities_and_indices(REPO_ROOT)
    eq = res["equities"]
    idx = res["indices"]
    assert eq["rows"] == 7440
    assert eq["ohlc_envelope_violations"] == 0
    assert eq["zero_volume_rows"] == 29
    assert eq["delivery_pct_missing_rows"] == 7384
    assert eq["stock_splits_positive_rows"] == 2
    assert eq["dividends_positive_rows"] == 44
    assert idx["has_NIFTY_50_underscore"] is True
    assert idx["has_NIFTY_50_space"] is False
    assert idx["nifty_naming_mismatch_bug"] is True
    assert res["severity"] == "HIGH"


def test_structured_events_and_paper_trades_live_audit() -> None:
    se = audit_structured_events(REPO_ROOT)
    assert se["valid_events"] == 254
    assert se["quarantined_events"] == 32
    assert se["null_collection_timestamp_rows"] == 156
    assert se["is_genuinely_forward_collected"] is False
    assert se["severity"] == "HIGH"

    pt = audit_paper_trades(REPO_ROOT)
    assert pt["total_rows"] == 219
    assert pt["legacy_rejected_action_rows"] == 15
    assert pt["lookahead_entry_executed_count"] == 1
    assert pt["non_null_exit_date_rows"] == 0
    assert pt["severity"] == "HIGH"


def test_run_full_audit_is_read_only_and_writes_report(tmp_path: Path) -> None:
    fp_path = REPO_ROOT / "data" / "predictions" / "forward_predictions.parquet"
    mtime_before = fp_path.stat().st_mtime_ns
    out_file = tmp_path / "DATA_QUALITY_REPORT.json"

    report = run_full_data_integrity_audit(REPO_ROOT, output_json_path=out_file)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["overall_status"] == "CRITICAL"
    assert loaded["read_only_guarantee"] is True
    assert set(loaded["sections"].keys()) == {
        "metadata_version_drift",
        "forward_predictions_and_outcomes",
        "prediction_log",
        "equities_and_indices",
        "structured_events",
        "paper_trades",
    }
    assert fp_path.stat().st_mtime_ns == mtime_before
    assert report["overall_status"] == "CRITICAL"
