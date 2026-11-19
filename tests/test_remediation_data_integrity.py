"""Non-vacuous tests for Workstream 1 (Data Integrity) Remediation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.audit.identity_contract import (
    CANONICAL_FEATURE_SET_VERSION,
    CANONICAL_MODEL_VERSION,
    MANDATORY_PROVENANCE_FIELDS,
    REMEDIATION_VERSION,
    SCHEMA_VERSION,
    build_remediated_shadow_layer,
    canonical_prediction_id,
    check_provenance_completeness,
    daily_canonical_key,
    enforce_provenance,
    resolve_metadata_version_drift,
    validate_event_records,
    validate_prediction_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deterministic_identity_keys_and_validation() -> None:
    k1 = canonical_prediction_id("birlacable", "2026-11-17T00:00:00", "pre_market")
    k2 = canonical_prediction_id("BIRLACABLE", "2026-11-17", "PRE_MARKET")
    assert k1 == k2 == "cpid_2026-11-17_BIRLACABLE_PRE_MARKET"

    d1 = daily_canonical_key("itc", "2026-11-18")
    d2 = daily_canonical_key("ITC", "2026-11-18T09:15:00+05:30")
    assert d1 == d2 == "dck_2026-11-18_ITC"

    with pytest.raises(ValueError):
        canonical_prediction_id("", "2026-11-17", "PRE_MARKET")
    with pytest.raises(ValueError):
        canonical_prediction_id("ITC", "not-a-date", "PRE_MARKET")
    with pytest.raises(ValueError):
        canonical_prediction_id("ITC", "2026-11-17", "OVERNIGHT")
    with pytest.raises(ValueError):
        daily_canonical_key("ITC", "invalid")


def test_duplicate_prediction_and_lookahead_gate() -> None:
    raw = pd.DataFrame(
        [
            # 1. Valid prospective pre-market
            {
                "prediction_id": "p1",
                "symbol": "ITC",
                "observation_date": "2026-11-17",
                "prediction_timestamp": "2026-11-18T03:00:00+05:30",
                "nse_session": "PRE_MARKET",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "entry_date": "2026-11-18",
            },
            # 2. Duplicate of (ITC, 2026-11-17, PRE_MARKET)
            {
                "prediction_id": "p2",
                "symbol": "ITC",
                "observation_date": "2026-11-17",
                "prediction_timestamp": "2026-11-18T03:05:00+05:30",
                "nse_session": "PRE_MARKET",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "entry_date": "2026-11-18",
            },
            # 3. Intraday look-ahead (10:25 IST > 09:15 IST entry_date)
            {
                "prediction_id": "p3",
                "symbol": "BIRLACABLE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T10:25:29+05:30",
                "nse_session": "INTRADAY",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "entry_date": "2026-11-17",
            },
            # 4. Missing prediction_timestamp -> QUARANTINED
            {
                "prediction_id": "p4",
                "symbol": "TCS",
                "observation_date": "2026-11-17",
                "prediction_timestamp": None,
                "nse_session": "PRE_MARKET",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "entry_date": "2026-11-18",
            },
        ]
    )
    res = validate_prediction_records(raw, source_dataset="synthetic_test")
    assert len(res) == 4  # Raw rows preserved
    by_id = res.set_index("prediction_id")

    assert by_id.loc["p1", "validation_status"] == "PROSPECTIVE"
    assert pd.isna(by_id.loc["p1", "rejection_reason"])
    assert by_id.loc["p1", "canonical_prediction_id"] == "cpid_2026-11-17_ITC_PRE_MARKET"
    assert by_id.loc["p1", "daily_canonical_key"] == "dck_2026-11-18_ITC"

    assert by_id.loc["p2", "validation_status"] == "DUPLICATE"
    assert by_id.loc["p2", "rejection_reason"] == "DUPLICATE_CANONICAL_PREDICTION_ID"

    assert by_id.loc["p3", "validation_status"] == "INVALID_LOOKAHEAD"
    assert "LOOKAHEAD_VIOLATION_DELTA_70.48_MIN" == by_id.loc["p3", "rejection_reason"]

    assert by_id.loc["p4", "validation_status"] == "QUARANTINED"
    assert by_id.loc["p4", "rejection_reason"] == "MISSING_OR_INVALID_PREDICTION_TIMESTAMP"


def test_prediction_log_live_deduplication_and_provenance() -> None:
    pl = pd.read_parquet(REPO_ROOT / "data" / "predictions" / "prediction_log.parquet")
    res = validate_prediction_records(
        pl,
        source_dataset="data/predictions/prediction_log.parquet",
        dedupe_by_symbol_obs_date=True,
        default_feature_set_version=CANONICAL_FEATURE_SET_VERSION,
    )
    assert len(res) == 90
    counts = res["validation_status"].value_counts().to_dict()
    assert counts == {"DUPLICATE": 45, "PROSPECTIVE": 30, "HISTORICAL": 15}
    dedup = res[res["validation_status"] != "DUPLICATE"]
    assert len(dedup) == 45
    assert int(dedup.duplicated(subset=["symbol", "observation_date"]).sum()) == 0
    prov = check_provenance_completeness(res)
    assert prov["is_complete"] is True


def test_event_validation_missing_collection_ts_and_duplicates() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "ticker": "ITC",
                "publication_timestamp": "2026-11-17T08:00:00+05:30",
                "collection_timestamp": "2026-11-17T09:00:00+05:30",
                "is_duplicate_report": False,
            },
            {
                "event_id": "e1",  # duplicate event_id
                "ticker": "ITC",
                "publication_timestamp": "2026-11-17T08:00:00+05:30",
                "collection_timestamp": "2026-11-17T09:05:00+05:30",
                "is_duplicate_report": False,
            },
            {
                "event_id": "e2",
                "ticker": "TCS",
                "publication_timestamp": "2026-11-10T07:00:00+05:30",
                "collection_timestamp": None,  # missing collection_timestamp
                "is_duplicate_report": False,
            },
            {
                "event_id": "e3",
                "ticker": "INFY",
                "publication_timestamp": "2026-11-18T10:00:00+05:30",
                "collection_timestamp": "2026-11-18T09:00:00+05:30",  # pub > coll
                "is_duplicate_report": False,
            },
        ]
    )
    res = validate_event_records(events, source_dataset="synthetic_events")
    assert list(res["validation_status"]) == [
        "PROSPECTIVE",
        "DUPLICATE",
        "QUARANTINED",
        "INVALID_LOOKAHEAD",
    ]
    assert res.loc[1, "rejection_reason"] == "DUPLICATE_EVENT_REPORT"
    assert res.loc[2, "rejection_reason"] == "MISSING_COLLECTION_TIMESTAMP"
    assert res.loc[3, "rejection_reason"] == "PUBLICATION_AFTER_COLLECTION"

    # Also verify live structured_events.parquet quarantines all 156 missing collection_timestamp rows
    live_se = pd.read_parquet(REPO_ROOT / "data" / "predictions" / "structured_events.parquet")
    live_res = validate_event_records(live_se)
    assert len(live_res) == 254
    assert int((live_res["validation_status"] == "QUARANTINED").sum()) == 156
    assert int((live_res["validation_status"] == "PROSPECTIVE").sum()) == 96
    assert int((live_res["validation_status"] == "DUPLICATE").sum()) == 2


def test_provenance_enforcement_and_completeness_check() -> None:
    incomplete = pd.DataFrame({"symbol": ["ITC"]})
    check_fail = check_provenance_completeness(incomplete)
    assert check_fail["is_complete"] is False
    assert set(check_fail["missing_columns"]) == set(MANDATORY_PROVENANCE_FIELDS)

    enforced = enforce_provenance(
        incomplete,
        source_dataset="unit_test",
        default_model_version=CANONICAL_MODEL_VERSION,
        default_feature_set_version=CANONICAL_FEATURE_SET_VERSION,
    )
    assert enforced.loc[0, "schema_version"] == SCHEMA_VERSION
    assert enforced.loc[0, "remediation_version"] == REMEDIATION_VERSION
    assert enforced.loc[0, "model_version"] == CANONICAL_MODEL_VERSION
    assert enforced.loc[0, "feature_set_version"] == CANONICAL_FEATURE_SET_VERSION
    assert check_provenance_completeness(enforced)["is_complete"] is True


def test_model_version_drift_detection_and_resolution(tmp_path: Path) -> None:
    live_manifest = resolve_metadata_version_drift(REPO_ROOT)
    assert live_manifest["drift_detected_in_raw"] is True
    assert live_manifest["raw_unique_versions"] == ["5.0.0-pilot", "5.3.0-pilot"]
    assert live_manifest["canonical_model_version"] == "5.3.0-pilot"
    assert live_manifest["drift_resolved"] is True
    assert all(v == "5.3.0-pilot" for v in live_manifest["shadow_resolved_versions"].values())

    # Clean fixture
    (tmp_path / "data" / "predictions").mkdir(parents=True)
    (tmp_path / "docs" / "data").mkdir(parents=True)
    for rel in (
        "data/predictions/phase5_intelligence.json",
        "docs/data/phase5_intelligence.json",
        "data/predictions/forward_validation.json",
        "docs/data/forward_validation.json",
    ):
        (tmp_path / rel).write_text(json.dumps({"model_version": "5.3.0-pilot"}), encoding="utf-8")
    pd.DataFrame({"model_version": ["5.3.0-pilot"]}).to_parquet(
        tmp_path / "data" / "predictions" / "forward_predictions.parquet"
    )
    clean_manifest = resolve_metadata_version_drift(tmp_path)
    assert clean_manifest["drift_detected_in_raw"] is False
    assert clean_manifest["raw_unique_versions"] == ["5.3.0-pilot"]


def test_shadow_layer_non_mutation_guarantee(tmp_path: Path) -> None:
    originals = [
        REPO_ROOT / "data" / "predictions" / "forward_predictions.parquet",
        REPO_ROOT / "data" / "predictions" / "forward_outcomes.parquet",
        REPO_ROOT / "data" / "predictions" / "prediction_log.parquet",
        REPO_ROOT / "data" / "predictions" / "structured_events.parquet",
        REPO_ROOT / "data" / "processed" / "equities_daily.parquet",
    ]
    hashes_before = {p: _sha256(p) for p in originals}
    mtimes_before = {p: p.stat().st_mtime_ns for p in originals}

    shadow_out = tmp_path / "v1_remediated"
    summary = build_remediated_shadow_layer(REPO_ROOT, output_dir=shadow_out)

    assert (shadow_out / "forward_predictions_remediated.parquet").exists()
    assert (shadow_out / "prediction_log_remediated.parquet").exists()
    assert (shadow_out / "prediction_log_deduplicated.parquet").exists()
    assert (shadow_out / "structured_events_remediated.parquet").exists()
    assert (shadow_out / "metadata_manifest.json").exists()
    assert (shadow_out / "remediation_summary.json").exists()

    assert summary["datasets"]["forward_predictions"]["status_counts"] == {
        "DUPLICATE": 35,
        "INVALID_LOOKAHEAD": 15,
        "PROSPECTIVE": 10,
        "HISTORICAL": 5,
    }
    assert summary["datasets"]["prediction_log"]["deduplicated_rows"] == 45

    for p in originals:
        assert _sha256(p) == hashes_before[p]
        assert p.stat().st_mtime_ns == mtimes_before[p]
