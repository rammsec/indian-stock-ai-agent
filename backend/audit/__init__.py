"""Read-only Data Integrity & Audit package for BharatAlpha AI.

Produces derived audit classifications and structured quality reports without
mutating any original Parquet, JSON, or model/feature artifacts.
"""

from .data_integrity import (
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
from .identity_contract import (
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

__all__ = [
    "CANONICAL_FEATURE_SET_VERSION",
    "CANONICAL_MODEL_VERSION",
    "DEFAULT_PROSPECTIVE_CUTOFF",
    "MANDATORY_PROVENANCE_FIELDS",
    "REMEDIATION_VERSION",
    "SCHEMA_VERSION",
    "audit_equities_and_indices",
    "audit_forward_predictions_and_outcomes",
    "audit_metadata_version_drift",
    "audit_paper_trades",
    "audit_prediction_log",
    "audit_structured_events",
    "build_remediated_shadow_layer",
    "canonical_prediction_id",
    "check_provenance_completeness",
    "classify_forward_snapshot_record",
    "daily_canonical_key",
    "enforce_provenance",
    "resolve_metadata_version_drift",
    "run_full_data_integrity_audit",
    "validate_event_records",
    "validate_prediction_records",
]
