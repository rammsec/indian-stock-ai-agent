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

__all__ = [
    "DEFAULT_PROSPECTIVE_CUTOFF",
    "audit_equities_and_indices",
    "audit_forward_predictions_and_outcomes",
    "audit_metadata_version_drift",
    "audit_paper_trades",
    "audit_prediction_log",
    "audit_structured_events",
    "classify_forward_snapshot_record",
    "run_full_data_integrity_audit",
]
