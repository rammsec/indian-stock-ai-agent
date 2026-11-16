"""Phase 2 — Reusable Feature Engineering Library, Anti-Leakage Validator & Feature Selector."""

from .feature_engine import (
    CANDIDATE_FEATURE_FAMILIES,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    build_features_for_symbol,
    build_historical_feature_matrix,
)
from .leakage_guard import (
    assert_point_in_time_integrity,
    get_candidate_feature_columns,
    verify_no_lookahead_leakage,
)
from .selector import evaluate_and_select_features

filter_and_rank_features = evaluate_and_select_features

__all__ = [
    "CANDIDATE_FEATURE_FAMILIES",
    "METADATA_COLUMNS",
    "TARGET_COLUMNS",
    "assert_point_in_time_integrity",
    "build_features_for_symbol",
    "build_historical_feature_matrix",
    "evaluate_and_select_features",
    "filter_and_rank_features",
    "get_candidate_feature_columns",
    "verify_no_lookahead_leakage",
]
