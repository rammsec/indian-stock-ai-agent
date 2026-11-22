"""Shared Intelligence Layer (CYCLE-009B) — canonical parsing, contracts, registry, MTF alignment, and bundles."""

from __future__ import annotations

from trader.intelligence.bundle import (
    FEATURE_SCHEMA_VERSION,
    EvidenceBundle,
    FeatureSnapshot,
    FeatureValue,
    build_evidence_bundle,
)
from trader.intelligence.data_contract import (
    NON_AUTHORITATIVE_PROVIDERS,
    LookaheadViolationError,
    OHLCVEvidenceFrame,
    ProvenanceTier,
    build_ohlcv_evidence_frame,
    cap_provenance_tier,
)
from trader.intelligence.mtf_aligner import (
    align_htf_to_ltf_ist,
    expected_5m_sub_bars_for_bucket,
)
from trader.intelligence.parsing import (
    compute_candle_series_hash,
    get_git_commit_sha,
    load_json_strict,
    parse_finite_float,
    parse_iso_to_ist,
    parse_obs_date,
    sha256_file,
    write_report_atomic,
)
from trader.intelligence.registry import (
    DEFAULT_FEATURE_REGISTRY,
    DuplicateFeatureRegistrationError,
    FeatureRegistry,
    FeatureSpec,
    compute_atr_series,
    compute_cpr_pivots_dict,
    compute_ema_series,
    compute_latest_session_vwap,
    compute_macd_series,
    compute_rsi_series,
    compute_rvol_series,
    compute_session_vwap_series,
    compute_supertrend_series,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "EvidenceBundle",
    "FeatureSnapshot",
    "FeatureValue",
    "build_evidence_bundle",
    "NON_AUTHORITATIVE_PROVIDERS",
    "LookaheadViolationError",
    "OHLCVEvidenceFrame",
    "ProvenanceTier",
    "build_ohlcv_evidence_frame",
    "cap_provenance_tier",
    "align_htf_to_ltf_ist",
    "expected_5m_sub_bars_for_bucket",
    "compute_candle_series_hash",
    "get_git_commit_sha",
    "load_json_strict",
    "parse_finite_float",
    "parse_iso_to_ist",
    "parse_obs_date",
    "sha256_file",
    "write_report_atomic",
    "DEFAULT_FEATURE_REGISTRY",
    "DuplicateFeatureRegistrationError",
    "FeatureRegistry",
    "FeatureSpec",
    "compute_atr_series",
    "compute_cpr_pivots_dict",
    "compute_ema_series",
    "compute_latest_session_vwap",
    "compute_macd_series",
    "compute_rsi_series",
    "compute_rvol_series",
    "compute_session_vwap_series",
    "compute_supertrend_series",
]
