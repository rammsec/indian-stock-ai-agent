"""Immutable EvidenceBundle, FeatureSnapshot, and FeatureValue contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from trader.intelligence.data_contract import (
    OHLCVEvidenceFrame,
    _ensure_tz_aware_ist,
)

FEATURE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class FeatureValue:
    """Single canonical feature value with explicit missingness semantics."""

    feature_id: str
    formula_variant: str
    value: float | int | str | bool | None
    is_missing: bool = False
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            object.__setattr__(self, "value", None)
            object.__setattr__(self, "is_missing", True)
            if self.missing_reason is None:
                object.__setattr__(self, "missing_reason", "NON_FINITE_VALUE")
        elif self.value is None and not self.is_missing:
            object.__setattr__(self, "is_missing", True)


@dataclass(frozen=True)
class FeatureSnapshot:
    """Immutable snapshot of canonical features for a symbol/timeframe at as_of_ist."""

    symbol: str
    timeframe: str
    as_of_ist: datetime
    features: Mapping[str, FeatureValue] = field(default_factory=dict)
    schema_version: str = FEATURE_SCHEMA_VERSION
    missing_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        norm_as_of = _ensure_tz_aware_ist(self.as_of_ist, "FeatureSnapshot.as_of_ist")
        object.__setattr__(self, "as_of_ist", norm_as_of)
        if not isinstance(self.features, MappingProxyType):
            object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


def _serialize_for_hash(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _serialize_for_hash(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_hash(x) for x in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, FeatureValue):
        return {
            "feature_id": obj.feature_id,
            "formula_variant": obj.formula_variant,
            "value": obj.value,
            "is_missing": obj.is_missing,
            "missing_reason": obj.missing_reason,
        }
    if isinstance(obj, FeatureSnapshot):
        return {
            "symbol": obj.symbol,
            "timeframe": obj.timeframe,
            "as_of_ist": obj.as_of_ist.isoformat(),
            "schema_version": obj.schema_version,
            "features": _serialize_for_hash(obj.features),
            "missing_flags": list(obj.missing_flags),
        }
    if isinstance(obj, OHLCVEvidenceFrame):
        return {
            "symbol": obj.symbol,
            "exchange": obj.exchange,
            "timeframe": obj.timeframe,
            "as_of_ist": obj.as_of_ist.isoformat(),
            "provenance_status": obj.provenance_status.value,
            "source_name": obj.source_name,
            "source_sha256": obj.source_sha256,
            "bar_count": obj.bar_count,
            "latest_ts_event_ist": obj.latest_ts_event_ist.isoformat() if obj.latest_ts_event_ist else None,
            "latest_ts_init_ist": obj.latest_ts_init_ist.isoformat() if obj.latest_ts_init_ist else None,
            "staleness_hours": obj.staleness_hours,
            "missing_reason": obj.missing_reason,
        }
    if hasattr(obj, "value") and not isinstance(obj, (str, bytes)):
        return obj.value
    return obj


@dataclass(frozen=True)
class EvidenceBundle:
    """Read-only, hash-sealed evidence bundle consumed by downstream BharatAlpha agents."""

    symbol: str
    as_of_ist: datetime
    market_frames: Mapping[str, OHLCVEvidenceFrame] = field(default_factory=dict)
    feature_snapshots: Mapping[str, FeatureSnapshot] = field(default_factory=dict)
    regime_evidence: Mapping[str, Any] = field(default_factory=dict)
    directional_inputs: Mapping[str, Any] = field(default_factory=dict)
    execution_context: Mapping[str, Any] = field(default_factory=dict)
    missing_or_stale_flags: tuple[str, ...] = ()
    confidence_calibration_status: str = "UNCALIBRATED"
    schema_version: str = FEATURE_SCHEMA_VERSION
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if self.confidence_calibration_status != "UNCALIBRATED":
            raise ValueError(
                f"confidence_calibration_status must be 'UNCALIBRATED', got {self.confidence_calibration_status!r}"
            )
        norm_as_of = _ensure_tz_aware_ist(self.as_of_ist, "EvidenceBundle.as_of_ist")
        object.__setattr__(self, "as_of_ist", norm_as_of)

        for attr_name in (
            "market_frames",
            "feature_snapshots",
            "regime_evidence",
            "directional_inputs",
            "execution_context",
        ):
            val = getattr(self, attr_name)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr_name, MappingProxyType(dict(val)))

        if not self.evidence_hash:
            payload = {
                "symbol": self.symbol,
                "as_of_ist": norm_as_of.isoformat(),
                "schema_version": self.schema_version,
                "confidence_calibration_status": self.confidence_calibration_status,
                "market_frames": _serialize_for_hash(self.market_frames),
                "feature_snapshots": _serialize_for_hash(self.feature_snapshots),
                "regime_evidence": _serialize_for_hash(self.regime_evidence),
                "directional_inputs": _serialize_for_hash(self.directional_inputs),
                "execution_context": _serialize_for_hash(self.execution_context),
                "missing_or_stale_flags": list(self.missing_or_stale_flags),
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            object.__setattr__(self, "evidence_hash", hashlib.sha256(canonical).hexdigest())


def build_evidence_bundle(
    *,
    symbol: str,
    as_of_ist: datetime,
    market_frames: Mapping[str, OHLCVEvidenceFrame] | None = None,
    feature_snapshots: Mapping[str, FeatureSnapshot] | None = None,
    regime_evidence: Mapping[str, Any] | None = None,
    directional_inputs: Mapping[str, Any] | None = None,
    execution_context: Mapping[str, Any] | None = None,
    missing_or_stale_flags: Sequence[str] = (),
    confidence_calibration_status: str = "UNCALIBRATED",
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> EvidenceBundle:
    """Factory for building a validated, immutable EvidenceBundle."""
    return EvidenceBundle(
        symbol=symbol,
        as_of_ist=as_of_ist,
        market_frames=market_frames or {},
        feature_snapshots=feature_snapshots or {},
        regime_evidence=regime_evidence or {},
        directional_inputs=directional_inputs or {},
        execution_context=execution_context or {},
        missing_or_stale_flags=tuple(missing_or_stale_flags),
        confidence_calibration_status=confidence_calibration_status,
        schema_version=schema_version,
    )
