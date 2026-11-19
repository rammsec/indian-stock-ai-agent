"""Phase 6 typed contracts (Pydantic v2, ``extra='forbid'``).

Every agent output derives from :class:`AgentReport`. Validators enforce:

* timezone-aware timestamps,
* ``data_cutoff_timestamp <= analysis_timestamp``,
* no evidence published/retrieved after the data cutoff (:class:`TimestampViolation`),
* every numeric claim cites an evidence id present in ``evidence_refs``
  (no invented numbers).

The evidence id of an :class:`EvidenceRef` is its ``source_id``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PHASE6_SCHEMA_VERSION = "p6-1.0.0"


class TimestampViolation(Exception):
    """Look-ahead: evidence newer than the data cutoff. Deliberately not a ValueError so
    pydantic propagates it unwrapped instead of converting to ValidationError."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_aware(v: Optional[datetime], name: str) -> Optional[datetime]:
    if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
        raise ValueError(f"{name} must be timezone-aware")
    return v


class EvidenceRef(_Strict):
    """Pointer to a piece of evidence; ``source_id`` is the citation id."""

    source_id: str
    dataset: str
    path: str
    record_key: str
    published_at: Optional[datetime] = None
    retrieved_at: datetime
    sha256: Optional[str] = None

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def _tz(cls, v, info):
        return _require_aware(v, info.field_name)


class NumericClaim(_Strict):
    """A number asserted by an agent, which must cite evidence."""

    name: str
    value: float
    evidence_id: str


class AgentReport(_Strict):
    """Base report produced by every agent."""

    agent_name: str
    agent_version: str
    prompt_version: str
    symbol: str
    analysis_timestamp: datetime
    data_cutoff_timestamp: datetime
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    numeric_claims: list[NumericClaim] = Field(default_factory=list)

    @field_validator("analysis_timestamp", "data_cutoff_timestamp")
    @classmethod
    def _tz(cls, v, info):
        return _require_aware(v, info.field_name)

    @model_validator(mode="after")
    def _check_temporal_and_claims(self):
        if self.data_cutoff_timestamp > self.analysis_timestamp:
            raise ValueError("data_cutoff_timestamp must be <= analysis_timestamp")
        for ref in self.evidence_refs:
            for fname in ("published_at", "retrieved_at"):
                ts = getattr(ref, fname)
                if ts is not None and ts > self.data_cutoff_timestamp:
                    raise TimestampViolation(
                        f"evidence {ref.source_id}.{fname}={ts.isoformat()} is after "
                        f"data cutoff {self.data_cutoff_timestamp.isoformat()}"
                    )
        ids = {r.source_id for r in self.evidence_refs}
        for c in self.numeric_claims:
            if c.evidence_id not in ids:
                raise ValueError(
                    f"numeric claim '{c.name}' cites unknown evidence_id '{c.evidence_id}'"
                )
        return self


class ResearchThesis(AgentReport):
    """Bull / Bear debater output."""

    side: Literal["BULL", "BEAR"]
    thesis: str
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    challenges_to_other_side: list[str] = Field(default_factory=list)


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


class RiskDimension(_Strict):
    level: RiskLevel = "UNKNOWN"
    note: str = ""


class RiskAssessment(AgentReport):
    """Risk manager output; verdict REJECT forces REJECTED_BY_RISK."""

    verdict: Literal["APPROVE", "REDUCE", "REJECT"]
    downside_risk: RiskDimension = Field(default_factory=RiskDimension)
    volatility_risk: RiskDimension = Field(default_factory=RiskDimension)
    liquidity_risk: RiskDimension = Field(default_factory=RiskDimension)
    concentration_risk: RiskDimension = Field(default_factory=RiskDimension)
    event_risk: RiskDimension = Field(default_factory=RiskDimension)
    drawdown_risk: RiskDimension = Field(default_factory=RiskDimension)
    risk_reward: RiskDimension = Field(default_factory=RiskDimension)
    rationale: str = ""
    rejection_reasons: list[str] = Field(default_factory=list)


OverallConclusion = Literal[
    "FAVOURABLE", "NEUTRAL", "UNFAVOURABLE", "REJECTED_BY_RISK", "INSUFFICIENT_EVIDENCE"
]


class ResearchManagerReport(AgentReport):
    """Final synthesis. ``quantitative_model_predictions`` is a read-only copy."""

    investment_thesis: str = ""
    bull_case: str = ""
    bear_case: str = ""
    fundamental_assessment: str = ""
    technical_assessment: str = ""
    news_assessment: str = ""
    promoter_assessment: str = ""
    competitor_assessment: str = ""
    quantitative_model_predictions: dict[str, Any] = Field(default_factory=dict)
    risk_assessment: Optional[RiskAssessment] = None
    agent_disagreements: list[str] = Field(default_factory=list)
    overall_conclusion: OverallConclusion
