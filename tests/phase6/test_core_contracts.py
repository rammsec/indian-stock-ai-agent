from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.phase6.contracts import (AgentReport, NumericClaim, PHASE6_SCHEMA_VERSION,
                                      ResearchManagerReport, RiskAssessment, TimestampViolation)
from p6helpers import ANA, CUT, base, ref


def test_valid_report_and_version():
    r = AgentReport(**base(numeric_claims=[NumericClaim(name="pe", value=20.5, evidence_id="ev1")]))
    assert r.numeric_claims[0].value == 20.5
    assert PHASE6_SCHEMA_VERSION == "p6-1.0.0"


def test_naive_timestamps_rejected():
    with pytest.raises(ValidationError):
        AgentReport(**base(analysis_timestamp=datetime(2025, 1, 15, 11)))
    with pytest.raises(ValidationError):
        AgentReport(**base(data_cutoff_timestamp=datetime(2025, 1, 15, 10)))


def test_cutoff_after_analysis_rejected():
    with pytest.raises(ValidationError, match="data_cutoff_timestamp"):
        AgentReport(**base(data_cutoff_timestamp=datetime(2025, 1, 16, tzinfo=timezone.utc)))


@pytest.mark.parametrize("kw", [
    {"pub": datetime(2025, 1, 15, 10, 0, 1, tzinfo=timezone.utc)},
    {"ret": datetime(2025, 1, 15, 12, tzinfo=timezone.utc)}])
def test_evidence_after_cutoff_raises_timestamp_violation(kw):
    with pytest.raises(TimestampViolation):
        AgentReport(**base(evidence_refs=[ref(**kw)]))


def test_evidence_exactly_at_cutoff_ok():
    AgentReport(**base(evidence_refs=[ref(pub=CUT, ret=CUT)]))


def test_numeric_claim_without_evidence_rejected():
    with pytest.raises(ValidationError, match="unknown evidence_id"):
        AgentReport(**base(numeric_claims=[NumericClaim(name="pe", value=1, evidence_id="nope")]))


def test_extra_forbidden_and_confidence_bounds():
    with pytest.raises(ValidationError):
        AgentReport(**base(bogus=1))
    with pytest.raises(ValidationError):
        AgentReport(**base(confidence=1.5))


def test_risk_and_manager_enums():
    with pytest.raises(ValidationError):
        RiskAssessment(**base(), verdict="MAYBE")
    with pytest.raises(ValidationError):
        ResearchManagerReport(**base(), overall_conclusion="BUY")
    ra = RiskAssessment(**base(), verdict="REJECT", rejection_reasons=["x"])
    assert ra.liquidity_risk.level == "UNKNOWN"
