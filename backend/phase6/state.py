"""Typed state for the shadow research graph (JSON-checkpointable)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.phase6.agents_base import EvidencePackage
from backend.phase6.contracts import (AgentReport, ResearchManagerReport, ResearchThesis,
                                      RiskAssessment)

NODES = ("load_evidence", "analysts", "debate", "research_manager", "risk_manager", "persist")


class ResearchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    symbol: str
    as_of: str
    status: Literal["RUNNING", "COMPLETE", "PARTIAL", "FAILED"] = "RUNNING"
    completed_nodes: list[str] = Field(default_factory=list)
    evidence_package: Optional[EvidencePackage] = None
    analyst_reports: dict[str, AgentReport] = Field(default_factory=dict)
    missing_analysts: list[str] = Field(default_factory=list)
    bull: Optional[ResearchThesis] = None
    bear: Optional[ResearchThesis] = None
    bull_input_hash: Optional[str] = None
    bear_input_hash: Optional[str] = None
    manager_report: Optional[ResearchManagerReport] = None
    risk_assessment: Optional[RiskAssessment] = None
    final_report: Optional[ResearchManagerReport] = None
    decision_record_hash: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    cost_summary: dict = Field(default_factory=dict)
