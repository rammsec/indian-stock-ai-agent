"""Agent interfaces for the shadow research graph.

Next stage implements concrete agents by subclassing these and registering them
in an :class:`AgentRegistry` (see ``graph.py``). Packages are strictly typed so
information flow is enforced structurally:

* ``EvidencePackage``  -> analysts and debaters (Bull/Bear get the *identical* object)
* ``ManagerPackage``   -> research manager (adds bull + bear theses)
* ``RiskPackage``      -> risk manager (adds the manager report)
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.phase6.contracts import (AgentReport, EvidenceRef, ResearchManagerReport,
                                      ResearchThesis, RiskAssessment)

ANALYST_NAMES = ("fundamental", "technical", "news", "promoter", "competitor", "quantitative")


class EvidencePackage(BaseModel):
    """Immutable, hashable evidence handed to analysts/debaters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    as_of: str                                  # YYYY-MM-DD decision date
    analysis_timestamp: datetime
    data_cutoff_timestamp: datetime
    evidence: tuple[EvidenceRef, ...] = ()
    data: dict[str, Any] = Field(default_factory=dict)          # JSON-safe raw inputs
    quantitative_predictions: dict[str, Any] = Field(default_factory=dict)  # frozen model, read-only
    analyst_reports: dict[str, AgentReport] = Field(default_factory=dict)   # empty for analysts
    missing_analysts: tuple[str, ...] = ()

    def input_hash(self) -> str:
        """sha256 over canonical JSON of the whole package."""
        blob = json.dumps(json.loads(self.model_dump_json()), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ManagerPackage(EvidencePackage):
    bull: Optional[ResearchThesis] = None
    bear: Optional[ResearchThesis] = None


class RiskPackage(ManagerPackage):
    manager_report: Optional[ResearchManagerReport] = None


@dataclass
class AgentContext:
    """Runtime services handed to each agent."""

    run_id: str
    config: Any                      # Phase6Config
    llm: Any = None                  # LLMClient
    cost_tracker: Any = None         # CostTracker
    extras: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base agent. ``name`` keys prompts/costs; ``run`` returns a validated report."""

    name: str = ""
    version: str = "0.0.0"
    prompt_version: str = "v1"

    @abstractmethod
    def run(self, evidence_package: EvidencePackage, ctx: AgentContext) -> AgentReport: ...


class AnalystAgent(Agent):
    """One of the six independent analysts; returns a plain ``AgentReport``."""

    @abstractmethod
    def run(self, evidence_package: EvidencePackage, ctx: AgentContext) -> AgentReport: ...


class DebaterAgent(Agent):
    """Bull or Bear. ``side`` must be 'BULL' or 'BEAR'; returns ``ResearchThesis``."""

    side: str = ""

    @abstractmethod
    def run(self, evidence_package: EvidencePackage, ctx: AgentContext) -> ResearchThesis: ...


class ManagerAgent(Agent):
    """Receives a :class:`ManagerPackage`; returns ``ResearchManagerReport``."""

    @abstractmethod
    def run(self, evidence_package: ManagerPackage, ctx: AgentContext) -> ResearchManagerReport: ...


class RiskAgent(Agent):
    """Receives a :class:`RiskPackage`; returns ``RiskAssessment``."""

    @abstractmethod
    def run(self, evidence_package: RiskPackage, ctx: AgentContext) -> RiskAssessment: ...
