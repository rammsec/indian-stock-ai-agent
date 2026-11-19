"""Shared deterministic test helpers for Phase 6 core tests."""
from datetime import datetime, timezone

from backend.phase6.agents_base import (AnalystAgent, DebaterAgent, EvidencePackage, ManagerAgent,
                                        RiskAgent, ANALYST_NAMES)
from backend.phase6.contracts import (AgentReport, EvidenceRef, NumericClaim, ResearchManagerReport,
                                      ResearchThesis, RiskAssessment)

CUT = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
ANA = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)


def ref(i="ev1", pub=None, ret=None):
    return EvidenceRef(source_id=i, dataset="d", path="p", record_key="k",
                       published_at=pub or datetime(2025, 1, 14, tzinfo=timezone.utc),
                       retrieved_at=ret or datetime(2025, 1, 15, 9, tzinfo=timezone.utc))


def base(name="x", **kw):
    d = dict(agent_name=name, agent_version="1", prompt_version="v1", symbol="TCS",
             analysis_timestamp=ANA, data_cutoff_timestamp=CUT, evidence_refs=[ref()],
             confidence=0.5)
    d.update(kw)
    return d


def make_pkg():
    return EvidencePackage(symbol="TCS", as_of="2025-01-15", analysis_timestamp=ANA,
                           data_cutoff_timestamp=CUT, evidence=(ref(),),
                           data={"close": 100.0}, quantitative_predictions={"p_up": 0.55})


def loader(symbol, as_of):
    return make_pkg()


class StubAnalyst(AnalystAgent):
    version = "1"

    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def run(self, pkg, ctx):
        if self.fail:
            raise RuntimeError("boom")
        return AgentReport(**base(self.name, findings=[f"{self.name} ok"]))


class StubDebater(DebaterAgent):
    version = "1"

    def __init__(self, side, fail=False):
        self.side, self.name, self.fail = side, side.lower(), fail
        self.seen = None

    def run(self, pkg, ctx):
        self.seen = pkg
        if self.fail:
            raise RuntimeError("debater down")
        return ResearchThesis(**base(self.name), side=self.side, thesis=f"{self.side} thesis")


class StubManager(ManagerAgent):
    name, version = "research_manager", "1"

    def __init__(self, conclusion="FAVOURABLE"):
        self.conclusion, self.seen = conclusion, None

    def run(self, pkg, ctx):
        self.seen = pkg
        return ResearchManagerReport(**base("research_manager"), overall_conclusion=self.conclusion,
                                     quantitative_model_predictions=dict(pkg.quantitative_predictions))


class StubRisk(RiskAgent):
    name, version = "risk_manager", "1"

    def __init__(self, verdict="APPROVE", fail=False):
        self.verdict, self.fail = verdict, fail

    def run(self, pkg, ctx):
        if self.fail:
            raise RuntimeError("risk down")
        return RiskAssessment(**base("risk_manager"), verdict=self.verdict,
                              rejection_reasons=["too risky"] if self.verdict == "REJECT" else [])


def registry(failing=(), verdict="APPROVE", conclusion="FAVOURABLE", bull_fail=False, risk_fail=False):
    from backend.phase6.graph import AgentRegistry
    return AgentRegistry(
        analysts={n: StubAnalyst(n, fail=n in failing) for n in ANALYST_NAMES},
        bull=StubDebater("BULL", bull_fail), bear=StubDebater("BEAR"),
        manager=StubManager(conclusion), risk=StubRisk(verdict, risk_fail))
