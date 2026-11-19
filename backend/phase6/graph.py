"""Explicit typed state-machine orchestrator (LangGraph is intentionally not used).

Flow::

    load_evidence -> analysts (6, concurrent, failure-isolated)
                  -> debate (Bull & Bear get the IDENTICAL EvidencePackage, run independently)
                  -> research_manager -> risk_manager (REJECT overrides) -> persist

A JSON checkpoint is written after every node; re-running with the same
``run_id`` resumes after the last completed node.

Registry usage::

    registry = AgentRegistry(analysts={"fundamental": ..., ...}, bull=..., bear=...,
                             manager=..., risk=...)
    graph = ShadowResearchGraph(registry, config, evidence_loader=loader,
                                llm=client, decision_log=log)
    state = graph.run("RELIANCE", "2025-01-15")
"""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend.phase6.agents_base import (ANALYST_NAMES, AgentContext, AnalystAgent, DebaterAgent,
                                        EvidencePackage, ManagerAgent, ManagerPackage, RiskAgent,
                                        RiskPackage)
from backend.phase6.config import Phase6Config
from backend.phase6.contracts import AgentReport
from backend.phase6.cost import CostTracker
from backend.phase6.prompts import all_prompt_hashes
from backend.phase6.state import NODES, ResearchState


@dataclass
class AgentRegistry:
    """Agents injected into the graph. Missing analysts are recorded as missing."""

    analysts: dict[str, AnalystAgent] = field(default_factory=dict)
    bull: Optional[DebaterAgent] = None
    bear: Optional[DebaterAgent] = None
    manager: Optional[ManagerAgent] = None
    risk: Optional[RiskAgent] = None


EvidenceLoader = Callable[[str, str], EvidencePackage]


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


class ShadowResearchGraph:
    def __init__(self, registry: AgentRegistry, config: Optional[Phase6Config] = None, *,
                 evidence_loader: EvidenceLoader, llm=None,
                 cost_tracker: Optional[CostTracker] = None, decision_log=None,
                 namespace_dir: Optional[str] = None, max_workers: int = 6):
        self.registry = registry
        self.config = config or Phase6Config()
        self.evidence_loader = evidence_loader
        self.llm = llm
        self.cost = cost_tracker or CostTracker(self.config.model_prices)
        self.decision_log = decision_log
        self.ns = Path(namespace_dir or self.config.namespace_dir)
        self.max_workers = max_workers
        self._nodes = {
            "load_evidence": self._load_evidence, "analysts": self._analysts,
            "debate": self._debate, "research_manager": self._manager,
            "risk_manager": self._risk, "persist": self._persist,
        }

    # ---- checkpointing -------------------------------------------------
    def checkpoint_path(self, run_id: str) -> Path:
        return self.ns / "checkpoints" / f"{_safe(run_id)}.json"

    def _save(self, state: ResearchState) -> None:
        p = self.checkpoint_path(state.run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=1), encoding="utf-8")
        os.replace(tmp, p)

    def load_checkpoint(self, run_id: str) -> Optional[ResearchState]:
        p = self.checkpoint_path(run_id)
        if not p.exists():
            return None
        return ResearchState.model_validate_json(p.read_text(encoding="utf-8"))

    # ---- run -----------------------------------------------------------
    def run(self, symbol: str, as_of: str, run_id: Optional[str] = None,
            resume: bool = True) -> ResearchState:
        run_id = run_id or f"{symbol}-{as_of}"
        state = (self.load_checkpoint(run_id) if resume else None) or ResearchState(
            run_id=run_id, symbol=symbol, as_of=as_of)
        if state.status == "FAILED":  # resuming a failed run retries the failed node
            state.status, state.errors = "RUNNING", []
        ctx = AgentContext(run_id=run_id, config=self.config, llm=self.llm,
                           cost_tracker=self.cost)
        for node in NODES:
            if node in state.completed_nodes:
                continue
            try:
                self._nodes[node](state, ctx)
            except Exception as e:  # fatal node failure
                state.errors.append(f"{node}: {type(e).__name__}: {e}")
                state.status = "FAILED"
                state.cost_summary = self.cost.summary()
                self._save(state)
                return state
            if state.status == "FAILED":
                break
            state.completed_nodes.append(node)
            self._save(state)
        if state.status != "FAILED":
            state.status = "PARTIAL" if (state.missing_analysts or state.errors
                                         or state.warnings) else "COMPLETE"
        state.cost_summary = self.cost.summary()
        self._save(state)
        return state

    # ---- nodes ---------------------------------------------------------
    def _load_evidence(self, state: ResearchState, ctx: AgentContext) -> None:
        state.evidence_package = self.evidence_loader(state.symbol, state.as_of)

    def _analysts(self, state, ctx) -> None:
        pkg = state.evidence_package
        lock = threading.Lock()

        def one(name: str):
            agent = self.registry.analysts.get(name)
            if agent is None:
                raise LookupError("analyst not registered")
            rep = agent.run(pkg, ctx)
            if not isinstance(rep, AgentReport):
                raise TypeError("analyst returned non-AgentReport")
            if rep.data_cutoff_timestamp > pkg.data_cutoff_timestamp:
                raise ValueError("analyst data cutoff is after package cutoff (look-ahead)")
            # coerce to the plain base type so checkpoints round-trip
            return AgentReport.model_validate(
                {k: getattr(rep, k) for k in AgentReport.model_fields})

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {n: ex.submit(one, n) for n in ANALYST_NAMES}
        for n, f in futs.items():
            try:
                state.analyst_reports[n] = f.result()
            except Exception as e:
                state.missing_analysts.append(n)
                state.warnings.append(f"analyst '{n}' failed: {type(e).__name__}: {e}")
        if not state.analyst_reports:
            state.errors.append("analysts: all analysts failed")
            state.status = "FAILED"

    def _debate(self, state, ctx) -> None:
        base = state.evidence_package
        pkg = EvidencePackage(**{**{k: getattr(base, k) for k in EvidencePackage.model_fields},
                                 "analyst_reports": dict(state.analyst_reports),
                                 "missing_analysts": tuple(state.missing_analysts)})
        h = pkg.input_hash()
        state.bull_input_hash = h
        state.bear_input_hash = h
        results = {}

        def one(side, agent):
            if agent is None:
                raise LookupError(f"{side} agent not registered")
            return agent.run(pkg, ctx)  # the SAME frozen object for both sides

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {"bull": ex.submit(one, "bull", self.registry.bull),
                    "bear": ex.submit(one, "bear", self.registry.bear)}
        for side, f in futs.items():
            try:
                results[side] = f.result()
            except Exception as e:
                state.warnings.append(f"{side} failed: {type(e).__name__}: {e}")
        if pkg.input_hash() != h:  # defensive: debaters must not mutate the package
            state.warnings.append("debate input package changed during debate")
        state.bull, state.bear = results.get("bull"), results.get("bear")

    def _manager_package(self, state, cls, **extra):
        base = state.evidence_package
        return cls(**{**{k: getattr(base, k) for k in EvidencePackage.model_fields},
                      "analyst_reports": dict(state.analyst_reports),
                      "missing_analysts": tuple(state.missing_analysts),
                      "bull": state.bull, "bear": state.bear, **extra})

    def _manager(self, state, ctx) -> None:
        if self.registry.manager is None:
            raise LookupError("research manager not registered")
        state.manager_report = self.registry.manager.run(
            self._manager_package(state, ManagerPackage), ctx)

    def _risk(self, state, ctx) -> None:
        mgr = state.manager_report
        final = mgr
        if self.registry.risk is None:
            state.warnings.append("risk manager not registered")
        else:
            try:
                state.risk_assessment = self.registry.risk.run(
                    self._manager_package(state, RiskPackage, manager_report=mgr), ctx)
            except Exception as e:
                state.warnings.append(f"risk manager failed: {type(e).__name__}: {e}")
        ra = state.risk_assessment
        if ra is not None:
            update = {"risk_assessment": ra}
            if ra.verdict == "REJECT":
                update["overall_conclusion"] = "REJECTED_BY_RISK"
            final = mgr.model_copy(update=update)
        elif mgr.overall_conclusion == "FAVOURABLE":
            # an un-assessed risk can never yield a favourable conclusion
            final = mgr.model_copy(update={"overall_conclusion": "INSUFFICIENT_EVIDENCE"})
            state.warnings.append("risk unassessed: FAVOURABLE downgraded to INSUFFICIENT_EVIDENCE")
        state.final_report = final

    def _persist(self, state, ctx) -> None:
        if self.decision_log is None or state.final_report is None:
            return
        pkg = state.evidence_package
        rec, _created = self.decision_log.append(
            run_id=state.run_id, symbol=state.symbol, as_of=state.as_of,
            analysis_timestamp=state.final_report.analysis_timestamp,
            data_cutoff_timestamp=pkg.data_cutoff_timestamp,
            config_snapshot=self.config.snapshot(),
            prompt_hashes=all_prompt_hashes(self.config.prompt_versions),
            input_hash=pkg.input_hash(), report=state.final_report)
        state.decision_record_hash = rec["record_hash"]
