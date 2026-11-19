"""Shadow research runner and CLI for BharatAlpha Phase 6 Multi-Agent Intelligence Layer.

Runs multi-agent shadow research across pilot stocks:
- RELIANCE
- TCS
- TATAELXSI
- ITC
- BIRLACABLE

Maintains immutable decision logs, snapshot checks, and writes reports under data/phase6/.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.phase6.config import Phase6Config
from backend.phase6.cost import CostTracker
from backend.phase6.decision_log import DecisionLog
from backend.phase6.evidence import load_evidence
from backend.phase6.frozen_guard import assert_unchanged, snapshot_frozen_files
from backend.phase6.graph import AgentRegistry, ShadowResearchGraph
from backend.phase6.llm import MockLLMClient
from backend.phase6.providers import get_provider

logger = logging.getLogger(__name__)

PILOT_STOCKS = ("RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE")


def build_default_registry(config: Optional[Phase6Config] = None, llm: Any = None) -> AgentRegistry:
    from backend.phase6.analysts import (
        CompetitorAnalystAgent,
        FundamentalAnalystAgent,
        NewsAnalystAgent,
        PromoterAnalystAgent,
        TechnicalAnalystAgent,
    )
    from backend.phase6.debate import (
        BearResearcherAgent,
        BullResearcherAgent,
        ResearchManagerAgent,
    )
    from backend.phase6.quant_agent import QuantitativeAnalystAgent
    from backend.phase6.risk import RiskManagerAgent

    cfg = config or Phase6Config()
    analysts = {
        "fundamental": FundamentalAnalystAgent(cfg),
        "technical": TechnicalAnalystAgent(cfg),
        "news": NewsAnalystAgent(cfg),
        "promoter": PromoterAnalystAgent(cfg),
        "competitor": CompetitorAnalystAgent(cfg),
        "quantitative": QuantitativeAnalystAgent(cfg),
    }

    return AgentRegistry(
        analysts=analysts,
        bull=BullResearcherAgent(cfg),
        bear=BearResearcherAgent(cfg),
        manager=ResearchManagerAgent(cfg),
        risk=RiskManagerAgent(cfg),
    )


def run_shadow_research(
    symbols: tuple[str, ...] = PILOT_STOCKS,
    as_of: str = "2026-11-18",
    config: Optional[Phase6Config] = None,
    llm: Any = None,
    registry: Optional[AgentRegistry] = None,
    repo_root: str = ".",
    namespace_dir: Optional[str] = None,
) -> dict[str, Any]:
    cfg = config or Phase6Config.from_env()
    ns = Path(namespace_dir or cfg.namespace_dir)
    ns.mkdir(parents=True, exist_ok=True)

    # 1. Take frozen guard snapshot before run
    frozen_before = snapshot_frozen_files(root=repo_root)

    cost_tracker = CostTracker(cfg.model_prices)
    dlog = DecisionLog(directory=str(ns / "decisions"))

    if registry is None:
        registry = build_default_registry(cfg, llm)

    def _loader(sym: str, dt: str):
        return load_evidence(sym, dt, repo_root=repo_root)

    graph = ShadowResearchGraph(
        registry=registry,
        config=cfg,
        evidence_loader=_loader,
        llm=llm,
        cost_tracker=cost_tracker,
        decision_log=dlog,
        namespace_dir=str(ns),
    )

    results: dict[str, Any] = {}
    reports_dir = ns / "reports" / as_of
    reports_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        logger.info(f"Running Phase 6 shadow research for {sym} as_of {as_of}...")
        state = graph.run(symbol=sym, as_of=as_of)
        results[sym] = {
            "status": state.status,
            "overall_conclusion": state.final_report.overall_conclusion if state.final_report else "FAILED",
            "decision_record_hash": state.decision_record_hash,
            "warnings": state.warnings,
            "errors": state.errors,
        }

        # Save individual stock JSON & Markdown report
        if state.final_report:
            rep_dict = state.final_report.model_dump(mode="json")
            (reports_dir / f"{sym}.json").write_text(json.dumps(rep_dict, indent=2), encoding="utf-8")

            # Markdown presentation
            md_lines = [
                f"# BharatAlpha Phase 6 Multi-Agent Research Report: {sym}",
                f"**As Of:** {as_of} | **Generated:** {state.final_report.analysis_timestamp.isoformat()} | **Status:** {state.status}",
                f"**Overall Conclusion:** `{state.final_report.overall_conclusion}` | **Confidence:** `{state.final_report.confidence:.2f}`",
                "",
                "## 1. Investment Thesis",
                state.final_report.investment_thesis,
                "",
                "## 2. Debate Summary",
                f"### Bull Case\n{state.final_report.bull_case}",
                "",
                f"### Bear Case\n{state.final_report.bear_case}",
                "",
                "## 3. Specialist Analyst Assessments",
                f"### Fundamental\n{state.final_report.fundamental_assessment}",
                f"### Technical\n{state.final_report.technical_assessment}",
                f"### News Intelligence\n{state.final_report.news_assessment}",
                f"### Promoter & Governance\n{state.final_report.promoter_assessment}",
                f"### Competitor & Industry\n{state.final_report.competitor_assessment}",
                "",
                "## 4. Quantitative Model Predictions (Read-Only)",
                "```json",
                json.dumps(state.final_report.quantitative_model_predictions, indent=2),
                "```",
                "",
                "## 5. Risk Assessment",
            ]
            if state.final_report.risk_assessment:
                ra = state.final_report.risk_assessment
                md_lines.append(f"**Verdict:** `{ra.verdict}`")
                md_lines.append(f"**Rationale:** {ra.rationale}")
                if ra.rejection_reasons:
                    md_lines.append(f"**Rejection Reasons:** {', '.join(ra.rejection_reasons)}")
            else:
                md_lines.append("*Risk assessment unavailable.*")

            if state.final_report.agent_disagreements:
                md_lines.append("")
                md_lines.append("## 6. Agent Disagreements")
                for dis in state.final_report.agent_disagreements:
                    md_lines.append(f"- {dis}")

            (reports_dir / f"{sym}.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 2. Write cost summary
    cost_dir = ns / "cost"
    cost_dir.mkdir(parents=True, exist_ok=True)
    cost_summary = cost_tracker.summary()
    (cost_dir / f"{as_of}.json").write_text(json.dumps(cost_summary, indent=2), encoding="utf-8")

    # 3. Verify frozen files unchanged & write integrity report
    assert_unchanged(frozen_before, root=repo_root)
    integrity_dir = ns / "integrity"
    integrity_dir.mkdir(parents=True, exist_ok=True)
    integrity_data = {
        "as_of": as_of,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "frozen_files_checked": len(frozen_before),
        "status": "PASSED_UNTOUCHED",
    }
    (integrity_dir / f"{as_of}.json").write_text(json.dumps(integrity_data, indent=2), encoding="utf-8")

    return {
        "as_of": as_of,
        "symbols": list(symbols),
        "results": results,
        "cost": cost_summary,
        "integrity": integrity_data,
    }


def main():
    parser = argparse.ArgumentParser(description="BharatAlpha Phase 6 Multi-Agent Research Runner")
    parser.add_argument("--symbols", nargs="+", default=list(PILOT_STOCKS), help="Symbols to evaluate")
    parser.add_argument("--as-of", default="2026-11-18", help="As-of evaluation date YYYY-MM-DD")
    parser.add_argument("--provider", default=None, help="LLM Provider override")
    parser.add_argument("--mock", action="store_true", help="Use deterministic MockLLMClient")
    parser.add_argument("--namespace-dir", default="data/phase6", help="Namespace output directory")
    args = parser.parse_args()

    cfg = Phase6Config.from_env()
    if args.provider:
        cfg.provider = args.provider

    if args.mock:
        llm = MockLLMClient()
    else:
        llm = get_provider(cfg.provider)

    res = run_shadow_research(
        symbols=tuple(args.symbols),
        as_of=args.as_of,
        config=cfg,
        llm=llm,
        namespace_dir=args.namespace_dir,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
