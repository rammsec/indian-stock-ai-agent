import pytest
from datetime import datetime, timezone

from backend.phase6.config import Phase6Config
from backend.phase6.contracts import (
    AgentReport,
    EvidenceRef,
    NumericClaim,
    ResearchManagerReport,
    ResearchThesis,
    RiskAssessment,
)
from backend.phase6.agents_base import (
    AgentContext,
    EvidencePackage,
    ManagerPackage,
    RiskPackage,
)
from backend.phase6.cost import CostTracker
from backend.phase6.debate import (
    BearResearcherAgent,
    BullResearcherAgent,
    ResearchManagerAgent,
    build_debate_user_prompt,
    completeness_ratio,
)
from backend.phase6.graph import AgentRegistry, ShadowResearchGraph
from backend.phase6.llm import MockLLMClient
from backend.phase6.risk import RiskManagerAgent


def _make_dummy_evidence_package(completeness=1.0):
    now = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)
    ev_ref = EvidenceRef(
        source_id="f:TEST:revenue_cr",
        dataset="dummy",
        path="dummy.json",
        record_key="key",
        retrieved_at=now,
    )
    analyst_reports = {}
    analysts = ["fundamental", "technical", "news", "promoter", "competitor", "quantitative"]
    num_included = int(round(completeness * len(analysts)))
    for i, a in enumerate(analysts):
        if i < num_included:
            analyst_reports[a] = AgentReport(
                agent_name=a,
                agent_version="1.0.0",
                prompt_version="v1",
                symbol="TEST",
                analysis_timestamp=now,
                data_cutoff_timestamp=now,
                findings=[f"{a} findings text with 100 cr value."],
                positive_factors=["positive factor"],
                negative_factors=["negative factor"],
                confidence=0.8,
                missing_information=[],
                warnings=[],
                numeric_claims=[],
            )

    return EvidencePackage(
        symbol="TEST",
        as_of="2026-11-19",
        analysis_timestamp=now,
        data_cutoff_timestamp=now,
        evidence=(ev_ref,),
        data={
            "fundamental": {"revenue_cr": 100.0, "evidence_ids": {"revenue_cr": "f:TEST:revenue_cr"}},
            "price_risk": {
                "volatility_ann_pct": 20.0,
                "var_95_1d_pct": 2.0,
                "max_drawdown_pct": 15.0,
                "avg_daily_turnover_cr": 100.0,
                "position_weight_pct": 3.0,
            },
            "market_regime": {"regime": "BULL", "india_vix_close": 14.0},
        },
        quantitative_predictions={"prob_next_day_up": 0.60, "confidence_score": 0.75},
        analyst_reports=analyst_reports,
        missing_analysts=tuple(analysts[num_included:]),
    )


def test_bull_bear_independence_and_identical_prompt():
    pkg = _make_dummy_evidence_package()
    prompt_bull = build_debate_user_prompt(pkg)
    prompt_bear = build_debate_user_prompt(pkg)
    # The prompt generated from package must be byte-identical
    assert prompt_bull == prompt_bear
    assert "TEST" in prompt_bull
    assert "revenue_cr" in prompt_bull

    # Test execution
    mock_llm = MockLLMClient()
    mock_llm.set("bull", {
        "side": "BULL",
        "thesis": "Strong growth",
        "catalysts": ["Earnings expansion"],
        "risks": ["Competition"],
        "challenges_to_other_side": ["Bear ignores moat"],
        "findings": ["Findings"],
        "positive_factors": ["Growth"],
        "negative_factors": [],
        "confidence": 0.85,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })
    mock_llm.set("bear", {
        "side": "BEAR",
        "thesis": "Overvalued",
        "catalysts": [],
        "risks": ["Valuation bubble"],
        "challenges_to_other_side": ["Bull ignores macro"],
        "findings": ["Findings"],
        "positive_factors": [],
        "negative_factors": ["Valuation"],
        "confidence": 0.70,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })

    ctx = AgentContext(run_id="r1", config=Phase6Config(), llm=mock_llm, cost_tracker=CostTracker())
    bull_agent = BullResearcherAgent()
    bear_agent = BearResearcherAgent()

    res_bull = bull_agent.run(pkg, ctx)
    res_bear = bear_agent.run(pkg, ctx)

    assert isinstance(res_bull, ResearchThesis)
    assert res_bull.side == "BULL"
    assert isinstance(res_bear, ResearchThesis)
    assert res_bear.side == "BEAR"

    # Verify Bull cannot see Bear output
    bull_call = [c for c in mock_llm.call_log if c["agent"] == "bull"][0]
    assert "Overvalued" not in bull_call["user"]


def test_research_manager_consolidation():
    pkg = _make_dummy_evidence_package()
    now = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)
    bull_thesis = ResearchThesis(
        agent_name="bull",
        agent_version="1.0.0",
        prompt_version="v1",
        symbol="TEST",
        analysis_timestamp=now,
        data_cutoff_timestamp=now,
        side="BULL",
        thesis="Bull thesis text",
        catalysts=["C1"],
        risks=["R1"],
        challenges_to_other_side=["Ch1"],
        findings=["Findings"],
        positive_factors=["P1"],
        negative_factors=[],
        confidence=0.85,
        missing_information=[],
        warnings=[],
        numeric_claims=[],
    )
    bear_thesis = ResearchThesis(
        agent_name="bear",
        agent_version="1.0.0",
        prompt_version="v1",
        symbol="TEST",
        analysis_timestamp=now,
        data_cutoff_timestamp=now,
        side="BEAR",
        thesis="Bear thesis text",
        catalysts=[],
        risks=["R2"],
        challenges_to_other_side=["Ch2"],
        findings=["Findings"],
        positive_factors=[],
        negative_factors=["N1"],
        confidence=0.65,
        missing_information=[],
        warnings=[],
        numeric_claims=[],
    )

    mgr_pkg = ManagerPackage(
        symbol=pkg.symbol,
        as_of=pkg.as_of,
        analysis_timestamp=pkg.analysis_timestamp,
        data_cutoff_timestamp=pkg.data_cutoff_timestamp,
        evidence=pkg.evidence,
        data=pkg.data,
        quantitative_predictions=pkg.quantitative_predictions,
        analyst_reports=pkg.analyst_reports,
        missing_analysts=pkg.missing_analysts,
        bull=bull_thesis,
        bear=bear_thesis,
    )

    mock_llm = MockLLMClient()
    mock_llm.set("research_manager", {
        "investment_thesis": "Favourable overall setup",
        "bull_case": "Bull argument",
        "bear_case": "Bear argument",
        "fundamental_assessment": "Solid fundamentals",
        "technical_assessment": "Good trend",
        "news_assessment": "Clean",
        "promoter_assessment": "Good",
        "competitor_assessment": "Leader",
        "quantitative_model_predictions": {},
        "risk_assessment": None,
        "agent_disagreements": ["Valuation disagreement"],
        "overall_conclusion": "FAVOURABLE",
        "findings": ["Findings"],
        "positive_factors": ["Growth"],
        "negative_factors": [],
        "confidence": 0.8,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })

    ctx = AgentContext(run_id="r1", config=Phase6Config(), llm=mock_llm, cost_tracker=CostTracker())
    mgr_agent = ResearchManagerAgent()
    mgr_rep = mgr_agent.run(mgr_pkg, ctx)

    assert isinstance(mgr_rep, ResearchManagerReport)
    assert mgr_rep.overall_conclusion == "FAVOURABLE"
    assert mgr_rep.bull_case == "Bull argument"
    # Quantitative predictions must be deterministically copied from pkg, not LLM
    assert mgr_rep.quantitative_model_predictions == pkg.quantitative_predictions


def test_risk_manager_rejection_on_liquidity():
    # Construct a package with high liquidity risk (low turnover)
    now = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)
    risk_pkg = RiskPackage(
        symbol="TEST",
        as_of="2026-11-19",
        analysis_timestamp=now,
        data_cutoff_timestamp=now,
        evidence=(),
        data={
            "price_risk": {
                "volatility_ann_pct": 20.0,
                "var_95_1d_pct": 2.0,
                "max_drawdown_pct": 10.0,
                "avg_daily_turnover_cr": 2.0,  # Below high threshold of 5.0 -> HIGH liquidity risk!
                "position_weight_pct": 3.0,
            },
            "market_regime": {"regime": "BULL", "india_vix_close": 14.0},
        },
        quantitative_predictions={},
        analyst_reports={},
        missing_analysts=(),
        bull=None,
        bear=None,
        manager_report=None,
    )

    mock_llm = MockLLMClient()
    mock_llm.set("risk_manager", {
        "verdict": "APPROVE",  # LLM tries to approve!
        "downside_risk": {"level": "LOW", "note": ""},
        "volatility_risk": {"level": "LOW", "note": ""},
        "liquidity_risk": {"level": "LOW", "note": ""},
        "concentration_risk": {"level": "LOW", "note": ""},
        "event_risk": {"level": "LOW", "note": ""},
        "drawdown_risk": {"level": "LOW", "note": ""},
        "risk_reward": {"level": "LOW", "note": ""},
        "rationale": "LLM thinks it is fine",
        "rejection_reasons": [],
        "findings": ["Findings"],
        "positive_factors": [],
        "negative_factors": [],
        "confidence": 0.8,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })

    ctx = AgentContext(run_id="r1", config=Phase6Config(), llm=mock_llm, cost_tracker=CostTracker())
    risk_agent = RiskManagerAgent()
    res = risk_agent.run(risk_pkg, ctx)

    assert isinstance(res, RiskAssessment)
    # Code must override LLM approval to REDUCE or REJECT due to turnover < 5.0 cr
    assert res.verdict in ("REDUCE", "REJECT")
    assert res.liquidity_risk.level == "HIGH"
