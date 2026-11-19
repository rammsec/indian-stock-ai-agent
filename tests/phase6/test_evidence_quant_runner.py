import pytest
from datetime import datetime, timezone
from pathlib import Path

from backend.phase6.config import Phase6Config
from backend.phase6.contracts import AgentReport
from backend.phase6.evidence import load_evidence
from backend.phase6.quant_agent import QuantitativeAnalystAgent
from backend.phase6.runner import PILOT_STOCKS, build_default_registry, run_shadow_research
from backend.phase6.llm import MockLLMClient
from backend.phase6.agents_base import AgentContext
from backend.phase6.cost import CostTracker


def test_load_evidence_pilot_stocks():
    for sym in PILOT_STOCKS:
        pkg = load_evidence(sym, "2026-11-18")
        assert pkg.symbol == sym
        assert pkg.as_of == "2026-11-18"
        assert "fundamental" in pkg.data
        assert "technical" in pkg.data
        assert "news" in pkg.data
        assert "promoter" in pkg.data
        assert "competitor" in pkg.data
        assert "quantitative" in pkg.data
        assert "market_regime" in pkg.data
        assert "price_risk" in pkg.data
        assert len(pkg.evidence) > 0


def test_quantitative_analyst_agent_deterministic():
    pkg = load_evidence("RELIANCE", "2026-11-19")
    ctx = AgentContext(run_id="test", config=Phase6Config(), llm=MockLLMClient(), cost_tracker=CostTracker())
    agent = QuantitativeAnalystAgent()
    rep = agent.run(pkg, ctx)
    assert isinstance(rep, AgentReport)
    assert rep.agent_name == "quantitative"
    assert rep.confidence > 0.0
    assert len(rep.findings) > 0
    # Assert numbers claim evidence_ids
    assert len(rep.numeric_claims) > 0
    eids = {e.source_id for e in rep.evidence_refs}
    for c in rep.numeric_claims:
        assert c.evidence_id in eids


def test_run_shadow_research_with_mock(tmp_path):
    # Set up mock responses for the LLM agents
    mock_llm = MockLLMClient()
    mock_llm.set("fundamental", {
        "findings": ["Solid revenue growth and margins."],
        "positive_factors": ["High OPM"],
        "negative_factors": ["Moderate debt"],
        "confidence": 0.85,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })
    mock_llm.set("technical", {
        "findings": ["Bullish moving average alignment."],
        "positive_factors": ["Above SMA200"],
        "negative_factors": [],
        "confidence": 0.8,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })
    mock_llm.set("news", {
        "findings": ["Positive news coverage."],
        "positive_factors": ["Contract win"],
        "negative_factors": [],
        "confidence": 0.75,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })
    mock_llm.set("promoter", {
        "findings": ["Stable promoter holding."],
        "positive_factors": ["Zero pledging"],
        "negative_factors": [],
        "confidence": 0.85,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })
    mock_llm.set("competitor", {
        "findings": ["Strong moat."],
        "positive_factors": ["Market leader"],
        "negative_factors": [],
        "confidence": 0.8,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": []
    })
    mock_llm.set("bull", {
        "findings": ["Strong thesis."],
        "positive_factors": ["Growth"],
        "negative_factors": [],
        "confidence": 0.8,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": [],
        "side": "BULL",
        "thesis": "High earnings potential",
        "catalysts": ["New contracts"],
        "risks": ["Execution delay"],
        "challenges_to_other_side": ["Bear case too pessimistic"]
    })
    mock_llm.set("bear", {
        "findings": ["Caution warranted."],
        "positive_factors": [],
        "negative_factors": ["Valuation elevated"],
        "confidence": 0.7,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": [],
        "side": "BEAR",
        "thesis": "Valuation headroom limited",
        "catalysts": [],
        "risks": ["Margin pressure"],
        "challenges_to_other_side": ["Growth projections too aggressive"]
    })
    mock_llm.set("research_manager", {
        "findings": ["Overall favourable setup."],
        "positive_factors": ["Earnings growth"],
        "negative_factors": ["Rich valuation"],
        "confidence": 0.8,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": [],
        "investment_thesis": "Selective buy for medium-term",
        "bull_case": "Expanding ROE",
        "bear_case": "Margin contraction risk",
        "fundamental_assessment": "Robust balance sheet",
        "technical_assessment": "Upward momentum",
        "news_assessment": "Positive news flow",
        "promoter_assessment": "High promoter stake",
        "competitor_assessment": "Peer ahead",
        "quantitative_model_predictions": {},
        "risk_assessment": None,
        "agent_disagreements": ["Valuation vs Growth"],
        "overall_conclusion": "FAVOURABLE"
    })
    mock_llm.set("risk_manager", {
        "findings": ["Manageable risk."],
        "positive_factors": ["Adequate liquidity"],
        "negative_factors": [],
        "confidence": 0.85,
        "missing_information": [],
        "warnings": [],
        "numeric_claims": [],
        "verdict": "APPROVE",
        "downside_risk": {"level": "LOW", "note": "Low VaR"},
        "volatility_risk": {"level": "LOW", "note": "Below 25%"},
        "liquidity_risk": {"level": "LOW", "note": "High turnover"},
        "concentration_risk": {"level": "LOW", "note": "3% weight"},
        "event_risk": {"level": "LOW", "note": "No imminent event"},
        "drawdown_risk": {"level": "LOW", "note": "Minimal DD"},
        "risk_reward": {"level": "LOW", "note": "Favourable ratio"},
        "rationale": "All dimensions within safety limits",
        "rejection_reasons": []
    })

    res = run_shadow_research(
        symbols=("RELIANCE",),
        as_of="2026-11-18",
        config=Phase6Config(namespace_dir=str(tmp_path)),
        llm=mock_llm,
        namespace_dir=str(tmp_path),
    )

    assert "RELIANCE" in res["results"]
    assert res["results"]["RELIANCE"]["status"] in ("COMPLETE", "PARTIAL")
    assert (tmp_path / "reports" / "2026-11-18" / "RELIANCE.json").exists()
    assert (tmp_path / "reports" / "2026-11-18" / "RELIANCE.md").exists()
    assert (tmp_path / "cost" / "2026-11-18.json").exists()
    assert (tmp_path / "integrity" / "2026-11-18.json").exists()
    assert (tmp_path / "decisions" / "decisions.jsonl").exists()
