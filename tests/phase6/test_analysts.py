"""Deterministic tests for the five Phase 6 analysts (MockLLMClient only; no network)."""
import copy
from datetime import datetime, timezone

import pytest

from backend.phase6 import analysts as A
from backend.phase6.agents_base import AgentContext, EvidencePackage
from backend.phase6.config import Phase6Config
from backend.phase6.contracts import EvidenceRef
from backend.phase6.cost import CostTracker
from backend.phase6.llm import AgentTimeout, InvalidStructuredOutput, MockLLMClient

CUT = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
ANA = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)
CFG = Phase6Config(max_retries=0, timeout_s=5.0, quick_think_model="mock")


def ids(prefix, fields):
    return {f: f"{prefix}_{f}" for f in fields}


FUND = dict(revenue_cr=1000.0, profit_cr=150.0, revenue_growth_pct=12.5, profit_growth_pct=-4.0,
            opm_pct=22.0, net_margin_pct=15.0, operating_cash_flow_cr=180.0,
            free_cash_flow_cr=-20.0, debt_to_equity=0.3, current_ratio=1.8, interest_coverage=9.0,
            pe=24.0, pb=3.5, roe_pct=18.0, roce_pct=21.0, piotroski_score=7)
TECH = dict(close=100.0, rsi_14=72.0, macd_hist=0.8, sma_20=98.0, sma_50=95.0, sma_200=85.0,
            atr_pct=2.1, return_1m_pct=4.0, return_3m_pct=9.0, volume_surge_ratio=1.8,
            support=90.0, resistance=110.0, adx=30.0, trend="up")
PROM = dict(promoters_pct=55.0, promoter_change_qoq_pct=-0.5, pledged_pct=30.0, fii_pct=12.0,
            dii_pct=15.0, insider_transactions=[{"x": 1}], management_commentary="steady",
            leadership_changes=[], governance_flags=[])
COMP = dict(competitors=["INFY", "WIPRO"], peer_metrics={"INFY": {"pe": 25.0}},
            cross_stock_impacts=[dict(source_symbol="INFY", target_symbol="TCS",
                                      relation="peer", direction="negative", rationale="deal win",
                                      evidence_id="imp1")], industry_growth_pct=8.0)


def ev(i, cat, **kw):
    d = dict(event_id=i, headline=f"headline {i}", category=cat, sentiment="positive",
             impact_magnitude=0.6, publication_timestamp="2025-01-14T10:00:00Z",
             effective_trading_date="2025-01-15", is_duplicate_report=False, evidence_id=f"n_{i}")
    d.update(kw)
    return d


def full_data():
    d = {
        "fundamental": dict(FUND, evidence_ids=ids("f", FUND)),
        "technical": dict(TECH, evidence_ids=ids("t", TECH)),
        "market_regime": {"regime": "bull", "india_vix_close": 13.5,
                          "evidence_ids": ids("m", ["regime", "india_vix_close"])},
        "promoter": dict(PROM, evidence_ids=ids("p", PROM)),
        "competitor": dict(COMP, evidence_ids=ids("c", COMP)),
        "news": {"events": [ev("e1", "company_news"), ev("e2", "company_news", is_duplicate_report=True),
                            ev("e3", "competitor"),
                            ev("e4", "industry_sector", headline="TCS share price today live")],
                 "quarantined_count": 2, "evidence_ids": {}},
    }
    return d


def make_pkg(data):
    eids = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "evidence_ids" and isinstance(v, dict):
                    eids.update(x for x in v.values() if x)
                elif k == "evidence_id" and isinstance(v, str):
                    eids.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(data)
    refs = tuple(EvidenceRef(source_id=i, dataset="d", path="p", record_key=i,
                             published_at=datetime(2025, 1, 14, tzinfo=timezone.utc),
                             retrieved_at=datetime(2025, 1, 15, 9, tzinfo=timezone.utc))
                 for i in sorted(eids))
    return EvidencePackage(symbol="TCS", as_of="2025-01-15", analysis_timestamp=ANA,
                           data_cutoff_timestamp=CUT, evidence=refs, data=data)


def good(**kw):
    d = dict(findings=["Qualitative synthesis based on the calculated facts."],
             positive_factors=["supportive picture"], negative_factors=["some caution"],
             confidence=0.9, missing_information=[], warnings=[], numeric_claims=[])
    d.update(kw)
    return d


def ctx_for(agent_name, resp, **kw):
    llm = MockLLMClient()
    llm.set(agent_name, resp, **kw)
    return AgentContext(run_id="r", config=CFG, llm=llm, cost_tracker=CostTracker()), llm


AGENTS = [(A.FundamentalAnalystAgent, "fundamental", len(A.FundamentalAnalystAgent.FIELDS)),
          (A.TechnicalAnalystAgent, "technical", 16),
          (A.NewsAnalystAgent, "news", None),
          (A.PromoterAnalystAgent, "promoter", 9),
          (A.CompetitorAnalystAgent, "competitor", 4)]
IDS = [a[1] for a in AGENTS]


def test_agent_names_and_constructor():
    for cls, name, _ in AGENTS:
        a = cls()
        assert a.name == name and isinstance(a, A.AnalystAgent)
        assert cls(config=CFG).config is CFG


@pytest.mark.parametrize("cls,name,_n", AGENTS, ids=IDS)
def test_happy_path(cls, name, _n):
    pkg = make_pkg(full_data())
    ctx, llm = ctx_for(name, good(confidence=0.95))
    rep = cls(CFG).run(pkg, ctx)
    assert llm.calls(name) == 1
    assert rep.agent_name == name and rep.symbol == "TCS"
    assert rep.findings[-1] == "Qualitative synthesis based on the calculated facts."
    assert any(f.startswith("[calc]") for f in rep.findings)
    assert rep.prompt_version.startswith("v1+sha256:")
    assert ctx.cost_tracker.summary()["per_agent"][name]["calls"] == 1
    refs = {r.source_id for r in rep.evidence_refs}
    assert all(c.evidence_id in refs for c in rep.numeric_claims)
    assert "NO DATA" not in rep.warnings
    # user digest actually carries the missing list + facts
    assert "missing_information" in llm.call_log[0]["user"]


def test_fundamental_deterministic_facts_and_claims():
    pkg = make_pkg(full_data())
    ctx, _ = ctx_for("fundamental", good(confidence=1.0))
    rep = A.FundamentalAnalystAgent(CFG).run(pkg, ctx)
    txt = " | ".join(rep.findings)
    assert "profit growth -4% (declining)" in txt
    assert "revenue growth 12.5% (growing)" in txt
    assert "operating margin 22% (high)" in txt
    assert "free cash flow -20 cr (non-positive)" in txt
    claims = {c.name: c for c in rep.numeric_claims}
    assert claims["revenue_growth_pct"].value == 12.5
    assert claims["revenue_growth_pct"].evidence_id == "f_revenue_growth_pct"
    assert rep.confidence == 0.95 or rep.confidence == 1.0  # complete data -> not capped


def test_technical_facts():
    pkg = make_pkg(full_data())
    ctx, _ = ctx_for("technical", good())
    rep = A.TechnicalAnalystAgent(CFG).run(pkg, ctx)
    txt = " | ".join(rep.findings)
    assert "RSI(14) 72 zone: overbought" in txt
    assert "MA alignment: bullish" in txt
    assert "close is 11.11% above support 90" in txt or "11.11% above support 90" in txt
    assert "resistance 110 is 10% above close" in txt
    assert "market regime: bull" in txt and "India VIX close 13.5" in txt
    names = {c.name: c for c in rep.numeric_claims}
    assert names["pct_above_support"].evidence_id == "t_support"
    assert names["india_vix_close"].evidence_id == "m_india_vix_close"


def test_technical_support_broken():
    d = full_data()
    d["technical"]["close"] = 80.0
    ctx, _ = ctx_for("technical", good())
    rep = A.TechnicalAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert any("support broken" in f for f in rep.findings)
    assert any("bearish" in f or "MA alignment: mixed" in f for f in rep.findings)


@pytest.mark.parametrize("cls,name,_n", AGENTS, ids=IDS)
def test_missing_section_no_llm_call(cls, name, _n):
    d = full_data()
    d.pop(name)
    ctx, llm = ctx_for(name, good())
    rep = cls(CFG).run(make_pkg(d), ctx)
    assert llm.calls(name) == 0 and llm.call_log == []
    assert rep.confidence <= 0.1
    assert "NO DATA" in rep.warnings
    assert rep.missing_information[0] == f"{name}: entire section missing"
    assert rep.evidence_refs == [] and rep.numeric_claims == []
    assert rep.analysis_timestamp == ANA and rep.data_cutoff_timestamp == CUT


def test_all_none_section_is_no_data():
    d = full_data()
    d["fundamental"] = {f: None for f in FUND} | {"evidence_ids": {}}
    ctx, llm = ctx_for("fundamental", good())
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert llm.calls("fundamental") == 0 and "NO DATA" in rep.warnings
    assert len(rep.missing_information) == 1 + len(FUND)


def test_fundamental_partial_missing_exact_and_confidence_cap():
    d = full_data()
    for f in ("pe", "roe_pct", "debt_to_equity", "piotroski_score"):
        d["fundamental"][f] = None
    d["fundamental"].pop("pb")                       # absent key == missing
    ctx, llm = ctx_for("fundamental", good(confidence=0.99))
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(d), ctx)
    expected = ["fundamental.debt_to_equity", "fundamental.pe", "fundamental.pb",
                "fundamental.roe_pct", "fundamental.piotroski_score"]
    assert rep.missing_information == expected
    assert rep.confidence == pytest.approx(11 / 16)
    assert not any(c.name in ("pe", "pb") for c in rep.numeric_claims)


def test_technical_partial_missing_includes_regime_context():
    d = full_data()
    d["technical"]["adx"] = None
    d["market_regime"]["regime"] = None
    ctx, _ = ctx_for("technical", good())
    rep = A.TechnicalAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert rep.missing_information == ["technical.adx", "market_regime.regime"]
    assert rep.confidence == pytest.approx(14 / 16)


def test_promoter_partial_missing_exact():
    d = full_data()
    d["promoter"]["insider_transactions"] = None
    d["promoter"]["fii_pct"] = None
    ctx, _ = ctx_for("promoter", good())
    rep = A.PromoterAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert rep.missing_information == ["promoter.fii_pct", "promoter.insider_transactions"]


def test_competitor_partial_missing_exact():
    d = full_data()
    d["competitor"]["peer_metrics"] = None
    d["competitor"]["industry_growth_pct"] = None
    ctx, _ = ctx_for("competitor", good())
    rep = A.CompetitorAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert rep.missing_information == ["competitor.peer_metrics", "competitor.industry_growth_pct"]
    assert rep.confidence == pytest.approx(0.5)
    assert any("peer metrics: unavailable" in f for f in rep.findings)


# ---- LLM post-validation ---------------------------------------------------------------
def test_unknown_evidence_claim_stripped_and_flagged():
    resp = good(numeric_claims=[
        {"name": "pe_fake", "value": 24.0, "evidence_id": "ghost_id"},
        {"name": "pe_ok", "value": 24.0, "evidence_id": "f_pe"},
        {"name": "invented_value", "value": 777.0, "evidence_id": "f_pe"}])
    ctx, _ = ctx_for("fundamental", resp)
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    names = [c.name for c in rep.numeric_claims]
    assert "pe_fake" not in names and "invented_value" not in names and "pe_ok" in names
    assert any("ghost_id" in w for w in rep.warnings)
    assert any("invented_value" in w for w in rep.warnings)


def test_fabricated_number_in_prose_warns_and_downgrades():
    clean_ctx, _ = ctx_for("fundamental", good(confidence=0.8))
    clean = A.FundamentalAnalystAgent(CFG).run(make_pkg(full_data()), clean_ctx)
    resp = good(confidence=0.8, findings=["Target price of 4321 implies strong upside, PE 24 fair."])
    ctx, _ = ctx_for("fundamental", resp)
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert any("fabricated" in w and "4321" in w for w in rep.warnings)
    assert not any("fabricated" in w for w in clean.warnings)
    assert rep.confidence == pytest.approx(clean.confidence * 0.5)


def test_rounding_tolerant_numbers_not_flagged():
    resp = good(findings=["Revenue growth about 12.5% and PE roughly 24, ROCE 21 percent; 12 or 13."])
    ctx, _ = ctx_for("fundamental", resp)
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert not any("fabricated" in w for w in rep.warnings)


def test_llm_cannot_drop_missing_information():
    d = full_data()
    d["fundamental"]["pe"] = None
    resp = good(missing_information=["something extra"])
    ctx, _ = ctx_for("fundamental", resp)
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert rep.missing_information == ["fundamental.pe", "something extra"]
    ctx2, _ = ctx_for("fundamental", good(missing_information=[]))
    rep2 = A.FundamentalAnalystAgent(CFG).run(make_pkg(d), ctx2)
    assert rep2.missing_information == ["fundamental.pe"]


def test_confidence_capped_by_completeness_even_if_llm_overconfident():
    d = full_data()
    for f in list(FUND)[:8]:
        d["fundamental"][f] = None
    ctx, _ = ctx_for("fundamental", good(confidence=1.0))
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert rep.confidence == pytest.approx(0.5)


def test_invalid_json_after_repair_raises():
    ctx, llm = ctx_for("technical", good(), invalid_first=2)
    with pytest.raises(InvalidStructuredOutput):
        A.TechnicalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert llm.calls("technical") == 2          # one repair retry


def test_invalid_json_once_is_repaired():
    ctx, llm = ctx_for("technical", good(), invalid_first=1)
    rep = A.TechnicalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert llm.calls("technical") == 2 and rep.agent_name == "technical"


def test_out_of_range_confidence_is_invalid_structure():
    ctx, _ = ctx_for("technical", good(confidence=1.7))
    with pytest.raises(InvalidStructuredOutput):
        A.TechnicalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)


def test_empty_findings_rejected():
    ctx, _ = ctx_for("technical", good(findings=[" "]))
    with pytest.raises(A.AnalystOutputRejected):
        A.TechnicalAnalystAgent(CFG).run(make_pkg(full_data()), ctx)


def test_timeout_raises_agent_timeout():
    cfg = Phase6Config(max_retries=0, timeout_s=0.05, quick_think_model="mock")
    llm = MockLLMClient()
    llm.set("promoter", good(), delay_s=0.5)
    ctx = AgentContext(run_id="r", config=cfg, llm=llm, cost_tracker=CostTracker())
    with pytest.raises(AgentTimeout):
        A.PromoterAnalystAgent(cfg).run(make_pkg(full_data()), ctx)


def test_timestamps_overridden_to_package_values():
    resp = good()
    resp.update(analysis_timestamp="2030-01-01T00:00:00Z",
                data_cutoff_timestamp="2031-01-01T00:00:00Z", agent_name="evil")
    ctx, _ = ctx_for("competitor", resp)
    rep = A.CompetitorAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert rep.analysis_timestamp == ANA and rep.data_cutoff_timestamp == CUT
    assert rep.agent_name == "competitor"


def test_evidence_after_cutoff_refused():
    from backend.phase6.contracts import TimestampViolation
    pkg = make_pkg(full_data())
    late = EvidenceRef(source_id="f_pe", dataset="d", path="p", record_key="k",
                       published_at=datetime(2025, 1, 16, tzinfo=timezone.utc),
                       retrieved_at=datetime(2025, 1, 15, 9, tzinfo=timezone.utc))
    pkg2 = pkg.model_copy(update={"evidence": tuple(late if r.source_id == "f_pe" else r
                                                    for r in pkg.evidence)})
    ctx, _ = ctx_for("fundamental", good())
    with pytest.raises(TimestampViolation):
        A.FundamentalAnalystAgent(CFG).run(pkg2, ctx)


def test_missing_evidence_id_omits_claim_with_warning():
    d = full_data()
    d["fundamental"]["evidence_ids"].pop("pe")
    ctx, _ = ctx_for("fundamental", good())
    rep = A.FundamentalAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert not any(c.name == "pe" for c in rep.numeric_claims)
    assert any("'pe' omitted" in w for w in rep.warnings)


def test_package_not_mutated():
    pkg = make_pkg(full_data())
    before = pkg.input_hash()
    for cls, name, _ in AGENTS:
        ctx, _l = ctx_for(name, good())
        cls(CFG).run(pkg, ctx)
    assert pkg.input_hash() == before


def test_agent_only_sees_its_own_section():
    ctx, llm = ctx_for("promoter", good())
    A.PromoterAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    user = llm.call_log[0]["user"]
    assert "pledged_pct" in user and "revenue_growth_pct" not in user and "rsi_14" not in user


# ---- news ------------------------------------------------------------------------------
def test_news_ignores_duplicates_generic_and_reports_empty_categories():
    ctx, llm = ctx_for("news", good())
    rep = A.NewsAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    f = {x.split(":")[0].replace("[calc] ", ""): x for x in rep.findings if x.startswith("[calc]")}
    assert "1 event(s) (positive=1); 1 duplicate report(s) ignored" in f["company_news"]
    assert "no events" in f["promoter_management"]
    assert "no events" in f["regulatory_government"] and "no events" in f["global_macro"]
    assert "no events; 0" not in f["industry_sector"]
    assert "no events" in f["industry_sector"] and "generic price-page" in f["industry_sector"]
    assert any("generic price-page style headline" in w for w in rep.warnings)
    assert "NOT positive for the target" in f["competitor"]
    assert any("competitor's win is NOT positive" in w for w in rep.warnings)
    claim_names = [c.name for c in rep.numeric_claims]
    assert claim_names == ["impact_magnitude[e1]", "impact_magnitude[e3]"]   # e2 dup, e4 generic
    refs = {r.source_id for r in rep.evidence_refs}
    assert "n_e1" in refs and "n_e2" not in refs
    assert any("quarantined items excluded: 2" in x for x in rep.findings)


def test_news_empty_events_all_categories_no_events():
    d = full_data()
    d["news"] = {"events": [], "quarantined_count": 0, "evidence_ids": {}}
    ctx, llm = ctx_for("news", good())
    rep = A.NewsAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert llm.calls("news") == 1
    calc = [x for x in rep.findings if x.startswith("[calc]")]
    assert sum("no events" in x for x in calc) == 6
    assert rep.missing_information == ["news.events"]
    assert rep.confidence == pytest.approx(0.5)


def test_news_missing_event_fields_listed():
    d = full_data()
    d["news"]["events"][0]["sentiment"] = None
    ctx, _ = ctx_for("news", good())
    rep = A.NewsAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert rep.missing_information == ["news.events[e1].sentiment"]


def test_news_events_none_is_no_data():
    d = full_data()
    d["news"] = {"events": None, "quarantined_count": None, "evidence_ids": {}}
    ctx, llm = ctx_for("news", good())
    rep = A.NewsAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert llm.calls("news") == 0 and "NO DATA" in rep.warnings


# ---- promoter --------------------------------------------------------------------------
def test_promoter_insider_none_reports_unavailable():
    d = full_data()
    d["promoter"]["insider_transactions"] = None
    ctx, llm = ctx_for("promoter", good())
    rep = A.PromoterAnalystAgent(CFG).run(make_pkg(d), ctx)
    calc = " | ".join(x for x in rep.findings if x.startswith("[calc]"))
    assert "insider transactions: unavailable" in calc
    assert "record(s) reported" not in calc
    assert "promoter.insider_transactions" in rep.missing_information
    assert "insider_transactions\": null" in llm.call_log[0]["user"]


@pytest.mark.parametrize("pct,lvl", [(0.0, "none"), (10.0, "low"), (30.0, "high"), (60.0, "critical")])
def test_promoter_pledge_thresholds(pct, lvl):
    d = full_data()
    d["promoter"]["pledged_pct"] = pct
    ctx, _ = ctx_for("promoter", good())
    rep = A.PromoterAnalystAgent(CFG).run(make_pkg(d), ctx)
    assert any(f"{lvl} pledge risk" in x for x in rep.findings)


def test_promoter_empty_lists_are_present_not_missing():
    ctx, _ = ctx_for("promoter", good())
    rep = A.PromoterAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert rep.missing_information == []
    assert any("governance flags: none raised" in x for x in rep.findings)


# ---- competitor ------------------------------------------------------------------------
def test_competitor_impacts_cited():
    ctx, _ = ctx_for("competitor", good())
    rep = A.CompetitorAnalystAgent(CFG).run(make_pkg(full_data()), ctx)
    assert any("INFY -> TCS: peer, direction negative" in x for x in rep.findings)
    assert "imp1" in {r.source_id for r in rep.evidence_refs}


def test_prompt_hash_stable_and_in_version():
    from backend.phase6.prompt_text import ANALYST_PROMPTS_V1, ANALYST_PROMPTS_V1_SHA256
    assert set(ANALYST_PROMPTS_V1) == {"fundamental", "technical", "news", "promoter", "competitor"}
    assert len(ANALYST_PROMPTS_V1_SHA256) == 64
    assert A.NewsAnalystAgent.prompt_version.endswith(ANALYST_PROMPTS_V1_SHA256[:16])
