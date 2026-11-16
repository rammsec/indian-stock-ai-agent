"""Phase 5 Multi-Dimensional Stock Intelligence & News/Cross-Stock Tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import ENABLE_GEMINI_EXTRACTOR, IST_TZ
from backend.main import app
from backend.models import CrossStockImpactItem, StructuredNewsEvent
from backend.news_intel.collector import RawNewsItem
from backend.news_intel.combiner import build_multi_horizon_outlooks
from backend.news_intel.cross_stock import compute_cross_stock_impacts
from backend.news_intel.extractor import (
    compute_effective_trading_date,
    extract_structured_events,
    jaccard_similarity,
)
from backend.news_intel.features import NEWS_FEATURE_NAMES, compute_pit_news_features
from backend.news_intel.monitoring import build_monitoring_snapshot
from backend.news_intel.relationships import PILOT_UNIVERSE_5, RELATIONSHIP_MAP
from backend.storage.duckdb_store import DuckDBStore


def test_pilot_universe_and_relationship_map() -> None:
    assert PILOT_UNIVERSE_5 == ("RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE")
    for sym in PILOT_UNIVERSE_5:
        assert sym in RELATIONSHIP_MAP
        assert len(RELATIONSHIP_MAP[sym]["competitors"]) >= 3
        assert len(RELATIONSHIP_MAP[sym]["regulatory_exposures"]) >= 2
        assert len(RELATIONSHIP_MAP[sym]["global_macro_links"]) >= 2


def test_gemini_disabled_by_default() -> None:
    assert ENABLE_GEMINI_EXTRACTOR is False


def test_after_market_pit_effective_trading_date() -> None:
    # Friday 2026-04-10 15:45 IST (post-market) -> Monday 2026-04-13
    _, eff_post = compute_effective_trading_date("2026-04-10T15:45:00+05:30")
    assert eff_post == "2026-04-13"

    # Friday 2026-04-10 14:15 IST (intra-day) -> same day 2026-04-10
    _, eff_intra = compute_effective_trading_date("2026-04-10T14:15:00+05:30")
    assert eff_intra == "2026-04-10"


def test_deduplication_and_structured_extraction() -> None:
    raw_items = [
        RawNewsItem(
            symbol="TCS",
            company_name="Tata Consultancy Services Ltd",
            event_category="company_news",
            title="TCS wins $500M European BFSI cloud transformation contract",
            source="Economic Times",
            published="2026-04-10T10:00:00+05:30",
            url="https://example.com/1",
        ),
        RawNewsItem(
            symbol="TCS",
            company_name="Tata Consultancy Services Ltd",
            event_category="company_news",
            title="TCS wins $500M European BFSI cloud transformation contract deal",
            source="Moneycontrol",
            published="2026-04-10T11:00:00+05:30",
            url="https://example.com/2",
        ),
    ]
    assert jaccard_similarity(raw_items[0].title, raw_items[1].title) >= 0.55
    events = extract_structured_events(raw_items, enable_gemini=False)
    assert len(events) == 2
    assert isinstance(events[0], StructuredNewsEvent)
    assert events[0].is_duplicate_report is False
    assert events[1].is_duplicate_report is True
    assert events[1].canonical_event_id == events[0].event_id
    assert events[0].extraction_method == "deterministic"
    assert events[0].sentiment == "positive"


def test_17_news_features_and_strict_pit_cutoff() -> None:
    past_evt = StructuredNewsEvent(
        event_id="evt_past",
        company_name="ITC Ltd",
        ticker="ITC",
        event_category="company_news",
        event_description="ITC launches new FMCG product line with strong growth",
        original_source="Mint",
        publication_timestamp="2026-04-09T10:00:00+05:30",
        effective_trading_date="2026-04-09",
        sentiment="positive",
        impact_magnitude="high",
        classification_confidence=0.85,
        novelty_score=0.90,
        source_credibility=0.86,
    )
    future_evt = StructuredNewsEvent(
        event_id="evt_future",
        company_name="ITC Ltd",
        ticker="ITC",
        event_category="regulatory_government",
        event_description="Sudden excise tax penalty negative shock",
        original_source="Reuters",
        publication_timestamp="2026-04-12T10:00:00+05:30",
        effective_trading_date="2026-04-12",
        sentiment="negative",
        impact_magnitude="high",
        classification_confidence=0.95,
        novelty_score=0.95,
        source_credibility=0.90,
    )
    cutoff = datetime.fromisoformat("2026-04-10T15:30:00+05:30").astimezone(IST_TZ)
    feats = compute_pit_news_features([past_evt, future_evt], as_of_ts=cutoff)
    assert set(feats.keys()) == set(NEWS_FEATURE_NAMES)
    assert len(feats) == 17
    assert feats["news_company_sentiment_5d"] > 0.0
    # Future regulatory event must NOT leak into features before cutoff!
    assert feats["news_regulatory_risk_score"] == 0.0


def test_asymmetric_cross_stock_impacts() -> None:
    rival_win = StructuredNewsEvent(
        event_id="c1",
        company_name="Reliance Industries Ltd",
        ticker="RELIANCE",
        related_competitor_or_industry="BHARTIARTL",
        event_category="competitor",
        event_description="Bharti Airtel wins major enterprise 5G contract",
        original_source="ET",
        publication_timestamp="2026-04-10T10:00:00+05:30",
        effective_trading_date="2026-04-10",
        sentiment="positive",
        impact_magnitude="high",
        classification_confidence=0.85,
        novelty_score=0.9,
        source_credibility=0.86,
    )
    jio_capex = StructuredNewsEvent(
        event_id="r1",
        company_name="Reliance Industries Ltd",
        ticker="RELIANCE",
        event_category="company_news",
        event_description="Reliance Jio expands FTTH fiber rollout capex across India",
        original_source="Mint",
        publication_timestamp="2026-04-10T11:00:00+05:30",
        effective_trading_date="2026-04-10",
        sentiment="positive",
        impact_magnitude="high",
        classification_confidence=0.88,
        novelty_score=0.9,
        source_credibility=0.86,
    )
    impacts = compute_cross_stock_impacts({"RELIANCE": [rival_win, jio_capex]})
    rel_imp = impacts["RELIANCE"]
    birla_imp = impacts["BIRLACABLE"]
    assert any(c.expected_impact_direction == "negative" and c.relationship_type == "competitor" for c in rel_imp)
    assert any(c.expected_impact_direction == "positive" and c.relationship_type == "supplier_customer" for c in birla_imp)


def test_multi_horizon_combiner_and_high_vol_block() -> None:
    feats = {k: 0.0 for k in NEWS_FEATURE_NAMES}
    feats["news_company_sentiment_5d"] = 0.65
    feats["news_cross_stock_net_impact"] = 0.5
    q_pred = {
        "prob_next_day_up": 0.58,
        "expected_next_day_return_pct": 0.55,
        "expected_5d_return_pct": 1.65,
        "confidence_score": 72.0,
        "roundtrip_delivery_cost_pct": 0.22,
    }
    out = build_multi_horizon_outlooks("TCS", q_pred, feats, [], regime_label="Weak Bull", trade_eligible=True)
    assert set(out.keys()) == {"1D", "5D", "10D"}
    assert out["5D"].quant_signal == "BULLISH"
    assert out["5D"].news_signal == "BULLISH"
    assert out["5D"].decision == "BUY"

    blocked = build_multi_horizon_outlooks("TCS", q_pred, feats, [], regime_label="High Volatility", trade_eligible=True)
    assert blocked["1D"].decision == "NO TRADE"
    assert blocked["5D"].decision == "NO TRADE"


def test_duckdb_store_events_and_monitoring(tmp_path: Path) -> None:
    st = DuckDBStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
    )
    evt = StructuredNewsEvent(
        event_id="e_db_1",
        company_name="Birla Cable Ltd",
        ticker="BIRLACABLE",
        event_category="company_news",
        event_description="Birla Cable bags optical fiber order",
        original_source="BSE",
        publication_timestamp="2026-04-10T10:00:00+05:30",
        effective_trading_date="2026-04-10",
        sentiment="positive",
        impact_magnitude="high",
        classification_confidence=0.9,
        novelty_score=0.9,
        source_credibility=0.95,
    )
    ci = CrossStockImpactItem(
        affected_company="BIRLACABLE",
        originating_event_id="e_db_1",
        originating_entity="RELIANCE",
        relationship_type="supplier_customer",
        expected_impact_direction="positive",
        impact_magnitude="medium",
        confidence=0.8,
        rationale="Jio FTTH demand",
    )
    assert st.upsert_structured_events([evt]) == 1
    assert st.upsert_cross_stock_impacts([ci]) == 1
    df_ev = st.query_df("SELECT event_id, ticker FROM v_structured_events")
    df_ci = st.query_df("SELECT affected_company, relationship_type FROM v_cross_stock_impacts")
    assert len(df_ev) == 1 and df_ev.loc[0, "ticker"] == "BIRLACABLE"
    assert len(df_ci) == 1 and df_ci.loc[0, "relationship_type"] == "supplier_customer"

    mon = build_monitoring_snapshot(store=st)
    assert mon.drift_status == "INSUFFICIENT_DATA"


def test_api_stock_intelligence_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/api/stock-intelligence?refresh=false")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pilot_universe"] == ["RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE"]
    assert "stocks" in body and "monitoring" in body
