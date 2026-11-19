"""Regression tests for all News Intelligence Quality defects (Phase 5.5 Audit).

Tests the isolated prototype implementations in `backend/news_intel/quality_audit.py`
without modifying any frozen production modules or the 17 numerical news features.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from backend.config import IST_TZ
from backend.models import StructuredNewsEvent
from backend.news_intel.collector import RawNewsItem
from backend.news_intel.quality_audit import (
    audit_news_intelligence_datasets,
    classify_sentiment_v2,
    compute_cross_stock_impacts_v2,
    compute_effective_trading_date_v2,
    extract_structured_events_v2,
    is_truncated_rss_timestamp,
    jaccard_similarity_v2,
    normalize_headline_v2,
    parse_rss_timestamp_ist_v2,
    resolve_category_and_entity_v2,
    strip_source_suffix,
)


def test_full_rfc2822_gmt_to_ist_and_truncated_recovery() -> None:
    # Full Google News GMT string must parse to +05:30 IST (10:15 GMT -> 15:45 IST -> next trading day)
    full_gmt = "Fri, 13 Nov 2026 10:15:00 GMT"
    dt_ist = parse_rss_timestamp_ist_v2(full_gmt)
    assert dt_ist is not None
    assert dt_ist.tzinfo == IST_TZ
    assert dt_ist.hour == 15 and dt_ist.minute == 45

    pub_iso, eff_date = compute_effective_trading_date_v2(full_gmt)
    assert pub_iso == "2026-11-13T15:45:00+05:30"
    assert eff_date == "2026-11-16"  # After 15:30 IST on Friday -> Monday

    # Legacy [:22]-truncated RSS string must be detected and recovered from UTC -> IST
    trunc = "Fri, 13 Nov 2026 10:15"
    assert is_truncated_rss_timestamp(trunc) is True
    dt_recovered = parse_rss_timestamp_ist_v2(trunc, assume_truncated_rfc2822_is_utc=True)
    assert dt_recovered == dt_ist

    # Missing/invalid timestamps must still return None (never datetime.now())
    assert parse_rss_timestamp_ist_v2("") is None
    assert parse_rss_timestamp_ist_v2("not-a-timestamp") is None


def test_source_suffix_stripping_prevents_insider_false_positive_and_dedup_dilution() -> None:
    title = "Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech - PRESS Insider"
    clean, src = strip_source_suffix(title, known_source="PRESS Insider")
    assert clean == "Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech"
    assert src == "PRESS Insider"

    # Must NOT match \binsider\b from publisher "- PRESS Insider" as promoter_management
    raw = RawNewsItem(
        symbol="TCS",
        company_name="Tata Consultancy Services Ltd",
        event_category="promoter_management",
        title=title,
        source="PRESS Insider",
        published="Wed, 18 Nov 2026 05:45:00 GMT",
        url="https://example.com/cognizant",
    )
    cat, ent, rej = resolve_category_and_entity_v2(raw)
    assert rej is None
    assert cat == "competitor"
    assert ent in ("HCLTECH", "COGNIZANT")

    # Cross-source identical headlines must normalize identically regardless of "- Indiatimes" vs "- ET Telecom"
    a = "L&T Technology Services, Nokia sign five-year engineering deal in India - Indiatimes"
    b = "L&T Technology Services, Nokia sign five-year engineering deal in India - ET Telecom"
    assert normalize_headline_v2(a, "Indiatimes") == normalize_headline_v2(b, "ET Telecom")
    assert jaccard_similarity_v2(a, b, "Indiatimes", "ET Telecom") == 1.0


def test_cross_category_deduplication_and_no_self_referential_canonical_id() -> None:
    pub = "Fri, 30 Oct 2026 01:30:00 GMT"
    items = [
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="company_news",
            title="RIL posts record Q2 revenue on strong Jio, O2C show; PAT rises 41% to Rs 25,548 crore - The New Indian Express",
            source="The New Indian Express",
            published=pub,
            url="https://example.com/1",
        ),
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="promoter_management",
            title="RIL posts record Q2 revenue on strong Jio, O2C show; PAT rises 41% to Rs 25,548 crore - The New Indian Express",
            source="The New Indian Express",
            published=pub,
            url="https://example.com/1",
        ),
        RawNewsItem(
            symbol="TATAELXSI",
            company_name="Tata Elxsi Ltd",
            event_category="company_news",
            title="Tata Elxsi expands European SDV engineering hub in Germany - Reuters",
            source="Reuters",
            published=pub,
            url="https://example.com/2",
        ),
        RawNewsItem(
            symbol="TATAELXSI",
            company_name="Tata Elxsi Ltd",
            event_category="promoter_management",
            title="Tata Elxsi expands European SDV engineering hub in Germany - Mint",
            source="Mint",
            published=pub,
            url="https://example.com/3",
        ),
    ]
    events, quarantined = extract_structured_events_v2(items)
    assert quarantined == []
    assert len(events) == 4

    rel_evts = [e for e in events if e.ticker == "RELIANCE"]
    assert rel_evts[0].is_duplicate_report is False
    assert rel_evts[1].is_duplicate_report is True
    assert rel_evts[1].canonical_event_id == rel_evts[0].event_id
    assert rel_evts[1].event_id != rel_evts[1].canonical_event_id

    elxsi_evts = [e for e in events if e.ticker == "TATAELXSI"]
    assert elxsi_evts[0].is_duplicate_report is False
    assert elxsi_evts[1].is_duplicate_report is True
    assert elxsi_evts[1].canonical_event_id == elxsi_evts[0].event_id
    assert elxsi_evts[1].event_id != elxsi_evts[1].canonical_event_id


def test_generic_headlines_seo_spam_and_acronym_entity_quarantine() -> None:
    pub = "Tue, 17 Nov 2026 06:00:00 GMT"
    bad_cases = [
        ("ITC", "regulatory_government", "Draft GST return changes aim to cut tax notices, ease ITC claims for firms - Business Standard", "irrelevant_entity_contamination"),
        ("ITC", "promoter_management", "ITC Holdings CEO says data center growth is helping cut Midwest transmission rates 13% - CNBC", "irrelevant_entity_contamination"),
        ("ITC", "company_news", "ITC Share Price - Upstox", "generic_price_headline"),
        ("RELIANCE", "company_news", "RELIANCE Share Price — NSE:RELIANCE - TradingView", "generic_price_headline"),
        ("TCS", "company_news", "Tata Consultancy Services (NSEI:TCS) Stock Forecast & Analyst Predictions - Simply Wall St", "generic_price_headline"),
        ("BIRLACABLE", "company_news", "Price Band Hitters Today, 6 November 2026 at 12:00 PM IST: MM Forgings Ltd, SML Mahindra Ltd, Birla Cable Ltd - HDFC Sky", "generic_price_headline"),
        ("TATAELXSI", "company_news", "Top 6 Semiconductor Stocks In India To Invest In November 2026 - Samco", "generic_price_headline"),
        ("TATAELXSI", "competitor", "Intellivate Cap Nine Monthly Results, Nine Monthly Earnings for Intellivate Cap - The Economic Times", "irrelevant_entity_contamination"),
        ("TATAELXSI", "competitor", "Upcoming Honda Compact SUV Launch Timeline, Engine Options and More - CarLelo", "irrelevant_entity_contamination"),
        ("TATAELXSI", "global_macro", "EV Charging Station Market Size, Share, Trends 2036 - Future Market Insights", "generic_seo_noise"),
        ("ITC", "regulatory_government", "Rise With Profit: New RBI Governor, Vodafone Idea's Fundraise Plans, AMFI Data | Podcast - NDTV Profit", "generic_seo_noise"),
    ]
    raw_items = [
        RawNewsItem(symbol=s, company_name=s, event_category=c, title=t, source="Test", published=pub, url="https://example.com")
        for s, c, t, _ in bad_cases
    ]
    events, quarantined = extract_structured_events_v2(raw_items)
    assert events == []
    assert len(quarantined) == len(bad_cases)
    for (_, _, _, expected_reason), q in zip(bad_cases, quarantined):
        assert q["quarantine_reason"] == expected_reason


def test_competitor_token_collisions_and_third_party_exec_attribution() -> None:
    pub = "Wed, 11 Nov 2026 02:13:00 GMT"
    # "Indian" in "The New Indian Express" or prose must NOT match IOC
    crude = RawNewsItem(
        symbol="RELIANCE",
        company_name="Reliance Industries Ltd",
        event_category="competitor",
        title="Some relief as crude oil slips below USD 90 for the first time since August - The New Indian Express",
        source="The New Indian Express",
        published=pub,
        url="https://example.com/crude",
    )
    cat, ent, rej = resolve_category_and_entity_v2(crude)
    assert rej is None
    assert cat == "global_macro"
    assert ent != "IOC"

    # "Tata Group" / "Tata Sons" must NOT match TATATECH as competitor for TATAELXSI
    tata_grp = RawNewsItem(
        symbol="TATAELXSI",
        company_name="Tata Elxsi Ltd",
        event_category="competitor",
        title="Tata Group shares rise as RBI reiterates Tata Sons listing requirement; Tata Chemicals gains up to 6% - Fortune India",
        source="Fortune India",
        published=pub,
        url="https://example.com/tata-group",
    )
    _, ent2, rej2 = resolve_category_and_entity_v2(tata_grp)
    assert ent2 != "TATATECH"
    assert rej2 == "irrelevant_entity_contamination"

    # Multi-word third-party CEO ("Britannia CEO & MD") must NOT be RELIANCE promoter_management
    britannia = RawNewsItem(
        symbol="RELIANCE",
        company_name="Reliance Industries Ltd",
        event_category="promoter_management",
        title="Reliance may find it difficult to replicate cola disruption in biscuits: Britannia CEO & MD - ET BrandEquity",
        source="ET BrandEquity",
        published=pub,
        url="https://example.com/britannia",
    )
    cat3, _, rej3 = resolve_category_and_entity_v2(britannia)
    assert rej3 is None
    assert cat3 == "company_news"


def test_competitor_win_sentiment_polarity_and_vocabulary() -> None:
    # Rival wins contract beating target -> negative for TCS
    sent, conf, _ = classify_sentiment_v2(
        "Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech - PRESS Insider",
        target_symbol="TCS",
    )
    assert sent == "negative"
    assert conf >= 0.70

    # HFCL Secures Export Orders -> positive (secures + orders)
    sent_hfcl, _, _ = classify_sentiment_v2(
        "HFCL Secures ₹834.36 Crore Export Orders For Optical Fiber Cables - Sahi",
        target_symbol="BIRLACABLE",
    )
    assert sent_hfcl == "positive"

    # "Downgraded to Buy" should recognize "downgraded" as negative token (conflict or negative, not uncontested positive)
    sent_dg, _, has_conflict = classify_sentiment_v2(
        "Birla Cable Ltd Downgraded to Buy Amid Mixed Technical Signals and Strong Financials - MarketsMojo",
        target_symbol="BIRLACABLE",
    )
    assert sent_dg in ("neutral", "negative") or has_conflict


def test_word_boundary_cross_stock_spillover_blocks_limited_mumbai_and_jio_ipo() -> None:
    tcs_limited = StructuredNewsEvent(
        event_id="tcs_lim",
        company_name="Tata Consultancy Services Ltd",
        ticker="TCS",
        event_category="company_news",
        event_description="Tata Consultancy Services Limited (TCS.NS) stock price update",
        original_source="Yahoo Finance UK",
        publication_timestamp="2026-11-17T10:00:00+05:30",
        effective_trading_date="2026-11-17",
        sentiment="neutral",
        impact_magnitude="low",
        classification_confidence=0.54,
        novelty_score=0.72,
        source_credibility=0.68,
    )
    tcs_mumbai = StructuredNewsEvent(
        event_id="tcs_mum",
        company_name="Tata Consultancy Services Ltd",
        ticker="TCS",
        event_category="company_news",
        event_description="TCS schedules analyst meetings in Mumbai and Singapore for November 2026",
        original_source="scanx.trade",
        publication_timestamp="2026-11-10T10:00:00+05:30",
        effective_trading_date="2026-11-10",
        sentiment="neutral",
        impact_magnitude="low",
        classification_confidence=0.54,
        novelty_score=0.72,
        source_credibility=0.68,
    )
    tcs_genuine_ai = StructuredNewsEvent(
        event_id="tcs_ai",
        company_name="Tata Consultancy Services Ltd",
        ticker="TCS",
        event_category="industry_sector",
        event_description="AI, cloud investments to drive India IT spending to 16.9 trillion in 2027",
        original_source="The Hindu",
        publication_timestamp="2026-11-17T11:00:00+05:30",
        effective_trading_date="2026-11-17",
        sentiment="positive",
        impact_magnitude="medium",
        classification_confidence=0.75,
        novelty_score=0.80,
        source_credibility=0.86,
    )
    rel_ipo = StructuredNewsEvent(
        event_id="rel_ipo",
        company_name="Reliance Industries Ltd",
        ticker="RELIANCE",
        event_category="company_news",
        event_description="Billionaire Ambani cries tears of joy as Jio Platforms becomes second most valued Indian firm",
        original_source="CNBC",
        publication_timestamp="2026-11-01T10:00:00+05:30",
        effective_trading_date="2026-11-02",
        sentiment="neutral",
        impact_magnitude="low",
        classification_confidence=0.54,
        novelty_score=0.72,
        source_credibility=0.86,
    )
    rel_capex = StructuredNewsEvent(
        event_id="rel_capex",
        company_name="Reliance Industries Ltd",
        ticker="RELIANCE",
        event_category="company_news",
        event_description="Reliance Jio accelerates 5G and FTTH optical fiber network rollout capex",
        original_source="Mint",
        publication_timestamp="2026-11-10T10:00:00+05:30",
        effective_trading_date="2026-11-10",
        sentiment="positive",
        impact_magnitude="high",
        classification_confidence=0.88,
        novelty_score=0.90,
        source_credibility=0.86,
    )

    impacts = compute_cross_stock_impacts_v2(
        {"TCS": [tcs_limited, tcs_mumbai, tcs_genuine_ai], "RELIANCE": [rel_ipo, rel_capex]}
    )

    elxsi_orig = {c.originating_event_id for c in impacts["TATAELXSI"]}
    assert "tcs_lim" not in elxsi_orig  # "limited" must NOT match "it"
    assert "tcs_mum" not in elxsi_orig  # "mumbai" must NOT match "ai"
    assert "tcs_ai" in elxsi_orig

    birla_orig = {c.originating_event_id for c in impacts["BIRLACABLE"]}
    assert "rel_ipo" not in birla_orig  # Jio IPO/speech must NOT spill to fiber vendor
    assert "rel_capex" in birla_orig


def test_live_parquet_audit_metrics_and_freeze_boundary() -> None:
    pred_dir = Path(__file__).resolve().parent.parent / "data" / "predictions"
    if not (pred_dir / "structured_events.parquet").exists():
        return
    report = audit_news_intelligence_datasets(pred_dir)
    assert report.total_structured_events == 254
    assert report.effective_trading_date_shift_count == 31
    assert report.v2_additional_quarantined_count > 0
    assert report.v2_spillover_after["RELIANCE->BIRLACABLE"] < report.v2_spillover_before["RELIANCE->BIRLACABLE"]
    assert report.v2_spillover_after["TCS->TATAELXSI"] < report.v2_spillover_before["TCS->TATAELXSI"]
