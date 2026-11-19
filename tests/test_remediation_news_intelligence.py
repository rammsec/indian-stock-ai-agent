"""Comprehensive non-vacuous remediation tests for Workstream 3 (News Intelligence).

Tests `backend/news_intel/prototype_v2.py` (`news_v2.0.0`) against all defect classes
and verifies the labelled evaluation thresholds on `tests/fixtures/labelled_news_eval.json`,
while asserting the freeze boundary on all production modules.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from backend.config import IST_TZ
from backend.models import StructuredNewsEvent
from backend.news_intel.collector import RawNewsItem
from backend.news_intel.prototype_v2 import (
    PROTOTYPE_VERSION,
    classify_sentiment_v2,
    compute_cross_stock_impacts_v2,
    compute_effective_trading_date_v2,
    evaluate_labelled_news,
    extract_structured_events_v2,
    is_truncated_rss_timestamp,
    jaccard_similarity_v2,
    normalize_headline_v2,
    parse_rss_timestamp_ist_v2,
    resolve_category_and_entity_v2,
    strip_source_suffix,
)


def test_prototype_version_and_freeze_boundary() -> None:
    assert PROTOTYPE_VERSION == "news_v2.0.0"

    repo_root = Path(__file__).resolve().parent.parent
    frozen_files = [
        "backend/news_intel/collector.py",
        "backend/news_intel/extractor.py",
        "backend/news_intel/cross_stock.py",
        "backend/news_intel/features.py",
        "backend/news_intel/combiner.py",
        "backend/news_intel/pipeline.py",
    ]
    res = subprocess.run(
        ["git", "diff", "--name-only", "--", *frozen_files],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", f"Frozen production files modified: {res.stdout}"


def test_news_timestamp_conversion_gmt_ist_and_truncated_recovery() -> None:
    # 1. Full RFC-2822 GMT -> IST (+05:30) and post-15:30 rollover to next NSE trading day
    gmt_ts = "Fri, 13 Nov 2026 10:15:00 GMT"
    dt_ist = parse_rss_timestamp_ist_v2(gmt_ts)
    assert dt_ist is not None
    assert dt_ist.tzinfo == IST_TZ
    assert (dt_ist.hour, dt_ist.minute) == (15, 45)
    pub_iso, eff_date = compute_effective_trading_date_v2(gmt_ts)
    assert pub_iso == "2026-11-13T15:45:00+05:30"
    assert eff_date == "2026-11-16"  # Friday 15:45 IST -> Monday

    # 2. Explicit RFC-2822 IST must stay IST (never double-shifted by +5h30m)
    ist_ts = "Tue, 17 Nov 2026 14:20:00 IST"
    dt_explicit_ist = parse_rss_timestamp_ist_v2(ist_ts)
    assert dt_explicit_ist is not None
    assert dt_explicit_ist.isoformat() == "2026-11-17T14:20:00+05:30"

    # 3. Legacy [:22]-truncated Google News GMT header must be detected and recovered from UTC -> IST
    trunc_ts = "Sun, 15 Nov 2026 06:51"
    assert is_truncated_rss_timestamp(trunc_ts) is True
    dt_rec = parse_rss_timestamp_ist_v2(trunc_ts, assume_truncated_rfc2822_is_utc=True)
    assert dt_rec is not None
    assert dt_rec.isoformat() == "2026-11-15T12:21:00+05:30"

    # 4. Missing/invalid timestamps must return None (never datetime.now())
    assert parse_rss_timestamp_ist_v2("") is None
    assert parse_rss_timestamp_ist_v2("invalid-ts") is None


def test_news_entity_disambiguation_acronyms_and_spam_filtering() -> None:
    pub = "Tue, 17 Nov 2026 06:00:00 GMT"

    # ITC GST Input Tax Credit collision -> quarantined
    itc_gst = RawNewsItem(
        symbol="ITC",
        company_name="ITC Ltd",
        event_category="regulatory_government",
        title="Draft GST return changes aim to cut tax notices, ease ITC claims for firms - Business Standard",
        source="Business Standard",
        published=pub,
        url="https://example.com/itc-gst",
    )
    _, _, rej_gst = resolve_category_and_entity_v2(itc_gst)
    assert rej_gst == "irrelevant_entity_contamination"

    # IOC / "The New Indian Express" collision -> must NOT match IOC
    crude_news = RawNewsItem(
        symbol="RELIANCE",
        company_name="Reliance Industries Ltd",
        event_category="competitor",
        title="Some relief as crude oil slips below USD 90 for the first time since August - The New Indian Express",
        source="The New Indian Express",
        published=pub,
        url="https://example.com/crude",
    )
    cat_c, ent_c, rej_c = resolve_category_and_entity_v2(crude_news)
    assert rej_c is None
    assert cat_c == "global_macro"
    assert ent_c != "IOC"

    # TATATECH / "Tata Group" collision -> must NOT match TATATECH
    tata_grp = RawNewsItem(
        symbol="TATAELXSI",
        company_name="Tata Elxsi Ltd",
        event_category="competitor",
        title="Tata Group shares rise as RBI reiterates Tata Sons listing requirement; Tata Chemicals gains up to 6% - Fortune India",
        source="Fortune India",
        published=pub,
        url="https://example.com/tata-group",
    )
    _, ent_t, rej_t = resolve_category_and_entity_v2(tata_grp)
    assert ent_t != "TATATECH"
    assert rej_t == "irrelevant_entity_contamination"

    # Generic price & SEO spam -> quarantined
    for sym, title, expected_reason in [
        ("ITC", "ITC Share Price - Upstox", "generic_price_headline"),
        ("TATAELXSI", "EV Charging Station Market Size, Share, Trends 2036 - Future Market Insights", "generic_seo_noise"),
        ("ITC", "Rise With Profit: New RBI Governor, AMFI Data | Podcast - NDTV Profit", "generic_seo_noise"),
    ]:
        raw = RawNewsItem(
            symbol=sym, company_name=sym, event_category="company_news",
            title=title, source="Test", published=pub, url="https://example.com",
        )
        _, _, rej = resolve_category_and_entity_v2(raw)
        assert rej == expected_reason


def test_source_suffix_stripping_and_competitor_sentiment_polarity() -> None:
    title = "Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech - PRESS Insider"
    clean, src = strip_source_suffix(title, known_source="PRESS Insider")
    assert clean == "Cognizant wins $175 million UK rail IT contract, beating TCS and HCLTech"
    assert src == "PRESS Insider"

    # Must classify as competitor (not promoter_management from "- PRESS Insider")
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
    assert ent == "COGNIZANT"

    # Sentiment polarity: rival beating target must be negative for TCS
    sent, conf, _ = classify_sentiment_v2(title, target_symbol="TCS", resolved_category=cat)
    assert sent == "negative"
    assert conf >= 0.70

    # "Downgraded to Buy" -> negative (downgraded outweighs Buy)
    sent_dg, _, _ = classify_sentiment_v2(
        "Birla Cable Ltd Downgraded to Buy Amid Mixed Technical Signals - MarketsMojo",
        target_symbol="BIRLACABLE",
    )
    assert sent_dg == "negative"

    # Contrastive "despite volume growth" -> negative main clause ("cloud outlook")
    sent_cl, _, _ = classify_sentiment_v2(
        "Oil prices, rural stress cloud outlook for consumer goods makers despite volume growth - Economic Times",
        target_symbol="ITC",
    )
    assert sent_cl == "negative"


def test_cross_category_and_near_duplicate_detection() -> None:
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
            symbol="BIRLACABLE",
            company_name="Birla Cable Ltd",
            event_category="competitor",
            title="HFCL Secures ₹834.36 Crore Export Orders For Optical Fiber Cables - Sahi",
            source="Sahi",
            published="Tue, 10 Nov 2026 04:00:00 GMT",
            url="https://example.com/2",
        ),
        RawNewsItem(
            symbol="BIRLACABLE",
            company_name="Birla Cable Ltd",
            event_category="industry_sector",
            title="HFCL bags export orders worth ₹834 crore for optical fibre cables - NDTV Profit",
            source="NDTV Profit",
            published="Tue, 10 Nov 2026 04:15:00 GMT",
            url="https://example.com/3",
        ),
    ]
    assert normalize_headline_v2(items[0].title, items[0].source) == normalize_headline_v2(items[1].title, items[1].source)
    assert jaccard_similarity_v2(items[2].title, items[3].title, items[2].source, items[3].source) >= 0.50

    events, quarantined = extract_structured_events_v2(items)
    assert quarantined == []
    assert len(events) == 4

    rel = [e for e in events if e.ticker == "RELIANCE"]
    assert rel[0].is_duplicate_report is False
    assert rel[1].is_duplicate_report is True
    assert rel[1].canonical_event_id == rel[0].event_id
    assert rel[1].event_id != rel[1].canonical_event_id

    birla = [e for e in events if e.ticker == "BIRLACABLE"]
    assert birla[0].is_duplicate_report is False
    assert birla[1].is_duplicate_report is True
    assert birla[1].canonical_event_id == birla[0].event_id
    assert birla[1].event_id != birla[1].canonical_event_id


def test_word_boundary_cross_stock_attribution() -> None:
    tcs_limited = StructuredNewsEvent(
        event_id="tcs_lim",
        company_name="Tata Consultancy Services Ltd",
        ticker="TCS",
        event_category="company_news",
        event_description="Tata Consultancy Services Limited secures domestic PSU maintenance contract",
        original_source="BusinessLine",
        publication_timestamp="2026-11-17T10:30:00+05:30",
        effective_trading_date="2026-11-17",
        sentiment="positive",
        impact_magnitude="high",
        classification_confidence=0.80,
        novelty_score=0.90,
        source_credibility=0.86,
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
    tcs_ai = StructuredNewsEvent(
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
        {"TCS": [tcs_limited, tcs_mumbai, tcs_ai], "RELIANCE": [rel_ipo, rel_capex]}
    )
    elxsi_ids = {c.originating_event_id for c in impacts["TATAELXSI"]}
    assert "tcs_lim" not in elxsi_ids
    assert "tcs_mum" not in elxsi_ids
    assert "tcs_ai" in elxsi_ids

    birla_ids = {c.originating_event_id for c in impacts["BIRLACABLE"]}
    assert "rel_ipo" not in birla_ids
    assert "rel_capex" in birla_ids


def test_labelled_evaluation_thresholds() -> None:
    report = evaluate_labelled_news()
    assert report.prototype_version == "news_v2.0.0"
    assert 25 <= report.total_items <= 35
    assert report.entity_precision >= 0.90
    assert report.entity_recall >= 0.90
    assert report.duplicate_precision >= 0.95
    assert report.duplicate_recall >= 0.95
    assert report.sentiment_accuracy >= 0.85
    assert report.cross_stock_attribution_accuracy >= 0.95
    assert report.timestamp_correctness == 1.0
    assert report.quarantine_accuracy >= 0.95
    assert report.per_item_failures == []
