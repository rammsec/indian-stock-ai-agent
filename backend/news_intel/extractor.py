"""Structured News Extraction, Deduplication & PIT Alignment (Phase 5 — Sections 3 & 4).

Key guarantees:
1. Strict `StructuredNewsEvent` Pydantic validation.
2. Headline normalization + token Jaccard deduplication (`is_duplicate_report`, `canonical_event_id`).
3. Cost-guarded optional Gemini extraction (`ENABLE_GEMINI_EXTRACTOR=false` by default,
   per-day cap `MAX_GEMINI_CALLS_PER_DAY`, and SQLite cache lookup before any API call).
4. Deterministic fallback extractor so the pipeline works at zero cost and offline.
5. Point-in-time (PIT) trading-session alignment: news published after `15:30 IST` or on a
   weekend/NSE holiday is assigned `effective_trading_date` = next NSE trading session.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

from ..config import (
    ENABLE_GEMINI_EXTRACTOR,
    IST_TZ,
    MAX_GEMINI_CALLS_PER_DAY,
    is_nse_trading_day,
)
from ..data_engine import cache_get, cache_set
from ..models import StructuredNewsEvent
from .collector import RawNewsItem

_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "ltd", "limited", "india",
    "rs", "cr", "crore", "percent", "pct", "shares", "stock", "stocks", "says",
}

_POSITIVE_TERMS = {
    "order", "contract", "win", "wins", "won", "bag", "bags", "beat", "beats",
    "surge", "jump", "growth", "profit", "margin", "expansion", "upgrade",
    "buyback", "dividend", "partnership", "approval", "launch", "launches",
    "record", "rally", "outperform", "accumulate", "tariff", "hike", "award",
    "pli", "incentive", "release", "strong", "robust", "accelerat",
}

_NEGATIVE_TERMS = {
    "loss", "miss", "misses", "drop", "drops", "fall", "falls", "slump",
    "decline", "cut", "cuts", "downgrade", "probe", "penalty", "fine",
    "litigation", "lawsuit", "resign", "resignation", "pledge", "default",
    "shutdown", "closure", "delay", "shortage", "tariff", "tax", "excise",
    "recall", "disruption", "ban", "restrict", "weak", "slowdown", "sell",
}

_HIGH_IMPACT_TERMS = {
    "earnings", "results", "profit", "order", "contract", "acquisition", "merger",
    "sebi", "rbi", "budget", "excise", "tax", "resign", "auditor", "pledge",
    "guidance", "buyback", "demerger", "tariff", "windfall", "bharatnet",
}

_TIER1_SOURCES = {
    "nse", "bse", "sebi", "rbi", "reuters", "bloomberg", "economic times",
    "livemint", "mint", "business standard", "moneycontrol", "cnbc", "ndtv profit",
    "financial express", "the hindu businessline",
}


def normalize_headline(text: str) -> str:
    """Lowercase, strip punctuation/source suffixes, and normalize whitespace."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity between two normalized headlines."""
    ta = set(normalize_headline(text_a).split())
    tb = set(normalize_headline(text_b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_publication_timestamp_ist(raw_ts: str | None) -> datetime:
    """Parse RFC-2822 / ISO-8601 / YYYY-MM-DD timestamps into `Asia/Kolkata` datetime."""
    if not raw_ts or not str(raw_ts).strip():
        return datetime.now(IST_TZ)
    s = str(raw_ts).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST_TZ)
        return dt.astimezone(IST_TZ)
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST_TZ)
        return dt.astimezone(IST_TZ)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%a, %d %b %Y %H:%M", "%d %b %Y"):
        try:
            dt = datetime.strptime(s[:20], fmt).replace(tzinfo=IST_TZ)
            return dt
        except Exception:
            continue
    return datetime.now(IST_TZ)


def next_nse_trading_day(d: date) -> date:
    """Return the first NSE trading day strictly after `d`."""
    cur = d + timedelta(days=1)
    while not is_nse_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def compute_effective_trading_date(raw_ts: str | None) -> tuple[str, str]:
    """Return `(publication_timestamp_ist_iso, effective_trading_date_yyyy_mm_dd)`.

    Point-in-time rule:
    - If published on a non-trading day (weekend/NSE holiday) -> next NSE trading day.
    - If published after NSE close (`15:30 IST`) -> next NSE trading day.
    - Otherwise -> same trading day.
    """
    dt_ist = parse_publication_timestamp_ist(raw_ts)
    d = dt_ist.date()
    if not is_nse_trading_day(d):
        eff = next_nse_trading_day(d)
    elif dt_ist.hour > 15 or (dt_ist.hour == 15 and dt_ist.minute > 30):
        eff = next_nse_trading_day(d)
    else:
        eff = d
    return dt_ist.isoformat(), eff.strftime("%Y-%m-%d")


def _source_credibility(source_name: str) -> float:
    s_low = (source_name or "").lower()
    if any(t in s_low for t in ("nse", "bse", "sebi", "rbi")):
        return 0.96
    if any(t in s_low for t in _TIER1_SOURCES):
        return 0.86
    return 0.68


def _deterministic_extract(raw: RawNewsItem, is_dup: bool, canonical_id: str | None) -> StructuredNewsEvent:
    """Zero-cost deterministic structured event extraction with strict schema validation."""
    pub_iso, eff_date = compute_effective_trading_date(raw.published)
    event_id = hashlib.sha256(
        f"{raw.symbol}|{raw.event_category}|{normalize_headline(raw.title)}|{eff_date}".encode()
    ).hexdigest()[:16]

    h_low = raw.title.lower()
    pos_hits = sum(1 for t in _POSITIVE_TERMS if t in h_low)
    neg_hits = sum(1 for t in _NEGATIVE_TERMS if t in h_low)

    # NSE Bulk/Block deal metadata override
    if raw.metadata and "side" in raw.metadata:
        if raw.metadata["side"] == "BUY":
            pos_hits += 2
        elif raw.metadata["side"] == "SELL":
            neg_hits += 2

    if pos_hits > neg_hits:
        sentiment = "positive"
    elif neg_hits > pos_hits:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    high_hits = sum(1 for t in _HIGH_IMPACT_TERMS if t in h_low)
    if high_hits >= 1 or (raw.metadata and float(raw.metadata.get("value_cr", 0.0) or 0.0) >= 25.0):
        impact_magnitude = "high"
        relevant_horizon = "5D"
        duration = "5-10 trading sessions"
    elif pos_hits + neg_hits >= 2:
        impact_magnitude = "medium"
        relevant_horizon = "5D"
        duration = "2-5 trading sessions"
    else:
        impact_magnitude = "low"
        relevant_horizon = "1D"
        duration = "1-2 trading sessions"

    if raw.event_category in ("regulatory_government", "global_macro"):
        relevant_horizon = "10D"
        duration = "5-15 trading sessions"

    margin = abs(pos_hits - neg_hits)
    confidence = min(0.92, 0.55 + 0.10 * margin + (0.08 if high_hits > 0 else 0.0))
    credibility = _source_credibility(raw.source)
    novelty = 0.25 if is_dup else (0.90 if high_hits > 0 else 0.72)

    facts_summary = f"Reported by {raw.source} ({eff_date} session): {raw.title}"
    if sentiment == "positive":
        interp = f"Constructive {raw.event_category.replace('_', ' ')} signal for {raw.symbol} ({impact_magnitude} magnitude)."
    elif sentiment == "negative":
        interp = f"Adverse {raw.event_category.replace('_', ' ')} risk for {raw.symbol} ({impact_magnitude} magnitude)."
    else:
        interp = f"Informational {raw.event_category.replace('_', ' ')} update with balanced immediate directional skew."

    return StructuredNewsEvent(
        event_id=event_id,
        company_name=raw.company_name,
        ticker=raw.symbol,
        related_competitor_or_industry=raw.related_competitor_or_industry,
        event_category=raw.event_category,  # type: ignore[arg-type]
        event_description=raw.title,
        original_source=raw.source,
        source_url=raw.url,
        publication_timestamp=pub_iso,
        event_timestamp=pub_iso,
        effective_trading_date=eff_date,
        sentiment=sentiment,  # type: ignore[arg-type]
        impact_magnitude=impact_magnitude,  # type: ignore[arg-type]
        classification_confidence=round(float(confidence), 3),
        expected_impact_duration=duration,
        relevant_horizon=relevant_horizon,  # type: ignore[arg-type]
        novelty_score=round(float(novelty), 3),
        source_credibility=round(float(credibility), 3),
        related_companies=[raw.symbol] + ([raw.related_competitor_or_industry] if raw.related_competitor_or_industry and raw.event_category == "competitor" else []),
        related_sectors=[raw.related_competitor_or_industry] if raw.related_competitor_or_industry else [],
        is_duplicate_report=is_dup,
        canonical_event_id=canonical_id or event_id,
        extraction_method="deterministic",
        facts_summary=facts_summary,
        interpretation_summary=interp,
    )


def _daily_gemini_counter_key() -> str:
    today_ist = datetime.now(IST_TZ).strftime("%Y-%m-%d")
    return f"gemini_calls_count::{today_ist}"


def _try_gemini_extract(raw: RawNewsItem, base_event: StructuredNewsEvent) -> StructuredNewsEvent:
    """Optional cost-guarded Gemini extraction (only when explicitly enabled & under daily cap)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return base_event

    cnt_key = _daily_gemini_counter_key()
    calls_today = int(cache_get(cnt_key) or 0)
    if calls_today >= MAX_GEMINI_CALLS_PER_DAY:
        return base_event

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            "Extract structured Indian equity event JSON with keys: "
            "sentiment (positive|negative|neutral), impact_magnitude (low|medium|high), "
            "classification_confidence (0.0-1.0), relevant_horizon (1D|5D|10D|LONGER), "
            "interpretation_summary (1 concise sentence). "
            f"Company: {raw.company_name} ({raw.symbol}). Category: {raw.event_category}. "
            f"Headline: {raw.title}. Return ONLY valid JSON."
        )
        resp = model.generate_content(prompt)
        cache_set(cnt_key, calls_today + 1, ttl_seconds=86400)
        txt = (resp.text or "").strip().strip("`")
        if txt.startswith("json"):
            txt = txt[4:].strip()
        parsed = json.loads(txt)
        return base_event.model_copy(
            update={
                "sentiment": parsed.get("sentiment", base_event.sentiment),
                "impact_magnitude": parsed.get("impact_magnitude", base_event.impact_magnitude),
                "classification_confidence": float(parsed.get("classification_confidence", base_event.classification_confidence)),
                "relevant_horizon": parsed.get("relevant_horizon", base_event.relevant_horizon),
                "interpretation_summary": str(parsed.get("interpretation_summary", base_event.interpretation_summary)),
                "extraction_method": "gemini",
            }
        )
    except Exception:
        return base_event


def extract_structured_events(
    raw_items: list[RawNewsItem],
    enable_gemini: bool | None = None,
    similarity_threshold: float = 0.55,
) -> list[StructuredNewsEvent]:
    """Deduplicate raw news items, enforce PIT alignment, cache & validate StructuredNewsEvent."""
    use_gemini = ENABLE_GEMINI_EXTRACTOR if enable_gemini is None else enable_gemini
    canonical_items: list[tuple[str, str]] = []  # (headline, event_id)
    events: list[StructuredNewsEvent] = []

    for raw in raw_items:
        if not raw.title or not raw.title.strip():
            continue

        is_dup = False
        canonical_id: str | None = None
        for prev_title, prev_id in canonical_items:
            if jaccard_similarity(raw.title, prev_title) >= similarity_threshold:
                is_dup = True
                canonical_id = prev_id
                break

        cache_key = f"phase5_evt::{raw.symbol}::{raw.event_category}::{hashlib.md5(raw.title.encode()).hexdigest()}"
        cached = cache_get(cache_key)
        if cached is not None:
            try:
                evt = StructuredNewsEvent.model_validate(cached)
                if is_dup and not evt.is_duplicate_report:
                    evt = evt.model_copy(update={"is_duplicate_report": True, "canonical_event_id": canonical_id, "novelty_score": 0.25})
                events.append(evt)
                if not is_dup:
                    canonical_items.append((raw.title, evt.event_id))
                continue
            except Exception:
                pass

        evt = _deterministic_extract(raw, is_dup=is_dup, canonical_id=canonical_id)
        if use_gemini and not is_dup:
            evt = _try_gemini_extract(raw, evt)

        if not is_dup:
            canonical_items.append((raw.title, evt.event_id))

        cache_set(cache_key, evt.model_dump(), ttl_seconds=21600)
        events.append(evt)

    return events
