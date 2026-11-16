"""Structured News Extraction, Deduplication, Entity Resolution & PIT Alignment (Phase 5.1).

Key guarantees:
1. Strict timestamp validation: articles without valid publication timestamps are
   rejected/quarantined — `datetime.now()` is NEVER substituted.
2. Entity resolution & category separation: separates `company_news`,
   `promoter_management`, `competitor`, `industry_sector`, `regulatory_government`,
   and `global_macro` based on headline content and `RELATIONSHIP_MAP`.
3. Contextual deterministic sentiment with word-boundary matching, multi-word phrases,
   and contrastive clause weighting (`despite`, `but`, `however`, `amid`).
4. Headline normalization + token Jaccard deduplication (`is_duplicate_report`, `canonical_event_id`).
5. Cost-guarded optional Gemini extraction (`ENABLE_GEMINI_EXTRACTOR=false` by default).
6. Point-in-time (PIT) trading-session alignment (`> 15:30 IST` or holiday -> next NSE session).
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
from .relationships import RELATIONSHIP_MAP

_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "ltd", "limited", "india",
    "rs", "cr", "crore", "percent", "pct", "shares", "stock", "stocks", "says",
}

_COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "RELIANCE": ("reliance", "ril", "jio", "reliance industries", "reliance retail"),
    "TCS": ("tcs", "tata consultancy"),
    "TATAELXSI": ("tata elxsi", "tataelxsi"),
    "ITC": ("itc", "itc ltd", "itc hotels"),
    "BIRLACABLE": ("birla cable", "birlacable"),
}

_REGULATORY_PATTERNS = (
    r"\bsebi\b", r"\brbi\b", r"\btrai\b", r"\bdot\b", r"\bcci\b", r"\bnclt\b",
    r"\bfssai\b", r"\bgst\b", r"\bexcise\b", r"\bwindfall tax\b", r"\banti-dumping\b",
    r"\bh-1b\b", r"\bfda\b", r"\btribunal\b", r"\bsupreme court\b", r"\bhigh court\b",
    r"\bunion budget\b", r"\bimport duty\b", r"\bexport ban\b", r"\bregulator\b",
)

_PROMOTER_PATTERNS = (
    r"\bpromoter\b", r"\bpromoters\b", r"\binsider\b", r"\bpledge\b", r"\bpledged\b",
    r"\bpledging\b", r"\bceo\b", r"\bcfo\b", r"\bchairman\b", r"\bmanaging director\b",
    r"\bauditor\b", r"\bresign\b", r"\bresigns\b", r"\bresignation\b",
    r"\bappointed\b", r"\bappointment\b", r"\bstake sale\b", r"\bstake buy\b",
)

_GLOBAL_MACRO_PATTERNS = (
    r"\bfederal reserve\b", r"\bfed rate\b", r"\bus fed\b", r"\bbrent\b", r"\bopec\b",
    r"\bcrude oil\b", r"\busd/inr\b", r"\brupee\b", r"\brepo rate\b", r"\binflation\b",
    r"\bmonsoon\b", r"\bcopper price\b", r"\bus recession\b", r"\bgeopolitical\b",
)

_INDUSTRY_PATTERNS = (
    r"\bindustry\b", r"\bsector\b", r"\bbharatnet\b", r"\b5g capex\b", r"\bftth\b",
    r"\bit services\b", r"\bfmcg\b", r"\brural demand\b", r"\ber&d\b", r"\bsupply chain\b",
    r"\brefining margin\b", r"\btelecom tariff\b", r"\boptical fiber\b",
)

_POSITIVE_PHRASES: list[tuple[str, float]] = [
    (r"\bwins?\s+(?:major\s+|large\s+)?(?:order|contract|deal|tender)\b", 2.0),
    (r"\bbags?\s+(?:major\s+|large\s+)?(?:order|contract|deal|tender)\b", 2.0),
    (r"\bbeats?\s+(?:estimates|expectations|consensus)\b", 2.0),
    (r"\bprofit\s+(?:jumps?|surges?|rises?|grows?|beats?)\b", 2.0),
    (r"\bmargin\s+(?:expansion|improves?|expands?|beats?)\b", 1.8),
    (r"\braises?\s+guidance\b", 1.8),
    (r"\bpledge\s+(?:release|released|revoked)\b", 1.8),
    (r"\btax\s+cut\b", 1.5),
    (r"\btariff\s+hike\b", 1.5),
    (r"\bpli\s+incentive\b", 1.5),
]

_NEGATIVE_PHRASES: list[tuple[str, float]] = [
    (r"\bprofit\s+(?:falls?|drops?|declines?|slumps?|plunges?|misses?)\b", 2.0),
    (r"\bmisses?\s+(?:estimates|expectations|consensus|profit)\b", 2.0),
    (r"\bmargin\s+(?:pressure|contraction|shrinks?|falls?|drops?)\b", 1.8),
    (r"\bguidance\s+cut\b", 2.0),
    (r"\bcuts?\s+guidance\b", 2.0),
    (r"\bgrowth\s+(?:slows?|slowdown)\b", 1.6),
    (r"\b(?:sebi|rbi|regulatory)\s+(?:penalty|fine|probe|ban|notice)\b", 2.0),
    (r"\bauditor\s+resigns?\b", 2.2),
    (r"\bpromoter\s+pledge\b", 1.8),
    (r"\btax\s+hike\b", 1.6),
    (r"\bwindfall\s+tax\b", 1.5),
    (r"\bproduct\s+recall\b", 1.8),
]

_POSITIVE_WORDS = {
    "order", "contract", "win", "wins", "won", "bag", "bags", "beat", "beats",
    "surge", "surges", "jump", "jumps", "growth", "expansion", "upgrade", "upgrades",
    "buyback", "dividend", "partnership", "approval", "launch", "launches",
    "record", "rally", "outperform", "accumulate", "award", "incentive",
    "strong", "robust", "accelerates", "release",
}

_NEGATIVE_WORDS = {
    "loss", "losses", "miss", "misses", "drop", "drops", "fall", "falls", "slump",
    "decline", "declines", "downgrade", "downgrades", "probe", "penalty", "fine",
    "litigation", "lawsuit", "resign", "resigns", "resignation", "pledge", "default",
    "shutdown", "closure", "delay", "delays", "shortage", "recall", "disruption",
    "ban", "restrict", "restrictions", "weak", "slowdown", "headwind", "pressure",
}

_HIGH_IMPACT_WORDS = {
    "earnings", "results", "profit", "order", "contract", "acquisition", "merger",
    "sebi", "rbi", "budget", "excise", "resign", "resigns", "auditor", "pledge",
    "guidance", "buyback", "demerger", "tariff", "windfall", "bharatnet", "penalty",
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


def try_parse_publication_timestamp_ist(raw_ts: str | None) -> datetime | None:
    """Parse RFC-2822 / ISO-8601 / YYYY-MM-DD timestamps into IST, or return None if invalid."""
    if not raw_ts or not str(raw_ts).strip():
        return None
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
    return None


def parse_publication_timestamp_ist(raw_ts: str | None) -> datetime:
    """Strict timestamp parser: raises ValueError if missing or unparseable (never uses now())."""
    dt = try_parse_publication_timestamp_ist(raw_ts)
    if dt is None:
        raise ValueError(f"Invalid or missing publication timestamp: {raw_ts!r}")
    return dt


def next_nse_trading_day(d: date) -> date:
    """Return the first NSE trading day strictly after `d`."""
    cur = d + timedelta(days=1)
    while not is_nse_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def compute_effective_trading_date(raw_ts: str | None) -> tuple[str, str]:
    """Return `(publication_timestamp_ist_iso, effective_trading_date_yyyy_mm_dd)`."""
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


def _matches_any(text_low: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text_low) for p in patterns)


def resolve_category_and_entity(raw: RawNewsItem) -> tuple[str, str | None, str | None]:
    """Resolve true event_category & entity, preventing cross-category contamination.

    Returns `(resolved_category, resolved_related_entity, rejection_reason)`.
    """
    sym = raw.symbol.upper()
    h_low = raw.title.lower()
    rel = RELATIONSHIP_MAP.get(sym, {})

    target_aliases = _COMPANY_ALIASES.get(sym, (sym.lower(),))
    mentions_target = any(re.search(rf"\b{re.escape(a)}\b", h_low) for a in target_aliases)

    matched_competitor: str | None = None
    for comp in rel.get("competitors", []):
        c_sym = str(comp.get("symbol", "")).upper()
        c_name = str(comp.get("name", "")).lower()
        c_tokens = [c_sym.lower()] + ([c_name] if c_name else []) + ([c_name.split()[0]] if len(c_name.split()) > 1 and len(c_name.split()[0]) > 3 else [])
        if any(re.search(rf"\b{re.escape(tok)}\b", h_low) for tok in c_tokens if tok):
            matched_competitor = c_sym
            break

    # Explicit NSE Bulk/Block Deal separation (Issue 3)
    if raw.metadata and "is_verified_promoter" in raw.metadata:
        if raw.metadata["is_verified_promoter"]:
            return "promoter_management", raw.related_competitor_or_industry, None
        return "company_news", "Institutional Bulk/Block Deal", None

    # 1. Regulatory & government developments
    if _matches_any(h_low, _REGULATORY_PATTERNS) and not matched_competitor:
        return "regulatory_government", "SEBI / RBI / Govt Policy", None

    # 2. Competitor developments (only when competitor is mentioned and target is NOT primary)
    if matched_competitor and not mentions_target:
        return "competitor", matched_competitor, None

    # 3. Promoter & management developments (requires target or explicit promoter/board context)
    if _matches_any(h_low, _PROMOTER_PATTERNS) and (mentions_target or raw.event_category == "promoter_management"):
        return "promoter_management", raw.related_competitor_or_industry, None

    # 4. Global macro (without target or competitor)
    if _matches_any(h_low, _GLOBAL_MACRO_PATTERNS) and not mentions_target and not matched_competitor:
        return "global_macro", "Global Macro", None

    # 5. Industry/sector developments (without single company target)
    if _matches_any(h_low, _INDUSTRY_PATTERNS) and not mentions_target and not matched_competitor:
        return "industry_sector", rel.get("sector", raw.related_competitor_or_industry), None

    # 6. Company news (must mention target if fetched as company_news)
    if mentions_target:
        return "company_news", rel.get("sector", raw.related_competitor_or_industry), None

    if raw.event_category == "company_news":
        return raw.event_category, raw.related_competitor_or_industry, "irrelevant_entity_contamination"

    return raw.event_category, raw.related_competitor_or_industry, None


def _score_clause(clause_low: str) -> tuple[float, float]:
    pos = 0.0
    neg = 0.0
    for pat, w in _POSITIVE_PHRASES:
        if re.search(pat, clause_low):
            pos += w
    for pat, w in _NEGATIVE_PHRASES:
        if re.search(pat, clause_low):
            neg += w
    tokens = re.findall(r"\b[a-z0-9]+\b", clause_low)
    for tok in tokens:
        if tok in _POSITIVE_WORDS:
            pos += 1.0
        if tok in _NEGATIVE_WORDS:
            neg += 1.0
    return pos, neg


def classify_sentiment_contextual(headline: str, metadata: dict[str, Any] | None = None) -> tuple[str, float, bool]:
    """Contextual deterministic sentiment with word boundaries & contrastive clause weighting.

    Returns `(sentiment, confidence, has_conflict)`.
    """
    h_low = headline.lower()
    contrast_split = re.split(r"\b(?:despite|although|though)\b", h_low)
    but_split = re.split(r"\b(?:but|however|yet)\b", h_low)

    pos_total = 0.0
    neg_total = 0.0

    if len(but_split) > 1:
        p1, n1 = _score_clause(but_split[0])
        p2, n2 = _score_clause(" ".join(but_split[1:]))
        pos_total = 0.6 * p1 + 1.5 * p2
        neg_total = 0.6 * n1 + 1.5 * n2
    elif len(contrast_split) > 1 and "," in h_low:
        # e.g., "Despite higher revenue, profit falls 12%" -> second part after comma is main clause
        parts = h_low.split(",", 1)
        p1, n1 = _score_clause(parts[0])
        p2, n2 = _score_clause(parts[1])
        pos_total = 0.5 * p1 + 1.6 * p2
        neg_total = 0.5 * n1 + 1.6 * n2
    else:
        pos_total, neg_total = _score_clause(h_low)

    if metadata and "side" in metadata:
        if metadata["side"] == "BUY":
            pos_total += 2.0
        elif metadata["side"] == "SELL":
            neg_total += 2.0

    has_conflict = pos_total > 0.5 and neg_total > 0.5
    diff = pos_total - neg_total
    if abs(diff) < 0.5:
        sentiment = "neutral"
        conf = 0.50 if has_conflict else 0.54
    elif diff > 0:
        sentiment = "positive"
        conf = min(0.92, (0.55 if has_conflict else 0.62) + 0.08 * min(diff, 3.5))
    else:
        sentiment = "negative"
        conf = min(0.92, (0.55 if has_conflict else 0.62) + 0.08 * min(abs(diff), 3.5))

    return sentiment, round(float(conf), 3), has_conflict


def _deterministic_extract(
    raw: RawNewsItem,
    resolved_cat: str,
    resolved_entity: str | None,
    pub_iso: str,
    eff_date: str,
    is_dup: bool,
    canonical_id: str | None,
) -> StructuredNewsEvent:
    """Zero-cost deterministic structured event extraction with strict schema validation."""
    event_id = hashlib.sha256(
        f"{raw.symbol}|{resolved_cat}|{normalize_headline(raw.title)}|{eff_date}".encode()
    ).hexdigest()[:16]

    sentiment, confidence, _ = classify_sentiment_contextual(raw.title, metadata=raw.metadata)
    h_low = raw.title.lower()
    tokens = set(re.findall(r"\b[a-z0-9]+\b", h_low))
    high_hits = len(tokens & _HIGH_IMPACT_WORDS)

    if high_hits >= 1 or (raw.metadata and float(raw.metadata.get("value_cr", 0.0) or 0.0) >= 25.0):
        impact_magnitude = "high"
        relevant_horizon = "5D"
        duration = "5-10 trading sessions"
    elif confidence >= 0.72:
        impact_magnitude = "medium"
        relevant_horizon = "5D"
        duration = "2-5 trading sessions"
    else:
        impact_magnitude = "low"
        relevant_horizon = "1D"
        duration = "1-2 trading sessions"

    if resolved_cat in ("regulatory_government", "global_macro"):
        relevant_horizon = "10D"
        duration = "5-15 trading sessions"

    credibility = _source_credibility(raw.source)
    novelty = 0.25 if is_dup else (0.90 if high_hits > 0 else 0.72)

    facts_summary = f"Reported by {raw.source} ({eff_date} session): {raw.title}"
    if sentiment == "positive":
        interp = f"Constructive {resolved_cat.replace('_', ' ')} signal for {raw.symbol} ({impact_magnitude} magnitude)."
    elif sentiment == "negative":
        interp = f"Adverse {resolved_cat.replace('_', ' ')} risk for {raw.symbol} ({impact_magnitude} magnitude)."
    else:
        interp = f"Informational {resolved_cat.replace('_', ' ')} update with balanced immediate directional skew."

    related_companies = [raw.symbol]
    if resolved_cat == "competitor" and resolved_entity:
        related_companies.append(resolved_entity)

    meta = raw.metadata if isinstance(raw.metadata, dict) else {}
    prom_status = str(meta.get("promoter_verification_status") or "NOT_APPLICABLE")
    if prom_status not in {"CONFIRMED", "INFERRED", "UNKNOWN", "NOT_APPLICABLE"}:
        prom_status = "NOT_APPLICABLE"
    prom_evidence = [str(x) for x in meta.get("classification_evidence", [])] if isinstance(meta.get("classification_evidence"), list) else []
    prom_conf = meta.get("promoter_classification_confidence")
    prom_conf_val = float(prom_conf) if prom_conf is not None else None

    return StructuredNewsEvent(
        event_id=event_id,
        company_name=raw.company_name,
        ticker=raw.symbol,
        related_competitor_or_industry=resolved_entity,
        event_category=resolved_cat,  # type: ignore[arg-type]
        event_description=raw.title,
        original_source=raw.source,
        source_url=raw.url,
        publication_timestamp=pub_iso,
        event_timestamp=pub_iso,
        effective_trading_date=eff_date,
        sentiment=sentiment,  # type: ignore[arg-type]
        impact_magnitude=impact_magnitude,  # type: ignore[arg-type]
        classification_confidence=confidence,
        expected_impact_duration=duration,
        relevant_horizon=relevant_horizon,  # type: ignore[arg-type]
        novelty_score=round(float(novelty), 3),
        source_credibility=round(float(credibility), 3),
        related_companies=related_companies,
        related_sectors=[resolved_entity] if resolved_entity else [],
        is_duplicate_report=is_dup,
        canonical_event_id=canonical_id or event_id,
        extraction_method="deterministic",
        facts_summary=facts_summary,
        interpretation_summary=interp,
        promoter_verification_status=prom_status,  # type: ignore[arg-type]
        classification_evidence=prom_evidence,
        promoter_classification_confidence=prom_conf_val,
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
            f"Company: {raw.company_name} ({raw.symbol}). Category: {base_event.event_category}. "
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


def extract_structured_events_with_quarantine(
    raw_items: list[RawNewsItem],
    enable_gemini: bool | None = None,
    similarity_threshold: float = 0.55,
) -> tuple[list[StructuredNewsEvent], list[dict[str, str]]]:
    """Validate timestamps, resolve categories/entities, deduplicate, and quarantine invalid items."""
    use_gemini = ENABLE_GEMINI_EXTRACTOR if enable_gemini is None else enable_gemini
    canonical_items: list[tuple[str, str]] = []
    events: list[StructuredNewsEvent] = []
    quarantined: list[dict[str, str]] = []

    for raw in raw_items:
        if not raw.title or not raw.title.strip():
            continue

        # 1. Strict timestamp check (NEVER substitute current time)
        try:
            pub_iso, eff_date = compute_effective_trading_date(raw.published)
        except ValueError:
            quarantined.append(
                {"symbol": raw.symbol, "title": raw.title[:100], "reason": "invalid_or_missing_timestamp"}
            )
            continue

        # 2. Entity resolution & category separation
        resolved_cat, resolved_entity, rej_reason = resolve_category_and_entity(raw)
        if rej_reason is not None:
            quarantined.append({"symbol": raw.symbol, "title": raw.title[:100], "reason": rej_reason})
            continue

        # 3. Deduplication
        is_dup = False
        canonical_id: str | None = None
        for prev_title, prev_id in canonical_items:
            if jaccard_similarity(raw.title, prev_title) >= similarity_threshold:
                is_dup = True
                canonical_id = prev_id
                break

        cache_key = f"phase5_1_evt::{raw.symbol}::{resolved_cat}::{eff_date}::{hashlib.md5(raw.title.encode()).hexdigest()}"
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

        evt = _deterministic_extract(
            raw,
            resolved_cat=resolved_cat,
            resolved_entity=resolved_entity,
            pub_iso=pub_iso,
            eff_date=eff_date,
            is_dup=is_dup,
            canonical_id=canonical_id,
        )
        if use_gemini and not is_dup:
            evt = _try_gemini_extract(raw, evt)

        if not is_dup:
            canonical_items.append((raw.title, evt.event_id))

        cache_set(cache_key, evt.model_dump(), ttl_seconds=21600)
        events.append(evt)

    return events, quarantined


def extract_structured_events(
    raw_items: list[RawNewsItem],
    enable_gemini: bool | None = None,
    similarity_threshold: float = 0.55,
) -> list[StructuredNewsEvent]:
    """Return validated StructuredNewsEvent list (quarantining invalid timestamps/entities)."""
    events, _ = extract_structured_events_with_quarantine(
        raw_items,
        enable_gemini=enable_gemini,
        similarity_threshold=similarity_threshold,
    )
    return events
