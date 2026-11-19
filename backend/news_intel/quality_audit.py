"""Isolated News Intelligence Quality Audit & Prototype Validator/Classifier (Phase 5.5 Research).

FREEZE BOUNDARY:
This module is strictly isolated from the frozen production pipeline (`collector.py`,
`extractor.py`, `cross_stock.py`, `features.py`, `combiner.py`, `pipeline.py`).
It does NOT modify the 17 numerical news features or production Parquet/JSON artifacts.

Provides:
1. Full RFC-2822 GMT -> IST timestamp parsing (without `[:22]` truncation) + detection
   of legacy `[:22]`-truncated GMT strings.
2. `" - <Source>"` suffix stripping from RSS titles prior to entity/promoter matching and
   headline normalization.
3. Tightened entity resolution, generic-headline rejection, acronym disambiguation
   (`ITC` Input Tax Credit, `IOC`/`indian`, `TATATECH`/`tata`, SEO market-research spam),
   and relevance gating for `competitor`, `industry_sector`, `regulatory_government`,
   and `global_macro` events.
4. Cross-category `canonical_event_id` deduplication (independent of `resolved_cat` and
   without source-suffix noise).
5. Competitor-win sentiment polarity (`<Rival> wins ..., beating <Target>` -> negative for
   target, plus `downgraded`/`secures`/`orders` vocabulary fixes).
6. Word-boundary cross-stock spillover (`\\bit\\b`, `\\bai\\b`, Jio capex/network vs IPO).
7. Live Parquet dataset auditor (`audit_news_intelligence_datasets`) producing before/after
   metrics across `structured_events.parquet`, `quarantined_events.parquet`, and
   `cross_stock_impacts.parquet`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import IST_TZ, PREDICTIONS_DATA_DIR, is_nse_trading_day
from ..models import CrossStockImpactItem, StructuredNewsEvent
from .collector import RawNewsItem
from .extractor import (
    _COMPANY_ALIASES,
    _FALSE_POSITIVE_ENTITY_PATTERNS,
    _GLOBAL_MACRO_PATTERNS,
    _HIGH_IMPACT_WORDS,
    _INDUSTRY_PATTERNS,
    _NEGATIVE_PHRASES,
    _NEGATIVE_WORDS,
    _POSITIVE_PHRASES,
    _POSITIVE_WORDS,
    _PROMOTER_PATTERNS,
    _REGULATORY_PATTERNS,
    _STOPWORDS,
    _source_credibility,
    next_nse_trading_day,
)
from .relationships import PILOT_UNIVERSE_5, RELATIONSHIP_MAP

# ---------------------------------------------------------------------------
# 1. Title / Source Suffix Stripping & Normalization
# ---------------------------------------------------------------------------

_SOURCE_SUFFIX_RE = re.compile(
    r"\s+[-–—|]\s+(?:"
    r"[A-Z][A-Za-z0-9.&'’!/ ]{1,55}"
    r"|(?:[a-z0-9-]+\.)+[a-z]{2,}"
    r")\s*$"
)


def strip_source_suffix(title: str, known_source: str | None = None) -> tuple[str, str]:
    """Strip trailing `' - <Source>'` attribution from RSS headlines.

    Returns `(clean_title, extracted_source)`.
    """
    t = (title or "").strip()
    src = (known_source or "").strip()
    if not t:
        return "", src

    if src and src.lower() not in ("unknown", "indian financial media"):
        for sep in (" - ", " – ", " — ", " | "):
            suffix = f"{sep}{src}"
            if t.lower().endswith(suffix.lower()):
                return t[: -len(suffix)].strip(), src

    m = _SOURCE_SUFFIX_RE.search(t)
    if m:
        candidate_src = t[m.start() :].lstrip(" -–—|").strip()
        clean = t[: m.start()].strip()
        # Avoid stripping legitimate numeric/date tails if short or obviously not a source
        if clean and len(candidate_src) >= 2:
            return clean, src or candidate_src

    return t, src


def normalize_headline_v2(text: str, known_source: str | None = None) -> str:
    """Strip trailing source suffix, lowercase, remove punctuation/stopwords."""
    clean_title, _ = strip_source_suffix(text, known_source=known_source)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", clean_title.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def jaccard_similarity_v2(
    text_a: str,
    text_b: str,
    source_a: str | None = None,
    source_b: str | None = None,
) -> float:
    """Token Jaccard similarity after stripping source suffixes."""
    ta = set(normalize_headline_v2(text_a, known_source=source_a).split())
    tb = set(normalize_headline_v2(text_b, known_source=source_b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# 2. Full RFC-2822 / GMT -> IST Timestamp Parsing (No [:22] Truncation)
# ---------------------------------------------------------------------------

_TRUNCATED_RFC2822_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}$"
)


def is_truncated_rss_timestamp(raw_ts: str | None) -> bool:
    """Detect `entry.get('published', '')[:22]` truncated Google News GMT strings."""
    if not raw_ts:
        return False
    return bool(_TRUNCATED_RFC2822_RE.match(str(raw_ts).strip()))


def parse_rss_timestamp_ist_v2(
    raw_ts: str | None,
    assume_truncated_rfc2822_is_utc: bool = True,
) -> datetime | None:
    """Parse full RFC-2822 (with GMT/UTC/+0000) or ISO-8601 timestamps into IST.

    - Never truncates `raw_ts` to `[:22]`.
    - When `assume_truncated_rfc2822_is_utc=True`, naive 22-char RSS headers
      like `'Sun, 15 Nov 2026 06:51'` (produced by Google News `[:22]`) are
      recovered as UTC before converting to IST (`+05:30`).
    - Returns `None` if missing or unparseable (never substitutes `datetime.now()`).
    """
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
            tz = timezone.utc if (assume_truncated_rfc2822_is_utc and is_truncated_rss_timestamp(s)) else IST_TZ
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(IST_TZ)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%a, %d %b %Y %H:%M", "%d %b %Y"):
        try:
            parsed = datetime.strptime(s, fmt)
            tz = timezone.utc if (assume_truncated_rfc2822_is_utc and fmt == "%a, %d %b %Y %H:%M") else IST_TZ
            return parsed.replace(tzinfo=tz).astimezone(IST_TZ)
        except Exception:
            continue

    return None


def compute_effective_trading_date_v2(
    raw_ts: str | None,
    assume_truncated_rfc2822_is_utc: bool = True,
) -> tuple[str, str]:
    """Compute `(publication_timestamp_ist_iso, effective_trading_date)` cleanly."""
    dt_ist = parse_rss_timestamp_ist_v2(
        raw_ts, assume_truncated_rfc2822_is_utc=assume_truncated_rfc2822_is_utc
    )
    if dt_ist is None:
        raise ValueError(f"Invalid or missing publication timestamp: {raw_ts!r}")
    d = dt_ist.date()
    if not is_nse_trading_day(d):
        eff = next_nse_trading_day(d)
    elif dt_ist.hour > 15 or (dt_ist.hour == 15 and dt_ist.minute > 30):
        eff = next_nse_trading_day(d)
    else:
        eff = d
    return dt_ist.isoformat(), eff.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 3. Tightened Generic-Headline, Acronym, Entity & SEO Spam Filters
# ---------------------------------------------------------------------------

_GENERIC_HEADLINE_PATTERNS_V2: tuple[str, ...] = (
    r"\bshare\s+price\s+(?:today|live)\b",
    r"\bstock\s+price\s+(?:today|live|nse|bse)\b",
    r"\bstock\s+price,\s+news,\s+quote\b",
    r"\blive\s+nse/bse\b",
    r"\bothers\s+to\s+post\s+(?:q[1-4]|earnings)\b",
    r"^\s*[^-]{2,80}\bshare\s+price\s*$",
    r"\bshare\s+price\s*[—–-]\s*nse:\w+\s*$",
    r"\bshare/stock\s+price\s+live\s+today\b",
    r"\bstock\s+forecast\s*&\s*analyst\s+predictions\b",
    r"\bprice\s+band\s+hitters\b",
    r"\b(?:upper|lower)\s*&\s*lower\s+circuit\s+stocks\b",
    r"\btop\s+gainers\s+and\s+top\s+losers\b",
    r"\bseasonal\s+returns\s+chart\b",
    r"\btop\s+\d+\s+(?:semiconductor|it|fmcg|telecom|ai|growth)\s+stocks\b",
    r"\bq[1-4]\s+results\s+(?:today|this\s+week)\s*:",
    r"\bfour\s+stocks\s+to\s+buy\s+today\b",
    r"\bweekly\s+market\s+commentary\b",
)

_SEO_SPAM_PATTERNS_V2: tuple[str, ...] = (
    r"\bmarket\s+size,?\s+share,?\s+trends\s+20\d{2}\b",
    r"\bmarket\s+to\s+20\d{2}\b",
    r"\bforecast\s+20\d{2}-20\d{2}\b",
    r"\|\s*podcast\b",
    r"\bpodcast\s*$",
)

_FALSE_POSITIVE_ENTITY_PATTERNS_V2: dict[str, tuple[str, ...]] = {
    "ITC": (
        *_FALSE_POSITIVE_ENTITY_PATTERNS.get("ITC", ()),
        r"\bitc\s+claims?\b",
        r"\binput\s+tax\s+credit\b",
        r"\bease\s+itc\b",
        r"\bitc\s+fraud\b",
        r"\bfake\s+itc\b",
        r"\bineligible\s+itc\b",
        r"\bgst\s+itc\b",
    ),
    "RELIANCE": (
        *_FALSE_POSITIVE_ENTITY_PATTERNS.get("RELIANCE", ()),
    ),
    "TCS": (
        r"\btata\s+sons\s+(?:listing|power\s+struggle|issues|dispute)\b",
        r"\btata\s+legacy\b",
        r"\bjimmy\s+tata\b",
        r"\bnoel\s+tata\b",
    ),
}

# Explicit safe competitor match tokens (prevents IOC -> "indian", TATATECH -> "tata", etc.)
_COMPETITOR_EXACT_TOKENS_V2: dict[str, dict[str, tuple[str, ...]]] = {
    "RELIANCE": {
        "BHARTIARTL": ("bhartiartl", "bharti airtel", "airtel"),
        "ONGC": ("ongc", "oil and natural gas"),
        "DMART": ("dmart", "avenue supermarts"),
        "IOC": ("ioc", "indian oil", "indianoil"),
    },
    "TCS": {
        "INFY": ("infy", "infosys"),
        "HCLTECH": ("hcltech", "hcl technologies", "hcl tech", "hcl"),
        "WIPRO": ("wipro",),
        "ACN": ("acn", "accenture"),
    },
    "TATAELXSI": {
        "KPITTECH": ("kpittech", "kpit technologies", "kpit"),
        "LTTS": ("ltts", "l&t technology", "l&t technology services"),
        "TATATECH": ("tatatech", "tata technologies"),
        "PERSISTENT": ("persistent systems", "persistent"),
    },
    "ITC": {
        "HINDUNILVR": ("hindunilvr", "hindustan unilever", "hul", "unilever india"),
        "NESTLEIND": ("nestleind", "nestle india", "nestlé india", "nestle"),
        "GODFRYPHLP": ("godfryphlp", "godfrey phillips"),
        "DABUR": ("dabur", "dabur india"),
    },
    "BIRLACABLE": {
        "HFCL": ("hfcl",),
        "STLTECH": ("stltech", "sterlite technologies", "sterlite tech"),
        "VINDHYATEL": ("vindhyatel", "vindhya telelinks"),
        "FINCABLES": ("fincables", "finolex cables", "finolex"),
    },
}

_REGULATORY_PATTERNS_V2: tuple[str, ...] = (
    *_REGULATORY_PATTERNS,
    r"\bh-?1b\b",
    r"\bgreen\s+card\b",
    r"\bvisa\s+curbs?\b",
    r"\bimmigration\s+(?:restrictions?|reset)\b",
    r"\bgst\s+(?:rationalisation|council|notification|rate|return)\b",
    r"\bmonetary\s+policy\b",
    r"\bmpc\b",
)

_GLOBAL_MACRO_PATTERNS_V2: tuple[str, ...] = (
    *_GLOBAL_MACRO_PATTERNS,
    r"\bfed\b",
    r"\bfederal\s+reserve\b",
    r"\bcrude\b",
    r"\boil\s+(?:prices?|slide|volatility|rise)\b",
    r"\brate\s+panel\b",
    r"\busd\s*\d+",
    r"\bfx\b",
    r"\bcurrency\b",
    r"\bpmi\b",
    r"\bgrowth\s+seen\s+losing\s+momentum\b",
    r"\beurope\s+carmakers\b",
    r"\bev\s+shift\b",
)

_INDUSTRY_PATTERNS_V2: tuple[str, ...] = (
    *_INDUSTRY_PATTERNS,
    r"\bit\s+spending\b",
    r"\bindia\s+it\s+spending\b",
    r"\bindian\s+it\s+stocks?\b",
    r"\bdata\s+cent(?:re|er)\s+spending\b",
    r"\bfibre\s+optics?\b",
    r"\boptical\s+fibr[eo]\b",
    r"\bofc\b",
    r"\bfiber\s+broadband\b",
    r"\bconsumer\s+goods\b",
    r"\brural\s+(?:india|stress)\b",
    r"\beuropean\s+oems?\b",
    r"\bauto\s+component\b",
    r"\badas\b",
    r"\bai\s+spending\b",
    r"\bbanking\s+and\s+capital\s+markets\b",
)

_SYMBOL_RELEVANCE_TERMS_V2: dict[str, tuple[str, ...]] = {
    "RELIANCE": (
        "reliance", "ril", "jio", "o2c", "telecom", "trai", "refining", "crude",
        "brent", "opec", "oil", "diesel", "atf", "windfall", "retail", "rupee",
        "usd/inr", "sebi", "fii", "ltcg", "satellite",
    ),
    "TCS": (
        "tcs", "tata consultancy", "it services", "it spending", "it stocks",
        "software", "bfsi", "genai", "ai", "cloud", "data centre", "data center",
        "h-1b", "visa", "green card", "immigration", "fed", "federal reserve",
        "rupee", "usd/inr", "dollar", "us tech", "nasdaq", "infosys", "wipro",
        "hcl", "accenture", "banking", "tech",
    ),
    "TATAELXSI": (
        "tata elxsi", "tataelxsi", "er&d", "sdv", "adas", "ev", "electric vehicle",
        "automotive", "jlr", "oem", "oems", "carmakers", "embedded", "kpit",
        "ltts", "tata technologies", "auto", "vehicle",
    ),
    "ITC": (
        "itc", "fmcg", "consumer goods", "cigarette", "tobacco", "gst", "excise",
        "fssai", "rural", "monsoon", "palm oil", "agri", "rbi", "mpc",
        "inflation", "repo rate", "rate panel", "pmi", "hul", "unilever",
        "nestle", "godfrey", "dabur",
    ),
    "BIRLACABLE": (
        "birla cable", "birlacable", "optical fiber", "optical fibre", "ofc",
        "fibre", "fiber", "bharatnet", "ftth", "5g", "telecom cable",
        "broadband", "copper", "hfcl", "sterlite", "vindhya", "finolex", "bsnl", "dot",
    ),
}

_THIRD_PARTY_EXEC_MULTIWORD_RE = re.compile(
    r"\b([a-z0-9&'’.-]+(?:\s+[a-z0-9&'’.-]+){0,2})\s+(?:ceo|cfo|chairman|managing director|md)\b"
)


def _has_third_party_exec_attribution_v2(h_low: str, target_aliases: tuple[str, ...]) -> bool:
    """Detect executive titles attributed to a non-target company (single or multi-word)."""
    alias_words = {w for a in target_aliases for w in a.split()} | {
        "new", "former", "ex", "interim", "group", "its", "the", "a", "an",
        "company", "joint", "deputy", "vice", "chief", "and", "&", "as", "by",
        "says", "said", "for", "of", "in", "on", "to", "with", "at",
    }
    for m in _THIRD_PARTY_EXEC_MULTIWORD_RE.finditer(h_low):
        prefix = m.group(1).strip()
        prefix_clean = re.sub(r"[’']s\b", "", prefix)
        words = [w for w in prefix_clean.split() if w not in alias_words]
        if words:
            return True
    return False


def resolve_category_and_entity_v2(raw: RawNewsItem) -> tuple[str, str | None, str | None]:
    """Tightened category & entity resolution with source-suffix stripping and relevance gating."""
    sym = raw.symbol.upper()
    clean_title, clean_src = strip_source_suffix(raw.title, known_source=raw.source)
    h_low = clean_title.lower()
    src_low = (clean_src or "").lower()
    rel = RELATIONSHIP_MAP.get(sym, {})

    # 0. Generic price / quote / roundup / SEO market-report spam
    if any(re.search(p, h_low) for p in _GENERIC_HEADLINE_PATTERNS_V2):
        return raw.event_category, raw.related_competitor_or_industry, "generic_price_headline"
    if any(re.search(p, h_low) for p in _SEO_SPAM_PATTERNS_V2):
        return raw.event_category, raw.related_competitor_or_industry, "generic_seo_noise"

    # 0b. Known false-positive namesakes / acronym collisions
    fp_pats = _FALSE_POSITIVE_ENTITY_PATTERNS_V2.get(sym, ())
    if fp_pats and any(re.search(p, h_low) for p in fp_pats):
        return raw.event_category, raw.related_competitor_or_industry, "irrelevant_entity_contamination"

    target_aliases = _COMPANY_ALIASES.get(sym, (sym.lower(),))
    mentions_target = any(re.search(rf"\b{re.escape(a)}\b", h_low) for a in target_aliases)

    # Strict competitor matching using curated alias dictionary (check title + corporate IR source)
    matched_competitor: str | None = None
    comp_map = _COMPETITOR_EXACT_TOKENS_V2.get(sym, {})
    for c_sym, tokens in comp_map.items():
        if any(re.search(rf"\b{re.escape(tok)}\b", h_low) or re.search(rf"\b{re.escape(tok)}\b", src_low) for tok in tokens):
            matched_competitor = c_sym
            break

    # NSE Bulk/Block deal explicit metadata path
    if raw.metadata and "is_verified_promoter" in raw.metadata:
        if raw.metadata["is_verified_promoter"]:
            return "promoter_management", raw.related_competitor_or_industry, None
        return "company_news", "Institutional Bulk/Block Deal", None

    # Competitor wins / deals that explicitly beat the target ("Cognizant wins..., beating TCS")
    rival_win_against_target = bool(
        re.search(r"\bbeating\s+(?:" + "|".join(re.escape(a) for a in target_aliases) + r")\b", h_low)
    )
    if rival_win_against_target:
        rival_name = clean_title.split()[0].upper() if clean_title else "COMPETITOR"
        return "competitor", matched_competitor or rival_name, None

    # 1. Regulatory & government
    if any(re.search(p, h_low) for p in _REGULATORY_PATTERNS_V2) and not matched_competitor:
        rel_terms = _SYMBOL_RELEVANCE_TERMS_V2.get(sym, ())
        if mentions_target or any(re.search(rf"\b{re.escape(t)}\b", h_low) for t in rel_terms):
            return "regulatory_government", "SEBI / RBI / Govt Policy", None
        return raw.event_category, raw.related_competitor_or_industry, "irrelevant_entity_contamination"

    # 2. Competitor
    if matched_competitor and not mentions_target:
        return "competitor", matched_competitor, None

    # 3. Promoter & management (on clean_title without publisher suffix like "PRESS Insider")
    if any(re.search(p, h_low) for p in _PROMOTER_PATTERNS):
        if mentions_target and not _has_third_party_exec_attribution_v2(h_low, target_aliases):
            return "promoter_management", raw.related_competitor_or_industry, None
        if raw.event_category == "promoter_management" and not mentions_target:
            return raw.event_category, raw.related_competitor_or_industry, "unverified_promoter_association"
        if mentions_target and _has_third_party_exec_attribution_v2(h_low, target_aliases):
            return "company_news", rel.get("sector", raw.related_competitor_or_industry), None

    # 4. Global macro (must still satisfy symbol macro/sector relevance)
    if any(re.search(p, h_low) for p in _GLOBAL_MACRO_PATTERNS_V2) and not matched_competitor:
        rel_terms = _SYMBOL_RELEVANCE_TERMS_V2.get(sym, ())
        if any(re.search(rf"\b{re.escape(t)}\b", h_low) for t in rel_terms) and not mentions_target:
            return "global_macro", "Global Macro", None

    # 5. Industry / sector
    if any(re.search(p, h_low) for p in _INDUSTRY_PATTERNS_V2) and not mentions_target and not matched_competitor:
        rel_terms = _SYMBOL_RELEVANCE_TERMS_V2.get(sym, ())
        if any(re.search(rf"\b{re.escape(t)}\b", h_low) for t in rel_terms):
            return "industry_sector", rel.get("sector", raw.related_competitor_or_industry), None
        return raw.event_category, raw.related_competitor_or_industry, "irrelevant_entity_contamination"

    # 6. Company news (must mention target)
    if mentions_target:
        return "company_news", rel.get("sector", raw.related_competitor_or_industry), None

    # 7. Gate ALL categories against unverified fallback contamination (fixes line 311 leak!)
    return raw.event_category, raw.related_competitor_or_industry, "irrelevant_entity_contamination"


# ---------------------------------------------------------------------------
# 4. Improved Sentiment Classifier (Competitor-Win Polarity & Vocabulary)
# ---------------------------------------------------------------------------

_POSITIVE_PHRASES_V2: list[tuple[str, float]] = [
    *_POSITIVE_PHRASES,
    (r"\bsecures?\s+(?:major\s+|large\s+|export\s+)?(?:orders?|contracts?|deals?|tenders?)\b", 2.0),
    (r"\bbags?\s+(?:major\s+|large\s+|export\s+)?(?:orders?|contracts?|deals?|tenders?)\b", 2.0),
    (r"\bwins?\s+(?:major\s+|large\s+|export\s+)?(?:orders?|contracts?|deals?|tenders?)\b", 2.0),
    (r"\bremoval\s+of\s+(?:[\w.%-]+\s+){0,3}tax\b", 1.5),
]

_NEGATIVE_PHRASES_V2: list[tuple[str, float]] = [
    *_NEGATIVE_PHRASES,
    (r"\bprofits?\s+(?:falls?|drops?|declines?|slumps?|plunges?|misses?)\b", 2.0),
    (r"\bdowngraded?\s+to\b", 1.8),
    (r"\bunder\s+pressure\b", 1.5),
]

_POSITIVE_WORDS_V2 = _POSITIVE_WORDS | {"orders", "contracts", "secures", "secured", "rises", "gains", "soars", "surged"}
_NEGATIVE_WORDS_V2 = _NEGATIVE_WORDS | {"downgraded", "slips", "slipped", "cuts", "cut", "falls", "fell", "slides", "slid"}


def classify_sentiment_v2(
    headline: str,
    target_symbol: str | None = None,
    resolved_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, float, bool]:
    """Contextual sentiment with clean-title stripping, vocabulary fixes, and rival-win inversion."""
    clean_title, _ = strip_source_suffix(headline)
    h_low = clean_title.lower()

    def _score_clause_v2(clause_low: str) -> tuple[float, float]:
        pos = 0.0
        neg = 0.0
        for pat, w in _POSITIVE_PHRASES_V2:
            if re.search(pat, clause_low):
                pos += w
        for pat, w in _NEGATIVE_PHRASES_V2:
            if re.search(pat, clause_low):
                neg += w
        tokens = re.findall(r"\b[a-z0-9]+\b", clause_low)
        for tok in tokens:
            if tok in _POSITIVE_WORDS_V2:
                pos += 1.0
            if tok in _NEGATIVE_WORDS_V2:
                neg += 1.0
        return pos, neg

    contrast_split = re.split(r"\b(?:despite|although|though)\b", h_low)
    but_split = re.split(r"\b(?:but|however|yet)\b", h_low)

    if len(but_split) > 1:
        p1, n1 = _score_clause_v2(but_split[0])
        p2, n2 = _score_clause_v2(" ".join(but_split[1:]))
        pos_total = 0.6 * p1 + 1.5 * p2
        neg_total = 0.6 * n1 + 1.5 * n2
    elif len(contrast_split) > 1 and "," in h_low:
        parts = h_low.split(",", 1)
        p1, n1 = _score_clause_v2(parts[0])
        p2, n2 = _score_clause_v2(parts[1])
        pos_total = 0.5 * p1 + 1.6 * p2
        neg_total = 0.5 * n1 + 1.6 * n2
    else:
        pos_total, neg_total = _score_clause_v2(h_low)

    if metadata and "side" in metadata:
        if metadata["side"] == "BUY":
            pos_total += 2.0
        elif metadata["side"] == "SELL":
            neg_total += 2.0

    # Rival beat target -> negative competitive shock for target (e.g. "Cognizant wins ..., beating TCS")
    if target_symbol:
        aliases = _COMPANY_ALIASES.get(target_symbol.upper(), (target_symbol.lower(),))
        beating_pat = r"\bbeating\s+(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
        if re.search(beating_pat, h_low):
            neg_total += max(2.5, pos_total + 1.0)
            pos_total = 0.0

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


# ---------------------------------------------------------------------------
# 5. Cross-Category Deduplication & Extractor Prototype
# ---------------------------------------------------------------------------


def compute_cross_category_event_id(symbol: str, title: str, eff_date: str, source: str | None = None) -> str:
    """Category-independent canonical ID based on `(symbol, normalized_clean_title, eff_date)`."""
    norm = normalize_headline_v2(title, known_source=source)
    return hashlib.sha256(f"{symbol.upper()}|{norm}|{eff_date}".encode()).hexdigest()[:16]


def extract_structured_events_v2(
    raw_items: list[RawNewsItem],
    similarity_threshold: float = 0.55,
    assume_truncated_rfc2822_is_utc: bool = True,
) -> tuple[list[StructuredNewsEvent], list[dict[str, str]]]:
    """Prototype extractor fixing timestamp truncation, cross-category duplicates, and entity leakage."""
    canonical_items: list[tuple[str, str, str, str]] = []  # (symbol, clean_title, source, canonical_id)
    events: list[StructuredNewsEvent] = []
    quarantined: list[dict[str, str]] = []
    seen_exact_keys: set[tuple[str, str, str]] = set()

    col_ts = datetime.now(IST_TZ).isoformat()
    for raw in raw_items:
        if not raw.title or not raw.title.strip():
            continue

        try:
            pub_iso, eff_date = compute_effective_trading_date_v2(
                raw.published, assume_truncated_rfc2822_is_utc=assume_truncated_rfc2822_is_utc
            )
        except ValueError:
            quarantined.append(
                {
                    "symbol": raw.symbol,
                    "title": raw.title[:160],
                    "source": raw.source,
                    "url": raw.url,
                    "raw_published": str(raw.published or ""),
                    "collection_timestamp": col_ts,
                    "quarantine_reason": "invalid_or_missing_timestamp",
                }
            )
            continue

        resolved_cat, resolved_entity, rej_reason = resolve_category_and_entity_v2(raw)
        if rej_reason is not None:
            quarantined.append(
                {
                    "symbol": raw.symbol,
                    "title": raw.title[:160],
                    "source": raw.source,
                    "url": raw.url,
                    "raw_published": str(raw.published or ""),
                    "collection_timestamp": col_ts,
                    "quarantine_reason": rej_reason,
                }
            )
            continue

        sym = raw.symbol.upper()
        norm_clean = normalize_headline_v2(raw.title, known_source=raw.source)
        canon_key = (sym, norm_clean, eff_date)

        # Cross-category deduplication: identical article fetched by two category collectors
        # resolves to ONE canonical event (no self-referential canonical_event_id == event_id)
        is_exact_cross_cat_dup = canon_key in seen_exact_keys
        seen_exact_keys.add(canon_key)

        base_canon_id = compute_cross_category_event_id(sym, raw.title, eff_date, source=raw.source)
        event_id = (
            hashlib.sha256(f"{sym}|{resolved_cat}|{norm_clean}|{eff_date}".encode()).hexdigest()[:16]
            if is_exact_cross_cat_dup
            else base_canon_id
        )

        is_dup = is_exact_cross_cat_dup
        canonical_id: str | None = base_canon_id if is_exact_cross_cat_dup else None

        if not is_dup:
            for prev_sym, prev_title, prev_src, prev_id in canonical_items:
                if prev_sym != sym:
                    continue
                if jaccard_similarity_v2(raw.title, prev_title, source_a=raw.source, source_b=prev_src) >= similarity_threshold:
                    is_dup = True
                    canonical_id = prev_id
                    break

        if is_dup and canonical_id == event_id:
            event_id = hashlib.sha256(f"{event_id}|dup|{len(events)}".encode()).hexdigest()[:16]

        sentiment, confidence, _ = classify_sentiment_v2(
            raw.title, target_symbol=sym, resolved_category=resolved_cat, metadata=raw.metadata
        )
        clean_title, clean_src = strip_source_suffix(raw.title, known_source=raw.source)
        tokens = set(re.findall(r"\b[a-z0-9]+\b", clean_title.lower()))
        high_hits = len(tokens & _HIGH_IMPACT_WORDS)
        impact_magnitude = (
            "high"
            if high_hits >= 1 or (raw.metadata and float(raw.metadata.get("value_cr", 0.0) or 0.0) >= 25.0)
            else ("medium" if confidence >= 0.72 else "low")
        )
        relevant_horizon = "10D" if resolved_cat in ("regulatory_government", "global_macro") else ("5D" if impact_magnitude != "low" else "1D")
        novelty = 0.25 if is_dup else (0.90 if high_hits > 0 else 0.72)

        evt = StructuredNewsEvent(
            event_id=event_id,
            company_name=raw.company_name,
            ticker=sym,
            related_competitor_or_industry=resolved_entity,
            event_category=resolved_cat,  # type: ignore[arg-type]
            event_description=clean_title,
            original_source=clean_src or raw.source,
            source_url=raw.url,
            publication_timestamp=pub_iso,
            event_timestamp=pub_iso,
            effective_trading_date=eff_date,
            sentiment=sentiment,  # type: ignore[arg-type]
            impact_magnitude=impact_magnitude,  # type: ignore[arg-type]
            classification_confidence=confidence,
            relevant_horizon=relevant_horizon,  # type: ignore[arg-type]
            novelty_score=round(float(novelty), 3),
            source_credibility=round(float(_source_credibility(clean_src or raw.source)), 3),
            related_companies=[sym, resolved_entity] if resolved_cat == "competitor" and resolved_entity else [sym],
            related_sectors=[resolved_entity] if resolved_entity else [],
            is_duplicate_report=is_dup,
            canonical_event_id=canonical_id or event_id,
            extraction_method="deterministic",
            facts_summary=f"Reported by {clean_src or raw.source} ({eff_date} session): {clean_title}",
            interpretation_summary=f"{resolved_cat.replace('_', ' ')} signal ({sentiment}) for {sym}.",
            collection_timestamp=col_ts,
            validation_status="VALID",
        )
        if not is_dup:
            canonical_items.append((sym, raw.title, raw.source, evt.event_id))
        events.append(evt)

    return events, quarantined


# ---------------------------------------------------------------------------
# 6. Word-Boundary Cross-Stock Spillover Prototype
# ---------------------------------------------------------------------------

_SPILLOVER_RULES_V2: list[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "RELIANCE",
        "BIRLACABLE",
        "supplier_customer",
        # Must relate to Jio / telecom network / capex / fiber deployment, NOT Jio IPO listing gossip
        (r"\b5g\b", r"\bftth\b", r"\bfiber\b", r"\bfibre\b", r"\bbroadband\b", r"\bcapex\b", r"\bnetwork\s+rollout\b", r"\btelecom\b"),
        (r"\bipo\b", r"\blisting\b", r"\bvaluation\b", r"\bgmp\b", r"\bcries\b", r"\bspeech\b", r"\bbell-ringing\b"),
    ),
    (
        "TCS",
        "TATAELXSI",
        "sector_peer",
        # Word boundaries prevent "limited" -> "it" and "Mumbai" -> "ai"
        (r"\bit\s+services\b", r"\bit\s+spending\b", r"\bit\s+stocks?\b", r"\bgenai\b", r"\bai\b", r"\bbfsi\b", r"\bguidance\b", r"\bdeal\b", r"\bmargin\b", r"\battrition\b"),
        (r"\bshare\s+price\b", r"\bstock\s+price\b", r"\banalyst\s+meetings?\b", r"\binterview\b"),
    ),
]


def compute_cross_stock_impacts_v2(
    events_by_symbol: dict[str, list[StructuredNewsEvent]],
    universe: tuple[str, ...] = PILOT_UNIVERSE_5,
) -> dict[str, list[CrossStockImpactItem]]:
    """Isolated word-boundary & capex-gated cross-stock spillover engine."""
    from .cross_stock import _asymmetric_competitor_direction

    impacts: dict[str, list[CrossStockImpactItem]] = {sym: [] for sym in universe}

    for sym in universe:
        rel = RELATIONSHIP_MAP.get(sym, {})
        comp_modes = {
            c.get("symbol", "").upper(): c.get("asymmetry", "demand_readacross_and_rival")
            for c in rel.get("competitors", [])
        }
        seen_ids: set[str] = set()
        for evt in events_by_symbol.get(sym, []):
            if evt.is_duplicate_report or evt.event_id in seen_ids:
                continue
            seen_ids.add(evt.event_id)

            if evt.event_category == "competitor":
                comp_entity = (evt.related_competitor_or_industry or "Competitor").upper()
                mode = comp_modes.get(comp_entity, "demand_readacross_and_rival")
                direction, rationale = _asymmetric_competitor_direction(evt, mode)
                impacts[sym].append(
                    CrossStockImpactItem(
                        affected_company=sym,
                        originating_event_id=evt.event_id,
                        originating_entity=evt.related_competitor_or_industry or "Sector Competitor",
                        relationship_type="competitor",
                        expected_impact_direction=direction,  # type: ignore[arg-type]
                        impact_magnitude=evt.impact_magnitude,
                        confidence=round(evt.classification_confidence * 0.85, 3),
                        expected_duration=evt.expected_impact_duration,
                        rationale=rationale,
                    )
                )
            elif evt.event_category == "regulatory_government":
                impacts[sym].append(
                    CrossStockImpactItem(
                        affected_company=sym,
                        originating_event_id=evt.event_id,
                        originating_entity=evt.original_source,
                        relationship_type="regulatory_exposure",
                        expected_impact_direction=evt.sentiment,
                        impact_magnitude=evt.impact_magnitude,
                        confidence=round(evt.classification_confidence * 0.90, 3),
                        expected_duration=evt.expected_impact_duration,
                        rationale=f"Regulatory / policy read-across: {evt.event_description[:110]}",
                    )
                )
            elif evt.event_category == "global_macro":
                impacts[sym].append(
                    CrossStockImpactItem(
                        affected_company=sym,
                        originating_event_id=evt.event_id,
                        originating_entity="Global Macro",
                        relationship_type="global_macro",
                        expected_impact_direction=evt.sentiment,
                        impact_magnitude=evt.impact_magnitude,
                        confidence=round(evt.classification_confidence * 0.80, 3),
                        expected_duration=evt.expected_impact_duration,
                        rationale=f"Macro / commodity link: {evt.event_description[:110]}",
                    )
                )

    for src_sym, dst_sym, rel_type, include_pats, exclude_pats in _SPILLOVER_RULES_V2:
        if dst_sym not in impacts:
            continue
        for evt in events_by_symbol.get(src_sym, []):
            if evt.is_duplicate_report or evt.event_category not in ("company_news", "industry_sector"):
                continue
            desc = evt.event_description.lower()
            if any(re.search(ex, desc) for ex in exclude_pats):
                continue
            if any(re.search(inc, desc) for inc in include_pats):
                impacts[dst_sym].append(
                    CrossStockImpactItem(
                        affected_company=dst_sym,
                        originating_event_id=evt.event_id,
                        originating_entity=src_sym,
                        relationship_type=rel_type,  # type: ignore[arg-type]
                        expected_impact_direction=evt.sentiment,
                        impact_magnitude="medium" if evt.impact_magnitude == "high" else "low",
                        confidence=round(evt.classification_confidence * 0.75, 3),
                        expected_duration="2-5 trading sessions",
                        rationale=f"{src_sym} {rel_type.replace('_', ' ')} spillover to {dst_sym}: {evt.event_description[:100]}",
                    )
                )

    return impacts


# ---------------------------------------------------------------------------
# 7. Dataset Auditor (Before / After Metrics on Live Parquet Files)
# ---------------------------------------------------------------------------


@dataclass
class NewsQualityAuditReport:
    total_structured_events: int
    total_quarantined_events: int
    total_cross_stock_impacts: int
    unstripped_source_suffix_count: int
    effective_trading_date_shift_count: int
    unflagged_same_ticker_duplicate_pairs: int
    cross_category_identical_headline_groups: int
    self_referential_canonical_dups: int
    null_collection_timestamp_count: int
    null_validation_status_count: int
    v2_retained_events_count: int
    v2_additional_quarantined_count: int
    v2_quarantine_reason_breakdown: dict[str, int]
    v2_reclassified_count: int
    v2_duplicate_events_count: int
    v2_cross_stock_impacts_count: int
    v2_spillover_before: dict[str, int]
    v2_spillover_after: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_news_intelligence_datasets(predictions_dir: Path | None = None) -> NewsQualityAuditReport:
    """Run read-only before/after quality audit on persisted Parquet artifacts."""
    p_dir = predictions_dir or PREDICTIONS_DATA_DIR
    se = pd.read_parquet(p_dir / "structured_events.parquet")
    qe = pd.read_parquet(p_dir / "quarantined_events.parquet")
    ci = pd.read_parquet(p_dir / "cross_stock_impacts.parquet")

    unstripped = sum(1 for d in se["event_description"] if " - " in str(d))

    # Count effective_trading_date errors caused by [:22] GMT->IST -5h30m shift
    eff_shifts = 0
    for _, r in se.iterrows():
        dt_wrong = pd.Timestamp(r["publication_timestamp"])
        dt_true_ist = dt_wrong + timedelta(hours=5, minutes=30)
        d = dt_true_ist.date()
        if not is_nse_trading_day(d):
            true_eff = next_nse_trading_day(d)
        elif dt_true_ist.hour > 15 or (dt_true_ist.hour == 15 and dt_true_ist.minute > 30):
            true_eff = next_nse_trading_day(d)
        else:
            true_eff = d
        if true_eff.strftime("%Y-%m-%d") != str(r["effective_trading_date"]):
            eff_shifts += 1

    # Unflagged same-ticker duplicates in existing structured_events
    from .extractor import jaccard_similarity

    unflagged_pairs = 0
    for _, grp in se.groupby("ticker"):
        recs = grp.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if not recs[i]["is_duplicate_report"] and not recs[j]["is_duplicate_report"]:
                    if jaccard_similarity(recs[i]["event_description"], recs[j]["event_description"]) >= 0.55:
                        unflagged_pairs += 1

    cross_cat_groups = int((se.groupby(["ticker", "event_description"])["event_category"].nunique() > 1).sum())
    self_ref_dups = int(((se["is_duplicate_report"]) & (se["event_id"] == se["canonical_event_id"])).sum())

    # Simulate v2 validator/deduplicator across stored structured_events
    raw_items: list[RawNewsItem] = []
    orig_cats: list[str] = []
    for _, r in se.iterrows():
        raw_items.append(
            RawNewsItem(
                symbol=str(r["ticker"]),
                company_name=str(r["company_name"]),
                event_category=str(r["event_category"]),
                title=str(r["event_description"]),
                source=str(r["original_source"]),
                published=str(r["publication_timestamp"]),
                url=str(r.get("source_url") or ""),
                related_competitor_or_industry=r.get("related_competitor_or_industry"),
            )
        )
        orig_cats.append(str(r["event_category"]))

    # Keep stored +05:30 ISO as-is for v2 simulation (it's already ISO; raw RSS check is separate)
    v2_events, v2_quarantined = extract_structured_events_v2(
        raw_items, assume_truncated_rfc2822_is_utc=False
    )

    q_breakdown: dict[str, int] = {}
    for q in v2_quarantined:
        reason = q["quarantine_reason"]
        q_breakdown[reason] = q_breakdown.get(reason, 0) + 1

    # Check reclassifications among retained events by matching (ticker, clean_title)
    reclassified = 0
    orig_by_key = {
        (str(r["ticker"]), strip_source_suffix(str(r["event_description"]), str(r["original_source"]))[0]): str(r["event_category"])
        for _, r in se.iterrows()
    }
    for e in v2_events:
        if not e.is_duplicate_report and orig_by_key.get((e.ticker, e.event_description)) not in (None, e.event_category):
            reclassified += 1

    evts_by_sym: dict[str, list[StructuredNewsEvent]] = {sym: [] for sym in PILOT_UNIVERSE_5}
    for e in v2_events:
        evts_by_sym.setdefault(e.ticker, []).append(e)
    ci_v2 = compute_cross_stock_impacts_v2(evts_by_sym)
    ci_v2_total = sum(len(v) for v in ci_v2.values())

    spill_before = {
        "RELIANCE->BIRLACABLE": int(((ci["affected_company"] == "BIRLACABLE") & (ci["relationship_type"] == "supplier_customer")).sum()),
        "TCS->TATAELXSI": int(((ci["affected_company"] == "TATAELXSI") & (ci["relationship_type"] == "sector_peer")).sum()),
    }
    spill_after = {
        "RELIANCE->BIRLACABLE": sum(1 for x in ci_v2["BIRLACABLE"] if x.relationship_type == "supplier_customer"),
        "TCS->TATAELXSI": sum(1 for x in ci_v2["TATAELXSI"] if x.relationship_type == "sector_peer"),
    }

    return NewsQualityAuditReport(
        total_structured_events=len(se),
        total_quarantined_events=len(qe),
        total_cross_stock_impacts=len(ci),
        unstripped_source_suffix_count=unstripped,
        effective_trading_date_shift_count=eff_shifts,
        unflagged_same_ticker_duplicate_pairs=unflagged_pairs,
        cross_category_identical_headline_groups=cross_cat_groups,
        self_referential_canonical_dups=self_ref_dups,
        null_collection_timestamp_count=int(se["collection_timestamp"].isna().sum()),
        null_validation_status_count=int(se["validation_status"].isna().sum()),
        v2_retained_events_count=len(v2_events),
        v2_additional_quarantined_count=len(v2_quarantined),
        v2_quarantine_reason_breakdown=q_breakdown,
        v2_reclassified_count=reclassified,
        v2_duplicate_events_count=sum(1 for e in v2_events if e.is_duplicate_report),
        v2_cross_stock_impacts_count=ci_v2_total,
        v2_spillover_before=spill_before,
        v2_spillover_after=spill_after,
    )
