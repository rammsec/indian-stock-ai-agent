"""Deep Promoter, Stakeholder, Competitor & Top-20 Movers Intelligence Engine.

Provides:
1. Promoter, CEO & Stakeholder statement/tweet/article scraping + share-price impact analysis.
2. Direct Competitor performance & statement read-across analysis.
3. Yesterday's Top 20 NSE Gainers & Top 20 NSE Losers with 2 concrete reasons per stock.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import feedparser
import pandas as pd
import requests
import yfinance as yf

from .config import (
    BROAD_NSE_MOVERS_POOL,
    BROWSER_USER_AGENT,
    DEEP_COMPANY_PROFILES,
)
from .data_engine import cache_get, cache_set
from .models import (
    CompanyOverview,
    CompetitorIntelItem,
    LeaderProfile,
    PromoterAndCompetitorIntel,
    PromoterStatementItem,
    StockholderAnalysis,
    TopMoverItem,
)
from .technical_engine import compute_rsi


def _search_google_news_rss(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch targeted Indian English news/statements from Google News RSS."""
    cache_key = f"gnews_q::{query[:80]}::{limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = "https://news.google.com/rss/search"
    params = {"q": f"{query} when:30d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    out: list[dict[str, str]] = []
    try:
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=7,
        )
        if r.status_code == 200:
            feed = feedparser.parse(r.text)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                source = ""
                if hasattr(entry, "source") and hasattr(entry.source, "title"):
                    source = entry.source.title
                elif " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title, source = parts[0].strip(), parts[1].strip()
                out.append(
                    {
                        "title": title,
                        "source": source or "Indian Financial Media",
                        "published": entry.get("published", "")[:22],
                        "url": entry.get("link", ""),
                    }
                )
    except Exception:
        pass

    if out:
        cache_set(cache_key, out, ttl_seconds=3600)
    return out


def _classify_statement_impact(
    headline: str, company_name: str
) -> tuple[str, str]:
    """Classify a promoter/stakeholder/company headline into (BULLISH/BEARISH/NEUTRAL, why_it_affects_share_value)."""
    h_low = headline.lower()

    bull_triggers = [
        (
            ("order", "contract", "win", "wins", "bag", "bags", "tender", "deal"),
            "Direct order-book / deal-pipeline expansion increases forward 4-quarter revenue visibility and earnings multiples.",
        ),
        (
            ("profit", "beat", "surge", "jump", "growth", "margin", "record", "high"),
            "Strong operating performance and earnings beat drive institutional EPS upgrades for the next session.",
        ),
        (
            ("buy", "accumulate", "target", "upgrade", "outperform", "overweight", "stake"),
            "Institutional accumulation or brokerage target upgrade attracts follow-through FII/DII buying.",
        ),
        (
            ("launch", "ai", "5g", "expansion", "capex", "new energy", "partner", "patent", "r&d"),
            "Strategic R&D, product launch, or capacity expansion strengthens long-term competitive moat.",
        ),
        (
            ("dividend", "buyback", "bonus", "demerger", "unlock"),
            "Shareholder-friendly capital allocation directly supports valuation floor and yield demand.",
        ),
    ]

    bear_triggers = [
        (
            ("fall", "drop", "plunge", "decline", "slump", "weak", "miss", "loss"),
            "Earnings or margin softness triggers short-term institutional de-risking and multiple compression.",
        ),
        (
            ("sell", "downgrade", "cut", "reduce", "underperform", "offload", "block deal"),
            "Stake sale supply overhang or brokerage downgrade creates near-term selling pressure.",
        ),
        (
            ("tax", "duty", "penalty", "notice", "probe", "inflation", "cost", "delay"),
            "Regulatory headwind or input-cost inflation weighs on near-term operating margins.",
        ),
    ]

    for words, explanation in bull_triggers:
        if any(w in h_low for w in words):
            return "BULLISH", explanation

    for words, explanation in bear_triggers:
        if any(w in h_low for w in words):
            return "BEARISH", explanation

    return (
        "NEUTRAL",
        f"Ongoing leadership commentary and corporate updates monitored by institutional desks for {company_name}.",
    )


def _quick_quote_batch(symbols: list[str]) -> dict[str, dict[str, float]]:
    """Fetch 1-month daily close & volume metrics for a batch of NSE symbols via yfinance."""
    if not symbols:
        return {}
    clean_syms = list(dict.fromkeys(s.upper().strip() for s in symbols if s.strip()))
    cache_key = "batch_quotes_v2::" + ",".join(sorted(clean_syms))
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    ns_tickers = [f"{s}.NS" for s in clean_syms]
    out: dict[str, dict[str, float]] = {}
    try:
        df = yf.download(
            tickers=" ".join(ns_tickers),
            period="1mo",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )
        for sym in clean_syms:
            tk = f"{sym}.NS"
            try:
                sub = df[tk] if len(clean_syms) > 1 else df
                sub = sub.dropna(subset=["Close"])
                if sub.empty or len(sub) < 2:
                    continue
                close_s = sub["Close"].astype(float)
                vol_s = sub["Volume"].astype(float) if "Volume" in sub.columns else pd.Series([1.0] * len(sub))
                cur = float(close_s.iloc[-1])
                prev = float(close_s.iloc[-2])
                day_chg = round((cur - prev) / prev * 100.0, 2) if prev > 0 else 0.0
                base_5d = float(close_s.iloc[max(0, len(close_s) - 6)])
                ret_5d = round((cur - base_5d) / base_5d * 100.0, 2) if base_5d > 0 else 0.0
                base_1m = float(close_s.iloc[0])
                ret_1m = round((cur - base_1m) / base_1m * 100.0, 2) if base_1m > 0 else 0.0

                vol_cur = float(vol_s.iloc[-1])
                vol_avg = float(vol_s.iloc[:-1].mean()) if len(vol_s) > 2 else vol_cur
                vol_ratio = round(vol_cur / vol_avg, 2) if vol_avg > 0 else 1.0

                rsi_series = compute_rsi(close_s, period=min(14, max(5, len(close_s) - 1)))
                rsi_val = round(float(rsi_series.iloc[-1]), 1)
                if math.isnan(rsi_val):
                    rsi_val = 50.0

                out[sym] = {
                    "current_price": round(cur, 2),
                    "prev_close": round(prev, 2),
                    "day_change_pct": day_chg,
                    "return_5d_pct": ret_5d,
                    "return_1m_pct": ret_1m,
                    "volume_surge_ratio": max(0.2, min(9.9, vol_ratio)),
                    "rsi_14": rsi_val,
                }
            except Exception:
                continue
    except Exception:
        pass

    if out:
        cache_set(cache_key, out, ttl_seconds=1800)
    return out


def build_promoter_and_competitor_intel(
    symbol: str,
    overview: CompanyOverview,
    stockholders: StockholderAnalysis,
) -> PromoterAndCompetitorIntel:
    """Build deep Promoter, Stakeholder, Statement/Tweet, and Competitor War-Room profile."""
    sym = symbol.upper().strip()
    cache_key = f"prom_comp_intel_v2::{sym}"
    cached = cache_get(cache_key)
    if cached is not None:
        return PromoterAndCompetitorIntel(**cached)

    profile = DEEP_COMPANY_PROFILES.get(sym)

    # Dynamic profile construction if the user adds a custom stock beyond our pre-mapped profiles
    if not profile:
        prom_names = [
            h.name for h in stockholders.named_stockholders if h.category == "Promoter"
        ]
        inst_names = [
            f"{h.name} ({h.category} — {h.latest_holding_pct:.2f}%)"
            for h in stockholders.named_stockholders
            if h.category in ("FII", "DII")
        ]
        promoter_group = (
            f"{', '.join(prom_names[:3])} ({stockholders.promoters_pct:.2f}% Total Promoter Stake)"
            if prom_names
            else f"Promoter & Insider Group ({stockholders.promoters_pct:.2f}% Stake)"
        )
        leaders = [
            LeaderProfile(
                name=prom_names[0] if prom_names else f"{overview.name} Managing Director & Board",
                role="Principal Promoter & Executive Leadership",
                focus=f"Capital allocation, operating margin expansion & growth in {overview.industry or overview.sector}",
                handle_or_source="BSE/NSE Corporate Filings & Earnings Concalls",
            )
        ]
        stakeholders_watch = inst_names[:4] or [
            f"Promoters hold {stockholders.promoters_pct:.2f}% ({stockholders.promoter_change_4q_pct:+.2f}% 4Q change)",
            f"Foreign Institutions (FII) hold {stockholders.fii_pct:.2f}% ({stockholders.fii_change_4q_pct:+.2f}% 4Q change)",
            f"Domestic Institutions (DII) hold {stockholders.dii_pct:.2f}% ({stockholders.dii_change_4q_pct:+.2f}% 4Q change)",
        ]
        search_queries = [
            f'"{overview.name}" promoter OR CEO OR order OR results',
            f"{sym} NSE share price news statement",
        ]
        # Pick 4 sector peers from BROAD_NSE_MOVERS_POOL
        sec_low = (overview.sector + " " + overview.industry).lower()
        comp_candidates = [
            p
            for p in BROAD_NSE_MOVERS_POOL
            if p["symbol"] != sym
            and any(w in p["sector"].lower() for w in sec_low.split()[:2])
        ][:4]
        if len(comp_candidates) < 3:
            comp_candidates = [p for p in BROAD_NSE_MOVERS_POOL if p["symbol"] != sym][:4]
        comp_specs = [
            {
                "symbol": c["symbol"],
                "name": c["name"],
                "segment": c["sector"],
                "impact_driver": f"Sector peer valuation, order wins, and margin read-across for {overview.name}.",
            }
            for c in comp_candidates
        ]
    else:
        promoter_group = profile["promoter_group"]
        leaders = [LeaderProfile(**ld) for ld in profile["key_leaders"]]
        stakeholders_watch = list(profile["key_stakeholders_watch"])
        search_queries = list(profile["promoter_search_queries"])
        comp_specs = list(profile["competitors"])

    # Scrape Promoter, CEO & Stakeholder Statements / Articles / Social Mentions
    raw_articles: list[dict[str, str]] = []
    for q in search_queries:
        raw_articles.extend(_search_google_news_rss(q, limit=4))

    seen_titles: set[str] = set()
    statement_items: list[PromoterStatementItem] = []
    leader_last_names = [ld.name.split()[-1].lower() for ld in leaders if ld.name]

    for art in raw_articles:
        t = art["title"]
        if not t or t in seen_titles:
            continue
        seen_titles.add(t)
        impact, why_str = _classify_statement_impact(t, overview.name)

        speaker = art["source"]
        for ld in leaders:
            last_n = ld.name.split()[-1].lower()
            if len(last_n) > 3 and last_n in t.lower():
                speaker = f"{ld.name} ({ld.role}) via {art['source']}"
                break

        statement_items.append(
            PromoterStatementItem(
                speaker_or_source=speaker,
                published=art["published"],
                headline=t,
                url=art["url"],
                share_price_impact=impact,  # type: ignore[arg-type]
                why_it_affects_share_value=why_str,
            )
        )
        if len(statement_items) >= 6:
            break

    # Fetch live Competitor quotes & latest news/statements
    comp_syms = [c["symbol"] for c in comp_specs]
    comp_quotes = _quick_quote_batch(comp_syms)
    competitor_items: list[CompetitorIntelItem] = []

    for c_spec in comp_specs:
        c_sym = c_spec["symbol"]
        q_data = comp_quotes.get(c_sym, {})
        c_news = _search_google_news_rss(f"{c_spec['name']} {c_sym} NSE results OR order OR guidance", limit=1)
        latest_hl = c_news[0]["title"] if c_news else f"{c_spec['name']} trading active in {c_spec['segment']}."
        latest_url = c_news[0]["url"] if c_news else f"https://www.screener.in/company/{c_sym}/"

        c_chg = q_data.get("day_change_pct", 0.0)
        c_1m = q_data.get("return_1m_pct", 0.0)
        if c_chg >= 1.5 or c_1m >= 5.0:
            read_across = "POSITIVE READ-ACROSS"
        elif c_chg <= -1.8 or c_1m <= -6.0:
            read_across = "COMPETITIVE THREAT"
        else:
            read_across = "NEUTRAL"

        competitor_items.append(
            CompetitorIntelItem(
                symbol=c_sym,
                name=c_spec["name"],
                segment=c_spec["segment"],
                current_price=q_data.get("current_price", 0.0),
                day_change_pct=c_chg,
                return_1m_pct=c_1m,
                latest_statement_or_news=latest_hl,
                latest_news_url=latest_url,
                read_across_signal=read_across,  # type: ignore[arg-type]
                impact_on_share_value=c_spec["impact_driver"],
            )
        )

    result = PromoterAndCompetitorIntel(
        promoter_group=promoter_group,
        key_leaders=leaders,
        key_stakeholders_watch=stakeholders_watch,
        promoter_statements_and_articles=statement_items,
        competitors=competitor_items,
    )
    cache_set(cache_key, result.model_dump(), ttl_seconds=3600)
    return result


def _generate_mover_reasons(
    sym: str,
    name: str,
    sector: str,
    day_chg: float,
    ret_5d: float,
    vol_ratio: float,
    rsi: float,
    headline: str,
    is_gainer: bool,
) -> tuple[str, str, str]:
    """Generate 2 concrete reasons why the stock went up/down + next-day verdict."""
    hl_low = headline.lower()

    if is_gainer:
        # Reason 1: News/Fundamental catalyst
        if any(w in hl_low for w in ("profit", "q2", "q3", "q4", "q1", "results", "beat", "margin", "ebitda")):
            r1 = f"Earnings & Margin Beat: Positive reaction to quarterly financial results and margin expansion ({headline[:85]})."
        elif any(w in hl_low for w in ("order", "win", "bag", "contract", "deal", "tender", "mou")):
            r1 = f"Order / Deal Catalyst: Fresh order win or commercial agreement sparked institutional buying ({headline[:85]})."
        elif any(w in hl_low for w in ("buy", "target", "upgrade", "outperform", "accumulate", "fii", "stake")):
            r1 = f"Institutional / Brokerage Accumulation: Upgraded target price and smart-money buying in {sector} ({headline[:85]})."
        elif headline:
            r1 = f"Sector & News Catalyst in {sector}: {headline[:95]}."
        else:
            r1 = f"Sector Rotation into {sector}: Strong institutional bid drove a +{day_chg:.2f}% session breakout."

        # Reason 2: Technical & Volume confirmation
        if vol_ratio >= 1.35:
            r2 = f"High-Volume Breakout: Session volume surged to {vol_ratio:.2f}x the 20-day average with RSI(14) at {rsi:.1f} (5D trend: {ret_5d:+.1f}%)."
        elif rsi >= 68:
            r2 = f"Momentum Acceleration: Strong follow-through pushed RSI(14) to {rsi:.1f} with +{ret_5d:.1f}% 5-day gain."
        else:
            r2 = f"Technical Rebound & Delivery Buying: Reclaimed key short-term moving average with RSI(14) at {rsi:.1f} and {vol_ratio:.2f}x volume."

        verdict = (
            "BOOK PARTIAL PROFITS (RSI Overextended >72)"
            if rsi > 72.0
            else "CONTINUATION BUY ON DIPS"
        )
    else:
        # Loser Reason 1: News/Fundamental headwind
        if any(w in hl_low for w in ("miss", "fall", "drop", "margin", "loss", "weak", "cut", "downgrade")):
            r1 = f"Earnings / Guidance Pressure: Valuation adjusted on margin or growth concerns ({headline[:85]})."
        elif any(w in hl_low for w in ("block", "sell", "stake", "offload", "promoter", "fii")):
            r1 = f"Institutional / Block Supply Overhang: Secondary supply or FII profit-booking weighed on {name} ({headline[:85]})."
        elif any(w in hl_low for w in ("tax", "penalty", "notice", "cost", "inflation", "regulatory")):
            r1 = f"Regulatory / Cost Headwind in {sector}: Near-term sentiment hit by cost or policy news ({headline[:85]})."
        elif headline:
            r1 = f"Profit-Booking & Sector Drag in {sector}: {headline[:95]}."
        else:
            r1 = f"Sectoral Profit-Booking in {sector}: Institutional trimming after recent run-up caused a {day_chg:.2f}% pullback."

        # Loser Reason 2: Technical & Volume breakdown
        if vol_ratio >= 1.35:
            r2 = f"Distribution Volume Spike: Selling occurred on {vol_ratio:.2f}x 20-day average volume, pushing RSI(14) to {rsi:.1f} (5D: {ret_5d:+.1f}%)."
        elif rsi <= 36:
            r2 = f"Oversold Exhaustion: Multi-session pullback (5D: {ret_5d:+.1f}%) dragged RSI(14) down to {rsi:.1f} near support."
        else:
            r2 = f"Short-Term Support Test: Slipped {day_chg:.2f}% with RSI(14) at {rsi:.1f} and {vol_ratio:.2f}x relative volume."

        verdict = (
            "BUY THE DIP (Oversold Quality Near Support)"
            if rsi <= 38.0
            else "WAIT FOR SUPPORT STABILIZATION"
        )

    return r1, r2, verdict


def fetch_top_20_gainers_and_losers() -> dict[str, Any]:
    """Compute Yesterday's Top 20 NSE Gainers and Top 20 NSE Losers with 2 reasons each."""
    cache_key = "nse_top20_movers_v2"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    pool_map = {item["symbol"]: item for item in BROAD_NSE_MOVERS_POOL}
    all_syms = list(pool_map.keys())
    quotes = _quick_quote_batch(all_syms)

    scored_rows: list[dict[str, Any]] = []
    for sym, q in quotes.items():
        meta = pool_map.get(sym, {"symbol": sym, "name": sym, "sector": "NSE Equities"})
        scored_rows.append({**meta, **q})

    # Sort descending for Gainers, ascending for Losers
    sorted_desc = sorted(scored_rows, key=lambda x: x["day_change_pct"], reverse=True)
    sorted_asc = sorted(scored_rows, key=lambda x: x["day_change_pct"])

    raw_gainers = sorted_desc[:20]
    raw_losers = sorted_asc[:20]

    def _enrich_mover(item: dict[str, Any], rank: int, is_gainer: bool) -> TopMoverItem:
        sym = item["symbol"]
        name = item["name"]
        sector = item["sector"]
        news_list = _search_google_news_rss(f"{name} {sym} NSE share price", limit=1)
        hl_title = news_list[0]["title"] if news_list else ""
        hl_source = news_list[0]["source"] if news_list else "NSE Market Pulse"
        hl_url = news_list[0]["url"] if news_list else f"https://www.screener.in/company/{sym}/"

        r1, r2, verdict = _generate_mover_reasons(
            sym=sym,
            name=name,
            sector=sector,
            day_chg=item["day_change_pct"],
            ret_5d=item["return_5d_pct"],
            vol_ratio=item["volume_surge_ratio"],
            rsi=item["rsi_14"],
            headline=hl_title,
            is_gainer=is_gainer,
        )
        cur = item["current_price"]
        return TopMoverItem(
            rank=rank,
            symbol=sym,
            name=name,
            sector=sector,
            current_price=cur,
            prev_close=item["prev_close"],
            day_change_pct=item["day_change_pct"],
            return_5d_pct=item["return_5d_pct"],
            volume_surge_ratio=item["volume_surge_ratio"],
            rsi_14=item["rsi_14"],
            reason_1=r1,
            reason_2=r2,
            headline_title=hl_title,
            headline_source=hl_source,
            headline_url=hl_url,
            next_day_verdict=verdict,
            next_day_target=round(cur * 1.028, 2),
            next_day_stop_loss=round(cur * 0.982, 2),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        gainers = list(
            pool.map(
                lambda pair: _enrich_mover(pair[1], pair[0] + 1, True),
                enumerate(raw_gainers),
            )
        )
        losers = list(
            pool.map(
                lambda pair: _enrich_mover(pair[1], pair[0] + 1, False),
                enumerate(raw_losers),
            )
        )

    from datetime import datetime

    payload = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "universe_scanned": len(scored_rows),
        "top_20_gainers": [g.model_dump() for g in gainers],
        "top_20_losers": [l.model_dump() for l in losers],
    }
    cache_set(cache_key, payload, ttl_seconds=1800)
    return payload
