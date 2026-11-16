"""6-Category News & Event Collector (Phase 5 — Section 2).

Separately collects raw news/events for:
A. Daily company news
B. Promoter and management intelligence
C. Industry and sector intelligence
D. Competitor intelligence
E. Regulatory and government intelligence
F. Global economic and geopolitical intelligence

Reuses `_search_google_news_rss` from `backend/intel_engine.py` (with SQLite caching)
and local NSE bulk/block deal records from `data/processed/nse_bulk_block_deals.parquet`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..config import DEEP_COMPANY_PROFILES, IST_TZ, PROCESSED_DATA_DIR
from ..intel_engine import _search_google_news_rss
from .relationships import PILOT_UNIVERSE_5, get_pilot_relationship


@dataclass
class RawNewsItem:
    symbol: str
    company_name: str
    event_category: str
    title: str
    source: str
    published: str
    url: str
    related_competitor_or_industry: str | None = None
    metadata: dict[str, Any] | None = None


def _safe_rss_fetch(query: str, limit: int = 4, use_network: bool = True) -> list[dict[str, str]]:
    if not use_network:
        return []
    try:
        return _search_google_news_rss(query, limit=limit)
    except Exception:
        return []


def collect_company_news(symbol: str, use_network: bool = True) -> list[RawNewsItem]:
    """A. Daily company-specific news, earnings, orders, launches, litigation, M&A."""
    sym = symbol.upper()
    rel = get_pilot_relationship(sym)
    cname = rel["company_name"]
    queries = [
        f'"{cname}" OR "{sym}" NSE earnings OR order OR contract OR launch OR expansion',
        f'"{cname}" India stock news',
    ]
    items: list[RawNewsItem] = []
    for q in queries:
        for row in _safe_rss_fetch(q, limit=4, use_network=use_network):
            items.append(
                RawNewsItem(
                    symbol=sym,
                    company_name=cname,
                    event_category="company_news",
                    title=row.get("title", ""),
                    source=row.get("source", "Indian Financial Media"),
                    published=row.get("published", ""),
                    url=row.get("url", ""),
                    related_competitor_or_industry=rel.get("sector"),
                )
            )
    return items


def collect_promoter_management_intel(
    symbol: str,
    use_network: bool = True,
    bulk_deals_df: pd.DataFrame | None = None,
) -> list[RawNewsItem]:
    """B. Promoter share purchases/sales, pledging, insider trades, management & governance."""
    sym = symbol.upper()
    rel = get_pilot_relationship(sym)
    cname = rel["company_name"]
    prof = DEEP_COMPANY_PROFILES.get(sym, {})
    promoter_group = prof.get("promoter_group", cname)
    leaders = [ldr.get("name", "") for ldr in prof.get("key_leaders", [])[:2] if ldr.get("name")]

    items: list[RawNewsItem] = []
    q_parts = [f'"{sym}" OR "{cname}" promoter OR insider OR pledge OR CEO OR resignation OR SEBI']
    if leaders:
        q_parts.append(f'"{leaders[0]}" "{cname}"')

    for q in q_parts:
        for row in _safe_rss_fetch(q, limit=3, use_network=use_network):
            items.append(
                RawNewsItem(
                    symbol=sym,
                    company_name=cname,
                    event_category="promoter_management",
                    title=row.get("title", ""),
                    source=row.get("source", "Corporate Governance Feed"),
                    published=row.get("published", ""),
                    url=row.get("url", ""),
                    related_competitor_or_industry=promoter_group,
                )
            )

    # Also ingest local NSE Bulk/Block deals if available
    if bulk_deals_df is None:
        p = PROCESSED_DATA_DIR / "nse_bulk_block_deals.parquet"
        if p.exists():
            try:
                bulk_deals_df = pd.read_parquet(p)
            except Exception:
                bulk_deals_df = None

    if bulk_deals_df is not None and not bulk_deals_df.empty and "symbol" in bulk_deals_df.columns:
        inst_exclusions = (
            "mutual fund", "mf", "etf", "asset management", "amc", "morgan", "goldman",
            "citigroup", "jpmorgan", "bofa", "merrill", "nomura", "ubs", "barclays",
            "hdfc", "icici", "sbi", "kotak", "axis", "nippon", "uti", "aditya birla sun life",
            "securities", "broking", "advisors", "partners", "llp", "fii", "fund",
        )
        promoter_hints = {
            "RELIANCE": ("sikka", "ambani", "reliance services", "deccan", "tattvam"),
            "TCS": ("tata sons",),
            "TATAELXSI": ("tata sons", "tata investment"),
            "ITC": ("tobacco manufacturers", "myddleton", "rothmans"),
            "BIRLACABLE": ("vindhya telelinks", "universal cables", "august agents", "insilco"),
        }.get(sym, ())

        sub = bulk_deals_df[bulk_deals_df["symbol"].astype(str).str.upper() == sym].tail(5)
        for _, r in sub.iterrows():
            side = str(r.get("side", "BUY")).upper()
            deal_type = str(r.get("deal_type", "BULK")).upper()
            client = str(r.get("client_name", "Institutional Investor"))
            c_low = client.lower()
            is_inst = any(tok in c_low for tok in inst_exclusions)
            is_verified_promoter = (not is_inst) and (
                "promoter" in c_low or "insider" in c_low or any(h in c_low for h in promoter_hints)
            )
            val_cr = float(r.get("value_cr", 0.0) or 0.0)
            deal_dt = str(r.get("deal_date", ""))
            cat = "promoter_management" if is_verified_promoter else "company_news"
            rel_label = promoter_group if is_verified_promoter else "Institutional Bulk/Block Deal"
            prefix = "Verified Promoter" if is_verified_promoter else "Institutional"
            items.append(
                RawNewsItem(
                    symbol=sym,
                    company_name=cname,
                    event_category=cat,
                    title=f"{prefix} NSE {deal_type} Deal {side}: {client} ({val_cr:.2f} Cr) in {sym}",
                    source="NSE Official Bulk/Block Deals",
                    published=f"{deal_dt}T15:45:00+05:30" if deal_dt else "",
                    url="https://www.nseindia.com/report-detail/display-bulk-and-block-deals",
                    related_competitor_or_industry=rel_label,
                    metadata={
                        "deal_type": deal_type,
                        "side": side,
                        "value_cr": val_cr,
                        "is_verified_promoter": is_verified_promoter,
                    },
                )
            )
    return items


def collect_industry_sector_intel(symbol: str, use_network: bool = True) -> list[RawNewsItem]:
    """C. Industry growth, demand trends, input costs, pricing, supply-chain & capacity."""
    sym = symbol.upper()
    rel = get_pilot_relationship(sym)
    cname = rel["company_name"]
    sector = rel.get("sector", "Indian Equities")

    sector_queries = {
        "RELIANCE": "India telecom tariff OR refining margin OR organized retail demand",
        "TCS": "India IT services demand OR BFSI tech spending OR GenAI deals",
        "TATAELXSI": "automotive ER&D SDV software OR EV R&D spending",
        "ITC": "India FMCG rural demand OR cigarette tax OR agri input cost",
        "BIRLACABLE": "India optical fiber cable BharatNet OR 5G FTTH capex",
    }
    q = sector_queries.get(sym, f"{sector} India industry demand supply chain")
    items: list[RawNewsItem] = []
    for row in _safe_rss_fetch(q, limit=4, use_network=use_network):
        items.append(
            RawNewsItem(
                symbol=sym,
                company_name=cname,
                event_category="industry_sector",
                title=row.get("title", ""),
                source=row.get("source", "Sector Intelligence"),
                published=row.get("published", ""),
                url=row.get("url", ""),
                related_competitor_or_industry=sector,
            )
        )
    return items


def collect_competitor_intel(symbol: str, use_network: bool = True) -> list[RawNewsItem]:
    """D. Competitor quarterly results, market share, launches, contracts, disruptions."""
    sym = symbol.upper()
    rel = get_pilot_relationship(sym)
    cname = rel["company_name"]
    comps = rel.get("competitors", [])[:3]
    items: list[RawNewsItem] = []
    for comp in comps:
        comp_name = comp.get("name", comp.get("symbol", ""))
        comp_sym = comp.get("symbol", comp_name)
        q = f'"{comp_name}" results OR contract OR launch OR expansion OR disruption India'
        for row in _safe_rss_fetch(q, limit=2, use_network=use_network):
            items.append(
                RawNewsItem(
                    symbol=sym,
                    company_name=cname,
                    event_category="competitor",
                    title=row.get("title", ""),
                    source=row.get("source", "Competitor Watch"),
                    published=row.get("published", ""),
                    url=row.get("url", ""),
                    related_competitor_or_industry=comp_sym,
                )
            )
    return items


def collect_regulatory_government_intel(symbol: str, use_network: bool = True) -> list[RawNewsItem]:
    """E. SEBI, RBI, government policy, taxation, import/export & legal developments."""
    sym = symbol.upper()
    rel = get_pilot_relationship(sym)
    cname = rel["company_name"]
    reg_queries = {
        "RELIANCE": "SEBI OR TRAI OR windfall tax OR CCI India telecom energy",
        "TCS": "US H-1B visa OR SEBI buyback OR India IT regulation",
        "TATAELXSI": "EU automotive ADAS regulation OR EV policy India",
        "ITC": "GST Council cigarette OR FSSAI OR RBI inflation India",
        "BIRLACABLE": "DoT BharatNet tender OR optical fiber anti-dumping duty India",
    }
    q = reg_queries.get(sym, f"SEBI OR RBI OR government policy India {rel.get('sector', '')}")
    items: list[RawNewsItem] = []
    for row in _safe_rss_fetch(q, limit=3, use_network=use_network):
        items.append(
            RawNewsItem(
                symbol=sym,
                company_name=cname,
                event_category="regulatory_government",
                title=row.get("title", ""),
                source=row.get("source", "Regulatory Monitor"),
                published=row.get("published", ""),
                url=row.get("url", ""),
                related_competitor_or_industry="SEBI / RBI / Govt Policy",
            )
        )
    return items


def collect_global_macro_intel(symbol: str, use_network: bool = True) -> list[RawNewsItem]:
    """F. Global interest rates, inflation, currency, crude oil & geopolitical developments."""
    sym = symbol.upper()
    rel = get_pilot_relationship(sym)
    cname = rel["company_name"]
    macro_queries = {
        "RELIANCE": "Brent crude oil price OPEC OR USD INR rupee",
        "TCS": "US Federal Reserve interest rate OR USD INR rupee OR US tech spending",
        "TATAELXSI": "European auto sales OR global EV demand OR USD INR",
        "ITC": "India monsoon OR palm oil price OR RBI repo rate inflation",
        "BIRLACABLE": "copper price OR global 5G FTTH capex OR US BEAD broadband",
    }
    q = macro_queries.get(sym, "Fed interest rate OR Brent crude OR USD INR OR inflation")
    items: list[RawNewsItem] = []
    for row in _safe_rss_fetch(q, limit=3, use_network=use_network):
        items.append(
            RawNewsItem(
                symbol=sym,
                company_name=cname,
                event_category="global_macro",
                title=row.get("title", ""),
                source=row.get("source", "Global Macro Feed"),
                published=row.get("published", ""),
                url=row.get("url", ""),
                related_competitor_or_industry="Global Macro",
            )
        )
    return items


def collect_stock_all_categories(
    symbol: str,
    use_network: bool = True,
    bulk_deals_df: pd.DataFrame | None = None,
) -> list[RawNewsItem]:
    """Collect all 6 news/event categories for a single symbol."""
    all_items: list[RawNewsItem] = []
    all_items.extend(collect_company_news(symbol, use_network=use_network))
    all_items.extend(collect_promoter_management_intel(symbol, use_network=use_network, bulk_deals_df=bulk_deals_df))
    all_items.extend(collect_industry_sector_intel(symbol, use_network=use_network))
    all_items.extend(collect_competitor_intel(symbol, use_network=use_network))
    all_items.extend(collect_regulatory_government_intel(symbol, use_network=use_network))
    all_items.extend(collect_global_macro_intel(symbol, use_network=use_network))
    return [it for it in all_items if it.title and it.title.strip()]


def collect_all_pilot_news(
    symbols: tuple[str, ...] | list[str] = PILOT_UNIVERSE_5,
    use_network: bool = True,
) -> dict[str, Any]:
    """Collect all 6 news categories across the 5-stock pilot universe."""
    bulk_df: pd.DataFrame | None = None
    p = PROCESSED_DATA_DIR / "nse_bulk_block_deals.parquet"
    if p.exists():
        try:
            bulk_df = pd.read_parquet(p)
        except Exception:
            bulk_df = None

    by_symbol: dict[str, list[RawNewsItem]] = {}
    warnings: list[str] = []
    for sym in symbols:
        s_up = sym.strip().upper()
        items = collect_stock_all_categories(s_up, use_network=use_network, bulk_deals_df=bulk_df)
        by_symbol[s_up] = items
        if not items:
            warnings.append(f"No live news items collected for {s_up} (offline or empty RSS).")

    return {
        "collected_at": datetime.now(IST_TZ).isoformat(),
        "by_symbol": by_symbol,
        "warnings": warnings,
    }
