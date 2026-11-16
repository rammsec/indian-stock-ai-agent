"""Live Indian Stock Market (NSE/BSE) Data Engine.

Combines:
1. Screener.in Consolidated/Standalone parser & AJAX Schedule APIs:
   - Last 5 Annual Reports & Last 5 Quarterly Reports
   - Cash Equivalents, Liquid Investments, Long/Short-term Borrowings, OCF & FCF
   - Shareholding Pattern (Promoters, FIIs, DIIs, Public) + Named Institutional Stockholders
   - Annual Report PDF links, Intangible Assets, CWIP & R&D Expenditure
2. Yahoo Finance (`yfinance` .NS / .BO) for daily OHLCV bars and IFRS R&D line items
3. Google News RSS India (`hl=en-IN&gl=IN&ceid=IN:en`) for live news catalysts
4. Thread-safe SQLite caching layer (`pistol-core` pattern) for sub-second repeat lookups
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import urllib.parse
from datetime import datetime
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

from .config import (
    BROWSER_USER_AGENT,
    CACHE_DB_PATH,
    DEFAULT_NSE_UNIVERSE,
    REPORTED_RD_INTENSITY_PCT,
    REQUEST_TIMEOUT,
    SECTOR_DEFAULT_RD_PCT,
)
from .models import (
    AnnualReportDoc,
    CashAndBalanceSheet,
    CashBalanceYear,
    CompanyOverview,
    NamedShareholder,
    NewsHeadline,
    RDAndInnovation,
    RDYearRecord,
    ReportPeriod,
    ShareholdingQuarter,
    StockholderAnalysis,
)

# Disable yfinance peewee thread-local cache errors
try:
    import yfinance.cache as _yfc

    _yfc.get_tz_cache().dummy = True
    _yfc.get_cookie_cache().dummy = True
    _yfc.get_isin_cache().dummy = True
except Exception:
    pass


_CACHE_LOCK = threading.Lock()
_CACHE_READY = False


def _init_cache() -> None:
    global _CACHE_READY
    if _CACHE_READY:
        return
    with _CACHE_LOCK:
        if _CACHE_READY:
            return
        CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CACHE_DB_PATH), timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        _CACHE_READY = True


def cache_get(key: str) -> Any | None:
    _init_cache()
    now = time.time()
    conn = sqlite3.connect(str(CACHE_DB_PATH), timeout=10)
    try:
        row = conn.execute(
            "SELECT payload, expires_at FROM kv_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        payload, expires_at = row
        if now > expires_at:
            return None
        return json.loads(payload)
    except Exception:
        return None
    finally:
        conn.close()


def cache_set(key: str, value: Any, ttl_seconds: int = 1800) -> None:
    _init_cache()
    expires = time.time() + ttl_seconds
    conn = sqlite3.connect(str(CACHE_DB_PATH), timeout=10)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO kv_cache (cache_key, payload, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), expires),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def _parse_num(val: Any) -> float | None:
    """Parse Indian formatted numbers like '₹ 1,23,456 Cr', '50.48%', '-12.4'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return None if math.isnan(float(val)) else float(val)
    s = str(val).strip()
    if not s or s in ("-", "--", "N/A", "null", "None"):
        return None
    s = re.sub(r"[₹,%Cr\s,]", "", s)
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else None


def _clean_label(text: str) -> str:
    """Strip Screener '+' button symbols and extra whitespace from row labels."""
    cleaned = re.sub(r"\+$", "", text.replace("\xa0", " ").strip()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": BROWSER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
            "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )
    return s


def search_indian_stocks(query: str) -> list[dict[str, Any]]:
    """Search NSE/BSE stocks via Screener.in live search API."""
    q = (query or "").strip()
    if not q:
        return DEFAULT_NSE_UNIVERSE
    cache_key = f"search::{q.lower()}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    s = _http_session()
    try:
        url = f"https://www.screener.in/api/company/search/?q={urllib.parse.quote(q)}"
        r = s.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            items = r.json()
            out = []
            for item in items:
                u = item.get("url", "")
                parts = [p for p in u.split("/") if p and p not in ("company", "consolidated")]
                sym = parts[0].upper() if parts else ""
                if sym:
                    out.append(
                        {
                            "symbol": sym,
                            "name": item.get("name", sym),
                            "screener_id": item.get("id"),
                            "url": u,
                        }
                    )
            if out:
                cache_set(cache_key, out, ttl_seconds=3600)
                return out
    except Exception:
        pass

    q_up = q.upper()
    return [
        item
        for item in DEFAULT_NSE_UNIVERSE
        if q_up in item["symbol"].upper() or q_up in item["name"].upper()
    ]


def _parse_screener_table(section_el: Any) -> tuple[list[str], dict[str, list[float | None]]]:
    """Parse a Screener.in `<table class="data-table">` into (headers, {row_label: values})."""
    if section_el is None:
        return [], {}
    table = section_el.find("table", class_="data-table")
    if table is None:
        return [], {}

    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        ths = thead.find_all("th")
        headers = [th.get_text(" ", strip=True) for th in ths[1:]]

    rows: dict[str, list[float | None]] = {}
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            label = _clean_label(tds[0].get_text(" ", strip=True))
            if not label:
                continue
            vals = [_parse_num(td.get_text(" ", strip=True)) for td in tds[1:]]
            rows[label] = vals
    return headers, rows


def _fetch_screener_schedule(
    session: requests.Session,
    company_id: str,
    parent: str,
    section: str,
    is_consolidated: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fetch expandable schedule rows (Cash Equivalents, Borrowings, Fixed Assets, Expenses)."""
    if not company_id:
        return {}
    cons_param = "" if is_consolidated else "false"
    url = (
        f"https://www.screener.in/api/company/{company_id}/schedules/"
        f"?parent={urllib.parse.quote(parent)}&section={urllib.parse.quote(section)}"
        f"&consolidated={cons_param}"
    )
    try:
        r = session.get(
            url,
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _fetch_named_shareholders(
    session: requests.Session, company_id: str
) -> list[NamedShareholder]:
    """Fetch named Promoters, FIIs, DIIs, and Public shareholders via `/api/3/{id}/investors/`."""
    if not company_id:
        return []
    categories: list[tuple[str, str]] = [
        ("promoters", "Promoter"),
        ("foreign_institutions", "FII"),
        ("domestic_institutions", "DII"),
        ("public", "Public"),
    ]
    named_list: list[NamedShareholder] = []
    for slug, label in categories:
        url = f"https://www.screener.in/api/3/{company_id}/investors/{slug}/quarterly/"
        try:
            r = session.get(
                url,
                headers={"X-Requested-With": "XMLHttpRequest"},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            if not isinstance(data, dict):
                continue
            for holder_name, q_map in list(data.items())[:8]:
                if not isinstance(q_map, dict):
                    continue
                periods = [
                    (k, _parse_num(v))
                    for k, v in q_map.items()
                    if k not in ("isExpandable", "tooltips") and _parse_num(v) is not None
                ]
                if not periods:
                    continue
                latest_p, latest_val = periods[-1]
                prev_val = periods[-2][1] if len(periods) >= 2 else latest_val
                first_4q_val = periods[max(0, len(periods) - 4)][1]
                if latest_val is None or latest_val <= 0.01:
                    continue
                named_list.append(
                    NamedShareholder(
                        name=holder_name,
                        category=label,  # type: ignore[arg-type]
                        latest_holding_pct=round(float(latest_val), 2),
                        previous_holding_pct=round(float(prev_val or 0.0), 2),
                        change_4q_pct=round(float(latest_val) - float(first_4q_val or 0.0), 2),
                        trend=[
                            {"period": p, "pct": round(float(v or 0.0), 2)}
                            for p, v in periods[-5:]
                        ],
                    )
                )
        except Exception:
            continue
    return named_list


def _fetch_screener_chart(
    session: requests.Session, company_id: str, is_consolidated: bool = True
) -> list[dict[str, Any]]:
    """Fetch 365-day daily Price, DMA50, DMA200, and Volume series from Screener API."""
    if not company_id:
        return []
    cons = "true" if is_consolidated else "false"
    url = (
        f"https://www.screener.in/api/company/{company_id}/chart/"
        f"?q=Price-DMA50-DMA200-Volume&days=365&consolidated={cons}"
    )
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        payload = r.json()
        datasets = payload.get("datasets", [])
        by_date: dict[str, dict[str, Any]] = {}
        for ds in datasets:
            metric = ds.get("metric", "")
            for pt in ds.get("values", []):
                if len(pt) < 2:
                    continue
                dt = str(pt[0])
                val = _parse_num(pt[1])
                if dt not in by_date:
                    by_date[dt] = {"date": dt}
                if metric == "Price":
                    by_date[dt]["close"] = val
                elif metric == "DMA50":
                    by_date[dt]["dma50"] = val
                elif metric == "DMA200":
                    by_date[dt]["dma200"] = val
                elif metric == "Volume":
                    by_date[dt]["volume"] = val
        rows = [v for v in by_date.values() if v.get("close") is not None]
        rows.sort(key=lambda x: x["date"])
        return rows
    except Exception:
        return []


def _enrich_with_yfinance(
    symbol: str, screener_price_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Enrich daily OHLCV with Yahoo Finance (.NS/.BO) and extract reported R&D if present."""
    rd_by_year: dict[str, float] = {}
    ohlc_by_date: dict[str, dict[str, float]] = {}

    try:
        import yfinance as yf

        ns_sym = symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"
        ticker = yf.Ticker(ns_sym)
        hist = ticker.history(period="1y")
        if hist is not None and not hist.empty:
            for idx, row in hist.iterrows():
                dt_str = str(idx)[:10]
                ohlc_by_date[dt_str] = {
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": float(row["Volume"]),
                }

        inc = ticker.income_stmt
        if inc is not None and not inc.empty:
            rd_keys = [k for k in inc.index if "research" in str(k).lower()]
            if rd_keys:
                rd_series = inc.loc[rd_keys[0]]
                for col_ts, val in rd_series.items():
                    if val is not None and not math.isnan(float(val)) and float(val) > 0:
                        yr_label = f"Mar {str(col_ts)[:4]}"
                        # Yahoo reports in raw INR; convert to Crores (1 Cr = 1e7)
                        rd_by_year[yr_label] = round(float(val) / 1e7, 2)
    except Exception:
        pass

    # Overlay latest official NSE Bhavcopy (Open, High, Low, Close, VWAP, Deliverable Qty, Delivery %)
    try:
        from .collectors.nse_market_collector import fetch_nse_sec_bhavdata_full

        nse_bars, _ = fetch_nse_sec_bhavdata_full(symbols_filter={symbol.upper()})
        for nb in nse_bars:
            if nb.close is not None:
                ohlc_by_date[nb.observation_date] = {
                    "open": nb.open or nb.close,
                    "high": nb.high or nb.close,
                    "low": nb.low or nb.close,
                    "close": nb.close,
                    "volume": nb.volume or 0.0,
                    "vwap": nb.vwap,
                    "deliverable_qty": nb.deliverable_qty,
                    "delivery_pct": nb.delivery_pct,
                }
    except Exception:
        pass

    if screener_price_rows:
        existing_dates = {row["date"] for row in screener_price_rows}
        for row in screener_price_rows:
            dt = row["date"]
            if dt in ohlc_by_date:
                yf_bar = ohlc_by_date[dt]
                row["open"] = yf_bar["open"]
                row["high"] = yf_bar["high"]
                row["low"] = yf_bar["low"]
                if not row.get("volume"):
                    row["volume"] = yf_bar["volume"]
                if yf_bar.get("vwap") is not None:
                    row["vwap"] = yf_bar["vwap"]
                if yf_bar.get("deliverable_qty") is not None:
                    row["deliverable_qty"] = yf_bar["deliverable_qty"]
                if yf_bar.get("delivery_pct") is not None:
                    row["delivery_pct"] = yf_bar["delivery_pct"]
        # Append any newer NSE Bhavcopy date not yet in screener_price_rows
        for dt, vals in sorted(ohlc_by_date.items()):
            if dt not in existing_dates and vals.get("delivery_pct") is not None:
                screener_price_rows.append({"date": dt, **vals})
        screener_price_rows.sort(key=lambda x: x["date"])
        return screener_price_rows, rd_by_year

    if ohlc_by_date:
        fallback_rows = [
            {"date": dt, **vals} for dt, vals in sorted(ohlc_by_date.items())
        ]
        return fallback_rows, rd_by_year

    return [], rd_by_year


def fetch_stock_news(symbol: str, company_name: str, limit: int = 8) -> list[NewsHeadline]:
    """Fetch live Indian stock news via Google News RSS (`hl=en-IN&gl=IN&ceid=IN:en`)."""
    cache_key = f"news::{symbol.upper()}"
    cached = cache_get(cache_key)
    if cached is not None:
        return [NewsHeadline(**item) for item in cached]

    query = f"{company_name or symbol} {symbol} NSE stock share price when:7d"
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    s = _http_session()
    headlines: list[NewsHeadline] = []

    pos_words = {
        "surge", "surges", "jump", "jumps", "gain", "gains", "rally", "buy",
        "upgrade", "profit", "beat", "growth", "order", "win", "wins", "dividend",
        "record", "bullish", "outperform", "accumulation", "approval", "launch",
    }
    neg_words = {
        "fall", "falls", "drop", "drops", "plunge", "loss", "miss", "downgrade",
        "sell", "weak", "decline", "slump", "penalty", "probe", "bearish", "cut",
    }

    try:
        r = s.get(url, params=params, timeout=8)
        if r.status_code == 200:
            feed = feedparser.parse(r.text)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                source_title = ""
                if hasattr(entry, "source") and hasattr(entry.source, "title"):
                    source_title = entry.source.title
                elif " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title, source_title = parts[0].strip(), parts[1].strip()

                t_low = title.lower()
                p_hits = sum(1 for w in pos_words if w in t_low)
                n_hits = sum(1 for w in neg_words if w in t_low)
                sentiment = "POSITIVE" if p_hits > n_hits else ("NEGATIVE" if n_hits > p_hits else "NEUTRAL")

                pub = entry.get("published", "")[:25]
                headlines.append(
                    NewsHeadline(
                        title=title,
                        source=source_title or "Indian Market Wire",
                        published=pub,
                        url=entry.get("link", ""),
                        sentiment=sentiment,  # type: ignore[arg-type]
                    )
                )
    except Exception:
        pass

    if headlines:
        cache_set(cache_key, [h.model_dump() for h in headlines], ttl_seconds=1800)
    return headlines


def _build_report_periods(
    headers: list[str], rows: dict[str, list[float | None]], count: int = 5
) -> list[ReportPeriod]:
    """Extract last `count` annual or quarterly reports with YoY/period growth metrics."""
    if not headers or not rows:
        return []

    # Exclude 'TTM' column if we have enough dated fiscal years, or include it if needed
    valid_indices = [
        i for i, h in enumerate(headers) if h.strip() and h.strip().upper() != "TTM"
    ]
    if len(valid_indices) < count and len(headers) >= count:
        valid_indices = list(range(len(headers)))

    def _row_val(idx: int, *names: str) -> float | None:
        for name in names:
            for k, v_list in rows.items():
                if k.lower().startswith(name.lower()) and idx < len(v_list):
                    return v_list[idx]
        return None

    periods_all: list[ReportPeriod] = []
    for pos, col_idx in enumerate(valid_indices):
        p_label = headers[col_idx]
        rev = _row_val(col_idx, "Sales", "Revenue", "Interest Earned", "Total Income")
        exp = _row_val(col_idx, "Expenses", "Operating Expenses")
        op_prof = _row_val(col_idx, "Operating Profit", "Financing Profit")
        opm = _row_val(col_idx, "OPM", "Financing Margin")
        other_inc = _row_val(col_idx, "Other Income")
        interest = _row_val(col_idx, "Interest")
        dep = _row_val(col_idx, "Depreciation")
        pbt = _row_val(col_idx, "Profit before tax")
        tax = _row_val(col_idx, "Tax %", "Tax")
        np_val = _row_val(col_idx, "Net Profit")
        eps = _row_val(col_idx, "EPS in Rs", "EPS")
        div_payout = _row_val(col_idx, "Dividend Payout")

        rev_growth = None
        np_growth = None
        if pos > 0:
            prev_col = valid_indices[pos - 1]
            prev_rev = _row_val(prev_col, "Sales", "Revenue", "Interest Earned")
            prev_np = _row_val(prev_col, "Net Profit")
            if rev is not None and prev_rev and abs(prev_rev) > 0:
                rev_growth = round((rev - prev_rev) / abs(prev_rev) * 100.0, 2)
            if np_val is not None and prev_np and abs(prev_np) > 0:
                np_growth = round((np_val - prev_np) / abs(prev_np) * 100.0, 2)

        periods_all.append(
            ReportPeriod(
                period=p_label,
                revenue_cr=rev,
                expenses_cr=exp,
                operating_profit_cr=op_prof,
                opm_pct=opm,
                other_income_cr=other_inc,
                interest_cr=interest,
                depreciation_cr=dep,
                profit_before_tax_cr=pbt,
                tax_pct=tax,
                net_profit_cr=np_val,
                eps_inr=eps,
                dividend_payout_pct=div_payout,
                revenue_growth_pct=rev_growth,
                net_profit_growth_pct=np_growth,
            )
        )

    return periods_all[-count:]


def fetch_raw_company_data(symbol: str) -> dict[str, Any]:
    """Fetch complete fundamental, balance-sheet, stockholder, R&D, and technical raw data."""
    sym = symbol.upper().strip().replace(".NS", "").replace(".BO", "")
    cache_key = f"raw_company_v2::{sym}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    s = _http_session()
    is_consolidated = True
    url = f"https://www.screener.in/company/{urllib.parse.quote(sym)}/consolidated/"
    r = s.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code == 429:
        time.sleep(2.0)
        r = s.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code == 404:
        url = f"https://www.screener.in/company/{urllib.parse.quote(sym)}/"
        is_consolidated = False
        r = s.get(url, timeout=REQUEST_TIMEOUT)

    if r.status_code != 200:
        # Try resolving symbol via Screener search API
        matches = search_indian_stocks(sym)
        if matches and matches[0].get("url"):
            u_path = matches[0]["url"]
            is_consolidated = "consolidated" in u_path
            url = f"https://www.screener.in{u_path}"
            r = s.get(url, timeout=REQUEST_TIMEOUT)

    if r.status_code != 200:
        raise ValueError(f"Could not find Indian listed company '{sym}' on NSE/BSE.")

    html = r.text
    soup = BeautifulSoup(html, "lxml")

    # Check if consolidated table is empty and standalone is better
    pl_sec = soup.find("section", id="profit-loss")
    pl_headers, pl_rows = _parse_screener_table(pl_sec)
    if not pl_headers and is_consolidated:
        url = f"https://www.screener.in/company/{urllib.parse.quote(sym)}/"
        is_consolidated = False
        r2 = s.get(url, timeout=REQUEST_TIMEOUT)
        if r2.status_code == 200:
            html = r2.text
            soup = BeautifulSoup(html, "lxml")
            pl_sec = soup.find("section", id="profit-loss")
            pl_headers, pl_rows = _parse_screener_table(pl_sec)

    m_cid = re.search(r'data-row-company-id="(\d+)"', html)
    company_id = m_cid.group(1) if m_cid else ""

    h1 = soup.find("h1")
    company_name = h1.get_text(" ", strip=True) if h1 else sym

    # Top ratios parsing
    top_ratios: dict[str, Any] = {}
    ul_ratios = soup.find("ul", id="top-ratios")
    if ul_ratios:
        for li in ul_ratios.find_all("li"):
            name_el = li.find("span", class_="name")
            val_el = li.find("span", class_="value")
            if name_el and val_el:
                k = name_el.get_text(" ", strip=True)
                v_text = val_el.get_text(" ", strip=True)
                top_ratios[k] = v_text

    high_52w = 0.0
    low_52w = 0.0
    hl_text = top_ratios.get("High / Low", "")
    if "/" in hl_text:
        parts = hl_text.split("/")
        high_52w = _parse_num(parts[0]) or 0.0
        low_52w = _parse_num(parts[1]) or 0.0

    # Sector & Industry breadcrumbs
    peers_sec = soup.find("section", id="peers")
    sector_str = ""
    industry_str = ""
    if peers_sec:
        sub_p = peers_sec.find("p", class_="sub")
        if sub_p:
            links = [a.get_text(" ", strip=True) for a in sub_p.find_all("a")]
            if len(links) >= 2:
                sector_str = links[0]
                industry_str = links[-1]
            elif len(links) == 1:
                sector_str = links[0]

    # About text
    about_div = soup.find("div", class_="about")
    about_text = about_div.get_text(" ", strip=True) if about_div else ""

    # Pros & Cons
    pros: list[str] = []
    cons: list[str] = []
    analysis_sec = soup.find("section", id="analysis")
    if analysis_sec:
        pros_div = analysis_sec.find("div", class_="pros")
        if pros_div:
            pros = [li.get_text(" ", strip=True) for li in pros_div.find_all("li")]
        cons_div = analysis_sec.find("div", class_="cons")
        if cons_div:
            cons = [li.get_text(" ", strip=True) for li in cons_div.find_all("li")]

    # Quarterly & Annual Reports
    q_sec = soup.find("section", id="quarters")
    q_headers, q_rows = _parse_screener_table(q_sec)
    last_5_quarters = _build_report_periods(q_headers, q_rows, count=5)
    last_5_annual = _build_report_periods(pl_headers, pl_rows, count=5)

    # Compute 5Y Revenue & Net Profit CAGR
    rev_cagr_5y = None
    prof_cagr_5y = None
    if len(last_5_annual) >= 2:
        first_r = last_5_annual[0].revenue_cr
        last_r = last_5_annual[-1].revenue_cr
        yrs = max(1, len(last_5_annual) - 1)
        if first_r and last_r and first_r > 0 and last_r > 0:
            rev_cagr_5y = round(((last_r / first_r) ** (1.0 / yrs) - 1.0) * 100.0, 2)
        first_p = last_5_annual[0].net_profit_cr
        last_p = last_5_annual[-1].net_profit_cr
        if first_p and last_p and first_p > 0 and last_p > 0:
            prof_cagr_5y = round(((last_p / first_p) ** (1.0 / yrs) - 1.0) * 100.0, 2)

    # Balance Sheet, Cash Flow, and Schedules (Cash Equivalents & Borrowings)
    bs_sec = soup.find("section", id="balance-sheet")
    bs_headers, bs_rows = _parse_screener_table(bs_sec)
    cf_sec = soup.find("section", id="cash-flow")
    cf_headers, cf_rows = _parse_screener_table(cf_sec)
    ratios_sec = soup.find("section", id="ratios")
    rat_headers, rat_rows = _parse_screener_table(ratios_sec)

    other_assets_sched = _fetch_screener_schedule(
        s, company_id, "Other Assets", "balance-sheet", is_consolidated
    )
    borrowings_sched = _fetch_screener_schedule(
        s, company_id, "Borrowings", "balance-sheet", is_consolidated
    )
    fixed_assets_sched = _fetch_screener_schedule(
        s, company_id, "Fixed Assets", "balance-sheet", is_consolidated
    )
    expenses_sched = _fetch_screener_schedule(
        s, company_id, "Expenses", "profit-loss", is_consolidated
    )

    # Build 5-Year Cash & Balance Sheet History
    bs_years = bs_headers[-5:] if len(bs_headers) >= 5 else bs_headers
    cash_history_5y: list[CashBalanceYear] = []

    for yr in bs_years:
        col_idx = bs_headers.index(yr)

        def _bs_val(*names: str) -> float:
            for nm in names:
                for k, v_list in bs_rows.items():
                    if k.lower().startswith(nm.lower()) and col_idx < len(v_list):
                        return float(v_list[col_idx] or 0.0)
            return 0.0

        eq_cap = _bs_val("Equity Capital")
        reserves = _bs_val("Reserves")
        total_eq = round(eq_cap + reserves, 2)
        investments = _bs_val("Investments")
        borrowings_total = _bs_val("Borrowings", "Deposits")
        total_assets = _bs_val("Total Assets")

        # Cash Equivalents from Other Assets schedule (or bank cash row)
        cash_eq = 0.0
        if "Cash Equivalents" in other_assets_sched:
            cash_eq = _parse_num(other_assets_sched["Cash Equivalents"].get(yr)) or 0.0
        if cash_eq == 0.0:
            cash_eq = _bs_val("Cash & Equivalents", "Cash")

        lt_debt = _parse_num(borrowings_sched.get("Long term Borrowings", {}).get(yr)) or 0.0
        st_debt = _parse_num(borrowings_sched.get("Short term Borrowings", {}).get(yr)) or 0.0
        lease_debt = _parse_num(borrowings_sched.get("Lease Liabilities", {}).get(yr)) or 0.0

        ocf_val = None
        icf_val = None
        fcf_val = None
        if yr in cf_headers:
            cf_idx = cf_headers.index(yr)
            for k, v_list in cf_rows.items():
                if "operating" in k.lower() and cf_idx < len(v_list):
                    ocf_val = v_list[cf_idx]
                elif "investing" in k.lower() and cf_idx < len(v_list):
                    icf_val = v_list[cf_idx]
            if ocf_val is not None and icf_val is not None:
                # Use reported Investing Cash Flow directly when negative (net capex/investments) without arbitrary * 0.65 multiplier
                fcf_val = round(float(ocf_val) + min(0.0, float(icf_val)), 2)

        total_cash_hand = round(cash_eq + investments, 2)
        net_cash = round(total_cash_hand - borrowings_total, 2)

        cash_history_5y.append(
            CashBalanceYear(
                period=yr,
                cash_equivalents_cr=round(cash_eq, 2),
                investments_cr=round(investments, 2),
                total_cash_in_hand_cr=total_cash_hand,
                long_term_debt_cr=round(lt_debt, 2),
                short_term_debt_cr=round(st_debt, 2),
                lease_liabilities_cr=round(lease_debt, 2),
                total_borrowings_cr=round(borrowings_total, 2),
                net_cash_cr=net_cash,
                total_equity_cr=total_eq,
                total_assets_cr=round(total_assets, 2),
                operating_cash_flow_cr=ocf_val,
                investing_cash_flow_cr=icf_val,
                free_cash_flow_cr=fcf_val,
            )
        )

    mcap_cr = _parse_num(top_ratios.get("Market Cap")) or 0.0
    latest_cb = cash_history_5y[-1] if cash_history_5y else CashBalanceYear(period="Latest")
    d_to_e = (
        round(latest_cb.total_borrowings_cr / latest_cb.total_equity_cr, 2)
        if latest_cb.total_equity_cr > 0
        else 0.0
    )
    cash_to_debt = (
        round(latest_cb.total_cash_in_hand_cr / latest_cb.total_borrowings_cr, 2)
        if latest_cb.total_borrowings_cr > 0
        else 99.0
    )
    cash_pct_mcap = (
        round(latest_cb.total_cash_in_hand_cr / mcap_cr * 100.0, 2)
        if mcap_cr > 0
        else 0.0
    )

    roce_series: list[dict[str, Any]] = []
    if rat_headers and rat_rows:
        for k, v_list in rat_rows.items():
            if "roce" in k.lower():
                for h_lbl, v_num in zip(rat_headers[-5:], v_list[-5:]):
                    if v_num is not None:
                        roce_series.append({"period": h_lbl, "roce_pct": v_num})

    cum_ocf = round(sum(float(x.operating_cash_flow_cr or 0.0) for x in cash_history_5y), 2)
    cum_fcf = round(sum(float(x.free_cash_flow_cr or 0.0) for x in cash_history_5y), 2)

    cash_and_bs = CashAndBalanceSheet(
        latest_period=latest_cb.period,
        cash_equivalents_cr=latest_cb.cash_equivalents_cr,
        liquid_investments_cr=latest_cb.investments_cr,
        total_cash_in_hand_cr=latest_cb.total_cash_in_hand_cr,
        total_borrowings_cr=latest_cb.total_borrowings_cr,
        net_cash_position_cr=latest_cb.net_cash_cr,
        debt_to_equity=d_to_e,
        cash_to_debt_ratio=cash_to_debt,
        cash_pct_of_mcap=cash_pct_mcap,
        latest_ocf_cr=float(latest_cb.operating_cash_flow_cr or 0.0),
        latest_fcf_cr=float(latest_cb.free_cash_flow_cr or 0.0),
        cumulative_5y_ocf_cr=cum_ocf,
        cumulative_5y_fcf_cr=cum_fcf,
        roce_5y_series=roce_series,
        history_5y=cash_history_5y,
    )

    # Shareholding Pattern + Named Stockholders
    shp_sec = soup.find("section", id="shareholding")
    shp_headers, shp_rows = _parse_screener_table(shp_sec)
    q_shp_list: list[ShareholdingQuarter] = []
    for idx, q_lbl in enumerate(shp_headers[-8:]):
        real_idx = shp_headers.index(q_lbl)

        def _shp_v(*names: str) -> float:
            for nm in names:
                for k, v_list in shp_rows.items():
                    if k.lower().startswith(nm.lower()) and real_idx < len(v_list):
                        return float(v_list[real_idx] or 0.0)
            return 0.0

        num_sh = int(_shp_v("No. of Shareholders")) if _shp_v("No. of Shareholders") > 0 else None
        q_shp_list.append(
            ShareholdingQuarter(
                period=q_lbl,
                promoters_pct=_shp_v("Promoters"),
                fii_pct=_shp_v("FIIs", "FII"),
                dii_pct=_shp_v("DIIs", "DII"),
                government_pct=_shp_v("Government"),
                public_pct=_shp_v("Public"),
                num_shareholders=num_sh,
            )
        )

    latest_shp = q_shp_list[-1] if q_shp_list else ShareholdingQuarter(period="Latest")
    base_4q_shp = q_shp_list[max(0, len(q_shp_list) - 5)] if q_shp_list else latest_shp
    named_holders = _fetch_named_shareholders(s, company_id)

    stockholders_model = StockholderAnalysis(
        latest_period=latest_shp.period,
        promoters_pct=latest_shp.promoters_pct,
        fii_pct=latest_shp.fii_pct,
        dii_pct=latest_shp.dii_pct,
        government_pct=latest_shp.government_pct,
        public_pct=latest_shp.public_pct,
        institutional_total_pct=round(latest_shp.fii_pct + latest_shp.dii_pct, 2),
        promoter_change_4q_pct=round(latest_shp.promoters_pct - base_4q_shp.promoters_pct, 2),
        fii_change_4q_pct=round(latest_shp.fii_pct - base_4q_shp.fii_pct, 2),
        dii_change_4q_pct=round(latest_shp.dii_pct - base_4q_shp.dii_pct, 2),
        num_shareholders=latest_shp.num_shareholders,
        quarterly_history=q_shp_list,
        named_stockholders=named_holders,
    )

    # Price series & yfinance enrichment (also returns Yahoo R&D line if present)
    screener_prices = _fetch_screener_chart(s, company_id, is_consolidated)
    price_rows, yf_rd_by_year = _enrich_with_yfinance(sym, screener_prices)

    # Annual Reports PDF links from #documents
    docs_sec = soup.find("section", id="documents")
    annual_report_docs: list[AnnualReportDoc] = []
    if docs_sec:
        ar_div = docs_sec.find("div", class_="annual-reports")
        if ar_div:
            for a_tag in ar_div.find_all("a", href=True)[:5]:
                txt = a_tag.get_text(" ", strip=True)
                yr_m = re.search(r"20\d{2}", txt)
                annual_report_docs.append(
                    AnnualReportDoc(
                        title=re.sub(r"\s+", " ", txt)[:60],
                        year=yr_m.group(0) if yr_m else "Recent",
                        url=a_tag["href"],
                    )
                )

    # R&D & Innovation 5-Year History (No synthetic ramp multiplier!)
    rd_meta = REPORTED_RD_INTENSITY_PCT.get(sym)
    statutory_rd_pct: float | None = (
        float(rd_meta["pct_of_sales"]) if rd_meta else None
    )
    rd_focus = (
        str(rd_meta["focus"])
        if rd_meta
        else f"Technology absorption, process automation, and product R&D in {industry_str or sector_str or 'core operations'}"
    )
    rd_source = (
        str(rd_meta["source"])
        if rd_meta
        else (
            "Yahoo Finance IFRS Income Statement (Research & Development)"
            if yf_rd_by_year
            else "MISSING (R&D not separately disclosed in consolidated P&L)"
        )
    )

    rd_history_5y: list[RDYearRecord] = []
    intangible_map = fixed_assets_sched.get("Intangible Assets", {})
    emp_cost_map = expenses_sched.get("Employee Cost %", {})

    for rep in last_5_annual:
        yr_lbl = rep.period
        rev_val = float(rep.revenue_cr or 0.0)
        if yr_lbl in yf_rd_by_year and yf_rd_by_year[yr_lbl] > 0:
            rd_cr = yf_rd_by_year[yr_lbl]
            rd_pct = round(rd_cr / rev_val * 100.0, 2) if rev_val > 0 else (statutory_rd_pct or 0.0)
        elif statutory_rd_pct is not None:
            # Constant statutory BRSR / Board Report R&D intensity % without synthetic ramp
            rd_pct = round(statutory_rd_pct, 2)
            rd_cr = round(rev_val * rd_pct / 100.0, 2)
        else:
            rd_pct = 0.0
            rd_cr = 0.0

        intang_val = _parse_num(intangible_map.get(yr_lbl)) or 0.0
        cwip_val = 0.0
        if yr_lbl in bs_headers:
            b_idx = bs_headers.index(yr_lbl)
            for k, v_list in bs_rows.items():
                if "cwip" in k.lower() and b_idx < len(v_list):
                    cwip_val = float(v_list[b_idx] or 0.0)

        emp_pct = _parse_num(emp_cost_map.get(yr_lbl))
        rd_history_5y.append(
            RDYearRecord(
                period=yr_lbl,
                revenue_cr=rev_val,
                rd_spend_cr=rd_cr,
                rd_pct_of_sales=rd_pct,
                intangible_assets_cr=round(intang_val, 2),
                cwip_cr=round(cwip_val, 2),
                employee_cost_pct=emp_pct,
            )
        )

    latest_rd = rd_history_5y[-1] if rd_history_5y else RDYearRecord(period="Latest")
    cum_5y_rd = round(sum(r.rd_spend_cr for r in rd_history_5y), 2)
    rd_cagr = None
    if len(rd_history_5y) >= 2 and rd_history_5y[0].rd_spend_cr > 0 and latest_rd.rd_spend_cr > 0:
        yrs = max(1, len(rd_history_5y) - 1)
        rd_cagr = round(
            ((latest_rd.rd_spend_cr / rd_history_5y[0].rd_spend_cr) ** (1.0 / yrs) - 1.0) * 100.0,
            2,
        )

    rd_model = RDAndInnovation(
        latest_rd_spend_cr=latest_rd.rd_spend_cr,
        latest_rd_pct_of_sales=latest_rd.rd_pct_of_sales,
        cumulative_5y_rd_cr=cum_5y_rd,
        intangible_assets_cr=latest_rd.intangible_assets_cr,
        cwip_cr=latest_rd.cwip_cr,
        rd_cagr_5y_pct=rd_cagr,
        innovation_focus=rd_focus,
        rd_data_source=rd_source,
        history_5y=rd_history_5y,
        annual_reports_5y=annual_report_docs,
    )

    # Compute Piotroski 9-point simplified quality score from 5-year reports
    pio_score = 0
    if len(last_5_annual) >= 2:
        cur_a = last_5_annual[-1]
        prev_a = last_5_annual[-2]
        if (cur_a.net_profit_cr or 0) > 0:
            pio_score += 1
        if cash_and_bs.latest_ocf_cr > 0:
            pio_score += 1
        if cash_and_bs.latest_ocf_cr > (cur_a.net_profit_cr or 0):
            pio_score += 1
        if (cur_a.opm_pct or 0) >= (prev_a.opm_pct or 0):
            pio_score += 1
        if (cur_a.revenue_growth_pct or 0) > 0:
            pio_score += 1
        if (cur_a.net_profit_growth_pct or 0) > 0:
            pio_score += 1
        if cash_and_bs.debt_to_equity < 0.8:
            pio_score += 1
        if cash_and_bs.net_cash_position_cr >= 0 or cash_and_bs.cash_to_debt_ratio >= 0.4:
            pio_score += 1
        if (_parse_num(top_ratios.get("ROE")) or 0) >= 12.0:
            pio_score += 1

    overview = CompanyOverview(
        symbol=sym,
        name=company_name,
        exchange="NSE",
        nse_code=sym,
        sector=sector_str or "Indian Equities",
        industry=industry_str or "Listed Enterprise",
        about=about_text,
        market_cap_cr=mcap_cr,
        current_price=_parse_num(top_ratios.get("Current Price")) or 0.0,
        high_52w=high_52w,
        low_52w=low_52w,
        stock_pe=_parse_num(top_ratios.get("Stock P/E")),
        book_value=_parse_num(top_ratios.get("Book Value")),
        dividend_yield_pct=_parse_num(top_ratios.get("Dividend Yield")),
        roce_pct=_parse_num(top_ratios.get("ROCE")),
        roe_pct=_parse_num(top_ratios.get("ROE")),
        face_value=_parse_num(top_ratios.get("Face Value")),
        revenue_cagr_5y_pct=rev_cagr_5y,
        profit_cagr_5y_pct=prof_cagr_5y,
        piotroski_score=pio_score,
        pros=pros[:6],
        cons=cons[:6],
    )

    news_items = fetch_stock_news(sym, company_name)

    out_bundle = {
        "overview": overview.model_dump(),
        "last_5_annual_reports": [r.model_dump() for r in last_5_annual],
        "last_5_quarterly_reports": [r.model_dump() for r in last_5_quarters],
        "cash_and_balance_sheet": cash_and_bs.model_dump(),
        "stockholders": stockholders_model.model_dump(),
        "rd_and_innovation": rd_model.model_dump(),
        "price_rows": price_rows,
        "news": [n.model_dump() for n in news_items],
    }
    cache_set(cache_key, out_bundle, ttl_seconds=1800)
    return out_bundle
