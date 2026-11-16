# DATA_SOURCES.md — Indian Market Data Sources, Reliability & Licensing

## 1. Source Hierarchy & Role Assignment

| Data Domain | Primary Source | Supplementary / Fallback Source | Access Method | Rate Limits & Polite Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Daily Equity OHLC, VWAP, Turnover, Trades & Delivery %** | **NSE India Official Archives** (`nsearchives.nseindia.com/products/content/sec_bhavdata_full_{DDMMYYYY}.csv`) | **Yahoo Finance** (`yfinance` `.NS` / `.BO`) for multi-year historical OHLCV + corporate actions backfill | Direct HTTPS GET of official daily CSV report (no anti-bot bypass) | Cached in `/data/raw/`; downloaded once per trading date |
| **Broad Market, Sector Indices & Index Valuation (P/E, P/B, Div Yield)** | **NSE India Official Index Archive** (`nsearchives.nseindia.com/content/indices/ind_close_all_{DDMMYYYY}.csv`) | **Yahoo Finance** (`^NSEI`, `^CRSLDX`, `^INDIAVIX`, `^NSEBANK`, `^CNXIT`, etc.) for multi-year index history | Direct HTTPS GET of official daily CSV report | Cached in `/data/raw/`; downloaded once per trading date |
| **Bulk & Block Deals** | **NSE India Official Archives** (`nsearchives.nseindia.com/content/equities/bulk.csv` & `block.csv`) | None (marked empty if no deals reported) | Direct HTTPS GET | Polled at most once per post-market run |
| **5-Year Consolidated / Standalone Financials, Balance Sheet Schedules & Shareholding** | **Screener.in Public Company Pages & Schedule Endpoints** (`screener.in/company/{SYMBOL}/consolidated/`) | **Yahoo Finance** (`ticker.income_stmt`) for IFRS R&D line items where available | Polite sequential HTTP GET with `1.0s–2.0s` delay and SQLite/Parquet caching | Max 1 request/sec; automatic `2.0s` backoff on `HTTP 429` |
| **Company, Promoter & Sector News** | **Google News RSS India** (`news.google.com/rss/search?hl=en-IN&gl=IN&ceid=IN:en`) | Direct BSE/NSE links from Screener `#documents` | Standard RSS XML parser (`feedparser`) | Cached for 30–60 minutes per query |

---

## 2. Known Limitations & Provenance Rules

1. **Historical Delivery Percentage Availability**:
   - Official NSE `sec_bhavdata_full_{DDMMYYYY}.csv` provides verified `DELIV_QTY` and `DELIV_PER` for each NSE session downloaded.
   - Multi-year historical bars backfilled via `yfinance` do **not** contain NSE delivery figures. In strict compliance with Section 5A and Section 6, those historical rows store `deliverable_qty = NULL`, `delivery_pct = NULL`, and `delivery_validation_status = "MISSING"` (never `0.0`).
2. **Financial Statement Publication Lag (Point-in-Time Discipline)**:
   - Indian companies report quarterly results up to **45 days** after quarter-end and annual results up to **60 days** after fiscal year-end (`Mar 31` → `May 30`) under SEBI LODR Regulation 33.
   - `backend/storage/quality.py` assigns conservative statutory `publication_timestamp` values (`May 30`, `Aug 14`, `Nov 14`, `Feb 14`) when exact exchange filing timestamps are not attached to historical summary tables, preventing look-ahead leakage during feature generation.
3. **Terms of Service & Compliance**:
   - All collectors use standard, polite HTTP requests to publicly published reports and RSS feeds.
   - The system never bypasses login walls, CAPTCHAs, or exchange access controls. When an upstream endpoint is unreachable, the collector logs the status and tags affected fields as `MISSING` or `STALE`.
