# COST_CONTROL.md — Google-First & Zero-Billing Architecture Policy

## 1. Core Cost Principle

BharatAlpha AI is engineered to operate at **$0.00 / ₹0.00 mandatory recurring monthly cost** while taking full advantage of your existing **Google AI Pro subscription + Google Antigravity** environment.

- **No paid APIs, cloud servers, or managed databases are enabled by default.**
- **No automatic AI credit overages or paid cloud resource provisioning exist in the codebase.**

---

## 2. Inventory of External Services & Billing Triggers

| Service / Component | Purpose | Default State | Cost Model | Possible Billing Trigger | Hard Code Safeguard |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Antigravity (IDE & Agentic Coding)** | Development, quant architecture, code generation, and debugging | **Active** | Included in your existing Google AI Pro subscription | None | Used for development and code reasoning only |
| **Python / Pandas / DuckDB / PyArrow (Parquet)** | Historical storage, analytical SQL queries, feature & risk math | **Enabled** | 100% Free & Open-Source | None | Executes locally or inside free GitHub Actions runners |
| **GitHub Pages & GitHub Actions** | Static web hosting (`docs/`) & scheduled post-market NSE pipelines | **Enabled** | **$0.00** (Free & unlimited for public GitHub repositories) | Private repo exceeding 2,000 CI minutes/month | Repository is public (`rammsec/indian-stock-ai-agent`); workflow skips weekends & NSE holidays |
| **NSE India Official Archives (`nsearchives.nseindia.com`)** | Daily Bhavcopy (`sec_bhavdata_full`), Delivery %, Index reports (`ind_close_all`), Bulk/Block deals | **Enabled** | **$0.00** (Public exchange reports) | None (IP rate-limit only if abused) | Raw CSVs cached in `/data/raw/`; fetched at most once per trading day |
| **Screener.in Public Web & Schedules** | Last 5 Annual & Quarterly reports, Balance Sheet, Cash Flow, Shareholding | **Enabled** | **$0.00** (Public pages) | None (HTTP 429 rate-limit if hit concurrently) | Sequential fetching with SQLite/Parquet caching & 2.0s backoff |
| **Yahoo Finance (`yfinance`)** | Supplementary historical OHLCV, corporate actions (splits/dividends) & index history | **Enabled** | **$0.00** | None | Batch ticker download + incremental Parquet append |
| **Google News RSS (`news.google.com/rss`)** | Company, promoter, and competitor news headlines | **Enabled** | **$0.00** | None | Cached with TTL & document fingerprinting |
| **Gemini API (`generativelanguage.googleapis.com`)** | Optional structured extraction from concall transcripts & corporate filings (Phase 5) | **DISABLED BY DEFAULT** (`ENABLE_GEMINI_EXTRACTOR=false`) | Free tier available in Google AI Studio; pay-as-you-go only if linked to a billed GCP project | Making uncached LLM calls on every stock or every run with a billed API key | 1. `ENABLE_GEMINI_EXTRACTOR=false` by default in `backend/config.py`<br/>2. SHA-256 document fingerprinting (never re-processes an already-extracted document)<br/>3. Hard daily circuit breaker `MAX_GEMINI_CALLS_PER_DAY = 10` |
| **Google BigQuery** | Optional cloud analytical warehouse (Stage 2) | **DISABLED BY DEFAULT** (`ENABLE_BIGQUERY=false`) | BigQuery Sandbox / Free Tier: 10 GB storage + 1 TB queries/month free | Queries exceeding 1 TB/month on a billed GCP account | 1. `ENABLE_BIGQUERY=false` by default in `backend/config.py`<br/>2. Hard per-query cap `BIGQUERY_MAX_BYTES_BILLED = 100_000_000` (100 MB) |
| **Google Drive / Google Sheets** | Optional manual export of research PDFs or summary review tables | **DISABLED BY DEFAULT** | Free within Google account storage quota | Exceeding personal Google Drive storage quota | Never used as primary historical database; Parquet + DuckDB is primary |

---

## 3. Configuration Flags (`backend/config.py`)

All cost-sensitive integrations are governed by explicit environment flags in [`backend/config.py`](file:///home/antigravity/indian-stock-ai-agent/backend/config.py):

```python
ENABLE_GEMINI_EXTRACTOR: bool = os.environ.get("ENABLE_GEMINI_EXTRACTOR", "false").lower() == "true"
MAX_GEMINI_CALLS_PER_DAY: int = int(os.environ.get("MAX_GEMINI_CALLS_PER_DAY", "10"))
ENABLE_BIGQUERY: bool = os.environ.get("ENABLE_BIGQUERY", "false").lower() == "true"
BIGQUERY_MAX_BYTES_BILLED: int = int(os.environ.get("BIGQUERY_MAX_BYTES_BILLED", "100000000"))
```

When run with default settings (no environment variables set), **external cloud/AI API expenditure is guaranteed to be $0.00**.
