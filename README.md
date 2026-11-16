# BharatAlpha AI v3.0 — Indian Market (NSE/BSE) Deep Research, Promoter/Competitor Intel & Next-Day Buy/Sell Terminal

An institutional-grade Multi-Agent AI Research & Next-Day Trading Decision Terminal built specifically for the **Indian Stock Market (NSE & BSE)** and hosted 100% on **GitHub Pages + GitHub Actions**.

- **Live Web App (GitHub Pages)**: https://rammsec.github.io/indian-stock-ai-agent/
- **GitHub Repository**: https://github.com/rammsec/indian-stock-ai-agent

Built by combining open-source patterns from:
- **[`virattt/ai-hedge-fund`](https://github.com/virattt/ai-hedge-fund)** — Multi-agent investment committee architecture where specialist fundamental, balance-sheet, moat, and technical agents debate and emit weighted conviction scores.
- **[`aswathas/pistol`](https://github.com/aswathas/pistol)** — Indian equity technical indicators (Wilder's RSI-14, MACD, Bollinger Bands, ATR-14, annualized log-return volatility), Yahoo Finance `.NS`/`.BO` adapters, and Google News RSS India (`hl=en-IN&gl=IN&ceid=IN:en`).
- **Screener.in Consolidated & Schedule APIs** — Live extraction of the **Last 5 Annual Reports**, **Last 5 Quarterly Reports**, **Cash Equivalents & Liquid Investments**, **Long/Short-Term Borrowings**, **Promoter / FII / DII / Public Shareholding**, **Named Institutional Stockholders** (`LIC`, `SBI Mutual Fund`, `Government of Singapore`, etc.), **R&D Intensity**, and **Official BSE/NSE Annual Report PDF links**.

---

## Key Features (v3.0)

1. **Focused 5-Stock Default Watchlist + Direct "+ Add Stock" on Page (Up to 10 Stocks)**
   - Starts with a clean, high-depth watchlist of 5 Indian stocks:
     1. **Reliance Industries (`RELIANCE`)**
     2. **Tata Consultancy Services (`TCS`)**
     3. **Tata Elxsi (`TATAELXSI`)**
     4. **ITC Ltd (`ITC`)**
     5. **Birla Cable (`BIRLACABLE`)**
   - Top company selector bar lets you switch companies in one click, add up to **10 NSE/BSE stocks** directly from the page (`+ Add Stock`), remove any stock (`×`), or reset to the default 5.
2. **Collapsible Section Headings (`Expand / Collapse`) & High-Readability UI**
   - Every section header (`01` through `07`) can be expanded or collapsed individually, or toggled globally via `↕ Collapse All / Expand All Sections`.
3. **7 Deep Research Sections per Stock**
   - **01. 5-Agent Investment Committee Debate** — Weighted multi-agent conviction scoring (`25%` 5-Report Performance, `20%` Cash & Balance Sheet, `20%` Stockholders, `15%` R&D Moat, `20%` Technical Trigger).
   - **02. Last 5 Reports (Annual FY21–FY25/26 & Quarterly)** — Interactive charts + full line-by-line P&L tables.
   - **03. Cash in Hand, Liquid Investments & Debt Fortress** — 5-year Cash Equivalents + Liquid Investments vs. Borrowings, Net Cash, and Operating Cash Flow.
   - **04. Stockholders, Promoters, FII/DII & Named Institutions** — Quarterly ownership trend + named >1% holders (`LIC`, `SBI MF`, etc.).
   - **05. R&D Spend, Innovation Moat & Official Annual Report PDFs** — 5-year R&D spend (`₹ Cr` & `% of sales`) + direct BSE/NSE Annual Report PDF links.
   - **06. Technical Chart (50/200-DMA) & Market Headlines** — 1-year price/DMA chart + live Indian news sentiment.
   - **07. Promoters, Stakeholders, Recent Statements/Articles & Competitor Intelligence (Bottom Deep-Dive Section)** — Tracks promoter groups & key executives, scrapes recent promoter/CEO/stakeholder statements and articles with **Share Price Impact (`BULLISH` / `BEARISH` / `NEUTRAL`)**, and benchmarks **4 direct NSE competitors** with live price moves, latest competitor statements, and read-across impact on share value.
4. **📈 Yesterday's Top 20 Gainers (NSE) Page**
   - Scans liquid NSE leaders from the latest trading session, ranks the **Top 20 Gainers**, and provides **2 concrete fundamental/technical reasons why each stock went up** plus a Next-Day Verdict (`CONTINUATION BUY` vs. `BOOK PROFITS`).
5. **📉 Yesterday's Top 20 Losers (NSE) Page**
   - Ranks the **Top 20 Losers** from the latest session and provides **2 concrete reasons why each stock went down** plus a Next-Day Verdict (`BUY THE DIP` vs. `WAIT FOR SUPPORT`).

---

## Quick Start (Local Optional — Runs 100% on GitHub Pages by Default)

```bash
# 1. Create virtual environment & install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Refresh static GitHub Pages data & bundle
.venv/bin/python scripts/generate_static_data.py
.venv/bin/python scripts/build_github_pages.py

# 3. Run the FastAPI server locally (optional)
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Run the test suite
.venv/bin/pytest -v
```
