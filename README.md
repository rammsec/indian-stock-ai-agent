# BharatAlpha AI — Indian Market (NSE/BSE) Research & Next-Day Buy/Sell Agent

An institutional-grade Multi-Agent AI Research & Next-Day Trading Decision Terminal built specifically for the **Indian Stock Market (NSE & BSE)**.

Built by combining open-source patterns from:
- **[`virattt/ai-hedge-fund`](https://github.com/virattt/ai-hedge-fund)** — Multi-agent investment committee architecture where specialist fundamental, balance-sheet, moat, and technical agents debate and emit weighted conviction scores.
- **[`aswathas/pistol`](https://github.com/aswathas/pistol)** — Indian equity technical indicators (Wilder's RSI-14, MACD, Bollinger Bands, ATR-14, annualized log-return volatility), SQLite WAL caching, Yahoo Finance `.NS`/`.BO` adapters, and Google News RSS India (`hl=en-IN&gl=IN&ceid=IN:en`).
- **Screener.in Consolidated & Schedule APIs** — Live extraction of the **Last 5 Annual Reports**, **Last 5 Quarterly Reports**, **Cash Equivalents & Liquid Investments**, **Long/Short-Term Borrowings**, **Promoter / FII / DII / Public Shareholding**, **Named Institutional Stockholders** (`LIC`, `SBI Mutual Fund`, `Government of Singapore`, etc.), **R&D Intensity**, and **Official BSE/NSE Annual Report PDF links**.

---

## Core Research Pillars

1. **Last 5 Reports (Annual & Quarterly Performance Agent — 25% Weight)**
   - Analyzes the last 5 consolidated Annual Reports (FY22–FY26) and last 5 Quarterly Reports.
   - Evaluates Revenue & Net Profit YoY/QoQ growth, 5-Year Revenue & Profit CAGR, Operating Profit Margin (`OPM %`), EPS trajectory, ROE, ROCE, and Piotroski 9-Point F-Score.
2. **Cash in Hand & Balance Sheet Fortress Agent (20% Weight)**
   - Extracts `Cash Equivalents` (from Screener's `Other Assets` schedule) + `Liquid Investments` to compute **Total Cash in Hand (₹ Cr)** across 5 years.
   - Compares Cash in Hand against `Long-Term Borrowings`, `Short-Term Borrowings`, and `Lease Liabilities` to determine **Net Cash / (Net Debt)**, Debt-to-Equity, Cash % of Market Cap, and 5-Year Cumulative Operating Cash Flow (`OCF`) & Free Cash Flow (`FCF`).
3. **Stockholders & Institutional Flow Agent (20% Weight)**
   - Tracks quarterly ownership across **Promoters**, **FIIs (Foreign Institutional Investors)**, **DIIs (Domestic Institutional Investors)**, **Government**, and **Public Float**, plus total shareholder accounts.
   - Pulls **Named Stockholders** (>1% stake) and their 4-quarter accumulation/distribution deltas.
4. **R&D Spend & Innovation Moat Agent (15% Weight)**
   - Tracks **5-Year R&D Expenditure (₹ Cr)**, **R&D as % of Turnover**, **Intangible Assets**, **Capital Work-in-Progress (CWIP)**, and **Employee Knowledge Cost %**, alongside direct links to the company's **Last 5 Official Annual Report PDFs**.
5. **Next-Day Technical & Catalyst Trigger Agent (20% Weight)**
   - Computes **RSI(14)**, **MACD(12,26,9)**, **50-DMA & 200-DMA** trend regime, **Bollinger Bands**, **Volume Surge Ratio** (vs. 20D Avg), **ATR(14)** daily range, and **Floor Pivot Points** (`P`, `S1`, `S2`, `R1`, `R2`).
   - Synthesizes an actionable **Next-Day Trading Ticket**:
     - **Signal**: `STRONG BUY`, `BUY`, `HOLD / ACCUMULATE`, `SELL / BOOK PROFIT`, or `AVOID / SHORT`
     - **Next-Day Entry Zone (₹)**
     - **Next-Day Target 1 (T1 ₹)** & **Swing Target 2 (T2 ₹)**
     - **Strict ATR/Pivot Stop-Loss (SL ₹)** & **Risk : Reward Ratio**
6. **Pluggable LLM Chief Investment Officer (CIO)**
   - Works out-of-the-box with the deterministic **Quantitative Multi-Agent Engine**, or plug in **Google Gemini**, **OpenAI**, **Anthropic Claude**, or **Local Ollama** via the UI settings modal.

---

## Quick Start

```bash
# 1. Create virtual environment & install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Run the FastAPI server
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. Run the test suite
.venv/bin/pytest -v
```
