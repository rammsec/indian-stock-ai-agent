# NSE 5-Minute Historical Market Data Provider Evaluation

> **Policy**: No paid subscriptions or broker APIs are activated without explicit user approval. Missing intervals are never fabricated or forward-filled.

---

## 1. Provider Comparison Matrix

| Provider | Historical Coverage (5m) | Supported Instruments | Missing-Candle Frequency | Adjustment Policy | Rate Limits | API Cost | Licensing & Restrictions | Reliability |
|---|---|---|---|---|---|---|---|---|
| **Yahoo Finance (`yfinance`)** | ~60 trading days rolling (`2026-08-25` to `2026-11-20`) | All 5 NSE symbols (`.NS`) & BSE (`.BO`) | 1–2 bars/day near `15:15–15:20 IST` on `RELIANCE`, `TCS`, `ITC`, `TATAELXSI` (`73–74/75` bars); `BIRLACABLE` `75/75` | Split & dividend adjusted | Unofficial (~2,000 req/hr) | **Free ($0)** | Non-commercial / personal research only; no SLA | Moderate (occasional closing-window omissions) |
| **NSE Authorised Vendors (DotEx / NSE Data)** | Multi-year (5–10+ years) | All NSE equities & indices | 0% (official exchange feed) | Unadjusted + separate corporate action feed | Contractual SLA | **Paid** (₹15,000–₹1,00,000+/mo) | Strict redistribution & commercial licensing | Institutional Grade |
| **Indian Broker APIs (Kite Connect / Upstox / SmartAPI)** | 60 days to 5 years (5m) | All NSE/BSE equities | < 0.1% | Unadjusted (splits/bonuses via master) | 3–10 req/sec | **Paid** (₹2,000/mo Kite) or broker account | Requires active Demat/Trading KYC + API keys | High |
| **NSE Official EOD Bhavcopy / Public Datasets** | 20+ years (Daily EOD only; **no 5m**) | All NSE equities | N/A (No 5m intraday) | Unadjusted | Public HTTP | **Free** | Official exchange public data | High for EOD; N/A for 5m |

---

## 2. Empirical Findings on Current 59-Day Dataset (`yfinance`)

An automated audit across all **59 cached trading days** (`2026-08-25` to `2026-11-20`) for the 5-stock universe (`RELIANCE`, `TCS`, `BIRLACABLE`, `ITC`, `TATAELXSI`) found:
- **Fully Ready Sessions (`75/75` bars across all 5 stocks)**: **0 / 59**
- **Partial Sessions (`>= 70/75` bars, complete `09:15–15:10 IST`, missing `15:15–15:20 IST` on 4 large-caps)**: **58 / 59** (`2026-11-04` has `63/75` on `BIRLACABLE`).
- **Root Cause**: Yahoo Finance consistently omits 1–2 bars in the `15:15–15:20 IST` window on `RELIANCE.NS`, `TCS.NS`, `ITC.NS`, and `TATAELXSI.NS`, while `BIRLACABLE.NS` has `75/75` bars.
- **Research Mitigation (No Fabrication)**: Sessions are strictly labelled `PARTIAL` in the Data Quality dashboard, missing intervals are preserved, and a configurable research completeness threshold (`min_bars_for_research = 70`) allows multi-week (`30+` sessions) evaluation across the `09:15–15:10 IST` active trading window while disclosing exact data quality.
