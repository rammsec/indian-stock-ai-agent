# Agent Design & Responsibilities Specification

This document details the roles, inputs, deterministic logic, LLM allocation, and output schemas for each specialized agent in the Intraday Trading Platform.

---

## Agent Directory & Overview

| # | Agent Name | Tier / Layer | Execution Mode | Model Allocation | Primary Responsibility |
|---|---|---|---|---|---|
| **1** | Market Intelligence Agent | Layer A (Council) | Scheduled / Event | Gemini Flash (routine) / Pro (regime) | Broader market trends, Nifty 50, Bank Nifty, India VIX, breadth |
| **2** | Technical Analysis Agent | Layer B (Trading) | Every 5m candle | Deterministic Python + Flash (ambiguous) | 5m candles, EMA 9/21/50, VWAP, RSI, MACD, S/R, volume spikes |
| **3** | News & Sentiment Agent | Layer A (Council) | Event-driven | Gemini Flash (filtering) / Pro (impact) | Corporate filings, BSE/NSE releases, macro news, sentiment |
| **4** | Market Microstructure Agent | Layer A & B | Every 5m candle | Deterministic Python + Flash (anomalies) | Spreads, order-book depth, price gaps, execution feasibility |
| **5** | Strategy Research Agent | Library / Research | Offline / Async | Gemini Pro (research) + Flash (parsing) | Catalog and synthesize Pine Script & quant strategies |
| **6** | Risk Management Agent | Governance | Every order | Deterministic Python (Strict Hard Veto) | Capital sizing, daily loss caps, exposure limits, cost deduction |
| **7** | Central Decision & Debate Council | Layer A (Council) | Scheduled / Event | Gemini Pro (synthesis) / Reasoning (deadlock)| Multi-agent debate, consensus, 4-tier posture output |
| **8** | Autonomous Trading Agent | Layer B (Trading) | Every 5m candle | Deterministic Python + Flash (setups) | Stock scanning, entry/exit decisions, bracket order management |
| **9** | Backtesting & Validation Agent | Validation | On-demand / CI | Deterministic Python + Flash (reporting) | Historical replay, walk-forward validation, overfitting checks |
| **10**| Learning & Performance Agent | Feedback Loop | Post-session | Gemini Pro (analysis) / Flash (summary) | Post-trade attribution, decision quality scoring, improvement proposals |
| **11**| Cross-Exchange Arbitrage Agent | Layer B (Trading) | Every 5m candle | Deterministic Python + Flash (liquidity) | NSE-BSE ISIN matching, executable net spread, paired order simulation |

---

## Detailed Agent Specifications

### Agent 1: Market Intelligence Agent
- **Inputs**: Real-time & historical 5m/daily data for Nifty 50, Nifty Bank, Nifty Financial Services, sectoral indices (IT, Auto, Metal, Pharma), India VIX, market breadth (Advances / Declines).
- **Deterministic Pipeline**:
  - Breadth ratio: `advance_decline_ratio = count(advances) / count(declines)`
  - VIX regime classification: `< 13` (Low Volatility / Complacent), `13 - 18` (Normal), `18 - 24` (Elevated), `> 24` (High Volatility / Defensive).
  - Trend assessment: 20-period vs 50-period EMA slope on Nifty 50 5m and 15m charts.
- **LLM Synthesis**: Translates macro data into structured market intelligence report.
- **Output Schema**: `MarketIntelligenceReport` (direction: `BULLISH | BEARISH | NEUTRAL`, confidence: `0.0 - 1.0`, leading/lagging sectors, risk appetite).

### Agent 2: Technical Analysis Agent
- **Inputs**: Completed 5-minute OHLCV candles for watchlist symbols.
- **Deterministic Pipeline**:
  - Exponential Moving Averages: EMA 9, EMA 21, EMA 50.
  - Volume Weighted Average Price (VWAP): Intraday cumulative `sum(price * volume) / sum(volume)` resetting at 09:15 IST daily.
  - Momentum: Wilder's RSI-14, MACD (12, 26, 9), Average True Range (ATR-14).
  - Volume Anomaly: Ratio of current bar volume to 20-bar rolling average volume (`volume / sma_volume_20`).
  - Candlestick Formations: Hammer, Bullish/Bearish Engulfing, Morning/Evening Star, Pin Bars.
  - Pivots: Classic Floor Pivots & CPR (Central Pivot Range) computed from prior session.
- **Output Schema**: `TechnicalSignal` (stance: `BULLISH | BEARISH | NEUTRAL`, indicators dictionary, key levels, setup type).

### Agent 3: News and Sentiment Agent
- **Inputs**: Exchange announcements (NSE/BSE filings), Corporate Action calendars, Google News India RSS, Earnings calendar.
- **Deterministic Pipeline**:
  - Point-in-Time Cutoff Enforcement: `published_at <= data_cutoff_timestamp <= analysis_timestamp`.
  - Entity matching: Strict ticker and ISIN resolution.
  - Deduplication: Semantic hashing of headlines to eliminate identical wire reports.
- **LLM Synthesis**: Classifies sentiment impact (`BULLISH | BEARISH | NEUTRAL`), expected duration, and financial magnitude.
- **Output Schema**: `NewsIntelligenceReport` (sentiment score, materiality score `0.0 - 1.0`, catalyst summary).

### Agent 4: Market Microstructure Agent
- **Inputs**: Bid/Ask quotes, top 5 order-book levels (where available), traded volume distribution, price tick anomalies.
- **Deterministic Pipeline**:
  - Effective Spread: `(Ask - Bid) / MidPrice`.
  - Amihud Illiquidity Metric: `|Return| / (Price * Volume)`.
  - Gap Analysis: Deviation of opening candle from previous bar close.
  - Execution Feasibility Score: Probability that limit/market orders fill without adverse price impact.
- **Output Schema**: `MicrostructureReport` (liquidity status: `HIGH | MODERATE | ILLIQUID`, estimated slippage in bps, execution feasibility: `PASS | CAUTION | REJECT`).

### Agent 5: Strategy Research Agent
- **Inputs**: Public Pine Script strategies, open-source quant strategies, academic papers.
- **Function**:
  - Catalogs strategies with full provenance metadata: name, origin URL, license, entry rules, exit rules, timeframe, indicator dependencies.
  - Converts human-readable logic into executable Python strategy modules.
  - Submits strategies to the Backtesting Agent for standardized walk-forward validation.
- **Output Schema**: `StrategyResearchEntry` stored in `STRATEGY_LIBRARY.md` and DuckDB catalog.

### Agent 6: Risk Management Agent (Deterministic Gatekeeper)
- **Inputs**: Candidate orders from Autonomous Trading Agent, current portfolio cash, open positions, unrealized P&L, daily drawdown.
- **Strict Hardcoded Rules (Zero LLM Override)**:
  - Position sizing via Fixed Fractional Risk: `Shares = (Portfolio_Equity * Risk_Pct) / (Entry_Price - Stop_Loss_Price)`.
  - Cap position size at `MAX_POSITION_PCT` (default: 15% of total capital).
  - Cap total portfolio exposure at `MAX_EXPOSURE_PCT` (default: 80%).
  - Maximum daily loss limit: If cumulative daily loss >= 2.5%, lock terminal into `HALT_TRADING_TODAY`.
  - Intraday time guard: No new entries after 14:45 IST; force-close all open positions at 15:15 IST.
- **Output Schema**: `RiskAssessment` (`decision: APPROVED | SIZED_DOWN | REJECTED`, rejected_reasons, approved_quantity, required_margin).

### Agent 7: Central Decision and Debate Council
- **Inputs**: Reports from Agents 1, 3, 4, 5, and historical predictive accuracy.
- **Debate Mechanism**:
  - Multi-agent debate between independent Bull and Bear stances.
  - Cross-examination: Verifies whether bullish claims are supported by technicals and volume, or contradicted by macro/news risks.
  - Eliminates unevidenced or hallucinated claims via numeric citation verification.
- **Output Schema**: `CouncilPostureReport` (posture: `BUY_BULLISH | SELL_BEARISH | NEUTRAL | HOLD_HIGH_RISK`, confidence score, rationale, sector focus, reassessment triggers).

### Agent 8: Autonomous Intraday Trading Agent
- **Inputs**: Layer A Council Posture, Layer B Technical signals, Microstructure assessment, Risk boundaries.
- **Workflow on Every 5-Minute Bar**:
  1. Check if trading session is active (09:15 - 15:15 IST).
  2. Filter approved stock universe against Council sector bias and market posture.
  3. Evaluate candidate setups against active Strategy Library rules.
  4. Submit order proposal with stop-loss and profit target to Agent 6 (Risk Manager).
  5. If approved, record order in the double-entry event ledger.
  6. Monitor open positions: adjust trailing stops, check stop-loss/target hit, execute 15:15 IST square-off.
- **Output Schema**: `TradeDecision` (`action: ENTER_LONG | EXIT | TRAIL_STOP | HOLD | WAIT`, order parameters, justification).

### Agent 11: Cross-Exchange Arbitrage Agent
- **Inputs**: Synchronized 5m/tick quotes for dual-listed stocks on NSE and BSE matched by ISIN.
- **Calculation Engine**:
  - Gross spread: `NSE_Executable_Bid - BSE_Executable_Ask` (or vice-versa).
  - Indian statutory friction deduction:
    - Brokerage (MIS intraday: 0.03% or ₹20/order)
    - STT (0.025% on sell side)
    - Exchange transaction charges (NSE: 0.00297%, BSE: 0.00375%)
    - GST (18% on brokerage + exchange charges + SEBI fee)
    - Stamp duty (0.003% on buy side)
    - SEBI turnover fee (₹10 / crore)
    - Modeled slippage (minimum 5 bps per leg)
  - Net executable spread: `Gross_Spread - Statutory_Costs - Slippage`.
  - Condition: `Net_Spread > Safety_Margin_Buffer` (e.g. 0.15%).
- **Execution Modeling**:
  - Simulates paired order execution.
  - Implements contingency logic if Leg 1 fills and Leg 2 is rejected or experiences adverse price slippage (immediate market square-off or trailing hedge).
- **Output Schema**: `ArbitrageOpportunity` (isin, buy_exchange, sell_exchange, gross_spread_pct, net_spread_pct, actionable_status, unhedged_risk_assessment).
