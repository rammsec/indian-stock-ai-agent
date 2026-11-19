# BharatAlpha Phase 6 — Specialist Agent Specifications

## 1. Specialist Analysts

### A. Fundamental Analyst (`FundamentalAnalystAgent`)
- **Objective:** Evaluate the financial quality, operating performance, and solvency of the enterprise.
- **Input Slots:** `revenue_cr`, `profit_cr`, `revenue_growth_pct`, `profit_growth_pct`, `opm_pct`, `net_margin_pct`, `operating_cash_flow_cr`, `free_cash_flow_cr`, `debt_to_equity`, `current_ratio`, `interest_coverage`, `pe`, `pb`, `roe_pct`, `roce_pct`, `piotroski_score`.
- **Deterministic Derivations:** Revenue/profit growth trajectory, OPM/NPM expansion or contraction, cash flow conversion quality, leverage categorization (<0.5 low, >1.5 high), liquidity adequacy, DuPont return evaluation.
- **Failure / Missing Handling:** Missing metrics are reported in `missing_information`. If the entire fundamental section is absent, confidence is capped at 0.10 and the LLM call is bypassed.

### B. Technical Analyst (`TechnicalAnalystAgent`)
- **Objective:** Assess momentum, trend direction, moving average structure, and support/resistance zones.
- **Input Slots:** `close`, `rsi_14`, `macd_hist`, `sma_20`, `sma_50`, `sma_200`, `atr_pct`, `return_1m_pct`, `return_3m_pct`, `volume_surge_ratio`, `support`, `resistance`, `adx`, `trend`, plus market regime context (`regime`, `india_vix_close`).
- **Deterministic Derivations:** MA alignment (bullish: Close > SMA50 > SMA200; bearish: Close < SMA50 < SMA200), RSI zone (<30 oversold, >70 overbought), MACD momentum polarity, distance to 52-week support/resistance, ADX trend strength (>25 trending).

### C. News Intelligence Analyst (`NewsAnalystAgent`)
- **Objective:** Analyze Point-in-Time structured news events across six official categories without sentiment confusion or duplicate contamination.
- **Input Slots:** Filtered `events` list from `data/predictions/structured_events.parquet` where `publication_timestamp <= cutoff`, and `quarantined_count`.
- **Deduplication & Hygiene:** Events marked `is_duplicate_report=True` are counted but excluded from substantive impact tallies. Generic price-page headlines are filtered.
- **Competitor Polarity Guard:** Headwinds or contract wins by named competitors are explicitly flagged as negative or neutral for the subject company, preventing sentiment inversion.

### D. Promoter & Governance Analyst (`PromoterAnalystAgent`)
- **Objective:** Scrutinize ownership stability, promoter alignment, pledging risk, institutional ownership changes, and governance signals.
- **Input Slots:** `promoters_pct`, `promoter_change_qoq_pct`, `pledged_pct`, `fii_pct`, `dii_pct`, `insider_transactions`, `management_commentary`, `leadership_changes`, `governance_flags`.
- **Safety Gate:** When `insider_transactions` is `None` or unavailable, the agent explicitly records "unavailable" and is strictly prohibited from inferring insider activity.

### E. Competitor & Industry Analyst (`CompetitorAnalystAgent`)
- **Objective:** Evaluate market share, peer valuation comparisons, cross-stock spillover effects, and supply-chain linkages.
- **Input Slots:** `competitors` list from relationship mapping, `peer_metrics` comparison table, `cross_stock_impacts` (asymmetric spillover events), and `industry_growth_pct`.

### F. Quantitative Analyst (`QuantitativeAnalystAgent`)
- **Objective:** Provide an honest, unvarnished interpretation of BharatAlpha's existing machine learning predictions.
- **Purely Deterministic:** Uses NO LLM. Never alters model weights, thresholds, features, or retraining logic.
- **Evidence Surfaces:** 1D, 5D, 10D directional probabilities (`p_up`), expected returns, ensemble agreement score, and forward validation status (`matured_count`, `pending_count`).
- **Honesty Guard:** Explicitly flags `INSUFFICIENT_DATA` warnings when matured sample thresholds (<10) are not met.

---

## 2. Debate & Synthesis Layer

### Bull Researcher (`BullResearcherAgent`)
- **Role:** Construct the strongest evidence-backed upside investment thesis.
- **Mandate:** Identify operational and structural catalysts, challenge pessimistic assumptions, and highlight positive asymmetry.
- **Constraint:** Must cite evidence IDs for all claims. Cannot view Bear output.

### Bear Researcher (`BearResearcherAgent`)
- **Role:** Construct the strongest evidence-backed downside investment thesis.
- **Mandate:** Identify valuation risks, cyclical headwinds, execution bottlenecks, and challenge management optimism.
- **Constraint:** Must cite evidence IDs for all claims. Cannot view Bull output.

### Research Manager (`ResearchManagerAgent`)
- **Role:** Impartial consensus builder and senior research reviewer.
- **Synthesis:** Consolidates Bull and Bear arguments, highlights explicit disagreements, extracts key risk factors, and synthesizes overall conclusions (`FAVOURABLE`, `NEUTRAL`, `UNFAVOURABLE`, `INSUFFICIENT_EVIDENCE`).
- **Guardrails:** Automatically downgrades conclusion if data completeness is below 80% or if quantitative validation is insufficient.

---

## 3. Risk Management Layer

### Risk Manager (`RiskManagerAgent`)
- **Role:** Independent risk controller with absolute veto authority.
- **7 Evaluated Dimensions:**
  1. Downside Risk (Historical 95% 1D VaR)
  2. Volatility Risk (Annualized 20D volatility)
  3. Liquidity Risk (Average daily turnover in INR crore)
  4. Concentration Risk (Portfolio position weight)
  5. Event Risk (Imminent corporate or regulatory events)
  6. Drawdown Risk (Trailing 1-year max drawdown)
  7. Risk/Reward Ratio (Expected upside vs. downside)
- **Veto Rules:**
  - `REJECT`: Triggered if >=2 dimensions are HIGH, or Liquidity is HIGH (<5 Cr turnover), or >=3 dimensions are UNKNOWN. Overrides Research Manager conclusion to `REJECTED_BY_RISK`.
  - `REDUCE`: Triggered if 1 dimension is HIGH or >=2 dimensions are UNKNOWN.
  - `APPROVE`: All dimensions within acceptable parameters.
