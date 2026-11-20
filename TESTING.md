# Comprehensive Testing & Quality Assurance Framework

## 1. Testing Philosophy & Separation of Concerns

Testing in the AI Multi-Agent Intraday Platform is **strictly deterministic**. We do **not** use LLM calls to test LLMs or run test suites that consume billable tokens.

All unit, integration, invariant, and regression tests are implemented using standard `pytest`, `pytest-cov`, and Python assertion suites.

---

## 2. Testing Modules & Coverage Requirements

### A. Market Data & 5-Minute Candle Validation (`tests/intraday/test_data_validation.py`)
- **Missing Candle Detection**: Identifies gaps between 09:15 and 15:30 IST; ensures non-trading holidays (Diwali, Holi, Republic Day) are respected.
- **Timestamp Hygiene**: Confirms timestamps are timezone-aware (`Asia/Kolkata` / `+05:30`) and monotonic.
- **OHLC Invariance**: Asserts `high >= max(open, close)` and `low <= min(open, close)` and `volume >= 0`.
- **Duplicate Bar Rejection**: Checks and deduplicates identical bar timestamps.
- **NSE / BSE Symbol & ISIN Matching**: Confirms that cross-exchange tickers resolve to the identical ISIN (`INE...`).

### B. Technical Indicators & Feature Engine (`tests/intraday/test_technical_engine.py`)
- **VWAP Calculation**: Validates that VWAP matches standard institutional formula and resets daily at 09:15 IST.
- **EMA 9/21/50 Precision**: Asserts exponential smoothing factors and zero-lag calculations against known reference values.
- **RSI-14 & MACD**: Validates Wilder's smoothing for RSI and (12, 26, 9) exponential parameters for MACD.
- **Lookahead Bias Prevention**: Verifies that candle `t` features depend strictly on candles `<= t` and never peek into candle `t + 1`.

### C. Deterministic Risk Engine & Position Sizing (`tests/intraday/test_risk_engine.py`)
- **Fixed Fractional Risk**: Validates position sizing adheres to `(Portfolio_Equity * Risk_Pct) / (Entry - Stop)`.
- **Max Capital Cap**: Rejects or truncates orders exceeding `MAX_POSITION_PCT` (e.g. 15%).
- **Exposure Ceiling**: Blocks orders that would push total invested capital over `MAX_EXPOSURE_PCT` (e.g. 80%).
- **Daily Loss Circuit Breaker**: Confirms that once daily realized + unrealized loss breaches 2.5%, all new entry orders are strictly rejected (`HALT_TRADING_TODAY`).
- **15:15 IST Mandatory Square-Off**: Validates that any simulated position open at 15:15 IST generates an automated market exit order.

### D. Double-Entry Event Ledger & Accounting (`tests/intraday/test_event_ledger.py`)
- **Conservation of Capital Invariance**: At every event step:
  $$\text{Cash}_{\text{after}} + \text{MTM Value}_{\text{after}} = \text{Total Equity}_{\text{after}}$$
  $$\text{Total Equity}_{\text{after}} = \text{Starting Capital} + \text{Realized P\&L}_{\text{total}} + \text{Unrealized P\&L}_{\text{total}}$$
- **Cryptographic Provenance**: Verifies that every event has a valid `event_seq`, deterministic `event_id`, and unbroken SHA-256 hash chain (`prev_hash` -> `current_hash`).
- **State Replay Verification**: Replaying the ledger from inception reproduces the exact equity curve and position state without divergence.

### E. Cross-Exchange Arbitrage & Friction Gate (`tests/intraday/test_cross_exchange.py`)
- **Statutory Deduction Verification**: Tests that STT, GST, SEBI fee, NSE/BSE charges, and stamp duty match official formulas down to the paisa.
- **Negative & Zero Arbitrage Filtering**: Asserts that apparent LTP differences where `Gross_Spread <= Statutory_Friction + Slippage` are rejected with `ARBITRAGE_UNECONOMIC`.
- **Paired Order Simulation**: Validates execution simulation when Leg 1 fills and Leg 2 is partially filled or delayed.

### F. Model Routing & Token Budget Circuit Breakers (`tests/intraday/test_model_routing.py`)
- **Routing Escalation**: Validates deterministic routing (L0 -> L1 -> L2 -> L3).
- **Daily Budget Exhaustion**: Simulates reaching `$0.50` daily spend and verifies that subsequent calls gracefully degrade to deterministic Python mode without crashing.
- **Zero API Key Leakage**: Scans code, fixtures, and generated artifacts to verify no credentials or private keys are committed or logged.

---

## 3. Adversarial QA Suite (Development Agent G Role)

The independent QA role intentionally injects toxic edge cases to break the engine:
1. **Flash Crash Simulation**: Injects a sudden 10% candle drop within one 5m bar to verify that stop-loss execution models slippage through the stop price.
2. **Missing Opening Bar**: Simulates an exchange feed disconnect between 09:15 and 09:45 IST to confirm the agent does not trade on stale data.
3. **Cross-Exchange Desync**: Injects stale BSE prices with fresh NSE prices to confirm the arbitrage agent detects stale timestamps and rejects the trade.
4. **LLM Hallucination Injection**: Simulates a mocked LLM response that invents a fictitious support level; verifies that the contract validator catches unmatched numeric claims and flags the report.
