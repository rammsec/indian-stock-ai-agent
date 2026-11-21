# NSE & BSE Cross-Exchange Arbitrage: Exchange, Broker & Settlement Restrictions

> **IMPORTANT**: Live broker execution is **STRICTLY DISABLED**. All arbitrage modules (`trader/arbitrage/`) operate exclusively in historical/paper simulation mode.

---

## 1. Settlement & Delivery Mechanics (T+1)
- Indian equities settle on a **T+1 rolling settlement cycle** across both NSE and BSE.
- **Same-Day Intraday Square-Off**: Buying on BSE and selling on NSE (or vice-versa) within the same intraday session creates a **cross-exchange open obligation** unless both positions are squared off on their respective exchanges before 15:15 IST or settled via **Demat delivery** (requiring pre-existing DP holdings).
- **Short-Selling Restriction**: Retail clients cannot carry naked intraday short positions overnight; if Leg 1 (Buy) fills on BSE and Leg 2 (Sell) fails on NSE without existing holdings in Demat, an **unhedged exposure** occurs. `PairedArbitrageExecutor` mitigates this by triggering an `EMERGENCY_MARKET_SQUAREOFF` on Leg 1's exchange.

## 2. Statutory & Exchange Fee Asymmetries
- **NSE Transaction Charge**: `0.00297%` per leg.
- **BSE Transaction Charge**: `0.00375%` per leg (higher than NSE).
- **STT (Intraday)**: `0.025%` on the Sell leg only.
- **Stamp Duty**: `0.003%` on the Buy leg only.
- **Brokerage & GST**: `0.03%` (capped at ₹20/order) + `18% GST` on brokerage, exchange fees, and SEBI turnover fee.
- **Minimum Viable Spread**: Because total two-leg statutory friction + 5 bps slippage per leg consumes ~`0.08%–0.14%` of turnover, any NSE–BSE spread below `0.15%` net return is automatically rejected as `UNECONOMIC_SPREAD`.

## 3. Quote Freshness, Order-Book Depth & ISIN Matching
- **ISIN Validation**: Symbols are matched strictly by official 12-character ISIN (`trader/data/instruments.py`) to prevent ticker collisions.
- **Quote Freshness**: Quotes older than `15.0 seconds` (`STALE_QUOTE`) are rejected to eliminate phantom spreads caused by asynchronous feed latency.
- **Order-Book Depth**: Available L1/L2 bid/ask quantities must meet or exceed the target execution quantity (`INSUFFICIENT_LIQUIDITY` otherwise).
