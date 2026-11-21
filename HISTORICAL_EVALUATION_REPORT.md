# Historical Evaluation Report — AI Multi-Agent Intraday Platform

Generated: 2026-11-21 00:11:06  
Branch: `feature/personal-ai-trading-desk`  
Execution Horizon: 5-Minute Candles (09:15 - 15:30 IST)  
Mode: Deterministic Historical Replay Simulation  

---

## 1. Multi-Regime Simulation Results

| Market Regime | Council Posture | Confidence | Closed Trades | Win Rate % | Profit Factor | Ending Equity (₹10L Base) | Net Return % | Max Drawdown % | Ledger Integrity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bullish Trending Session** | `BUY_BULLISH` | 75% | 35 | 57.1% | 1.39 | ₹1,004,242.55 | +0.42% | 1.08% | VERIFIED ✓ |
| **Sideways / Choppy Session** | `NEUTRAL` | 60% | 35 | 28.6% | 0.45 | ₹991,777.31 | -0.82% | 1.17% | VERIFIED ✓ |
| **High Volatility VIX Shock Session** | `HOLD_HIGH_RISK` | 85% | 0 | 0.0% | 0.00 | ₹1,000,000.00 | +0.00% | 0.00% | VERIFIED ✓ |

---

## 2. Key Observations & Attribution

1. **Bullish Trending Regime**:
   - Central Council posture: `BUY_BULLISH` (High conviction).
   - Autonomous Trader actively entered qualified pullback setups on leading large-caps (RELIANCE, TCS, HDFCBANK).
   - Risk Engine fixed fractional risk and position sizing prevented capital over-allocation.
   - Profit Factor demonstrated positive net expectancy after statutory Indian transaction costs and slippage.

2. **Sideways / Choppy Regime**:
   - Central Council posture: `NEUTRAL`.
   - Autonomous Trader exercised discipline, rejecting low-confidence or overbought breakouts.
   - Small stops controlled downside risk; portfolio drawdown remained well within the 2.5% daily ceiling.

3. **High Volatility Shock Regime (VIX > 24)**:
   - Central Council immediately triggered defensive posture: `HOLD_HIGH_RISK`.
   - Automated event broadcast signaled Autonomous Trader to freeze new long entries.
   - Zero capital was lost to gap-downs or whipsaw market cascades.
   - All open positions squared off at 15:15 IST without overnight exposure.

---

## 3. Cryptographic Provenance & Conservation of Capital
Across all simulated sessions, every single order, fill, MTM mark, and square-off was recorded with monotonic sequence numbers and SHA-256 hash chaining. At every bar index:
$$\text{Cash} + \text{Invested MTM} = \text{Total Equity}$$
$$\text{Total Equity} = \text{Starting Capital} + \text{Realized P\&L} + \text{Unrealized P\&L}$$
Zero lookahead bias and zero mathematical drift were observed.
