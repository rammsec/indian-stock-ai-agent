# CYCLE-009A — Open-Source Quantitative Trading Intelligence Research Audit

- **Cycle ID**: `CYCLE-009A`
- **Audit Type**: `READ_ONLY_RESEARCH_AUDIT` (zero source-code modifications, zero new dependencies, zero broker/live/paid-API/holdout actions, zero commits/PRs/merges)
- **BharatAlpha Baseline Commit (`main`)**: `1b079c2ac3bfebc2b49ad09e9bd9e85c4ee6c827` (post-merge CI `45016595765` passed; `18/18` protected files unchanged)
- **Final Classification**: **`RESEARCH_COMPLETE`**
- **Structured Feature Matrix**: `data/lab/cycle_009a_oss_feature_matrix.json`

> **Governance Notice**: Every item in this report has been **researched only** from actual source files, tests, and `LICENSE` files across all six repositories and BharatAlpha's `main` branch. Nothing in this report claims that any researched feature or strategy has been implemented, validated, or shown to be profitable.

---

## 1. Executive Summary

Six independent read-only research agents audited **NautilusTrader**, **vn.py**, **OpenAlgo**, **Freqtrade**, **je-suis-tm/quant-trading**, **lightweight-charts-python**, and **BharatAlpha (`main` @ `1b079c2`)**.

**Core Findings:**
1. **Direct External Dependencies — `REJECT` All 6 Packages**: Every single one of the six repositories should be **rejected as a direct Python runtime dependency** for BharatAlpha due to strong copyleft licenses (`AGPL-3.0` on OpenAlgo, `GPL-3.0-only` on Freqtrade, `LGPL-3.0-only` on NautilusTrader), heavy GUI/toolchain coupling (`PySide6`/`ta-lib` on vn.py, Rust `1.100`/PyO3 on NautilusTrader, `pywebview` on `lightweight-charts-python`), >2-year abandonment (`lightweight-charts-python`), or severe lookahead/math bugs (`je-suis-tm/quant-trading`).
2. **High-Value Architectural Patterns — Clean-Room Native Python (`ADAPT` / `STUDY`)**: Studying the source code yielded **10 concrete, high-leverage, license-clean native improvements** for BharatAlpha — led by:
   - Eliminating BharatAlpha's **4× duplicate indicator/regime implementations** into a single **Shared Intelligence Layer**,
   - Adding an automated **Lookahead & Recursive Convergence CI Validator** (inspired by Freqtrade's `LookaheadAnalysis` / `RecursiveAnalysis`, upgraded for IST bar-index and cross-sectional safety),
   - Enforcing **dual-timestamp (`ts_event` vs `ts_init`)** event contracts and a **3-state `RiskEngine` (`ACTIVE` / `REDUCING` / `HALTED`)** with pending cash reservations and order-rate throttling (inspired by NautilusTrader),
   - Upgrading BharatAlpha's static `docs/lab/` charts by vendoring **native TradingView `lightweight-charts` JS (Apache-2.0)** directly rather than adding `lightweight-charts-python`.
3. **Single Most Valuable Next Cycle**: **`CYCLE-009B — Shared BharatAlpha Intelligence Layer + Automated Lookahead/Convergence CI Gate + 3-State RiskEngine Hardening`**.

---

## 2. Current BharatAlpha Architecture Assessment (Phase 1 — `main` @ `1b079c2`)

BharatAlpha currently operates across **three parallel, loosely coupled stacks**:
1. **Daily EOD ML + News + Phase 6 Debate Stack (`backend/`)**: Collectors, `feature_engine.py`, `leakage_guard.py`, `ensemble_model.py`, DuckDB store, producing `data/predictions/` (`prediction_log.parquet`, `forward_predictions.parquet`, `structured_events.parquet`) and `data/processed/` (`market_regimes.parquet`).
2. **Legacy 5m Multi-Agent Desk (`trader/agents/`, `trader/runner.py`, `trader/strategies/`)**: Layer A (`CentralMarketCouncil`) + Layer B (`AutonomousTradingAgent`) using Pydantic v2 `contracts.py` and `trader/features/indicators.py`, currently driven by `generate_synthetic_5m_session()` in `runner.py` and **disconnected** from `CYCLE-006B/C/007/008`.
3. **Fail-Closed Research & Paper-Trading Stack (`trader/paper/`, `trader/timing/`, `trader/evaluation/`)**:
   - `CYCLE-006B/C`: `OfflineExecutionGuard`, `Cycle006BPaperTradingRunner`, `reconstruct_as_of_snapshot()`, `classify_timeframe_provenance()`.
   - `CYCLE-007`: `CentralDirectionalAgent` (Stage 1, `UNCALIBRATED`), `CandleIntelligenceEngine` + `CandleTimingAgent` (Stage 2 independent numerical timing), `Cycle007Pipeline`.
   - `CYCLE-008`: `audit_historical_data_readiness()`, `compute_strategy_metrics()`, `Cycle008WalkForwardEvaluator` (chronological 196-day pre-holdout walk-forward; overall classification `INSUFFICIENT_EVIDENCE`).
   - Authoritative Risk & Ledger: `DeterministicRiskEngine` (`trader/risk/engine.py`) + `IntradayPaperPortfolio` (`trader/portfolio/ledger.py`, SHA-256 hash-chained, long-only) + `calculate_intraday_costs()` (`trader/portfolio/cost_calculator.py`, full NSE statutory cost stack).

### Verified Gaps, Duplication & Technical Debt
- **4× Duplicate Indicator Math**: `trader/features/indicators.py` (Wilder EWM ATR) vs `trader/timing/candle_intelligence.py` (simple-mean `_compute_atr` — mathematically drifts from Wilder ATR) vs `trader/evaluation/walk_forward.py:726-728` (inline EMA 9/21) vs `trader/strategies/supertrend.py`.
- **4× Fragmented Regime Logic**: `market_intelligence.py`, `market_regimes.parquet`, `walk_forward.py:_compute_pre_cutoff_regime_proxy()`, and `docs/data/regime.json`.
- **Private `_`-Helper Cross-Imports**: `central_directional_agent.py` imports `_load_json_strict`, `_parse_iso_to_ist`, `_parse_obs_date`, `_parse_finite_float` from `trader.paper.cycle006b`; `loader_1h.py` / `data_service.py` import `_compute_candle_series_hash` from `trader.data.loader`.
- **Synthetic Token Telemetry**: `MarketIntelligenceAgent`, `TechnicalAnalysisAgent`, and `NewsSentimentAgent` record hardcoded token counts without invoking `GeminiProvider`.
- **Data & Evaluation Bottlenecks**: Only **3** committed Stage 1 archive dates (`2026-11-16..18`), `44.29%` (`124/280`) `collection_timestamp` completeness in `structured_events.parquet`, **`48 EXPOSED / 0 AVAILABLE / 0 SEALED`** holdout dates, **0** committed `5m`/`15m` files (`UNAVAILABLE`), `PARTIALLY_VERIFIED` Yahoo Finance `1h` provenance, `UNCALIBRATED` confidence, and negative 1h OOS net return (`-1.91%` / `-5.28%`).

---

## 3. Six Individual Repository Reviews (Phase 2)

### 3.1 NautilusTrader (`nautechsystems/nautilus_trader`) — `LGPL-3.0-only`
- **Verified Architecture**: v2 (`Cargo.toml` v0.69.0, Rust edition 2024, `python/pyproject.toml` v2.0.0, Python `>=3.13`, Cython removed, PyO3 bindings). Single-threaded deterministic core (`MessageBus`, `DataEngine`, `ExecutionEngine`, `RiskEngine`, `Portfolio`, `Cache`, `VirtualClock`). Zero built-in NSE/Indian broker adapters; zero built-in walk-forward/CV harness.
- **Strongest Transferable Ideas (Clean-Room Python `ADAPT`)**:
  1. **Dual Timestamps (`ts_event` vs `ts_init`) + Monotonic `VirtualClock`** (`crates/backtest/src/engine.rs`, `crates/common/src/clock/virtual.rs`): Steps replay strictly by `ts_init` (local receipt time), structurally preventing ingestion-lag lookahead.
  2. **3-State `RiskEngine` (`ACTIVE` / `REDUCING` / `HALTED`) + Cash Reservations + Rate Throttler** (`crates/risk/src/engine/mod.rs`, `crates/common/src/throttler.rs`): `REDUCING` maps directly to NSE 15:05–15:15 IST MIS wind-down (blocks new risk while allowing exits); pending `reservations` prevent concurrent multi-stock cash double-spend; throttler enforces SEBI/broker order-rate caps.
  3. **Adaptive OHLC Bar Path + Seeded Slippage + Latency Queue** (`crates/execution/src/matching_engine/mod.rs:bar_high_first`, `fill.rs`, `exchange.rs:InflightCommand`): Realistic intrabar traversal and deterministic slippage.
  4. **Pluggable `FeeModel` Context** (`crates/execution/src/models/fee.rs`): Clean side/product fee abstraction (already partially modeled in BharatAlpha's `cost_calculator.py`).
  5. **Hash-Chained Event Store** (`crates/event_store/src/lib.rs`): **`DEFER`** until live execution.
- **Direct Dependency Verdict**: **`REJECT`** (`LGPL-3.0-only`, heavy Rust toolchain, 0 NSE adapters).

### 3.2 vn.py (`vnpy/vnpy`) — `MIT`
- **Verified Architecture**: Single-process `EventEngine` (`queue.Queue` without handler exception isolation or backpressure) + `MainEngine` (calls `os.chdir(TRADER_DIR)` in `__init__`) + `OmsEngine` + `BaseGateway` + `BarGenerator`/`ArrayManager` + new Polars-based `vnpy/alpha/` (`AlphaLab`, `AlphaDataset`, `LgbModel`/`LassoModel`/`MlpModel`).
- **Critical NSE Gaps**: No `NSE`/`NFO`/`INR` enums (`Exchange.BSE` is *Beijing* Stock Exchange!), Chinese gettext enum values, `update_bar_hour_window` hardcodes `:59` hour closes (breaking NSE `09:15–10:15`..`15:15–15:30` hourly candles), `PySide6`/`ta-lib` unconditional core deps, `pickle`/`shelve` in `AlphaLab`.
- **Transferable Ideas (`ADAPT` / `STUDY`)**: `Decimal`-based `round_to`/`floor_to`/`ceil_to` tick snapping (`vnpy/trader/utility.py` — ideal for NSE `₹0.05` ticks), `AlphaDataset` point-in-time constituent filter (`vnpy/alpha/dataset/template.py`), clean `BaseGateway`/`BaseDatabase` ABC separation.
- **Direct Dependency Verdict**: **`REJECT`**.

### 3.3 OpenAlgo (`marketcalls/openalgo`) — `AGPL-3.0` (**CRITICAL LEGAL BLOCKER**)
- **Verified Architecture**: Massive Flask + SocketIO + React SPA monolith (62 blueprints, 189 services, 731 broker files across 39 Indian brokers, 1,304 React files, 100+ pinned deps). Highest native Indian NSE/NFO/MIS/CNC/SL/SL-M fidelity of any OSS repo.
- **Legal Blocker**: **`AGPL-3.0` Section 13 (Remote Network Interaction)** — copying code, importing modules, or modifying and serving over a network requires open-sourcing the combined work under AGPL-3.0.
- **Clean-Room Concepts (`STUDY` Only — Zero Code Copied)**: Canonical `SymToken` `(symbol, exchange) ↔ (brsymbol, token, tick_size, lotsize)` table (`database/symbol.py`), IST `03:00` daily broker-session token expiry (`database/auth_db.py`), `force_live` exit-leg safety (`services/place_order_service.py`), pre-send bounded broker backpressure + unanswered-write HTTP timeout guard (`utils/broker_backpressure.py`, `utils/httpx_client.py`), per-position locking (`sandbox/position_locks.py`), Argon2 + Fernet/PBKDF2 secret hygiene.
- **Direct Dependency / Code Reuse Verdict**: **`REJECT`**.

### 3.4 Freqtrade (`freqtrade/freqtrade`) — `GPL-3.0-only` (**HARD LEGAL BLOCKER**)
- **Verified Architecture**: Mature Python 3.12+ crypto bot (`ccxt`, SQLAlchemy 2.0, Optuna v4, `FreqAI` + `datasieve` + LightGBM/XGBoost/RL).
- **Crypto-Specific Assumptions Unsuitable for NSE**: 24/7/365 clock, `days_in_year=365` & `0%` Rf in `hyperopt_loss_sharpe_daily.py` (inflates NSE Sharpe by $\approx 1.203\times$), flat symmetric fee with zero slippage, perpetual funding fees, liquidation tiers, fractional quantities, closed-trades-only protections (blind to intraday unrealized MTM drawdown).
- **Strongest Clean-Room Concepts (`STUDY` Only — Zero GPL Code Copied)**:
  1. **`LookaheadAnalysis` & `RecursiveAnalysis`** (`freqtrade/optimize/analysis/lookahead.py`, `recursive.py`): Automated bar-truncation lookahead & warmup-convergence CI detectors (reimplement natively with IST bar-index slicing & cross-sectional support).
  2. **`FreqAI` Dissimilarity Index (`do_predict` OOD mask) + Rolling Prediction Normalization** (`freqtrade/freqai/freqai_interface.py`, `data_kitchen.py`): Abstain during feature-space regime shifts; use NSE trading-day windows & Purged/Embargoed CV.
  3. **`IProtection` + `PairLocks` Middleware** (`freqtrade/plugins/protections/`): `StoplossGuard`, `CooldownPeriod`, `MaxDrawdown`, `LowProfitPairs` — upgrade natively to include **unrealized intraday MTM drawdown**.
  4. **Lookahead-Safe HTF Merger & `BaseParameter` Static-Indicator Guard** (`strategy_helper.py`, `parameters.py`): Reimplement with `pd.merge_asof(..., direction="backward")` on explicit IST bar-close timestamps.
- **Direct Dependency / Code Reuse Verdict**: **`REJECT`**.

### 3.5 `je-suis-tm/quant-trading` — `Apache-2.0` *(NOT MIT)*
- **Verified Code Quality**: 12 standalone Jupyter-exported `.py` scripts (`tests/` = none, `pyproject.toml` = none). **Zero production-reusable code** (`ADOPT = 0`).
- **Verified Fatal Bugs in Source**:
  - `Dual Thrust backtest.py` (L83–85, 127–129): Unshifted `rolling(rg)` leaks day `t`'s full high/low/close into day `t`'s opening breakout trigger.
  - `Shooting Star backtest.py` (L40–42, 57–64): Uses 21-year global mean + `shift(-1)` future high/close peeking.
  - `Pair trading backtest.py` (L157–181, 252–253): Unit-mixing bug (`z ± std(resid)`), global `max(price)` position-sizing lookahead, 1:1 equal shares ignoring OLS $\beta$.
  - `Awesome Oscillator backtest.py` (L84–123, 285–286): Inverted saucer sign overwritten by MA cross; broken Sharpe formula.
  - `Heikin-Ashi backtest.py` (L280–372): Broken Student-t `omega()`/`sortino()`, mismatched 50-bar vs 750-bar sample length.
  - Universal: same-bar close fills, zero costs/slippage, unadjusted prices (`del df['Adj Close']`, `del df['Volume']`).
- **Verdict**: **`REJECT`** all source code; **`ADAPT`/`STUDY`** clean-room concepts only (IST ORB / Causal Dual Thrust, BB Squeeze, clean Parabolic SAR/EMA, log-price Kalman/OLS pairs).

### 3.6 `lightweight-charts-python` (`louisnw01/lightweight-charts-python`) — `MIT` + bundled `Apache-2.0`
- **Verified Status**: Unmaintained >25 months (last commit `v2.1` on 2024-09-28; bundles stale LWC v4.1.3 while upstream is v5.3.0 and BharatAlpha's `docs/lab/index.html` **already loads native `lightweight-charts@4.2.1`**).
- **Verified Source Defects**: `eval(str(e))` (`chart.py:90`), unescaped JS f-string interpolation, `pywebview>=5.0.5` import-time process/queue instantiation (hostile to headless Linux CI), dead callbacks in `StaticLWC`, and UTC vs local-OS timezone mismatch between `_df_datetime_format()` and `_single_datetime_format()` that silently shifts NSE intraday trade markers.
- **Verdict**: **`REJECT`**. Instead, vendor native TradingView `lightweight-charts` JS (Apache-2.0 + `NOTICE`) directly into `docs/vendor/` with IST `timeFormatter`, `₹` `priceFormatter`, and `series.createPriceLine()`.

---

## 4. Comparative Feature Matrix (P0–P4)

| ID | Priority | Feature | Source Reference | BharatAlpha Target | Mode | Rec. |
|---|---|---|---|---|---|---|
| **F-01** | **P0** | Automated Lookahead & Recursive Convergence CI Validator | Freqtrade `lookahead.py`, `recursive.py` | `trader/evaluation/bias_auditor.py` | Native Clean-Room | STUDY |
| **F-02** | **P0** | Dual-Timestamp (`ts_event` vs `ts_init`) & Monotonic `VirtualClock` | Nautilus `engine.rs`, `virtual.rs` | `trader/intelligence/data_contract.py` | Native Clean-Room | ADAPT |
| **F-03** | **P0** | 3-State `RiskEngine` (`ACTIVE`/`REDUCING`/`HALTED`) + Cash Reservations + Rate Throttler | Nautilus `risk/engine/mod.rs`, `throttler.rs` | `trader/risk/engine.py` | Native Clean-Room | ADAPT |
| **F-04** | **P0** | Single-Source-of-Truth Shared Intelligence & Feature Engine | vn.py `AlphaDataset` + Phase 1 Audit | `trader/intelligence/features.py` | Native Clean-Room | ADAPT |
| **F-05** | **P0** | IST Bar-Close Multi-Timeframe Merger (`merge_asof` backward) | Freqtrade `strategy_helper.py`, Nautilus `aggregation.rs` | `trader/intelligence/mtf_aligner.py` | Native Clean-Room | ADAPT |
| **F-06** | **P1** | Feature-Space OOD Abstention (`DissimilarityIndex`) & Score Calibration | Freqtrade `freqai_interface.py` | `trader/intelligence/calibration_and_ood.py` | Native Clean-Room | STUDY |
| **F-07** | **P1** | MTM-Aware Risk-Guard Plugins (`StoplossGuard`, `Cooldown`, Intraday MTM DD) | Freqtrade `plugins/protections/` | `trader/risk/protections.py` | Native Clean-Room | STUDY |
| **F-08** | **P1** | Adaptive OHLC Bar-Path Fill + `₹0.05` Decimal Tick Snap + Circuit Gate | Nautilus `matching_engine/mod.rs`, `fill.rs`; vn.py `round_to` | `trader/portfolio/execution_model.py` | Native Clean-Room | ADAPT |
| **F-09** | **P2** | Clean-Room NSE Causal Dual Thrust / ORB & BB Squeeze Strategies | `quant-trading` concepts (rewritten) | `trader/strategies/` | Native Clean-Room | ADAPT |
| **F-10** | **P3** | Self-Hosted Native TradingView `lightweight-charts` JS + IST/₹ Formatters | Native `tradingview/lightweight-charts` (Apache-2.0) | `docs/vendor/`, `docs/lab/app.js` | Vendored JS | ADAPT |
| **F-11** | **P2** | Canonical NSE Master-Contract Registry & PIT Constituent Filter | OpenAlgo `SymToken` concept + vn.py `filters` | `trader/data/instruments.py` | Native Clean-Room | STUDY |
| **F-12** | **P4** | Hash-Chained Embedded Event Store & Snapshot+Tail Crash Recovery | Nautilus `crates/event_store/` | Post-live-readiness | Native Clean-Room | DEFER |

---

## 5. Strategy Research Catalogue (Phase 3 — All 11 Categories)

*No strategy is assumed profitable. All candidates require survivor-free Nifty 100/200 walk-forward testing net of full NSE statutory taxes + slippage before promotion.*

| # | Category | Hypothesis | Required Data (NSE) | Entry / Exit Logic (Clean-Room) | Risk Controls & Biases to Guard | Rec. |
|---|---|---|---|---|---|---|
| 1 | **Trend Following** | Persistent directional drifts in liquid NSE large/mid-caps | Adjusted 15m/1D OHLCV, Nifty 100/200 | Close > EMA(50/200) + Parabolic SAR (`af=0.01–0.02`) flip; exit on SAR/2–3× ATR trail; `t+1` Open fill | Cap 0.5–1% risk/trade; circuit/gap filter; guard unadjusted splits & same-bar close fills | **ADAPT** |
| 2 | **Momentum** | Short-term midpoint acceleration precedes continuation | Adjusted 15m/1D OHLCV, ADV ≥ ₹20 Cr | Clean AO (`SMA5−SMA34` midpoint) zero-cross + slope; exit on opposite cross / time stop | Turnover cap (STT drag); guard whipsaw in sideways NSE regimes | **STUDY** |
| 3 | **Mean Reversion** | Oversold extremes in uptrending liquid equities revert | Adjusted 15m/1D OHLCV, earnings calendar | Wilder RSI(2/14) < 25–30 **only** when Close > SMA(200); exit RSI > 55 / SMA(10) / 5d time stop | Hard 2× ATR stop; zero uncapped pyramiding; exclude earnings & circuit-locked names | **ADAPT** |
| 4 | **Opening Range Breakout (ORB)** | 09:15–09:45 IST auction imbalance sets intraday bias | IST 1m/5m OHLCV, pre-open gap & volume, F&O stocks | Breakout of `[OR_High, OR_Low]` or **strictly `t-1..t-N`** Dual Thrust + RVOL surge; no entries > 13:30 IST; mandatory 15:10 IST MIS square-off | Skip opening gaps > 1.5× ATR; ₹0.05 tick snap; guard `Dual Thrust` unshifted lookahead | **ADAPT** |
| 5 | **Volatility Expansion / Contraction** | Low-bandwidth BB/Keltner squeeze precedes expansion | Adjusted 5m/15m/1D OHLCV, India VIX | Relative BB bandwidth in bottom 15th pct (100-bar rolling) + band breakout; exit ATR trail / mid-band | Use %/ATR thresholds (never raw FX price units); guard squeeze fakeouts | **ADAPT** |
| 6 | **Volume Confirmation** | Abnormal RVOL/VWAP alignment reduces false breakouts | Adjusted OHLCV + NSE volume & delivery % | Overlay requiring `Volume > 1.5× SMA(Vol, 20)` + VWAP alignment | Exclude ADV < ₹20 Cr & zero-volume circuit bars; ablation-test net Sharpe uplift | **STUDY** |
| 7 | **Relative Strength & Sector Rotation** | Institutional flows drive medium-term NSE sector persistence | Survivor-free Nifty 200 + NSE sectoral indices | Weekly/monthly top-decile risk-adjusted RS (skip last 5–21d) + sector RS > Nifty 50; rank buffer exit | Sector cap ≤ 30%, stock cap ≤ 10%, CNC delivery STT (0.1% both sides) | **STUDY** |
| 8 | **Stat-Arb / Pair Trading** | Same-sector NSE F&O pairs revert around cointegrated spread | Adjusted 15m/1D OHLCV, NSE F&O / MIS | Rolling Engle-Granger ($p<0.05$) on **log prices** + OLS/Kalman $\beta_t$; enter $\lvert z_t \rvert > 2.0$, exit $z_t \to 0$ | Hard stop $\lvert z_t \rvert > 3.5$; MIS intraday or Futures only (no overnight cash shorts); guard unit-mixing & `max(price)` lookahead | **ADAPT** |
| 9 | **Multi-Timeframe Confirmation** | Causal HTF trend filters LTF entry chop | Synchronized IST 5m/15m/1h/1D OHLCV | Completed `t-1` HTF trend gates LTF entry via `merge_asof(backward)` on IST close timestamps | Unit-test anti-lookahead invariant (afternoon truncation must not alter morning signals) | **ADAPT** |
| 10 | **Market-Regime-Specific** | Expectancy varies across India VIX & trend regimes | Nifty 50 OHLCV, official India VIX, breadth | Meta-gate: Trend/ORB in normal-VIX trends, Mean Reversion in low-VIX chop, reduce/halt in VIX > 30 crisis | 3-bar hysteresis to prevent regime-flip whipsaw; never use broken CME `VIX Calculator.py` | **STUDY** |
| 11 | **Microstructure / Execution-Aware** | Realized NSE alpha is dominated by spread, ticks, circuits & taxes | L1 spread, ₹0.05 tick, circuit bands, NSE tax table | Marketable-limit fills, avoid 09:15–09:16 auction noise, 15:10 IST MIS buffer | Max participation < 1–2% bar volume; spread cap < 15–20 bps; 1×/2×/3× slippage stress tests | **STUDY** |

---

## 6. BharatAlpha Shared Intelligence Layer Architecture Proposal (Phase 4)

To eliminate duplicate indicator drift and unify BharatAlpha's three parallel stacks without weakening `DeterministicRiskEngine`:

```
[Committed Archives / Feeds]
       │  (ts_event, ts_init, source_provenance, IST tz-aware)
       ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Data & Provenance Contract (trader/intelligence/data)   │
│     • Single strict OHLCV/JSON/timestamp validator          │
│     • IST bar-close MTF aligner (merge_asof backward)       │
│     • Explicit UNAVAILABLE / PARTIALLY_VERIFIED propagation │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Shared Evidence Store (Single Source of Truth)          │
│     • Features: Wilder ATR, EMA, VWAP, RSI, BB, RVOL,       │
│       10 candle patterns, swing structure (computed ONCE)   │
│     • Macro/Regime: India VIX, Nifty breadth, sector RS     │
│     • Events/News: PIT-filtered (pub<=cutoff & col<=cutoff) │
│     • Execution Context: spread, ₹0.05 tick, circuit band   │
│     • Portfolio Context: read-only exposure & MTM snapshot  │
└──────────────┬───────────────────────────────┬──────────────┘
               ▼                               ▼
   [Stage 1: Central Directional]   [Stage 2: Candle Timing]
   • Emits SignalEnvelope           • Independently evaluates
   • confidence_calibration_status    EvidenceBundle BEFORE
     = UNCALIBRATED (mandatory)       comparing with Stage 1
   • OOD mask (do_predict)          • Can CONTRADICT / WAIT / REJECT
               └───────────────┬───────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Disagreement & Uncertainty Escalation Bus               │
│     • Any missing/stale/OOD/contradicted evidence fails     │
│       closed to WAIT / NO_TRADE / REJECT with audit trail   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Independent Authoritative DeterministicRiskEngine       │
│     • 3-State: ACTIVE / REDUCING (15:05 IST) / HALTED       │
│     • Pending cash reservations + order-rate throttler      │
│     • MTM-aware Protections (StoplossGuard, Cooldown, DD)   │
│     • Absolute veto — agents NEVER bypass or override       │
└─────────────────────────────────────────────────────────────┘
```

**Common `EvidenceBundle` Contract Rules:**
- Every field carries `(value, ts_event_ist, ts_init_ist, source_provenance, staleness_hours, missing_status)`.
- Missing data propagates explicitly (`UNAVAILABLE` / `STALE` / `OOD_REJECTED`) — **never** silently imputed with `0.0` or forward-filled across sessions.
- Confidence scores carry mandatory `confidence_calibration_status = "UNCALIBRATED"` until out-of-sample Platt/isotonic calibration is independently verified on unexposed data.
- Agents query `EvidenceBundle` read-only; no agent independently recomputes EMA/ATR/VWAP/RSI.

---

## 7. Top 10 Recommended Improvements (Phase 5)

| Rank | Feature (Priority) | Source Reference | BharatAlpha Target | Dependencies | Required Tests | Complexity | Acceptance Criteria | Native vs External |
|---|---|---|---|---|---|---|---|---|
| **1** | **Shared Intelligence & Feature Engine** (`P0`) | vn.py `AlphaDataset` + Phase 1 Audit | `trader/intelligence/features.py` | `pandas`, `numpy` (existing) | Parity tests across `indicators.py` & `candle_intelligence.py`; zero duplicate ATR/EMA/VWAP; zero private `_` cross-imports | Medium | All agents/evaluators consume single feature module; Wilder vs SMA ATR explicit | **Native** |
| **2** | **Automated Lookahead & Recursive Convergence CI Validator** (`P0`) | Freqtrade `lookahead.py`, `recursive.py` | `trader/evaluation/bias_auditor.py` | Existing stack | Intentionally biased fixture strategies MUST be caught; clean strategies pass with 0 false positives | Medium | Runs in GitHub Actions CI gating every strategy/indicator | **Native** |
| **3** | **Dual-Timestamp (`ts_event`/`ts_init`) & Monotonic Clock Contract** (`P0`) | Nautilus `engine.rs`, `virtual.rs` | `trader/intelligence/data_contract.py` | Existing stack | Non-monotonic clock, `ts_init > as_of_ist`, and naive timestamps fail closed | Low–Med | Every bar, news event, and regime snapshot carries verified IST `ts_event` & `ts_init` | **Native** |
| **4** | **3-State `RiskEngine` + Cash Reservations + Rate Throttler** (`P0`) | Nautilus `risk/engine/mod.rs`, `throttler.rs` | `trader/risk/engine.py` | Existing stack | `REDUCING` blocks entries/allows exits; concurrent orders cannot double-spend cash; rate limit denies bursts | Low | Zero cash overspend; clean 15:05 IST wind-down state transition | **Native** |
| **5** | **IST Bar-Close Multi-Timeframe Aligner** (`P0`) | Freqtrade `strategy_helper.py`, Nautilus `aggregation.rs` | `trader/intelligence/mtf_aligner.py` | `pandas` (existing) | Afternoon truncation invariance test; NSE `15:15–15:30` final hourly bar alignment test | Low | Zero HTF→LTF lookahead across 5m/15m/1h/1D IST bars | **Native** |
| **6** | **Adaptive OHLC Bar-Path Fill + `₹0.05` Decimal Tick + Circuit Gate** (`P1`) | Nautilus `matching_engine/mod.rs`, `fill.rs`; vn.py `round_to` | `trader/portfolio/execution_model.py` | `decimal` (stdlib) | Deterministic seeded slippage replay; ₹0.05 rounding; circuit-locked bar rejects fill | Low–Med | Matches NSE tick/circuit mechanics; reproducible across runs | **Native** |
| **7** | **OOD Abstention (`DissimilarityIndex`) & Probability Calibration Harness** (`P1`) | Freqtrade `freqai_interface.py` | `trader/intelligence/calibration_and_ood.py` | `scikit-learn` (existing) | Synthetic OOD shift triggers `do_predict=False`; uncalibrated fallback stays `UNCALIBRATED` | Medium | Brier score & reliability curve verified strictly out-of-sample | **Native** |
| **8** | **MTM-Aware Risk-Guard Plugins (`StoplossGuard`, `Cooldown`, MTM DD)** (`P1`) | Freqtrade `plugins/protections/` | `trader/risk/protections.py` | Existing stack | Unrealized intraday MTM breach triggers halt; per-symbol cooldown blocks whipsaw re-entry | Low–Med | Protections decouple cleanly via `SymbolLock` without touching strategies | **Native** |
| **9** | **Clean-Room Causal Dual Thrust / ORB & BB Squeeze Candidates** (`P2`) | `quant-trading` concepts (rewritten) | `trader/strategies/` | Existing stack | Passed through `bias_auditor.py` + `Cycle008WalkForwardEvaluator` net of NSE costs | Low–Med | Zero lookahead; evaluated honestly against `no_trade_baseline` | **Native** |
| **10** | **Vendored Native TradingView `lightweight-charts` JS + IST/₹ Formatters** (`P3`) | Native `tradingview/lightweight-charts` (Apache-2.0) | `docs/vendor/`, `docs/lab/app.js` | None (static JS + `NOTICE`) | Offline HTML render check; IST `09:15–15:30` axis & `₹` formatting verified | Low | Zero CDN dependency; Apache-2.0 `NOTICE` preserved; SL/target `createPriceLine()` | **External JS (Apache-2.0)** |

---

## 8. Licence & Dependency Legal/Technical Review (Phase 6)

> **Disclaimer**: The licence and copyleft notes below are **technical research findings** derived from inspecting repository `LICENSE` files and source trees, and **do not constitute formal legal advice**. Consult qualified legal counsel for any binding licensing determination.

| Repository | SPDX License | Copyleft / Legal Impact | Key Technical & Security Flags |
|---|---|---|---|
| **OpenAlgo** | **`AGPL-3.0`** | **CRITICAL BLOCKER** (Section 13 network copyleft): no code copying, vendoring, or in-process import. | 100+ pinned deps, 39 broker plugins, 407 open issues, large web/webhook/LLM attack surface. |
| **Freqtrade** | **`GPL-3.0-only`** | **HARD BLOCKER** (strong copyleft): study concepts only; 100% clean-room native reimplementation required. | Crypto-coupled (`ccxt`, 24/7, 365d Sharpe, funding/liquidation, fractional qty). |
| **NautilusTrader** | **`LGPL-3.0-only`** | Weak copyleft on core modifications / static Rust linking; clean-room Python adaptation avoids all LGPL obligations. | Rust `1.100.0` + Python `3.13+` build weight; 0 Indian broker adapters. |
| **vn.py** | **`MIT`** | Permissive (safe with copyright notice). | Heavy GUI (`PySide6`/`ta-lib`), `os.chdir()` side effect, Chinese enums, `BSE` = Beijing, `pickle` in `AlphaLab`. |
| **quant-trading** | **`Apache-2.0`** | Permissive with Section 4 notice; zero code worth reusing anyway. | Severe lookahead & math bugs, hardcoded Windows paths, deprecated Yahoo imports. |
| **lightweight-charts-python** | **`MIT` + `Apache-2.0`** | Permissive, but omits upstream TradingView `NOTICE`. | Unmaintained >25m, `eval(str(e))`, `pywebview` GUI crash in headless CI, marker TZ bug. |

**Security & Secrets Policy**: Zero API keys/secrets required for any P0–P3 improvement. Never adopt `pickle`/`shelve` (vn.py `AlphaLab`) or `eval()` (`lightweight-charts-python`).

---

## 9. Implementation Roadmap

- **Wave 1 (`CYCLE-009B` — P0 Foundation)**: Build Shared Intelligence Layer (`F-04`, `F-02`, `F-05`), Automated Lookahead & Recursive Convergence CI Validator (`F-01`), and 3-State `RiskEngine` (`F-03`). Zero new strategies until the foundation is unified and bias-gated.
- **Wave 2 (`CYCLE-010` — P1 Realism & Archive Depth)**: Upgrade execution fill/tick/circuit model (`F-08`), add MTM-aware risk-guard plugins (`F-07`), and accumulate genuine multi-month pre-cutoff Stage 1 + 5m/15m archives with fresh sealed holdout dates (prerequisites for OOD/calibration `F-06`).
- **Wave 3 (`CYCLE-011` — P2/P3 Strategy & Dashboard)**: Evaluate clean-room NSE strategy candidates (`F-09`) with master-contract/PIT filters (`F-11`) on the hardened walk-forward engine, and vendor native `lightweight-charts` JS (`F-10`) in `docs/vendor/`.
- **Wave 4 (P4 — Deferred)**: Embedded hash-chained crash-recovery event store (`F-12`) only when live broker readiness is explicitly authorized.

---

## 10. Explicit List of Features That Should NOT Be Adopted

1. **OpenAlgo codebase / broker plugins / Flask monolith** (`AGPL-3.0` copyleft blocker + retail webhook scope mismatch).
2. **Freqtrade package / `ccxt` backtester / 365-day Sharpe hyperopt loss** (`GPL-3.0-only` blocker + 24/7 crypto assumptions).
3. **NautilusTrader v2 Rust/PyO3 runtime as a direct dependency** (`LGPL-3.0-only` + Rust toolchain weight + 0 NSE adapters).
4. **vn.py core package** (`PySide6`/`ta-lib` GUI coupling, `os.chdir()`, Chinese enums, wrong `BSE`).
5. **`lightweight-charts-python`** (abandoned >25m, `eval()`, `pywebview` CI incompatibility, marker TZ bug).
6. **Any verbatim script from `je-suis-tm/quant-trading`** (fatal lookahead bugs, broken Student-t metrics, zero costs, out-of-scope FX/CME/commodity scripts).
7. **FreqAI Reinforcement Learning (`freqai_rl`) & OpenAlgo LLM/MCP chat agents** (premature complexity while BharatAlpha still lacks multi-month Stage 1 archives and unexposed holdout dates).

---

## 11. Final Classification & Single Most Valuable Next Implementation Cycle

- **Final Classification**: **`RESEARCH_COMPLETE`**
- **Single Most Valuable Next Cycle**: **`CYCLE-009B — Shared BharatAlpha Intelligence Layer + Automated Lookahead/Convergence CI Gate + 3-State RiskEngine Hardening`**
- **Why**: `CYCLE-008` demonstrated empirically that BharatAlpha's binding constraint is **not** a lack of indicators, UI charts, or AI agents — it is (1) duplicate/drifting indicator & regime implementations across 4 codepaths, (2) lack of an automated CI lookahead/convergence gate before strategy evaluation, and (3) missing unified `ts_event`/`ts_init` + IST multi-timeframe alignment contracts and `REDUCING` wind-down risk states. Fixing these P0 architectural primitives cleanly in native Python eliminates technical debt, protects every future strategy test from silent leakage, and builds the exact foundation needed before accumulating fresh multi-month archives and sealed holdout dates.
