# BharatAlpha — Remediation Changelog (`BHARATALPHA-AUTO-QA-REMEDIATION-001`)

**Branch**: `feature/remed-01-authority-risk-capital` | **Base**: `1758a83ea852584170e85876365dca466eed7af3`

---

## Frontend, Dashboard & JS Parity (`docs/`)
- **`docs/candle-agent/index.html`**:
  - Fixed P0 `TypeError` in `renderWatchlistScanner()`: `sim.records` → `sim.decisions` (`JS-001`).
  - Changed `#select-human-perm` default to `UNSET (Fail-Closed — No Entries)` (`value=""`), `#hud-human` to `UNSET (FAIL-CLOSED)`, and passed `elPerm.value || null` to `IntradaySessionSimulator` (`JS-002`).
  - Replaced misleading "Live" replay labels (`JS-003`), added active `#select-speed` `change` listener (`JS-004`), fixed `rsi14=0`/`rvol=0` nullish checks and empty-session UI reset (`JS-005`).
- **`docs/candle-agent/engine.js`**:
  - Aligned `_sizeOrder()` risk budget from `eq * 0.02` to `eq * 0.01` matching Python `MAX_RISK_PER_TRADE_PCT = 0.01` (`PAR-01`).
  - Enforced strict conservative stop-first ordering (`GAP_STOP_LOSS` → `INTRABAR_STOP_LOSS` → `GAP_TARGET` → `INTRABAR_TARGET`) for both LONG and SHORT (`PAR-02`).
- **`docs/trader/index.html`, `docs/trader/app.js`, `docs/lab/index.html`**:
  - Replaced misleading "Live" replay strings and dynamically rendered provenance/walk-forward badges as `SYNTHETIC TEST DATA` when `source_mode === "SYNTHETIC_TEST"` (`JS-003`).

## Python Execution, Risk & Authority (`trader/`, `scripts/`)
- **`trader/risk/engine.py`**:
  - Used `max(portfolio.invested_capital, gross_portfolio_exposure)` in 80% portfolio exposure cap so losing SHORT positions cannot breach gross exposure (`R5-A`).
  - Rejected duplicate `order_id` reservations in `CashReservationLedger.reserve()` with `ValueError` (`R5-E`).
- **`trader/authority/gate.py`**:
  - Added `_normalize_scope()` (`strip().upper()` before wildcard matching) across all 5 scope-handling helpers (`R5-C`).
  - Rejected exits with `entry_price <= 0` (`INVALID_EXIT_PRICE`, `PY-08`).
  - Applied `round_to_nse_tick()` to default stop-losses and passed `round_ticks=True` to `portfolio.submit_order()` (`R5-F`).
- **`trader/timing/monitoring_session.py`**:
  - Bound `self.risk_engine = execution_gate.risk_engine` on injection and raised `ValueError` on conflicting instances (`PY-02`).
  - Excluded incomplete forming 5m buckets (`count < 5`) from `evaluate_causal_5m_mtf_trend()` (`D6-05`).
  - Added `causal_df is not None` guard in ATR fallback (`PY-09`) and included statutory entry friction in pending cash reservations (`R5-B`).
- **`trader/timing/live_paper_runner.py`**:
  - Bound `execution_gate.risk_engine` (`PY-02`), added trading-date rollover `reset_daily_session(day)` (`PY-03`), strict bar monotonicity guard (`D6-01`), shutdown partial-bar evaluation guard (`D6-04`), gap-through SL/TP fills (`PY-06`), `ts_close.time() < ENTRY_END_IST` entry window (`PY-07`), fee-inclusive pending reservation (`R5-B`), and `threading.RLock()` (`_state_lock`, `PY-10`).
- **`trader/timing/paper_feed_session.py`**:
  - Bound `execution_gate.risk_engine` (`PY-02`) and added `"1m"` primary timeframe fallback (`PY-05`).
- **`scripts/run_live_paper_session.py`**:
  - Anchored `offline_jsonl` Human/Central instruction validity window to the JSONL file's min/max `timestamp_ist` (`PY-01`).

## Data Integrity, Quant Hygiene, Test Isolation & CI
- **`trader/data/tick_recorder.py`**: `allow_inf_nan=False` on `TickEvent`/`RecordedBar`, fail-closed tick parsers (no `NaN`/`Inf`, no `datetime.now()` timestamp fabrication), `_flushed_buckets` late-tick drop + duplicate-tick dedup, and `is_nse_trading_day()` holiday check (`D6-01`, `D6-06`, `D6-07`, `D6-08`, `D6-10`).
- **`trader/data/market_feed.py`**: Preserved `DatetimeIndex` timezone via `pd.Series(ts, index=out.index)` (`D6-02`) and fail-closed `UNAVAILABLE` in `CompositeMarketFeed.get_bars()` (`D6-13`).
- **`trader/intelligence/data_contract.py`, `trader/data/validator.py`, `trader/data/resampler.py`, `trader/intelligence/mtf_aligner.py`**: Capped non-authoritative provenance tiers (`D6-03`), tightened `15:30:00` session-close upper bound (`D6-10`), and clamped `15:15–15:30` final NSE bars for `30m`/`1h` (`D6-12`).
- **`trader/research/orchestrator.py`, `tests/intraday/test_phase11_3_1_holdout_lifecycle.py`, `tests/intraday/conftest.py`**: Isolated all 3 lab JSON outputs and 5m parquet cache directory to temp paths so `pytest tests/intraday/` produces zero repo mutations (`DEF-7.1`, `DEF-7.2`).
- **`trader/evaluation/walkforward_optimizer.py`, `walkforward_1m.py`, `mtf_walkforward_optimizer.py`, `trader/intelligence/ml_setup_scorer.py`, `trader/lab/replay_engine.py`**: Modeled gap-through SL fills and 15m 15:15 open square-off, required non-empty matching `weights_sha256` & `NaN` sanitization, enforced disjoint `run_walk_forward` splits, and replaced `assert` chronological checks with `ValueError` (`DEF-7.3`–`DEF-7.7`).
- **`.github/workflows/intraday-lab-tests.yml` & `.github/workflows/bharatalpha-continuous-qa.yml`**: Added Node setup, JS syntax checks, post-test clean-tree & holdout immutability gate, and scheduled daily read-only continuous QA workflow (`DEF-7.8`).
- **`tests/intraday/test_autonomous_qa_remediation.py`**: Added 26 non-vacuous behavioral tests covering all remediated defects; updated 2 stale legacy expectations in `test_cycle019_022_auto_walkforward_live_ml.py` and `test_phase11_1_cycle002.py`.
