# BharatAlpha — Unified Defect Register (`BHARATALPHA-AUTO-QA-REMEDIATION-001`)

**Baseline HEAD**: `1758a83ea852584170e85876365dca466eed7af3` | **Branch**: `feature/remed-01-authority-risk-capital`

---

## 1. Mandatory Previous-Defect Revalidation

| Ref | Previous Defect | Revalidation Result | Evidence |
|---|---|---|---|
| PR34-01 | `central_hypothesis` NameError in `AUTO` direction | **CONFIRMED RESOLVED** | `trader/timing/monitoring_session.py` uses `hypothesis is None` |
| PR34-02 | Shutdown square-off symbol-key mismatch (`NSE:TCS`) | **CONFIRMED RESOLVED** | `live_paper_runner.py` & `gate.py` iterate `portfolio.positions.values()` |
| PR34-03 | `PAUSE` preventing 15:15 IST square-off | **CONFIRMED RESOLVED** | `monitoring_session.py` routes 15:15 via `ActionType.RISK_EXIT` |
| PR34-04 | Missing Human permission defaulting to `ALLOW_LONG` | **CONFIRMED RESOLVED** (+ UI fixed via `JS-002`) | Python `None`, JS `engine.js` `null`, HTML default `UNSET` |
| PR34-05 | Symbol-scoped `STOP_ALL` reservation leak | **CONFIRMED RESOLVED** | `CashReservationLedger.release_for_symbol` + `release_reservations_for_scope` |
| PR34-06 | Intrabar SL/TP bypassing mandatory gate | **CONFIRMED RESOLVED** | `monitoring_session.py` Step B routes via `ActionType.RISK_EXIT` + MFE tracking |
| PR34-07..10 | 4 Py↔JS parity mismatches (14:45, `PAUSE`, bar-0 `STOP_ALL`, `AUTO`) | **CONFIRMED RESOLVED** | Verified by Agent 4 (14/20 MATCH + 2 new parity fixes `PAR-01`/`PAR-02`) |
| PR34-11..12 | Weak/missing test coverage | **CONFIRMED RESOLVED** | 19 REMED-01 tests (16/16 STRONG) + 26 new QA remediation tests |
| E2E-01 | Browser dashboard runtime errors | **RE-OPENED (`JS-001`) → FIXED** | `docs/candle-agent/index.html:1094` `sim.records` → `sim.decisions` |
| E2E-02..05 | Live runner gate bypass, CLI `--token-map`, Gate 6 sizing | **CONFIRMED RESOLVED** (+ `PY-01`..`PY-03` fixed) | `live_paper_runner.py`, `run_live_paper_session.py`, `gate.py` |
| E2E-06 | Timestamp, tick-stream & partial-bar integrity | **RE-OPENED (`D6-01`..`D6-13`) → FIXED** | `tick_recorder.py`, `live_paper_runner.py`, `monitoring_session.py`, `market_feed.py` |
| E2E-07 | Holdout registry enforcement | **VERIFIED INTACT** (+ `DEF-7.7` `ValueError` hardening) | `sealed_holdout_registry.json` 0 bytes diff |
| E2E-08 | Test isolation & tracked-file mutation | **RE-OPENED (`DEF-7.1`/`DEF-7.2`) → FIXED** | `orchestrator.py` kwargs + `tests/intraday/conftest.py`; 0 mutations post-pytest |

---

## 2. Stage B Discovered Defects & Disposition

| ID | Agent | Severity | File & Line | Summary | Status | Test Coverage |
|---|---|---|---|---|---|---|
| `JS-001` | A3 | **P0** | `docs/candle-agent/index.html:1094` | `sim.records.length` uncaught `TypeError` on load & every replay click | **FIXED** | `test_js_inline_scripts_syntax_and_candle_agent_invariants` |
| `D6-01` | A6 | **P0** | `trader/data/tick_recorder.py:227-303`, `live_paper_runner.py:424` | Late ticks after bucket flush re-emit duplicate non-monotonic bars | **FIXED** | `test_d6_01_d6_02_late_tick_drop_and_duplicate_tick_dedup` |
| `JS-002` | A3 | **P1** | `docs/candle-agent/index.html:309-317,425,618,981` | UI Human permission defaulted to `ALLOW_BOTH` and rendered blank on `null` | **FIXED** | `test_js_inline_scripts_syntax_and_candle_agent_invariants` |
| `PAR-01` | A4 | **P1** | `docs/candle-agent/engine.js:811` | JS used 2.0% risk/trade vs Python's 1.0% (`MAX_RISK_PER_TRADE_PCT`) | **FIXED** | `test_par01_python_js_1pct_risk_per_trade_sizing_parity` |
| `R5-A` | A5 | **P1** | `trader/risk/engine.py:637-641` | Losing SHORT adverse MTM shrank `invested_capital`, breaching 80% gross exposure | **FIXED** | `test_r5a_losing_short_adverse_mtm_capped_at_80pct_gross_exposure` |
| `R5-B` | A5 | **P1** | `monitoring_session.py:885`, `live_paper_runner.py:722` | Pending cash reservations omitted statutory entry friction | **FIXED** | `test_remed01_authority_risk_capital.py` + `R5` suite |
| `R5-E` | A5 | **P1** | `trader/risk/engine.py:105-109` | Duplicate `order_id` silently overwrote prior cash reservation | **FIXED** | `test_r5b_duplicate_reservation_order_id_raises_value_error` |
| `PY-01` | A2 | **P1** | `scripts/run_live_paper_session.py:178-212` | `offline_jsonl` CLI anchored instruction validity to `datetime.now()` | **FIXED** | `test_py01_offline_jsonl_cli_historical_timestamp_anchoring` |
| `PY-02` | A2 | **P1** | `monitoring_session.py:336`, `live_paper_runner.py:277`, `paper_feed_session.py:83` | Injecting `execution_gate` without `risk_engine` caused split-brain risk state | **FIXED** | `test_py02_split_brain_execution_gate_and_risk_engine_binding_and_conflict` |
| `PY-03` | A2 | **P1** | `trader/timing/live_paper_runner.py:416-513` | Missing multi-day `reset_daily_session(day)` left Day 2 stuck in `REDUCING` | **FIXED** | `test_py03_multiday_live_paper_runner_rollover_resets_reducing_to_active` |
| `PY-04` | A2 | **P1** | `trader/paper/cycle006b.py`, `trader/timing/cycle007_pipeline.py` | Legacy historical benchmark runners call `portfolio.submit_order` directly | **DEFERRED (LEGACY)** | Out of active REMED-01/live-paper path; isolated from live runner |
| `D6-02` | A6 | **P1** | `trader/data/market_feed.py:72` | `_normalize_raw_ohlcv` stripped tz via `DatetimeIndex.values` | **FIXED** | `test_d6_08_normalize_raw_ohlcv_preserves_datetime_index_tz` |
| `D6-03` | A6 | **P1** | `trader/intelligence/data_contract.py:56-71` | Non-authoritative sources (`local_parquet`, `YahooFinanceAPI:*`, etc.) uncapped | **FIXED** | `test_d6_13_cap_provenance_tier_caps_non_authoritative_sources` |
| `D6-04` | A6 | **P1** | `trader/timing/live_paper_runner.py:810-815` | `finalize_all()` fed unclosed partial candles into `_evaluate_closed_1m_bar` | **FIXED** | `test_d6_04_unclosed_partial_bar_not_evaluated_on_shutdown` |
| `D6-05` | A6 | **P1** | `trader/timing/monitoring_session.py:217-251` | Forming partial 5m buckets entered `evaluate_causal_5m_mtf_trend()` | **FIXED** | `test_d6_05_incomplete_5m_bucket_excluded_from_causal_mtf_trend` |
| `D6-06` | A6 | **P1** | `trader/data/tick_recorder.py:58-62,475,523` | `NaN` crashed tick parsers & `Inf` bypassed `TickEvent` validation | **FIXED** | `test_d6_06_tick_parsers_fail_closed_on_nan_inf_and_missing_timestamp` |
| `D6-07` | A6 | **P1** | `trader/data/tick_recorder.py:478-485,527-533` | Broker tick parsers silently fabricated `datetime.now(IST_TZ)` on missing timestamp | **FIXED** | `test_d6_06_tick_parsers_fail_closed_on_nan_inf_and_missing_timestamp` |
| `DEF-7.1` | A7 | **P1** | `trader/research/orchestrator.py:1668`, `test_phase11_3_1_holdout_lifecycle.py:213` | Pytest mutated 3 tracked lab JSON files | **FIXED** | Verified 0 bytes diff post-full-pytest |
| `DEF-7.2` | A7 | **P1** | `tests/intraday/conftest.py` (new) | Pytest created 5 untracked `data/intraday/*_NSE_5m.parquet` files | **FIXED** | Verified `NO_5M_PARQUET` post-full-pytest |
| `DEF-7.3` | A7 | **P1** | `trader/evaluation/walkforward_optimizer.py:631` | 15m EOD square-off filled at 15:29 close instead of 15:15 open | **FIXED** | `walkforward_optimizer.py` multi-minute open fill |
| `DEF-7.8` | A7 | **P1** | `.github/workflows/intraday-lab-tests.yml` | PR CI lacked Node setup, JS syntax checks, & clean-tree/holdout gate | **FIXED** | Workflow updated + `bharatalpha-continuous-qa.yml` added |
| `PAR-02` | A4 | **P2** | `docs/candle-agent/engine.js:978-1012` | Gap-target checked before intrabar SL on reversal bars | **FIXED** | `test_par02_python_js_strict_conservative_stop_first_on_reversal_bars` |
| `JS-003` | A3 | **P2** | `docs/candle-agent/index.html`, `docs/trader/index.html`, `docs/trader/app.js`, `docs/lab/index.html` | Misleading "Live" & hardcoded "Verified ✓" labels on synthetic/replay pages | **FIXED** | `test_js_trader_app_synthetic_badge_logic` |
| `R5-C` | A5 | **P2** | `trader/authority/gate.py:229,247,293,355,379` | Lowercase `"portfolio"`/`"all"` failed to normalize to `"*"` | **FIXED** | `test_r5c_lowercase_portfolio_and_all_scope_normalization_in_gate` |
| `R5-D` | A5 | **P2** | `trader/portfolio/ledger.py:595` | Extreme >100% adverse short gap can drive `portfolio.cash` negative | **DOCUMENTED** | Accounting deficit on >100% gap; tracked for REMED-02 margin-call rule |
| `R5-F` | A5 | **P2** | `trader/authority/gate.py:541-544,695-708` | Default SL & `submit_order` lacked ₹0.05 NSE tick rounding | **FIXED** | `test_r5e_nse_tick_rounding_in_execute_with_gate_protection` |
| `PY-05` | A2 | **P2** | `trader/timing/paper_feed_session.py:149-154` | `primary_timeframe="1m"` raised `KeyError: '1m'` | **FIXED** | `test_py05_paper_feed_session_supports_primary_timeframe_1m` |
| `PY-06` | A2 | **P2** | `trader/timing/live_paper_runner.py:534-555` | Intrabar SL/TP ignored gap-through `bar.open` | **FIXED** | `test_py06_py07_live_paper_runner_gap_through_sl_fills_at_bar_open` |
| `PY-07` | A2 | **P2** | `trader/timing/live_paper_runner.py:576` | Entry window used `ts_open <= 14:45` instead of `ts_close < 14:45` | **FIXED** | `test_py06_py07_live_paper_runner_gap_through_sl_fills_at_bar_open` |
| `D6-08` | A6 | **P2** | `trader/data/tick_recorder.py:227-264` | Exact duplicate ticks double-counted volume & `tick_count` | **FIXED** | `test_d6_01_d6_02_late_tick_drop_and_duplicate_tick_dedup` |
| `D6-09` | A6 | **P2** | `trader/data/loader_1m.py:70`, `loader_1h.py:31` | Naive timestamps shifted +5h30m under `utc=True` | **DOCUMENTED** | Tracked for REMED-03 historical data loader migration |
| `D6-10` | A6 | **P2** | `tick_recorder.py:151`, `validator.py:15`, `resampler.py:61` | NSE holidays not checked in `tick_recorder`; `15:30:00` accepted | **FIXED** | `test_d6_07_nse_holiday_rejected`, `test_d6_12_*` |
| `D6-11` | A6 | **P2** | `scripts/generate_trader_data.py:170` | Unconditional synthetic 5m session in strategy comparison | **DOCUMENTED** | Tracked for REMED-03 dashboard data regeneration |
| `D6-12` | A6 | **P2** | `resampler.py:75`, `mtf_aligner.py:27`, `data_contract.py:194` | `15:15–15:30` final NSE bar mishandled in 30m/1h | **FIXED** | `test_d6_12_nse_final_1515_to_1530_bar_clamp_and_session_close_bound` |
| `D6-13` | A6 | **P2** | `trader/data/market_feed.py:517-526` | `CompositeMarketFeed.get_bars()` crashed with `AssertionError` | **FIXED** | `test_d6_10_composite_market_feed_unsupported_timeframe_returns_unavailable` |
| `DEF-7.4` | A7 | **P2** | `trader/evaluation/walkforward_optimizer.py:635-661` | Gap-through SL fill optimism in `simulate_fast_1m_session` | **FIXED** | `test_def7_4_simulate_fast_1m_session_gap_through_sl_fills_at_open` |
| `DEF-7.5` | A7 | **P2** | `trader/intelligence/ml_setup_scorer.py:197,290` | Skipped empty `weights_sha256` & unguarded `NaN` in `_vectorize` | **FIXED** | `test_def7_5_ml_setup_scorer_rejects_missing_sha256_and_sanitizes_nan` |
| `DEF-7.6` | A7 | **P2** | `trader/lab/replay_engine.py:1190-1208` | `run_walk_forward` allowed val/test date overlap | **FIXED** | `test_def7_6_replay_engine_run_walk_forward_disjoint_splits` |
| `DEF-7.7` | A7 | **P2** | `walkforward_optimizer.py`, `walkforward_1m.py`, `mtf_walkforward_optimizer.py`, `ml_setup_scorer.py` | Chronological split checked via `assert` (stripped under `-O`) | **FIXED** | Replaced with explicit `ValueError` |
| `JS-004/005` | A3 | **P3** | `docs/candle-agent/index.html` | `#select-speed` ignored mid-playback; `0` rendered as `"N/A"`; stale empty UI | **FIXED** | `test_js_inline_scripts_syntax_and_candle_agent_invariants` |
| `PY-08/09/10` | A2 | **P3** | `gate.py:575`, `monitoring_session.py:715`, `live_paper_runner.py` | Exit `entry_price <= 0`, `causal_df is not None` guard, `RLock` thread-safety | **FIXED** | `test_r5f_invalid_exit_price_rejected_by_gate` + regression suite |
| `DEF-7.9` | A7 | **P3** | `data/intraday_1m/*.json` (4 local untracked files) | Pre-existing untracked local research JSONs | **DOCUMENTED** | Never staged/committed |
