# CYCLE-009B — Shared Intelligence Layer: Implementation Readiness Audit

- **Cycle ID**: `CYCLE-009B` (Architecture & Implementation-Readiness Audit — **READ-ONLY**)
- **Verified Baseline (`main`)**: `1b079c2ac3bfebc2b49ad09e9bd9e85c4ee6c827` (zero tracked modifications, `18/18` protected files SHA-256 OK, `sealed_holdout_registry.json` untouched at `48 EXPOSED / 0 AVAILABLE / 0 SEALED`)
- **CYCLE-009A Artifacts Verified**: `docs/research/CYCLE_009A_OSS_QUANT_INTELLIGENCE_AUDIT.md` & `data/lab/cycle_009a_oss_feature_matrix.json` present and intact.
- **Companion Machine-Readable Artifacts**:
  - `data/lab/cycle_009b_feature_inventory.json`
  - `data/lab/cycle_009b_migration_parity_matrix.json`
- **Readiness Classification**: **`READY`** (zero external blockers; all 7 pre-implementation hazards are verified, quantified, and resolved in the design below)

> **Epistemic Hygiene**: Throughout this report, **VERIFIED FACTS** (observed directly in `main` @ `1b079c2`), **INFERRED RISKS**, and **RECOMMENDATIONS** are explicitly separated. Zero source code was modified, zero packages were installed, and no commits/PRs/merges were created.

---

## 1. Baseline & Multi-Agent Audit Summary (VERIFIED FACTS)

Five independent read-only specialist reviewers (Architecture, Indicator & Quant Math, Data Provenance & Lookahead, Integration, and Testing) audited `main` @ `1b079c2`:
- **Test Baseline**: `tests/intraday/` collects **264 tests** (plus **92 tests** in `tests/phase6/`). `.github/workflows/intraday-lab-tests.yml` currently runs `pytest tests/intraday/ tests/phase6/` in ~6m25s and installs **only** `pytest httpx pydantic pandas numpy pyarrow yfinance` (no `scipy`/`duckdb`/`scikit-learn`).
- **7 Critical Pre-Implementation Hazards Discovered (VERIFIED FACTS)**:
  1. **Wilder vs SMA ATR Divergence**: `indicators.py:73-91` uses Wilder EWM-14, while `candle_intelligence.py:521-539` uses SMA-14 — empirically diverging by **median 6.81%, p90 17.13%, p95 21.20%, max 50.51%** (`n=1,673` on committed NSE 1h archive).
  2. **RVOL Divergence**: 6 implementations diverge on inclusive vs exclusive current-bar denominator and window (`10`/`14`/`20`) — empirically diverging by **median 16.11%, p90 27.63%, max 1234.94%** (`n=1,450`).
  3. **Intraday Bar-Open vs Bar-Close Lookahead**: `candle_intelligence.py:313` / `walk_forward.py:373` / `cycle007_pipeline.py:254` filter on `ts_open < as_of_ist`, which is safe only at `09:15 IST` and would leak an in-progress `10:15–11:15` 1h bar at intraday `10:30 IST`.
  4. **Final `15:15–15:30` 1h Bar Silently Dropped**: `trader/data/resampler.py:75-77` requires `12` × 5m sub-bars for every 60m bucket, permanently dropping NSE's valid 15-minute closing bar (`15:15–15:30`, `3` × 5m bars).
  5. **Yahoo Finance Provenance Cap Gap**: `trader/data/provenance_1h.py:196-205` has no non-authoritative provider cap and could emit `"VERIFIED"` if 0 incomplete sessions exist.
  6. **RiskEngine Exit-Blocking Bug**: `trader/risk/engine.py:91-129` checks `is_daily_halted` and `>= 15:15 IST` *before* `if side_upper == "SELL"`, blocking exits during halts/square-offs, and `L88` silently misinterprets naive datetimes on UTC runners.
  7. **Hard-Asserted Compatibility Constraints**: `tests/intraday/test_cycle007_candle_timing.py:1275-1284` hard-asserts the exact 6 paths in `CYCLE007_IMPLEMENTATION_MODULES`; `trader/evaluation/walk_forward.py:365,796` directly mutates `risk_engine.is_daily_halted = False`; `trader/agents/contracts.py` uses Pydantic v2 `extra="forbid"`.

---

## 2. Audit of Duplicated Intelligence Implementations (9 Feature Families)

Complete file/line inventory is recorded in `data/lab/cycle_009b_feature_inventory.json`. Summary of verified duplication:

| Family | Impl. Count | Key Verified Discrepancies |
|---|---|---|
| **ATR** | 5 | 3 Wilder EWM (`indicators.py:73`, `technical_engine.py:62`, `feature_engine.py:93`, `0.0` diff for `bar>=14`) vs **SMA-14** (`candle_intelligence.py:521`, **6.81% median / 50.51% max diff**) vs H-L SMA fallback (`supertrend.py:15`, unused). |
| **EMA** | 5 | Mathematically identical `ewm(span, adjust=False)` (`0.0` diff once `bar >= min_periods`), duplicated across 5 files (`indicators.py:34`, `technical_engine.py:43`, `feature_engine.py:471`, `candle_intelligence.py:685`, `walk_forward.py:726-728`). |
| **VWAP** | 3 | `indicators.py:11` vs `candle_intelligence.py:542` match within **`1.82e-12`** on non-zero-volume sessions, but diverge on zero-volume sessions (`typical_price + ffill()` vs `(None, "UNAVAILABLE", None)`). |
| **RSI** | 3 | All Wilder EWM (`0.0` diff on normal bars), but `indicators.py:49-51` & `technical_engine.py:34-36` have a **bug mapping pure monotonic rallies (`avg_loss==0, avg_gain>0`) to `50.0` instead of `100.0`** (`feature_engine.py:89` emits `NaN`). |
| **Volume / RVOL** | 6 | Divergent windows (`10`/`14`/`20`), current-bar inclusion (`indicators.py:161`) vs exclusion (`candle_intelligence.py:724`, `mtf_strategies.py`), and zero-vol handling (**16.11% median / 1234.94% max diff**). |
| **Candle Anatomy & Patterns** | 2 | `candle_intelligence.py:370-496` uses unsigned `[0,1]` body/CLV & `range/open` (+10 patterns); `feature_engine.py:550-557` uses signed `[-1,1]` body/CLV (`= 2*CLV_01 - 1`) & `range/close`. |
| **Trend & Momentum** | 5 | MACD `(12,26,9)` matches (`0.0`); `SupertrendStrategy.evaluate_bar` never calls `compute_supertrend` (dead code); 5 different breakout windows (`3`/`4`/`5`/`6`/`10` bars); CPR pivots add ATR clamping in `technical_engine.py:203`. |
| **Market Regime** | 6 | 6 incompatible taxonomies & VIX thresholds (`18.5`, `21.0`, `24.0`, `25.0`) across `market_regime_collector.py`, `market_intelligence.py`, `trader/evaluation/walk_forward.py`, `backend/backtesting/walk_forward.py`, `feature_engine.py`, `phase6/risk.py`. |
| **MTF Alignment** | 3 | `resampler.py:75-77` drops valid `15:15–15:30` closing 1h bar; `mtf_strategies.py:499,549` calls `.dt.date` without `tz_convert("Asia/Kolkata")`; `candle_intelligence.py:823` blends 70/30 primary/confirming. |

**Private `_`-Helper Cross-Imports Verified (4 production + 5 test)**:
- `trader/timing/central_directional_agent.py:19-25` → `trader/paper/cycle006b.py` (`_load_json_strict`, `_parse_finite_float`, `_parse_iso_to_ist`, `_parse_obs_date`)
- `trader/data/loader_1h.py:13` & `trader/lab/data_service.py:12` → `trader/data/loader.py` (`_compute_candle_series_hash`)
- `trader/research/orchestrator.py:38` → `trader/research/holdout.py` (`_get_git_commit_sha`)

---

## 3. Shared Intelligence Layer Architecture (`trader/intelligence/`) — RECOMMENDATIONS

Pure additive native Python package (zero new dependencies — uses only `pandas`, `numpy`, `pydantic`, stdlib so `.github/workflows/intraday-lab-tests.yml` passes out of the box):

1. **`trader/intelligence/parsing.py`**: Promotes `load_json_strict`, `parse_iso_to_ist`, `parse_obs_date`, `parse_finite_float`, `compute_candle_series_hash`, and `collect_execution_provenance` to public APIs; leaves deprecated re-export shims in `cycle006b.py` and `loader.py` so all 264 existing tests stay green.
2. **`trader/intelligence/data_contract.py` (`OHLCVEvidenceFrame`)**:
   - Frozen dataclass with IST tz-aware `ts_open_ist`, **`ts_close_ist` (`ts_event_ist`)**, and **`ts_init_ist`** (receipt/materialization time).
   - Universal visibility invariant: **`ts_event_ist <= ts_init_ist <= as_of_ist`** (never gates on `ts_open_ist <= as_of_ist`).
   - Unified `ProvenanceTier` enum (`VERIFIED`, `PARTIALLY_VERIFIED`, `UNVERIFIED`, `UNAVAILABLE`) with hard non-authoritative provider cap (`yahoo_finance_*` can NEVER emit `VERIFIED`).
   - Fail-closed missing-data contract: `None`/`NaN` + explicit `missing_reason` (`INSUFFICIENT_BARS`, `ZERO_SESSION_VOLUME`, `TIMEFRAME_UNAVAILABLE`, `STALE`) — **zero silent `0.0` imputation**.
3. **`trader/intelligence/registry.py` (`FeatureRegistry` & `FeatureSpec`)**:
   - Immutable `MappingProxyType` registry enforcing **one canonical `formula_variant` per `feature_id`** (`DuplicateFeatureRegistrationError` on collision).
   - Explicit named variants preserve exact historical parity (`atr_wilder_14` canonical + `atr_sma_14` explicit legacy; `rvol_inclusive_20` + `rvol_prior_nonzero_14` + `rvol_prior_10`; `rsi_wilder` with `zero_loss_mode="legacy_50"|"strict_100"`).
   - Every recursive feature declares `is_recursive=True` and `min_warmup_bars` for CYCLE-009C convergence gating.
4. **`trader/intelligence/mtf_aligner.py`**:
   - Fixes `15:15–15:30` closing 1h bar via dynamic `expected_bar_count` (`3` × 5m bars for Bar 7, `12` × 5m bars for Bars 1–6).
   - Aligns HTF→LTF strictly via `pd.merge_asof(..., direction="backward")` on explicit IST **`ts_close_ist`**.
5. **`trader/intelligence/bundle.py` (`FeatureSnapshot`, `EvidenceBundle`, `FEATURE_SCHEMA_VERSION = "1.0.0"`)**:
   - Read-only `EvidenceBundle` carrying `market_frames`, `feature_snapshots`, `regime_evidence`, `directional_inputs`, `execution_context`, mandatory `confidence_calibration_status = "UNCALIBRATED"`, `missing_or_stale_flags`, and deterministic `evidence_hash`.
   - **Consumer Rules**: `CentralDirectionalAgent`, `CandleTimingAgent`, `QuantResearchAgent` (`Cycle008WalkForwardEvaluator`), `MarketRegimeAgent`, and future agents consume the exact same read-only `EvidenceBundle` and **never** compute competing inline versions of canonical features. Legacy `trader/agents/` (`extra="forbid"` Pydantic) bridges via pure adapter functions.

---

## 4. Historical Parity & Migration Strategy

Full parity table is in `data/lab/cycle_009b_migration_parity_matrix.json`.
- **Golden Rule**: **Never overwrite frozen `CYCLE-001`..`CYCLE-008` reports or alter `CYCLE007_IMPLEMENTATION_MODULES` / `CYCLE008_IMPLEMENTATION_MODULES`.**
- **Parity Tolerances**: `atol=1e-12, rtol=1e-12` for all preserved variants (`atr_wilder_14`, `atr_sma_14`, `ema`, `rsi_wilder`, `rvol_*`, `candle_anatomy_v1`, `macd_12_26_9`); `atol=1e-10` for `session_vwap_ist`; `100% exact match` for discrete regime tiers.
- **When a Versioned Rerun IS Required**: Only when a future cycle (`CYCLE-010+`) intentionally opts into a formula change (e.g., migrating `CandleTimingAgent` from `atr_sma_14` → `atr_wilder_14`, opting into `rsi_wilder(zero_loss_mode="strict_100")`, or wiring `compute_supertrend` into `STRAT-004`).

---

## 5. Lookahead & Recursive Convergence Gate Specification (`CYCLE-009C`)

Target module: `trader/evaluation/bias_auditor.py` (do NOT implement in CYCLE-009B). Covers 7 mandatory test families:
1. **Bar-Index Lookahead Truncation & Perturbation**: IST calendar boundaries (`09:15`, `10:30`, `15:15`, `15:30`, post-holiday); 5× future perturbation & unclosed bar injection (`ts_open < as_of < ts_close`) must leave outputs `<= as_of` invariant (`atol=1e-12`).
2. **Cross-Sectional Lookahead**: Perturbing `S_2..S_m` future bars or dropping `S_2`'s current bar while keeping future bars must not alter `S_1` at `<= as_of`.
3. **Historical `ts_init` Cutoff**: Records with `ts_event <= as_of < ts_init` must be invisible; `ts_init < ts_event - 2s` or naive timestamps fail closed.
4. **Recursive Indicator Convergence**: Evaluate every `is_recursive=True` feature across `W ∈ [50, 100, 200, 500]` vs `W_ref=500` (`δ(200) <= 1e-4` for `period<=14`, `<= 5e-4` for `15..30`, `<= 5e-3` for `31..60`).
5. **Warm-Up Sensitivity**: Below `min_warmup_bars` → `None`/`WARMUP_INSUFFICIENT`; above `min_warmup_bars` → `score_drift_p95 <= 0.02`, `signal_flip_rate <= 1.0%`.
6. **Multi-Timeframe Backward-Only Alignment**: Afternoon truncation (`12:15`/`13:30`) invariance + `12:30` in-progress 1h bar trap + `15:15–15:30` short closing bar visibility check.
7. **Future News & Event Exclusion**: 4 adversarial event classes (`pub > as_of`, `pub <= as_of < col`, missing/naive timestamps, extreme sentiment) must produce bit-for-bit identical outputs (`atol=1e-12`).

**CI Tier Split**:
- **Fast Normal CI (`<15s`, `@pytest.mark.fast_ci`, hard PR blocker)**: Synthetic 15-session IST calendar + 5 intentionally biased fixture strategies (`shift(-1)`, global z-score, unshifted HTF merge, future `ts_init` news, non-converged `EMA(50)`) that MUST fail + 1 clean control strategy that MUST pass.
- **Longer Research Validation (`1–5m`, `@pytest.mark.research_slow`, promotion/nightly blocker)**: Full 196-day committed 1h sweep across all 5 `LAB_DEFAULT_UNIVERSE` symbols, `W ∈ [25..1000]` real-data convergence matrix, and full archive replay.

---

## 6. Risk Engine Separation & `CYCLE-009D` Scope

`DeterministicRiskEngine` (`trader/risk/engine.py`) remains **100% independent and authoritative** (zero imports from `trader/intelligence/`, `trader/agents/`, or `trader/timing/`).
- **3-State Machine (`RiskState`)**: `ACTIVE` (`09:15–14:45` entries) → `REDUCING` (`15:05 IST` wind-down: blocks `BUY`, explicitly **allows `SELL` exits** up to held qty — fixing the current `L91-129` exit-blocking bug) → `HALTED` (daily loss or intraday peak-to-trough MTM drawdown breach: blocks all entries, allows `force_reduce_only=True` square-offs). Reset only via `reset_daily_session()` on a new IST date.
- **Backwards Compatibility**: Keep `is_daily_halted` as a `@property` + `@is_daily_halted.setter` so `walk_forward.py:365,796` and `test_risk_engine.py:92` remain 100% green.
- **Hardening Controls**: Pending `CashReservationLedger` (thread-safe anti-double-spend across concurrent signals + fill/cancel release), intraday unrealized MTM peak-to-trough drawdown trigger, sliding-window `OrderRateThrottler`, naive-datetime fail-closed rejection, and SHA-256 hash-chained `RiskAuditEvent` log with `verify_audit_chain()`.

---

## 7. Phased Implementation Sequencing & Rollback Plans

| Cycle | Scope | Files Created / Modified | Migration Risks | Rollback Plan | Acceptance Criteria |
|---|---|---|---|---|---|
| **`CYCLE-009B`** | Shared Intelligence Layer (`trader/intelligence/`) | *Create*: `trader/intelligence/{__init__,parsing,data_contract,registry,mtf_aligner,bundle}.py`, `tests/intraday/test_cycle009b_*.py`. *Modify*: thin shims/delegations in `indicators.py`, `candle_intelligence.py`, `central_directional_agent.py`, `candle_timing_agent.py`, `walk_forward.py`, `cycle006b.py`, `loader.py`, `resampler.py`, `provenance_1h.py`. | Accidental Wilder vs SMA ATR drift; mutating `CYCLE007_IMPLEMENTATION_MODULES` list; CI missing `scipy`/`duckdb`. | Single `git revert`; purely additive package with preserved public signatures and frozen reports. | 0 private `_` cross-imports; `1e-12` parity on all preserved variants; `ts_close_ist <= ts_init_ist <= as_of_ist`; Yahoo Finance capped at `PARTIALLY_VERIFIED`; `15:15–15:30` 1h bar fixed; all 264+92 tests green. |
| **`CYCLE-009C`** | Lookahead & Recursive Convergence CI Gate | *Create*: `trader/evaluation/bias_auditor.py`, `tests/intraday/test_cycle009c_*.py`. *Modify*: `pyproject.toml` (register `fast_ci`, `research_slow` markers), `.github/workflows/intraday-lab-tests.yml`. | False positives on tight recursive warmup tolerances; CI runtime bloat. | `git revert` workflow + auditor; zero runtime trading impact. | 5/5 intentionally biased fixtures caught; 0 false positives on clean control & registered features; Fast CI gate `<15s`. |
| **`CYCLE-009D`** | Risk Engine Hardening | *Modify*: `trader/config.py`, `trader/risk/engine.py`, `trader/timing/cycle007_pipeline.py`, `trader/paper/cycle006b.py`, `trader/evaluation/walk_forward.py`, `trader/agents/autonomous_trader.py`. *Create*: `tests/intraday/test_cycle009d_risk_engine_hardening.py`. | Breaking `walk_forward.py:365,796` or `test_risk_engine.py:92` if `is_daily_halted` setter missing; naive datetime in legacy tests. | `git revert`; `RiskEvaluationResult` fields strictly additive. | `ACTIVE→REDUCING→HALTED` verified; 15:05 IST allows `SELL`/blocks `BUY`; thread-safe reservations; MTM DD halt; 4/4 SHA-256 audit tamper tests caught; 0 imports from `intelligence`/`agents`/`timing`. |

---

## 8. Readiness Classification & Recommended Next Implementation Task

- **Final Classification**: **`READY`**
- **Recommended Next Implementation Task**: **`CYCLE-009B — Shared Intelligence Layer Implementation`** (strictly scoped to `trader/intelligence/` + backward-compatible shims/delegations + `test_cycle009b_intelligence_parity_and_contracts.py`; do **not** mix CYCLE-009C CI gate or CYCLE-009D Risk Engine changes into the same PR).
- **Measurable Acceptance Criteria for `CYCLE-009B` Execution**:
  1. Zero private `_`-helper cross-imports across `trader/`.
  2. Numerical parity `atol=1e-12, rtol=1e-12` verified across all 5 committed `data/intraday/*_1h.parquet` files for every preserved variant (`atr_wilder_14`, `atr_sma_14`, `ema`, `session_vwap_ist`, `rsi_wilder`, `rvol_inclusive_20`, `rvol_prior_nonzero_14`, `candle_anatomy_v1`).
  3. Dual-timestamp causal invariant (`ts_close_ist <= ts_init_ist <= as_of_ist`), `15:15–15:30` 1h bar resampling fix, and Yahoo Finance `PARTIALLY_VERIFIED` hard cap enforced with unit tests.
  4. `CYCLE007_IMPLEMENTATION_MODULES`, `CYCLE008_IMPLEMENTATION_MODULES`, all `18` protected files, and `data/lab/sealed_holdout_registry.json` (`48 EXPOSED`, `0 AVAILABLE`, `0 SEALED`) remain 100% untouched.
  5. Full regression suite (`tests/intraday/` + `tests/phase6/`) and GitHub Actions CI (`intraday-lab-tests.yml`) pass with zero skipped or weakened tests.

**Stopped — awaiting explicit human approval before beginning `CYCLE-009B` implementation.**
