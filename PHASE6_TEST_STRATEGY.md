# BharatAlpha Phase 6 — Test Strategy & Verification Matrix

## 1. Quality & Verification Philosophy

To ensure absolute reliability in investment research workflows, Phase 6 enforces non-vacuous, deterministic testing:
- **Zero Network Flakiness:** All unit and integration tests execute against `MockLLMClient` or deterministic rule engines.
- **Strict Anti-Hallucination Testing:** Explicit tests verify that agents reject or strip invented numbers and fail when required evidence is missing.
- **Frozen Baseline Immutability:** Pre- and post-execution checks verify that no production files or datasets have been altered.

---

## 2. Test Suite Organization

| Test Module | Coverage Scope | Test Count | Status |
| :--- | :--- | :---: | :---: |
| `tests/phase6/test_core_contracts.py` | Schema validation, timezone awareness, cutoff timestamps, numeric claim citation | 8 | PASSED |
| `tests/phase6/test_core_llm_cost_prompts.py` | Structured call retries, timeout exceptions, cost tracker arithmetic, prompt hashing | 9 | PASSED |
| `tests/phase6/test_core_graph.py` | Graph state machine, failure isolation, Bull/Bear identical input hash, checkpoint resume | 12 | PASSED |
| `tests/phase6/test_core_decision_log_frozen.py`| Hash-chained decision log, tamper detection, duplicate-run idempotency, frozen guard | 11 | PASSED |
| `tests/phase6/test_analysts.py` | 5 specialist analysts, deterministic derivations, anti-hallucination number checks, missing data | 46 | PASSED |
| `tests/phase6/test_evidence_quant_runner.py` | Point-in-time evidence loader, quantitative analyst deterministic report, runner pipeline | 3 | PASSED |
| `tests/phase6/test_debate_risk.py` | Bull/Bear prompt isolation, Research Manager synthesis, Risk Manager hard veto | 3 | PASSED |
| **Phase 6 Test Suite Total** | **Dedicated Phase 6 multi-agent research tests** | **92** | **100% PASSED** |

---

## 3. Full Repository Regression Matrix

In addition to the 92 Phase 6 tests, the entire existing repository test suite (Phases 1–5, audits, and remediations) was executed in the Phase 6 worktree:
- **Baseline Tests:** 148 passed
- **Phase 6 Tests:** 92 passed
- **Total Combined Tests:** **240 passed, 0 failed** in `57.83s`
