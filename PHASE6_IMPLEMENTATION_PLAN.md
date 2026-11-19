# BharatAlpha Phase 6 — Implementation Plan

## 1. Workstream Breakdown & Staging

The Phase 6 initiative is organized into distinct, modular workstreams executed in the isolated branch `feature/phase6-multi-agent-intelligence`:

| Workstream | Modules | Key Responsibilities | Isolation Boundary |
| :--- | :--- | :--- | :--- |
| **WS1: Core Contracts & State** | `contracts.py`, `state.py`, `agents_base.py` | Pydantic v2 schemas, typed state, immutable EvidencePackage | Read-only; zero mutation of existing schemas |
| **WS2: Orchestration & Checkpointing** | `graph.py`, `decision_log.py`, `frozen_guard.py` | DAG state machine, thread isolation, hash-chained log, frozen checks | Operates inside `data/phase6/` namespace only |
| **WS3: LLM & Provider Layer** | `llm.py`, `providers.py`, `config.py`, `cost.py`, `prompts.py` | Configurable multi-provider routing (OpenAI, Gemini, Anthropic), mock testing, token cost tracking | Environment variables only; no hardcoded keys |
| **WS4: Specialist Analysts** | `analysts.py`, `quant_agent.py`, `evidence.py` | 6 domain analysts (Fundamental, Technical, News, Promoter, Competitor, Quant) | Reads docs/data & data/processed read-only |
| **WS5: Debate & Synthesis** | `debate.py`, `prompt_text_debate.py` | Bull & Bear researchers with zero cross-visibility, Research Manager synthesis | Operates strictly on EvidencePackage digest |
| **WS6: Risk Management** | `risk.py` | 7-dimension risk assessment with algorithmic veto authority | Independent of narrative bias |
| **WS7: Runner & Shadow Evaluation** | `runner.py` | Pilot stock pipeline execution, report generation, integrity verification | Outputs to `data/phase6/reports/` and `decisions/` |

---

## 2. Integration & Deployment Milestones

1. **Stage 1: Core Framework (Completed)**
   - Pydantic contracts, mock LLM client, prompt registry, graph orchestrator, hash-chained decision log.
   - Comprehensive test suite in `tests/phase6/test_core_*.py` (40 passed).

2. **Stage 2: Specialist Agents & Evidence Pipeline (Completed)**
   - Point-in-time `load_evidence()` connecting docs/data JSONs, processed Parquet, and forward validation outputs.
   - 5 deterministic/LLM analysts with anti-hallucination number verification and completeness caps.
   - Deterministic `QuantitativeAnalystAgent` reading BharatAlpha predictions.
   - Unit tests in `tests/phase6/test_analysts.py` and `tests/phase6/test_evidence_quant_runner.py` (49 passed).

3. **Stage 3: Debate & Risk Layer (Completed)**
   - Bull & Bear debate agents with strict prompt isolation.
   - Research Manager with algorithmic guardrails and disagreement detection.
   - Risk Manager with hard thresholds and rejection triggers.
   - Unit tests in `tests/phase6/test_debate_risk.py` (3 passed).

4. **Stage 4: Shadow Evaluation across Pilot Universe (Completed)**
   - Shadow research executed across 5 pilot stocks: RELIANCE, TCS, TATAELXSI, ITC, BIRLACABLE.
   - Generated reports under `data/phase6/reports/2026-11-19/`.
   - Verified frozen model integrity (`frozen_guard.py`: 41 frozen files checked, 0 changed).

5. **Stage 5: Final Verification & PR Generation**
   - Full test suite execution across the entire repository.
   - Generation of pull request and controlled merge following user authorization.
