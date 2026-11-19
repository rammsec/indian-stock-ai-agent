# BharatAlpha Phase 6 — Model & Provider Configuration

## 1. Multi-Tier Model Routing

Following the architecture of TauricResearch/TradingAgents, Phase 6 decouples model execution into two operational tiers:

| Tier | Target Roles | Default Model | Characteristic Requirements |
| :--- | :--- | :--- | :--- |
| **Quick-Think Tier** | 5 Specialist Analysts, Bull Researcher, Bear Researcher | `gpt-4o-mini` / `gemini-1.5-flash` | Low latency, high throughput, strong structured JSON compliance |
| **Deep-Think Tier** | Research Manager, Risk Manager | `gpt-4o` / `gemini-1.5-pro` / `claude-3-5-sonnet` | Deep contextual reasoning, synthesis across conflicting perspectives, rigorous risk evaluation |

---

## 2. Supported LLM Providers (`backend/phase6/providers.py`)

Configuration is controlled via environment variables or `Phase6Config`:

```bash
# Global Provider Selection
export PHASE6_PROVIDER="openai"            # Options: openai, google, anthropic, openai_compatible, mock

# Tier-Specific Model Selection
export PHASE6_QUICK_MODEL="gpt-4o-mini"
export PHASE6_DEEP_MODEL="gpt-4o"

# Independent Tier Provider Overrides
export PHASE6_QUICK_PROVIDER="openai"
export PHASE6_DEEP_PROVIDER="anthropic"

# Local / Self-Hosted Compatible Endpoints (vLLM, Ollama, LM Studio)
export PHASE6_PROVIDER="openai_compatible"
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:11434/v1"
export OPENAI_COMPATIBLE_API_KEY="optional_key"
```

---

## 3. Provider Security & Secrets Management

- **Zero Hardcoded Secrets:** No API keys are stored in the codebase or git history.
- **Provider Registry:**
  - `openai`: Uses `OPENAI_API_KEY`
  - `google`: Uses `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - `anthropic`: Uses `ANTHROPIC_API_KEY`
  - `openai_compatible`: Configurable via `OPENAI_COMPATIBLE_API_KEY`
- **Testing Safety:** In all automated tests, `MockLLMClient` is injected, ensuring that test runs are 100% deterministic, offline, and consume zero live tokens.
