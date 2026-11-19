# BharatAlpha Phase 6 — Cost & Token Estimation

## 1. Unit Token Consumption per Stock Run

A complete multi-agent research evaluation of a single stock executes 9 agent calls (6 specialist analysts, Bull researcher, Bear researcher, Research Manager, Risk Manager):

| Agent Stage | Tier | Model | Est. Input Tokens | Est. Output Tokens | Calls | Total Input | Total Output |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Specialist Analysts (5 LLM, 1 Det.) | Quick | `gpt-4o-mini` | 1,500 | 400 | 5 | 7,500 | 2,000 |
| Bull Researcher | Quick | `gpt-4o-mini` | 2,500 | 600 | 1 | 2,500 | 600 |
| Bear Researcher | Quick | `gpt-4o-mini` | 2,500 | 600 | 1 | 2,500 | 600 |
| Research Manager | Deep | `gpt-4o` | 4,500 | 1,200 | 1 | 4,500 | 1,200 |
| Risk Manager | Deep | `gpt-4o` | 3,000 | 800 | 1 | 3,000 | 800 |
| **Total per Stock** | — | — | — | — | **9** | **20,000** | **5,200** |

---

## 2. Estimated Model Costs (Based on Standard Cloud Pricing)

- **Quick-Think Tier (`gpt-4o-mini` / `gemini-1.5-flash`):**
  - Input: $0.15 per 1M tokens
  - Output: $0.60 per 1M tokens
  - Cost per stock run: `(12,500 * $0.15/1M) + (3,200 * $0.60/1M) = $0.00188 + $0.00192 = $0.0038` (~₹0.32)
- **Deep-Think Tier (`gpt-4o` / `claude-3-5-sonnet`):**
  - Input: $2.50 per 1M tokens
  - Output: $10.00 per 1M tokens
  - Cost per stock run: `(7,500 * $2.50/1M) + (2,000 * $10.00/1M) = $0.01875 + $0.02000 = $0.03875` (~₹3.25)
- **Combined Cost per Stock:** **~$0.0425** (~₹3.57 INR)

---

## 3. Batch Budget Projections

| Run Scope | Stocks Evaluated | Frequency | Total Runs / Mo | Monthly USD | Monthly INR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pilot Universe** | 5 stocks | Daily EOD (22 trading days) | 110 | **$4.68** | **₹393** |
| **Nifty 50 Universe** | 50 stocks | Daily EOD (22 trading days) | 1,100 | **$46.75** | **₹3,927** |
| **Full Tracked (500)**| 500 stocks | Weekly review (4 weeks) | 2,000 | **$85.00** | **₹7,140** |

*Note: In local and CI/CD automated test runs, all LLM calls use `MockLLMClient`, resulting in $0.00 token cost.*
