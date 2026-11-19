# BharatAlpha Phase 6 — Data Contracts & Schemas

## 1. Schema Specifications (`backend/phase6/contracts.py`)

All schemas are implemented using Pydantic v2 with `model_config = ConfigDict(extra="forbid")`.

### `EvidenceRef`
Cryptographic provenance record for every datum ingested by the research graph:
```python
class EvidenceRef(BaseModel):
    source_id: str          # Format: "<section>:<SYMBOL>:<field>"
    dataset: str            # E.g., "docs_symbol_snapshot", "market_regime", "forward_validation"
    path: str               # File path relative to repository root
    record_key: str         # Key within dataset
    published_at: datetime | None   # Must be <= data_cutoff_timestamp
    retrieved_at: datetime          # Must be <= data_cutoff_timestamp
    sha256: str | None      # SHA-256 hash of the underlying source file
```

### `NumericClaim`
Anti-hallucination contract tying numeric assertions to evidence:
```python
class NumericClaim(BaseModel):
    name: str               # Metric name (e.g., "revenue_growth_pct")
    value: float            # Numerical value asserted
    evidence_id: str        # Must match an existing EvidenceRef.source_id
```

### `AgentReport`
Base output contract produced by all specialist and debater agents:
```python
class AgentReport(BaseModel):
    agent_name: str
    agent_version: str
    prompt_version: str
    symbol: str
    analysis_timestamp: datetime
    data_cutoff_timestamp: datetime
    evidence_refs: list[EvidenceRef]
    findings: list[str]
    positive_factors: list[str]
    negative_factors: list[str]
    confidence: float       # Bounded 0.0 to 1.0; capped by data completeness
    missing_information: list[str]
    warnings: list[str]
    numeric_claims: list[NumericClaim]
```

### `ResearchThesis` (Debater Output)
Extends `AgentReport` with structural debate elements:
- `side`: `"BULL"` or `"BEAR"`
- `thesis`: Core narrative investment thesis
- `catalysts`: List of identified drivers or catalysts
- `risks`: Downside vulnerabilities or counter-theses
- `challenges_to_other_side`: Explicit refutation of counter-arguments

### `RiskAssessment`
```python
class RiskDimension(BaseModel):
    level: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    note: str

class RiskAssessment(AgentReport):
    verdict: Literal["APPROVE", "REDUCE", "REJECT"]
    downside_risk: RiskDimension
    volatility_risk: RiskDimension
    liquidity_risk: RiskDimension
    concentration_risk: RiskDimension
    event_risk: RiskDimension
    drawdown_risk: RiskDimension
    risk_reward: RiskDimension
    rationale: str
    rejection_reasons: list[str]
```

### `ResearchManagerReport`
Consolidated institutional report produced at the end of the research pipeline:
- `investment_thesis`: Comprehensive executive thesis
- `bull_case`: Summary of bull arguments
- `bear_case`: Summary of bear arguments
- `fundamental_assessment`: Synthesis of financial statement quality
- `technical_assessment`: Synthesis of market action and momentum
- `news_assessment`: Synthesis of corporate and macro events
- `promoter_assessment`: Synthesis of management and governance signals
- `competitor_assessment`: Synthesis of peer and competitive landscape
- `quantitative_model_predictions`: Deep-copied read-only BharatAlpha predictions
- `risk_assessment`: Full nested RiskAssessment object
- `agent_disagreements`: List of detected contradictions across analysts/debaters
- `overall_conclusion`: `FAVOURABLE`, `NEUTRAL`, `UNFAVOURABLE`, `REJECTED_BY_RISK`, `INSUFFICIENT_EVIDENCE`

---

## 2. Ingestion Evidence Contract (`EvidencePackage.data`)

The loader strictly outputs the following normalized dictionary format:
```json
{
  "fundamental": {
    "revenue_cr": 1000.0,
    "profit_cr": 150.0,
    "revenue_growth_pct": 12.5,
    "evidence_ids": {
      "revenue_cr": "fundamental:RELIANCE:revenue_cr",
      "revenue_growth_pct": "fundamental:RELIANCE:revenue_growth_pct"
    }
  },
  "technical": { ... },
  "price_risk": { ... },
  "news": {
    "events": [ ... ],
    "quarantined_count": 0
  },
  "promoter": { ... },
  "competitor": { ... },
  "quantitative": { ... },
  "market_regime": { ... },
  "_missing": [ ... ],
  "_notes": [ ... ],
  "_cutoff": { ... }
}
```
