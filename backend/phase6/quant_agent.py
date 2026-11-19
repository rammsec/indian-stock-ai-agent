"""Deterministic Quantitative Analyst Agent for Phase 6.

Reads existing BharatAlpha model outputs from EvidencePackage without modifying them:
- 1D, 5D, 10D probabilities (prob_next_day_up, prob_5d_up, prob_10d_up)
- Expected returns (expected_next_day_return_pct, expected_5d_return_pct, etc.)
- Model agreement and confidence scores
- Model validation status from forward_validation.json (status, sample counts)
- Honest flagging of INSUFFICIENT_DATA when sample counts are insufficient
- Purely deterministic; NO LLM used. Never alters or retrains the frozen model.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from backend.phase6.agents_base import AnalystAgent, AgentContext, EvidencePackage
from backend.phase6.config import Phase6Config
from backend.phase6.contracts import AgentReport, NumericClaim


class QuantitativeAnalystAgent(AnalystAgent):
    name = "quantitative"
    version = "1.0.0"
    prompt_version = "deterministic_v1"

    def __init__(self, config: Optional[Phase6Config] = None):
        self.config = config

    def run(self, evidence_package: EvidencePackage, ctx: AgentContext) -> AgentReport:
        pkg = evidence_package
        q = pkg.data.get("quantitative")
        evidence_by_id = {ref.source_id: ref for ref in pkg.evidence}

        if not isinstance(q, dict) or not any(q.get(k) is not None for k in ("prob_next_day_up", "prob_5d_up", "prob_10d_up", "model_version")):
            return AgentReport(
                agent_name=self.name,
                agent_version=self.version,
                prompt_version=self.prompt_version,
                symbol=pkg.symbol,
                analysis_timestamp=pkg.analysis_timestamp,
                data_cutoff_timestamp=pkg.data_cutoff_timestamp,
                evidence_refs=[],
                findings=["NO DATA: Quantitative predictions unavailable for this symbol."],
                positive_factors=[],
                negative_factors=[],
                confidence=0.0,
                missing_information=["quantitative.predictions"],
                warnings=["NO DATA"],
                numeric_claims=[],
            )

        eids = q.get("evidence_ids") if isinstance(q.get("evidence_ids"), dict) else {}
        findings: list[str] = []
        positive_factors: list[str] = []
        negative_factors: list[str] = []
        warnings: list[str] = []
        missing_info: list[str] = []
        claims: list[NumericClaim] = []

        def _add_claim(claim_name: str, val: Any, field_key: str):
            if val is not None and isinstance(val, (int, float)):
                eid = eids.get(field_key)
                if eid and eid in evidence_by_id:
                    claims.append(NumericClaim(name=claim_name, value=float(val), evidence_id=eid))

        # 1. Model Identity
        mv = q.get("model_version") or "unknown"
        findings.append(f"Model version: {mv} (frozen pilot). Observation date: {q.get('observation_date')}")

        # 2. Probability & Predictions
        p1 = q.get("prob_next_day_up")
        p5 = q.get("prob_5d_up")
        p10 = q.get("prob_10d_up")

        _add_claim("prob_next_day_up", p1, "prob_next_day_up")
        _add_claim("prob_5d_up", p5, "prob_5d_up")
        _add_claim("prob_10d_up", p10, "prob_10d_up")

        exp1 = q.get("expected_next_day_return_pct")
        exp5 = q.get("expected_5d_return_pct")
        exp10 = q.get("expected_10d_return_pct")

        _add_claim("expected_next_day_return_pct", exp1, "expected_next_day_return_pct")
        _add_claim("expected_5d_return_pct", exp5, "expected_5d_return_pct")
        _add_claim("expected_10d_return_pct", exp10, "expected_10d_return_pct")

        if p1 is not None:
            findings.append(f"1D Horizon: P(Up)={p1:.1%}, Expected Return={exp1:+.2f}%" if exp1 is not None else f"1D Horizon: P(Up)={p1:.1%}")
            if p1 >= 0.55:
                positive_factors.append(f"1D quantitative directional bias is bullish (P(Up)={p1:.1%})")
            elif p1 <= 0.45:
                negative_factors.append(f"1D quantitative directional bias is bearish (P(Up)={p1:.1%})")
        else:
            missing_info.append("quantitative.prob_next_day_up")

        if p5 is not None:
            findings.append(f"5D Horizon: P(Up)={p5:.1%}, Expected Return={exp5:+.2f}%" if exp5 is not None else f"5D Horizon: P(Up)={p5:.1%}")
            if p5 >= 0.55:
                positive_factors.append(f"5D quantitative directional bias is bullish (P(Up)={p5:.1%})")
            elif p5 <= 0.45:
                negative_factors.append(f"5D quantitative directional bias is bearish (P(Up)={p5:.1%})")
        else:
            missing_info.append("quantitative.prob_5d_up")

        if p10 is not None:
            findings.append(f"10D Horizon: P(Up)={p10:.1%}, Expected Return={exp10:+.2f}%" if exp10 is not None else f"10D Horizon: P(Up)={p10:.1%}")
            if p10 >= 0.55:
                positive_factors.append(f"10D quantitative directional bias is bullish (P(Up)={p10:.1%})")
            elif p10 <= 0.45:
                negative_factors.append(f"10D quantitative directional bias is bearish (P(Up)={p10:.1%})")
        else:
            missing_info.append("quantitative.prob_10d_up")

        # 3. Model Agreement & Confidence
        conf = q.get("confidence_score")
        agree = q.get("model_agreement_score")
        _add_claim("confidence_score", conf, "confidence_score")
        _add_claim("model_agreement_score", agree, "model_agreement_score")

        if agree is not None:
            findings.append(f"Model sub-ensemble agreement score: {agree:.2f}")

        # 4. Action recommendation
        rec = q.get("recommended_action")
        if rec:
            findings.append(f"Model recommended action: {rec}")

        # 5. Validation evidence from forward validation
        val = q.get("validation")
        val_status_1d = "UNKNOWN"
        val_status_5d = "UNKNOWN"
        val_status_10d = "UNKNOWN"

        if isinstance(val, dict):
            for h, info in val.items():
                if isinstance(info, dict):
                    st = info.get("status", "UNKNOWN")
                    mc = info.get("matured_count", 0)
                    pc = info.get("pending_count", 0)
                    findings.append(f"Validation evidence {h}: status={st}, matured={mc}, pending={pc}")
                    if h in ("1D", "1d"):
                        val_status_1d = st
                    elif h in ("5D", "5d"):
                        val_status_5d = st
                    elif h in ("10D", "10d"):
                        val_status_10d = st

                    if st == "INSUFFICIENT_DATA" or (mc is not None and mc < 10):
                        warnings.append(f"{h} validation evidence has insufficient matured samples (matured={mc})")
        else:
            warnings.append("Forward validation evidence record is missing or incomplete")

        # 6. Overall Agent Confidence Calculation
        base_conf = float(conf) if conf is not None and isinstance(conf, (int, float)) else 0.5
        if val_status_1d == "INSUFFICIENT_DATA":
            agent_confidence = min(base_conf, 0.4)
        elif val_status_1d == "WARNING":
            agent_confidence = min(base_conf, 0.6)
        else:
            agent_confidence = min(base_conf, 0.85)

        used_eids = {c.evidence_id for c in claims}
        used_refs = [evidence_by_id[eid] for eid in used_eids if eid in evidence_by_id]

        return AgentReport(
            agent_name=self.name,
            agent_version=self.version,
            prompt_version=self.prompt_version,
            symbol=pkg.symbol,
            analysis_timestamp=pkg.analysis_timestamp,
            data_cutoff_timestamp=pkg.data_cutoff_timestamp,
            evidence_refs=used_refs,
            findings=findings,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            confidence=round(agent_confidence, 2),
            missing_information=missing_info,
            warnings=warnings,
            numeric_claims=claims,
        )
