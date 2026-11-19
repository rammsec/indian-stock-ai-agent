"""Bull / Bear researchers and the Research Manager (Phase 6).

Information flow
----------------
* Bull and Bear receive the IDENTICAL :class:`EvidencePackage`; each LLM user prompt is built
  ONLY from that package (``build_debate_user_prompt``) and is byte-identical for both sides.
  Only the *system* prompt differs (the side instruction).
* The manager receives a :class:`ManagerPackage` and is the only agent that sees both theses.

Contract notes / documented assumptions
---------------------------------------
* Completeness = fraction of the 6 analysts that reported successfully (present, not in
  ``missing_analysts``, no "NO DATA" marker in findings/warnings).
* Numeric-literal check: literals in prose are matched against numbers in the evidence digest
  (exact, rounding of a digest value, or x100 / x0.01 percent variants); bare integers 0..10
  are exempt. Unmatched literals are FLAGGED in ``warnings`` (not stripped).
* Quantitative stance parsing (``quant_stance``) reads ``signal``/``recommendation``/``action``
  strings or ``p_up``; ``validation_sufficient`` (bool) marks sufficient validation.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.phase6.agents_base import (ANALYST_NAMES, AgentContext, DebaterAgent,
                                        EvidencePackage, ManagerAgent, ManagerPackage)
from backend.phase6.config import Phase6Config
from backend.phase6.contracts import (AgentReport, EvidenceRef, NumericClaim,
                                      ResearchManagerReport, ResearchThesis)
from backend.phase6.llm import call_structured
from backend.phase6.prompt_text_debate import (SIDE_INSTRUCTIONS, DEBATER_COMMON, prompt_text)

# ---- thresholds (documented, tweakable) ---------------------------------
LOW_EVIDENCE_MISSING_ANALYSTS = 4        # >= this many unusable analysts -> no LLM call
LOW_EVIDENCE_CONFIDENCE = 0.1            # confidence of the low-evidence thesis
INSUFFICIENT_COMPLETENESS_BELOW = 0.5    # manager: completeness below -> INSUFFICIENT_EVIDENCE
FAVOURABLE_MIN_COMPLETENESS = 0.8        # manager: FAVOURABLE needs >= (5 of 6 analysts)
CLOSE_CONFIDENCE_GAP = 0.1               # bull/bear within this gap -> unresolved debate
ANALYST_CONFIDENCE_SPREAD = 0.4          # analyst confidence spread flagged as disagreement
ASSESSED_ANALYSTS = ("fundamental", "technical", "news", "promoter", "competitor")


# ---- helpers ---------------------------------------------------------------
def _is_no_data(rep: AgentReport) -> bool:
    return any("NO DATA" in s.upper() for s in list(rep.findings) + list(rep.warnings))


def unusable_analysts(analyst_reports: dict[str, AgentReport],
                      missing_analysts: Iterable[str] = ()) -> list[str]:
    missing = set(missing_analysts)
    return [n for n in ANALYST_NAMES
            if n in missing or n not in analyst_reports or _is_no_data(analyst_reports[n])]


def completeness_ratio(analyst_reports: dict[str, AgentReport],
                       missing_analysts: Iterable[str] = ()) -> float:
    """Fraction (0..1) of the six analysts that reported successfully."""
    return (len(ANALYST_NAMES) - len(unusable_analysts(analyst_reports, missing_analysts))) \
        / len(ANALYST_NAMES)


def analyst_sentiment(rep: AgentReport) -> str:
    p, n = len(rep.positive_factors), len(rep.negative_factors)
    return "BULLISH" if p > n else "BEARISH" if n > p else "NEUTRAL"


def quant_stance(preds: dict[str, Any]) -> str:
    """BULLISH | BEARISH | NEUTRAL | NO_TRADE | UNKNOWN from frozen quant predictions."""
    if not preds:
        return "UNKNOWN"
    for k in ("signal", "recommendation", "action"):
        v = preds.get(k)
        if isinstance(v, str) and v.strip():
            u = v.upper().replace("_", " ")
            if "NO TRADE" in u:
                return "NO_TRADE"
            if any(w in u for w in ("BUY", "LONG", "BULL", "UP")):
                return "BULLISH"
            if any(w in u for w in ("SELL", "SHORT", "BEAR", "DOWN")):
                return "BEARISH"
            return "NEUTRAL"
    p = preds.get("p_up")
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return "BULLISH" if p >= 0.55 else "BEARISH" if p <= 0.45 else "NEUTRAL"
    return "UNKNOWN"


def _all_refs(pkg: EvidencePackage) -> dict[str, EvidenceRef]:
    refs: dict[str, EvidenceRef] = {r.source_id: r for r in pkg.evidence}
    for rep in pkg.analyst_reports.values():
        for r in rep.evidence_refs:
            refs.setdefault(r.source_id, r)
    for attr in ("bull", "bear"):
        t = getattr(pkg, attr, None)
        if t is not None:
            for r in t.evidence_refs:
                refs.setdefault(r.source_id, r)
    return refs


def _report_digest(rep: AgentReport) -> dict[str, Any]:
    return {"findings": rep.findings, "positive_factors": rep.positive_factors,
            "negative_factors": rep.negative_factors, "confidence": rep.confidence,
            "missing_information": rep.missing_information, "warnings": rep.warnings,
            "numeric_claims": [c.model_dump() for c in rep.numeric_claims],
            "evidence_ids": [r.source_id for r in rep.evidence_refs]}


def build_evidence_digest(pkg: EvidencePackage) -> str:
    """Canonical text digest built ONLY from the package (deterministic)."""
    d: dict[str, Any] = {
        "symbol": pkg.symbol, "as_of": pkg.as_of,
        "data_cutoff": pkg.data_cutoff_timestamp.isoformat(),
        "input_hash": EvidencePackage.input_hash(
            EvidencePackage(**{k: getattr(pkg, k) for k in EvidencePackage.model_fields})),
        "evidence_ids": sorted(_all_refs(pkg)),
        "evidence": [{"id": r.source_id, "dataset": r.dataset, "path": r.path,
                      "record_key": r.record_key} for r in pkg.evidence],
        "data": pkg.data, "quantitative_predictions": pkg.quantitative_predictions,
        "analysts": {n: _report_digest(r) for n, r in sorted(pkg.analyst_reports.items())},
        "missing_analysts": sorted(pkg.missing_analysts),
    }
    for attr in ("bull", "bear"):
        t = getattr(pkg, attr, None)
        if t is not None:
            d[attr] = {**_report_digest(t), "thesis": t.thesis, "catalysts": t.catalysts,
                       "risks": t.risks, "challenges_to_other_side": t.challenges_to_other_side}
    return json.dumps(d, sort_keys=True, default=str, indent=1)


_NUM = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(text: str) -> list[tuple[float, int]]:
    out = []
    for m in _NUM.finditer(text):
        lit = m.group(0).replace(",", "")
        out.append((float(lit), len(lit.split(".")[1]) if "." in lit else 0))
    return out


def unmatched_numbers(prose: Iterable[str], digest: str) -> list[float]:
    """Numeric literals in ``prose`` not present in ``digest`` (see module docstring)."""
    known = [v for v, _ in _numbers(digest)]
    bad: list[float] = []
    for text in prose:
        for x, dec in _numbers(text):
            if dec == 0 and x <= 10:
                continue
            tol = 0.5 * 10 ** (-dec) + 1e-9
            if not any(abs(k - x) <= tol or abs(k * 100 - x) <= tol or abs(k / 100 - x) <= tol
                       for k in known):
                if x not in bad:
                    bad.append(x)
    return bad


def _dedupe(*lists: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for lst in lists:
        for s in lst:
            if isinstance(s, str) and s.strip():
                seen.setdefault(s, None)
    return list(seen)


def _cfg(agent_cfg, ctx: AgentContext) -> Phase6Config:
    return agent_cfg or getattr(ctx, "config", None) or Phase6Config()


def _call(ctx: AgentContext, cfg: Phase6Config, schema, system: str, user: str, *,
          agent: str, tier: str):
    if ctx.llm is None:
        raise RuntimeError(f"{agent}: no LLM client in AgentContext")
    return call_structured(ctx.llm, schema, system, user, model=cfg.model_for(tier), agent=agent,
                           timeout_s=cfg.timeout_s, max_retries=cfg.max_retries,
                           cost_tracker=ctx.cost_tracker)


# ---- Bull / Bear -----------------------------------------------------------
class _ClaimOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    value: float
    evidence_id: str


class _ThesisOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    thesis: str
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    challenges_to_other_side: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    evidence_ids: list[str] = Field(default_factory=list)
    numeric_claims: list[_ClaimOut] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


DEBATE_TASK = ("TASK: produce JSON with thesis, catalysts, risks, challenges_to_other_side, "
               "findings, positive_factors, negative_factors, confidence (0..1), evidence_ids "
               "(ids from the digest only), numeric_claims [{name,value,evidence_id}], "
               "missing_information, warnings.")


def build_debate_user_prompt(pkg: EvidencePackage) -> str:
    """User prompt from the package ONLY; identical for Bull and Bear."""
    return f"EVIDENCE DIGEST:\n{build_evidence_digest(pkg)}\n\n{DEBATE_TASK}"


class _Debater(DebaterAgent):
    version = "1.0.0"
    prompt_version = "v1"

    def __init__(self, config: Optional[Phase6Config] = None):
        self.config = config

    def system_prompt(self) -> str:
        return prompt_text(self.name, self.prompt_version)

    def run(self, pkg: EvidencePackage, ctx: AgentContext) -> ResearchThesis:
        cfg = _cfg(self.config, ctx)
        self.prompt_version = cfg.prompt_versions.get(self.name, self.prompt_version)
        comp = completeness_ratio(pkg.analyst_reports, pkg.missing_analysts)
        unusable = unusable_analysts(pkg.analyst_reports, pkg.missing_analysts)
        det_missing = _dedupe(
            [m for r in pkg.analyst_reports.values() for m in r.missing_information],
            [f"analyst '{n}' unavailable or NO DATA" for n in unusable])
        base = dict(agent_name=self.name, agent_version=self.version,
                    prompt_version=self.prompt_version, symbol=pkg.symbol,
                    analysis_timestamp=pkg.analysis_timestamp,
                    data_cutoff_timestamp=pkg.data_cutoff_timestamp)
        if len(unusable) >= LOW_EVIDENCE_MISSING_ANALYSTS:
            return ResearchThesis(
                **base, side=self.side,
                thesis=f"Insufficient evidence to build a {self.side} thesis.",
                confidence=min(LOW_EVIDENCE_CONFIDENCE, comp), missing_information=det_missing,
                warnings=[f"LOW EVIDENCE: {len(unusable)} of {len(ANALYST_NAMES)} analysts "
                          f"missing/NO DATA ({', '.join(unusable)}); LLM not called"])
        out = _call(ctx, cfg, _ThesisOut, self.system_prompt(), build_debate_user_prompt(pkg),
                    agent=self.name, tier="quick")
        refs = _all_refs(pkg)
        warnings = list(out.warnings)
        good_claims = []
        for c in out.numeric_claims:
            if c.evidence_id in refs:
                good_claims.append(c)
            else:
                warnings.append(f"stripped numeric claim '{c.name}': unknown evidence id "
                                f"'{c.evidence_id}'")
        cited = _dedupe([e for e in out.evidence_ids if e in refs],
                        [c.evidence_id for c in good_claims])
        for e in out.evidence_ids:
            if e not in refs:
                warnings.append(f"dropped citation of unknown evidence id '{e}'")
        digest = build_evidence_digest(pkg)
        prose = [out.thesis, *out.catalysts, *out.risks, *out.challenges_to_other_side,
                 *out.findings, *out.positive_factors, *out.negative_factors]
        for c in good_claims:
            if unmatched_numbers([str(c.value)], digest):
                warnings.append(f"numeric claim '{c.name}'={c.value} not found in evidence digest")
        bad = unmatched_numbers(prose, digest)
        if bad:
            warnings.append(f"numeric literals in prose not found in evidence digest: {bad}")
        conf = max(0.0, min(1.0, out.confidence))
        if conf > comp:
            warnings.append(f"confidence capped from {conf:.2f} to completeness {comp:.2f}")
            conf = comp
        return ResearchThesis(
            **base, side=self.side, thesis=out.thesis, catalysts=out.catalysts, risks=out.risks,
            challenges_to_other_side=out.challenges_to_other_side,
            evidence_refs=[refs[e] for e in cited], findings=out.findings,
            positive_factors=out.positive_factors, negative_factors=out.negative_factors,
            confidence=conf, missing_information=_dedupe(det_missing, out.missing_information),
            warnings=_dedupe(warnings),
            numeric_claims=[NumericClaim(**c.model_dump()) for c in good_claims])


class BullResearcherAgent(_Debater):
    name, side = "bull", "BULL"


class BearResearcherAgent(_Debater):
    name, side = "bear", "BEAR"


# ---- Research manager ------------------------------------------------------
class _ManagerOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    investment_thesis: str = ""
    bull_case: str = ""
    bear_case: str = ""
    fundamental_assessment: str = ""
    technical_assessment: str = ""
    news_assessment: str = ""
    promoter_assessment: str = ""
    competitor_assessment: str = ""
    agent_disagreements: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    proposed_conclusion: str = ""
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


MANAGER_TASK = ("TASK: produce JSON with investment_thesis, bull_case, bear_case, "
                "fundamental_assessment, technical_assessment, news_assessment, "
                "promoter_assessment, competitor_assessment, agent_disagreements, findings, "
                "positive_factors, negative_factors, confidence (0..1), proposed_conclusion, "
                "missing_information, warnings. Do NOT output quantitative predictions.")


def deterministic_disagreements(pkg: ManagerPackage) -> list[str]:
    out: list[str] = []
    sent = {n: analyst_sentiment(r) for n, r in pkg.analyst_reports.items()}
    names = sorted(sent)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if {sent[a], sent[b]} == {"BULLISH", "BEARISH"}:
                out.append(f"{a} ({sent[a]}) vs {b} ({sent[b]}) analysts disagree")
    # factor conflicts: same factor positive in one analyst, negative in another
    pos: dict[str, set[str]] = {}
    neg: dict[str, set[str]] = {}
    for n, r in pkg.analyst_reports.items():
        for f in r.positive_factors:
            pos.setdefault(f.strip().lower(), set()).add(n)
        for f in r.negative_factors:
            neg.setdefault(f.strip().lower(), set()).add(n)
    for f in sorted(set(pos) & set(neg)):
        if pos[f] != neg[f] or len(pos[f]) > 0:
            a, b = sorted(pos[f])[0], sorted(neg[f])[0]
            if a != b:
                out.append(f"factor conflict on '{f}': positive per {a}, negative per {b}")
    confs = [r.confidence for r in pkg.analyst_reports.values()]
    if len(confs) >= 2 and max(confs) - min(confs) >= ANALYST_CONFIDENCE_SPREAD:
        out.append(f"analyst confidence spread {min(confs):.2f}-{max(confs):.2f}")
    bull, bear = pkg.bull, pkg.bear
    qs = quant_stance(pkg.quantitative_predictions)
    if bull is not None and bear is not None:
        gap = bull.confidence - bear.confidence
        if abs(gap) < CLOSE_CONFIDENCE_GAP:
            out.append(f"Bull ({bull.confidence:.2f}) and Bear ({bear.confidence:.2f}) "
                       "confidences are close: debate unresolved")
        if qs == "BEARISH" and gap > 0:
            out.append("quantitative model BEARISH while Bull case is stronger")
        if qs == "BULLISH" and gap < 0:
            out.append("quantitative model BULLISH while Bear case is stronger")
    if qs == "NO_TRADE":
        out.append("quantitative model reports NO TRADE")
    return out


def enforce_conclusion(pkg: ManagerPackage, proposed: str) -> tuple[str, list[str]]:
    """Deterministic guardrail; returns (conclusion, reasons for any downgrade)."""
    comp = completeness_ratio(pkg.analyst_reports, pkg.missing_analysts)
    qs = quant_stance(pkg.quantitative_predictions)
    sufficient = pkg.quantitative_predictions.get("validation_sufficient") is not False
    bull, bear = pkg.bull, pkg.bear
    if comp < INSUFFICIENT_COMPLETENESS_BELOW:
        return "INSUFFICIENT_EVIDENCE", [f"completeness {comp:.2f} < {INSUFFICIENT_COMPLETENESS_BELOW}"]
    if bull is None or bear is None:
        return "INSUFFICIENT_EVIDENCE", ["Bull or Bear thesis missing"]
    valid = {"FAVOURABLE", "NEUTRAL", "UNFAVOURABLE", "INSUFFICIENT_EVIDENCE"}
    p = proposed if proposed in valid else ""
    reasons: list[str] = []
    if not p:  # no usable proposal: derive deterministically
        p = ("FAVOURABLE" if bull.confidence > bear.confidence + CLOSE_CONFIDENCE_GAP
             else "UNFAVOURABLE" if bear.confidence > bull.confidence + CLOSE_CONFIDENCE_GAP
             else "NEUTRAL")
        reasons.append("no valid LLM proposal; derived from Bull/Bear confidence")
    if p == "FAVOURABLE":
        if not bull.confidence > bear.confidence:
            reasons.append("Bull confidence not above Bear")
        if comp < FAVOURABLE_MIN_COMPLETENESS:
            reasons.append(f"completeness {comp:.2f} < {FAVOURABLE_MIN_COMPLETENESS}")
        if qs == "BEARISH":
            reasons.append("quantitative model bearish")
        if qs == "NO_TRADE" and sufficient:
            reasons.append("quantitative model NO TRADE with sufficient validation")
        if any(r for r in reasons if not r.startswith("no valid")):
            return "NEUTRAL", reasons
    if p == "UNFAVOURABLE" and not bear.confidence > bull.confidence:
        return "NEUTRAL", reasons + ["Bear confidence not above Bull"]
    return p, reasons


class ResearchManagerAgent(ManagerAgent):
    name = "research_manager"
    version = "1.0.0"
    prompt_version = "v1"

    def __init__(self, config: Optional[Phase6Config] = None):
        self.config = config

    def run(self, pkg: ManagerPackage, ctx: AgentContext) -> ResearchManagerReport:
        cfg = _cfg(self.config, ctx)
        self.prompt_version = cfg.prompt_versions.get(self.name, self.prompt_version)
        user = f"EVIDENCE DIGEST:\n{build_evidence_digest(pkg)}\n\n{MANAGER_TASK}"
        out = _call(ctx, cfg, _ManagerOut, prompt_text(self.name, self.prompt_version), user,
                    agent=self.name, tier="deep")
        comp = completeness_ratio(pkg.analyst_reports, pkg.missing_analysts)
        unusable = unusable_analysts(pkg.analyst_reports, pkg.missing_analysts)
        conclusion, reasons = enforce_conclusion(pkg, out.proposed_conclusion.strip().upper())
        warnings = list(out.warnings)
        if conclusion != out.proposed_conclusion.strip().upper():
            warnings.append(f"conclusion guardrail: LLM proposed "
                            f"'{out.proposed_conclusion}', enforced '{conclusion}' "
                            f"({'; '.join(reasons)})")
        digest = build_evidence_digest(pkg)
        prose = [out.investment_thesis, out.bull_case, out.bear_case, *out.findings,
                 out.fundamental_assessment, out.technical_assessment, out.news_assessment,
                 out.promoter_assessment, out.competitor_assessment]
        bad = unmatched_numbers(prose, digest)
        if bad:
            warnings.append(f"numeric literals in prose not found in evidence digest: {bad}")
        for t in (pkg.bull, pkg.bear):
            if t is not None:
                warnings.extend(t.warnings)
        for r in pkg.analyst_reports.values():
            warnings.extend(r.warnings)
        if not pkg.quantitative_predictions:
            warnings.append("quantitative predictions unavailable")
        missing = _dedupe(
            [m for r in pkg.analyst_reports.values() for m in r.missing_information],
            *(t.missing_information for t in (pkg.bull, pkg.bear) if t is not None),
            [f"analyst '{n}' unavailable or NO DATA" for n in unusable],
            out.missing_information)
        assess = {}
        for n in ASSESSED_ANALYSTS:
            if n in unusable:
                assess[n] = f"NOT AVAILABLE: {n} analyst missing or NO DATA"
            else:
                assess[n] = getattr(out, f"{n}_assessment") or "; ".join(
                    pkg.analyst_reports[n].findings) or "no findings"
        conf = max(0.0, min(1.0, out.confidence), 0.0)
        conf = min(conf, comp)
        if conclusion == "INSUFFICIENT_EVIDENCE":
            conf = min(conf, 0.3)
        refs = _all_refs(pkg)
        return ResearchManagerReport(
            agent_name=self.name, agent_version=self.version, prompt_version=self.prompt_version,
            symbol=pkg.symbol, analysis_timestamp=pkg.analysis_timestamp,
            data_cutoff_timestamp=pkg.data_cutoff_timestamp,
            evidence_refs=list(refs.values()), findings=out.findings,
            positive_factors=out.positive_factors, negative_factors=out.negative_factors,
            confidence=conf, missing_information=missing, warnings=_dedupe(warnings),
            investment_thesis=out.investment_thesis, bull_case=out.bull_case or
            (pkg.bull.thesis if pkg.bull else ""), bear_case=out.bear_case or
            (pkg.bear.thesis if pkg.bear else ""),
            fundamental_assessment=assess["fundamental"], technical_assessment=assess["technical"],
            news_assessment=assess["news"], promoter_assessment=assess["promoter"],
            competitor_assessment=assess["competitor"],
            quantitative_model_predictions=copy.deepcopy(pkg.quantitative_predictions),
            agent_disagreements=_dedupe(deterministic_disagreements(pkg), out.agent_disagreements),
            overall_conclusion=conclusion)
