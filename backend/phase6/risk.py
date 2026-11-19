"""Risk Manager (Phase 6): deterministic hard rules first, LLM rationale second.

Expected (documented) input shapes in the evidence ``data`` dict
---------------------------------------------------------------
``data['price_risk']`` (numbers; ``None``/missing -> UNKNOWN, never LOW)::

    volatility_ann_pct       annualised volatility, %
    var_95_1d_pct            1-day 95% historical VaR as a positive %
    max_drawdown_pct         trailing max drawdown magnitude, % (abs() is applied)
    avg_daily_turnover_cr    average daily traded value, INR crore
    expected_upside_pct / expected_downside_pct   for risk/reward (ratio)
    position_weight_pct      proposed portfolio weight (concentration); optional
``data['market_regime']``: ``{"label": "BULL|BEAR|SIDEWAYS|HIGH_VOLATILITY|CRISIS", "vix": float}``
``data['news_events']``: list of ``{"category","impact": HIGH|MEDIUM|LOW,
"sentiment": NEGATIVE|NEUTRAL|POSITIVE, "days_to_event": number|None, "imminent": bool}``;
an absent key -> event risk UNKNOWN, an empty list -> LOW.

All numeric cut-offs live in :data:`RISK_THRESHOLDS`. The LLM may only ADD rationale or make
the verdict stricter; the code takes the stricter verdict.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.phase6.agents_base import AgentContext, RiskAgent, RiskPackage
from backend.phase6.config import Phase6Config
from backend.phase6.contracts import RiskAssessment, RiskDimension
from backend.phase6.debate import (_call, _cfg, _dedupe, build_evidence_digest,  # noqa: F401
                                   completeness_ratio, quant_stance, unmatched_numbers)
from backend.phase6.llm import AgentTimeout, InvalidStructuredOutput, LLMProviderError
from backend.phase6.prompt_text_debate import prompt_text

# (medium_at, high_at); higher value = riskier unless "inverse" (lower = riskier).
RISK_THRESHOLDS: dict[str, dict[str, Any]] = {
    "volatility": {"metric": "volatility_ann_pct", "medium": 25.0, "high": 40.0},
    "downside": {"metric": "var_95_1d_pct", "medium": 2.5, "high": 4.5},
    "drawdown": {"metric": "max_drawdown_pct", "medium": 20.0, "high": 40.0},
    # liquidity: LOWER is riskier (INR crore average daily turnover)
    "liquidity": {"metric": "avg_daily_turnover_cr", "medium": 25.0, "high": 5.0, "inverse": True},
    # risk/reward: reward/risk ratio, LOWER is riskier
    "risk_reward": {"metric": "expected_upside_pct/expected_downside_pct", "medium": 2.0,
                    "high": 1.0, "inverse": True},
    "concentration": {"metric": "position_weight_pct", "medium": 5.0, "high": 10.0},
    "regime": {"stress_labels": ("HIGH_VOLATILITY", "CRISIS"), "vix_stress": 25.0},
    "event": {"imminent_days": 7},
    "reject_if_high_count_at_least": 2,
    "reject_if_unknown_count_at_least": 3,
    "reduce_if_unknown_count_at_least": 2,
}

# ASSUMPTIONS (not derived from data): used only to suggest a sizing, never to grade risk.
DEFAULT_POSITION_WEIGHT = {
    "label": "ASSUMPTION - not from user portfolio data",
    "base_weight_pct": 3.0,
    "reduce_factor": 0.5,
}

_ORDER = {"APPROVE": 0, "REDUCE": 1, "REJECT": 2}
_BUMP = {"LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "HIGH", "UNKNOWN": "UNKNOWN"}


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _grade(value: Optional[float], key: str, label: str) -> RiskDimension:
    t = RISK_THRESHOLDS[key]
    if value is None:
        return RiskDimension(level="UNKNOWN", note=f"{label} missing")
    if t.get("inverse"):
        lvl = "HIGH" if value < t["high"] else "MEDIUM" if value < t["medium"] else "LOW"
    else:
        lvl = "HIGH" if value >= t["high"] else "MEDIUM" if value >= t["medium"] else "LOW"
    return RiskDimension(level=lvl, note=f"{label}={value:g} (medium@{t['medium']}, high@{t['high']})")


def _event_risk(events: Any) -> RiskDimension:
    if not isinstance(events, list):
        return RiskDimension(level="UNKNOWN", note="news_events missing")
    win = RISK_THRESHOLDS["event"]["imminent_days"]
    level, note = "LOW", "no adverse events"
    for e in events:
        if not isinstance(e, dict):
            continue
        impact = str(e.get("impact", "")).upper()
        neg = str(e.get("sentiment", "")).upper() == "NEGATIVE"
        d = _num(e.get("days_to_event"))
        imminent = bool(e.get("imminent")) or (d is not None and d <= win)
        regulatory = str(e.get("category", "")).upper() in ("REGULATORY", "LITIGATION")
        if impact == "HIGH" and neg and (imminent or regulatory):
            return RiskDimension(level="HIGH", note=f"imminent/regulatory negative high-impact "
                                                    f"event: {e.get('category', '?')}")
        if impact == "HIGH" or (neg and impact == "MEDIUM"):
            level, note = "MEDIUM", f"notable event: {e.get('category', '?')}"
    return RiskDimension(level=level, note=note)


def assess_dimensions(data: dict[str, Any], quant: dict[str, Any]) -> dict[str, RiskDimension]:
    pr = data.get("price_risk")
    pr = pr if isinstance(pr, dict) else {}
    reg = data.get("market_regime")
    reg = reg if isinstance(reg, dict) else {}
    th = RISK_THRESHOLDS["regime"]
    stressed = (str(reg.get("label", "")).upper() in th["stress_labels"]
                or (_num(reg.get("vix")) or 0.0) >= th["vix_stress"])
    dd = _num(pr.get("max_drawdown_pct"))
    up, down = _num(pr.get("expected_upside_pct")), _num(pr.get("expected_downside_pct"))
    rr = up / down if up is not None and down not in (None, 0) else None
    dims = {
        "downside_risk": _grade(_num(pr.get("var_95_1d_pct")), "downside", "var_95_1d_pct"),
        "volatility_risk": _grade(_num(pr.get("volatility_ann_pct")), "volatility",
                                  "volatility_ann_pct"),
        "liquidity_risk": _grade(_num(pr.get("avg_daily_turnover_cr")), "liquidity",
                                 "avg_daily_turnover_cr"),
        "concentration_risk": _grade(_num(pr.get("position_weight_pct")), "concentration",
                                     "position_weight_pct"),
        "event_risk": _event_risk(data.get("news_events")),
        "drawdown_risk": _grade(abs(dd) if dd is not None else None, "drawdown",
                                "max_drawdown_pct"),
        "risk_reward": _grade(rr, "risk_reward", "reward/risk"),
    }
    if stressed:
        for k in ("downside_risk", "volatility_risk"):
            d = dims[k]
            dims[k] = RiskDimension(level=_BUMP[d.level], note=d.note + "; bumped: stress regime")
    return dims


def deterministic_verdict(dims: dict[str, RiskDimension], quant: dict[str, Any],
                          price_risk_present: bool) -> tuple[str, list[str]]:
    lv = {k: d.level for k, d in dims.items()}
    highs = [k for k, v in lv.items() if v == "HIGH"]
    unknown = [k for k, v in lv.items() if v == "UNKNOWN"]
    reasons: list[str] = []
    if len(highs) >= RISK_THRESHOLDS["reject_if_high_count_at_least"]:
        reasons.append(f"{len(highs)} HIGH risk dimensions: {', '.join(highs)}")
    if lv["liquidity_risk"] == "HIGH":
        reasons.append("liquidity risk HIGH")
    if lv["drawdown_risk"] == "HIGH" and quant_stance(quant) == "BEARISH":
        reasons.append("drawdown risk HIGH with bearish quantitative model")
    if len(unknown) >= RISK_THRESHOLDS["reject_if_unknown_count_at_least"]:
        reasons.append(f"{len(unknown)} UNKNOWN risk dimensions (insufficient risk data): "
                       f"{', '.join(unknown)}")
    if not price_risk_present:
        reasons.append("price_risk data missing")
    if reasons:
        return "REJECT", reasons
    if highs:
        return "REDUCE", [f"HIGH risk: {', '.join(highs)}"]
    if len(unknown) >= RISK_THRESHOLDS["reduce_if_unknown_count_at_least"]:
        return "REDUCE", [f"{len(unknown)} UNKNOWN risk dimensions"]
    return "APPROVE", []


class _RiskOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rationale: str = ""
    proposed_verdict: str = ""
    warnings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


RISK_TASK = ("TASK: produce JSON with rationale, proposed_verdict (APPROVE|REDUCE|REJECT), "
             "warnings, missing_information. You cannot soften the deterministic verdict.")


class RiskManagerAgent(RiskAgent):
    name = "risk_manager"
    version = "1.0.0"
    prompt_version = "v1"

    def __init__(self, config: Optional[Phase6Config] = None):
        self.config = config

    def run(self, pkg: RiskPackage, ctx: AgentContext) -> RiskAssessment:
        cfg = _cfg(self.config, ctx)
        self.prompt_version = cfg.prompt_versions.get(self.name, self.prompt_version)
        quant = pkg.quantitative_predictions
        dims = assess_dimensions(pkg.data, quant)
        pr_present = isinstance(pkg.data.get("price_risk"), dict) and bool(pkg.data["price_risk"])
        verdict, reasons = deterministic_verdict(dims, quant, pr_present)
        warnings: list[str] = []
        missing = [f"{k}: {d.note}" for k, d in dims.items() if d.level == "UNKNOWN"]
        if not isinstance(pkg.data.get("market_regime"), dict):
            missing.append("market_regime missing")
        # Narrative-independent prompt: deterministic facts only, no Bull/Bear/manager text.
        facts = {"symbol": pkg.symbol, "as_of": pkg.as_of,
                 "price_risk": pkg.data.get("price_risk"),
                 "market_regime": pkg.data.get("market_regime"),
                 "news_events": pkg.data.get("news_events"),
                 "quantitative_stance": quant_stance(quant),
                 "dimensions": {k: d.model_dump() for k, d in dims.items()},
                 "deterministic_verdict": verdict, "deterministic_reasons": reasons,
                 "thresholds": RISK_THRESHOLDS}
        import json
        user = f"RISK FACTS:\n{json.dumps(facts, sort_keys=True, default=str, indent=1)}\n\n{RISK_TASK}"
        llm_rationale = ""
        if ctx.llm is None:
            warnings.append("no LLM client: deterministic risk assessment only")
        else:
            try:
                out = _call(ctx, cfg, _RiskOut, prompt_text(self.name, self.prompt_version), user,
                            agent=self.name, tier="deep")
                llm_rationale = out.rationale
                warnings.extend(out.warnings)
                missing.extend(out.missing_information)
                prop = out.proposed_verdict.strip().upper()
                if prop in _ORDER and _ORDER[prop] > _ORDER[verdict]:
                    reasons.append(f"LLM risk review proposed stricter verdict {prop}")
                    verdict = prop
                elif prop in _ORDER and _ORDER[prop] < _ORDER[verdict]:
                    warnings.append(f"LLM proposed {prop} but deterministic {verdict} retained")
                bad = unmatched_numbers([out.rationale], json.dumps(facts, default=str))
                if bad:
                    warnings.append(f"numeric literals in LLM rationale not found in risk facts: {bad}")
            except (AgentTimeout, InvalidStructuredOutput, LLMProviderError) as e:
                warnings.append(f"LLM risk review failed ({type(e).__name__}); "
                                "deterministic verdict used")
        known = sum(1 for d in dims.values() if d.level != "UNKNOWN")
        w = DEFAULT_POSITION_WEIGHT
        factor = {"APPROVE": 1.0, "REDUCE": w["reduce_factor"], "REJECT": 0.0}[verdict]
        findings = [f"{k}: {d.level} - {d.note}" for k, d in dims.items()]
        findings.append(f"suggested position weight {w['base_weight_pct'] * factor:g}% "
                        f"[{w['label']}]")
        rationale = f"Deterministic verdict {verdict}. " + (
            "Reasons: " + "; ".join(reasons) + ". " if reasons else "") + llm_rationale
        return RiskAssessment(
            agent_name=self.name, agent_version=self.version, prompt_version=self.prompt_version,
            symbol=pkg.symbol, analysis_timestamp=pkg.analysis_timestamp,
            data_cutoff_timestamp=pkg.data_cutoff_timestamp,
            evidence_refs=list(pkg.evidence), findings=findings,
            confidence=known / len(dims), missing_information=_dedupe(missing),
            warnings=_dedupe(warnings), verdict=verdict, rationale=rationale.strip(),
            rejection_reasons=reasons if verdict == "REJECT" else [], **dims)
