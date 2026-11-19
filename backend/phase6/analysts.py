"""Five concrete analyst agents with code-enforced anti-hallucination guards.

Pipeline per agent (see module-level helpers):
1. read ONLY its own section of ``EvidencePackage.data``;
2. deterministically compute missing_information, completeness and derived facts (each fact
   carries numeric claims that cite section evidence ids);
3. if the section is entirely missing -> NO DATA report, the LLM is NOT called;
4. otherwise send a digest to the quick-think model via ``call_structured``;
5. POST-VALIDATE: strip unknown/unsupported numeric claims, scan prose for fabricated
   numbers, force-merge deterministic missing_information, cap confidence by completeness,
   override timestamps with the package's, refuse structurally invalid output.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.phase6.agents_base import AgentContext, AnalystAgent, EvidencePackage
from backend.phase6.config import Phase6Config
from backend.phase6.contracts import AgentReport, EvidenceRef, NumericClaim
from backend.phase6.llm import call_structured
from backend.phase6.prompt_text import ANALYST_PROMPTS_V1, ANALYST_PROMPTS_V1_SHA256

AGENT_VERSION = "1.0.0"
NO_DATA_CONFIDENCE = 0.1
FABRICATION_PENALTY = 0.5


class AnalystOutputRejected(Exception):
    """LLM output parsed as JSON but is structurally unacceptable (e.g. no findings)."""


# ----------------------------------------------------------------------------------------
# lenient LLM output schema (strict AgentReport validation happens on the FINAL report so
# that unknown evidence ids can be stripped instead of failing the whole call)
# ----------------------------------------------------------------------------------------
class LLMClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    value: float
    evidence_id: str


class AnalystLLMOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    findings: list[str] = Field(default_factory=list)
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    numeric_claims: list[LLMClaim] = Field(default_factory=list)


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def num(v: Any) -> Optional[float]:
    """Finite number or None (bools / strings / NaN are treated as missing)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            continue
    return out


def _close(x: float, a: float) -> bool:
    """Rounding-tolerant equality (sign-insensitive)."""
    x, a = abs(x), abs(a)
    return (abs(x - a) <= max(0.051, 0.01 * a) or x == round(a) or x == round(a, 1)
            or x == round(a, 2) or x == math.floor(a) or x == math.ceil(a))


def number_allowed(x: float, allowed: list[float]) -> bool:
    return any(_close(x, a) for a in allowed)


@dataclass
class Fact:
    text: str
    claims: list[tuple[str, float, Optional[str]]] = field(default_factory=list)


def _present(v: Any) -> bool:
    return v is not None


class _DeterministicAnalyst(AnalystAgent):
    """Template-method base; subclasses define slots + facts."""

    version = AGENT_VERSION
    prompt_version = f"v1+sha256:{ANALYST_PROMPTS_V1_SHA256[:16]}"
    FIELDS: tuple[str, ...] = ()
    EXTRA_FIELDS: tuple[tuple[str, str], ...] = ()   # (section, field) context fields

    def __init__(self, config: Optional[Phase6Config] = None):
        self.config = config

    # ---- section access --------------------------------------------------------------
    def section(self, pkg: EvidencePackage, key: Optional[str] = None) -> Optional[dict]:
        s = pkg.data.get(key or self.name)
        return s if isinstance(s, dict) else None

    @staticmethod
    def eid(pkg: EvidencePackage, sec_key: str, fld: str) -> Optional[str]:
        s = pkg.data.get(sec_key)
        ids = s.get("evidence_ids") if isinstance(s, dict) else None
        return ids.get(fld) if isinstance(ids, dict) else None

    # ---- hooks -----------------------------------------------------------------------
    def slots(self, pkg: EvidencePackage) -> list[tuple[str, bool]]:
        sec = self.section(pkg) or {}
        out = [(f"{self.name}.{f}", num(sec.get(f)) is not None or
                (not isinstance(sec.get(f), (int, float)) and _present(sec.get(f))
                 and f in self._non_numeric()))
               for f in self.FIELDS]
        for s, f in self.EXTRA_FIELDS:
            d = self.section(pkg, s) or {}
            out.append((f"{s}.{f}", _present(d.get(f))))
        return out

    def _non_numeric(self) -> tuple[str, ...]:
        return ()

    def has_data(self, pkg: EvidencePackage) -> bool:
        sec = self.section(pkg)
        return bool(sec) and any(_present(sec.get(f)) for f in self.FIELDS)

    def facts(self, pkg: EvidencePackage) -> tuple[list[Fact], list[str]]:
        raise NotImplementedError

    def evidence_ids(self, pkg: EvidencePackage) -> list[str]:
        ids: list[str] = []
        for s in (self.name,) + tuple(dict.fromkeys(s for s, _ in self.EXTRA_FIELDS)):
            d = self.section(pkg, s) or {}
            e = d.get("evidence_ids")
            if isinstance(e, dict):
                ids += [v for v in e.values() if isinstance(v, str)]
        return ids

    def raw_view(self, pkg: EvidencePackage) -> dict:
        out = {k: v for k, v in (self.section(pkg) or {}).items() if k != "evidence_ids"}
        for s in dict.fromkeys(s for s, _ in self.EXTRA_FIELDS):
            out[f"_{s}"] = {k: v for k, v in (self.section(pkg, s) or {}).items()
                            if k != "evidence_ids"}
        return out

    # ---- run -------------------------------------------------------------------------
    def run(self, evidence_package: EvidencePackage, ctx: AgentContext) -> AgentReport:
        pkg = evidence_package
        by_id = {r.source_id: r for r in pkg.evidence}
        slots = self.slots(pkg)
        missing = [lbl for lbl, ok in slots if not ok]
        completeness = (sum(ok for _, ok in slots) / len(slots)) if slots else 0.0
        if not self.has_data(pkg):
            return self._no_data(pkg, [f"{self.name}: entire section missing"] + missing)

        facts, det_warnings = self.facts(pkg)
        det_claims: list[NumericClaim] = []
        for f in facts:
            for n, v, e in f.claims:
                if e and e in by_id:
                    det_claims.append(NumericClaim(name=n, value=v, evidence_id=e))
                else:
                    det_warnings.append(f"claim '{n}' omitted: evidence id "
                                        f"{e!r} not in evidence package")
        agent_allowed_ids = set(self.evidence_ids(pkg))
        digest = {
            "symbol": pkg.symbol, "as_of": pkg.as_of, "agent": self.name,
            "deterministic_facts": [f.text for f in facts],
            "deterministic_numeric_claims": [c.model_dump() for c in det_claims],
            "missing_information": missing,
            "completeness": round(completeness, 4),
            "allowed_evidence_ids": sorted(agent_allowed_ids),
            "section": self.raw_view(pkg),
        }
        digest_json = json.dumps(digest, sort_keys=True, default=str)
        cfg = self.config or ctx.config or Phase6Config()
        out = call_structured(
            ctx.llm, AnalystLLMOutput, ANALYST_PROMPTS_V1[self.name],
            "DIGEST:\n" + digest_json,
            model=cfg.model_for("quick"), agent=self.name, timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries, cost_tracker=ctx.cost_tracker)
        return self._finalize(pkg, by_id, out, facts, det_claims, det_warnings, missing,
                              completeness, digest_json)

    # ---- post-validation -------------------------------------------------------------
    def _base(self, pkg: EvidencePackage, **kw) -> dict:
        return dict(agent_name=self.name, agent_version=self.version,
                    prompt_version=self.prompt_version, symbol=pkg.symbol,
                    analysis_timestamp=pkg.analysis_timestamp,
                    data_cutoff_timestamp=pkg.data_cutoff_timestamp, **kw)

    def _no_data(self, pkg: EvidencePackage, missing: list[str]) -> AgentReport:
        return AgentReport(**self._base(
            pkg, findings=[f"NO DATA: {self.name} section unavailable; no analysis performed"],
            confidence=NO_DATA_CONFIDENCE, missing_information=missing,
            warnings=["NO DATA"]))

    def _finalize(self, pkg, by_id, out: AnalystLLMOutput, facts, det_claims, det_warnings,
                  missing, completeness, digest_json) -> AgentReport:
        if not any(s.strip() for s in out.findings):
            raise AnalystOutputRejected(f"{self.name}: LLM returned no findings")
        warnings = list(det_warnings)
        allowed = extract_numbers(digest_json)

        # numeric claims: drop unknown evidence ids / unsupported values
        claims = list(det_claims)
        seen = {(c.name, c.value, c.evidence_id) for c in claims}
        for c in out.numeric_claims:
            if c.evidence_id not in by_id:
                warnings.append(f"stripped LLM numeric claim '{c.name}': unknown evidence_id "
                                f"'{c.evidence_id}'")
            elif not number_allowed(c.value, allowed):
                warnings.append(f"stripped LLM numeric claim '{c.name}': value {c.value} not "
                                "present in evidence digest")
            elif (c.name, c.value, c.evidence_id) not in seen:
                seen.add((c.name, c.value, c.evidence_id))
                claims.append(NumericClaim(**c.model_dump()))

        # fabricated numbers in prose
        fabricated: list[float] = []
        for text in out.findings + out.positive_factors + out.negative_factors:
            for x in extract_numbers(text):
                if not number_allowed(x, allowed) and x not in fabricated:
                    fabricated.append(x)
        confidence = min(out.confidence, completeness)
        if fabricated:
            warnings.append("possible fabricated numbers in LLM text not found in evidence: "
                            + ", ".join(f"{x:g}" for x in fabricated))
            confidence *= FABRICATION_PENALTY

        # missing_information: deterministic items always present, LLM extras appended
        merged = list(missing) + [m for m in out.missing_information if m not in missing]

        ref_ids = list(dict.fromkeys(
            [i for i in self.evidence_ids(pkg) if i in by_id]
            + [c.evidence_id for c in claims] + self.extra_ref_ids(pkg, by_id)))
        refs: list[EvidenceRef] = [by_id[i] for i in ref_ids]
        warnings += [w for w in out.warnings if w not in warnings]
        return AgentReport(**self._base(
            pkg, evidence_refs=refs,
            findings=[f"[calc] {f.text}" for f in facts] + out.findings,
            positive_factors=out.positive_factors, negative_factors=out.negative_factors,
            confidence=max(0.0, min(1.0, confidence)), missing_information=merged,
            warnings=warnings, numeric_claims=claims))

    def extra_ref_ids(self, pkg: EvidencePackage, by_id: dict) -> list[str]:
        return []


# ----------------------------------------------------------------------------------------
# Fundamental
# ----------------------------------------------------------------------------------------
def _level(v: float, hi: float, lo: float, hi_t: str, mid_t: str, lo_t: str) -> str:
    return hi_t if v >= hi else (lo_t if v < lo else mid_t)


class FundamentalAnalystAgent(_DeterministicAnalyst):
    name = "fundamental"
    FIELDS = ("revenue_cr", "profit_cr", "revenue_growth_pct", "profit_growth_pct", "opm_pct",
              "net_margin_pct", "operating_cash_flow_cr", "free_cash_flow_cr", "debt_to_equity",
              "current_ratio", "interest_coverage", "pe", "pb", "roe_pct", "roce_pct",
              "piotroski_score")

    def facts(self, pkg):
        sec = self.section(pkg)
        out: list[Fact] = []

        def add(fld: str, fn):
            v = num(sec.get(fld))
            if v is None:
                return
            out.append(Fact(fn(v), [(fld, v, self.eid(pkg, "fundamental", fld))]))

        def sign(v, up="growing", dn="declining"):
            return up if v > 0 else (dn if v < 0 else "flat")

        add("revenue_growth_pct", lambda v: f"revenue growth {v:g}% ({sign(v)})")
        add("profit_growth_pct", lambda v: f"profit growth {v:g}% ({sign(v)})")
        add("opm_pct", lambda v: f"operating margin {v:g}% "
            f"({_level(v, 20, 10, 'high', 'moderate', 'low')})")
        add("net_margin_pct", lambda v: f"net margin {v:g}%")
        add("operating_cash_flow_cr", lambda v: f"operating cash flow {v:g} cr "
            f"({'positive' if v > 0 else 'non-positive'})")
        add("free_cash_flow_cr", lambda v: f"free cash flow {v:g} cr "
            f"({'positive' if v > 0 else 'non-positive'})")
        add("debt_to_equity", lambda v: f"debt-to-equity {v:g} "
            f"({_level(v, 1.5, 0.5, 'high leverage', 'moderate leverage', 'low leverage')})")
        add("current_ratio", lambda v: f"current ratio {v:g} "
            f"({'weak liquidity' if v < 1 else 'adequate liquidity'})")
        add("interest_coverage", lambda v: f"interest coverage {v:g}x "
            f"({_level(v, 5, 2, 'comfortable', 'adequate', 'weak')})")
        add("pe", lambda v: f"PE {v:g}")
        add("pb", lambda v: f"PB {v:g}")
        add("roe_pct", lambda v: f"ROE {v:g}% ({_level(v, 15, 8, 'strong', 'moderate', 'weak')})")
        add("roce_pct", lambda v: f"ROCE {v:g}% "
            f"({_level(v, 15, 8, 'strong', 'moderate', 'weak')})")
        add("piotroski_score", lambda v: f"Piotroski score {v:g} "
            f"({_level(v, 7, 4, 'strong', 'average', 'weak')})")
        return out, []


# ----------------------------------------------------------------------------------------
# Technical
# ----------------------------------------------------------------------------------------
class TechnicalAnalystAgent(_DeterministicAnalyst):
    name = "technical"
    FIELDS = ("close", "rsi_14", "macd_hist", "sma_20", "sma_50", "sma_200", "atr_pct",
              "return_1m_pct", "return_3m_pct", "volume_surge_ratio", "support", "resistance",
              "adx", "trend")
    EXTRA_FIELDS = (("market_regime", "regime"), ("market_regime", "india_vix_close"))

    def slots(self, pkg):
        sec = self.section(pkg) or {}
        out = []
        for f in self.FIELDS:
            v = sec.get(f)
            out.append((f"technical.{f}", _present(v) if f == "trend" else num(v) is not None))
        for s, f in self.EXTRA_FIELDS:
            v = (self.section(pkg, s) or {}).get(f)
            out.append((f"{s}.{f}", _present(v) if f == "regime" else num(v) is not None))
        return out

    def facts(self, pkg):
        s = self.section(pkg)
        e = lambda f: self.eid(pkg, "technical", f)  # noqa: E731
        out: list[Fact] = []
        close = num(s.get("close"))
        if close is not None:
            out.append(Fact(f"close {close:g}", [("close", close, e("close"))]))
        for f in ("sma_20", "sma_50", "sma_200"):
            m = num(s.get(f))
            if m is not None and close is not None:
                pos = "above" if close > m else ("below" if close < m else "at")
                out.append(Fact(f"close is {pos} {f.upper()} ({m:g})", [(f, m, e(f))]))
        s50, s200 = num(s.get("sma_50")), num(s.get("sma_200"))
        if None not in (close, s50, s200):
            al = ("bullish (close > SMA50 > SMA200)" if close > s50 > s200 else
                  "bearish (close < SMA50 < SMA200)" if close < s50 < s200 else "mixed")
            out.append(Fact(f"MA alignment: {al}"))
        rsi = num(s.get("rsi_14"))
        if rsi is not None:
            z = "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral")
            out.append(Fact(f"RSI(14) {rsi:g} zone: {z}", [("rsi_14", rsi, e("rsi_14"))]))
        mh = num(s.get("macd_hist"))
        if mh is not None:
            out.append(Fact(f"MACD histogram {mh:g} ({'positive' if mh > 0 else 'non-positive'}"
                            " momentum)", [("macd_hist", mh, e("macd_hist"))]))
        sup, res = num(s.get("support")), num(s.get("resistance"))
        if close and sup is not None and sup > 0:
            d = round((close - sup) / sup * 100, 2)
            t = (f"close is {d:g}% above support {sup:g}" if d >= 0 else
                 f"close is {abs(d):g}% below support {sup:g} (support broken)")
            out.append(Fact(t, [("pct_above_support", d, e("support")), ("support", sup, e("support"))]))
        if close and res is not None:
            d = round((res - close) / close * 100, 2)
            t = (f"resistance {res:g} is {d:g}% above close" if d >= 0 else
                 f"close is {abs(d):g}% above resistance {res:g} (breakout)")
            out.append(Fact(t, [("pct_to_resistance", d, e("resistance")),
                                ("resistance", res, e("resistance"))]))
        for f, label in (("atr_pct", "ATR%"), ("return_1m_pct", "1m return %"),
                         ("return_3m_pct", "3m return %")):
            v = num(s.get(f))
            if v is not None:
                out.append(Fact(f"{label} {v:g}", [(f, v, e(f))]))
        vs = num(s.get("volume_surge_ratio"))
        if vs is not None:
            out.append(Fact(f"volume surge ratio {vs:g} "
                            f"({'elevated' if vs >= 1.5 else 'normal/low'})",
                            [("volume_surge_ratio", vs, e("volume_surge_ratio"))]))
        adx = num(s.get("adx"))
        if adx is not None:
            out.append(Fact(f"ADX {adx:g} ({'trending' if adx >= 25 else 'weak/no trend'})",
                            [("adx", adx, e("adx"))]))
        if _present(s.get("trend")):
            out.append(Fact(f"trend label: {s['trend']}"))
        mr = self.section(pkg, "market_regime") or {}
        if _present(mr.get("regime")):
            out.append(Fact(f"market regime: {mr['regime']}"))
        vix = num(mr.get("india_vix_close"))
        if vix is not None:
            out.append(Fact(f"India VIX close {vix:g}",
                            [("india_vix_close", vix, self.eid(pkg, "market_regime",
                                                               "india_vix_close"))]))
        return out, []


# ----------------------------------------------------------------------------------------
# News
# ----------------------------------------------------------------------------------------
NEWS_CATEGORIES = ("company_news", "promoter_management", "industry_sector", "competitor",
                   "regulatory_government", "global_macro")
_GENERIC_RE = re.compile(
    r"(share price|stock price|stock quote|live price|price today|price chart|"
    r"live updates?|live stock|today'?s? (?:stock|share)|nse/bse|quote,? chart)", re.I)
EVENT_SLOT_FIELDS = ("category", "sentiment", "impact_magnitude", "publication_timestamp",
                     "effective_trading_date", "evidence_id")


def is_generic_headline(h: Any) -> bool:
    return isinstance(h, str) and bool(_GENERIC_RE.search(h))


class NewsAnalystAgent(_DeterministicAnalyst):
    name = "news"

    def _events(self, pkg) -> Optional[list]:
        ev = (self.section(pkg) or {}).get("events")
        return ev if isinstance(ev, list) else None

    def slots(self, pkg):
        sec = self.section(pkg) or {}
        events = self._events(pkg)
        out = [("news.events", bool(events)),
               ("news.quarantined_count", num(sec.get("quarantined_count")) is not None)]
        for i, ev in enumerate(events or []):
            ev = ev if isinstance(ev, dict) else {}
            key = ev.get("event_id") or f"#{i}"
            for f in EVENT_SLOT_FIELDS:
                out.append((f"news.events[{key}].{f}", _present(ev.get(f))))
        return out

    def has_data(self, pkg):
        return self._events(pkg) is not None

    def evidence_ids(self, pkg):
        return []

    def extra_ref_ids(self, pkg, by_id):
        return [e.get("evidence_id") for e in (self._events(pkg) or [])
                if isinstance(e, dict) and not e.get("is_duplicate_report")
                and e.get("evidence_id") in by_id]

    def raw_view(self, pkg):
        return {k: v for k, v in (self.section(pkg) or {}).items() if k != "evidence_ids"}

    def facts(self, pkg):
        sec = self.section(pkg)
        events = self._events(pkg) or []
        warnings: list[str] = []
        buckets: dict[str, dict[str, Any]] = {
            c: {"counted": [], "dups": 0, "generic": 0} for c in NEWS_CATEGORIES}
        for i, ev in enumerate(events):
            if not isinstance(ev, dict):
                warnings.append(f"news event #{i} malformed; ignored")
                continue
            cat = ev.get("category")
            if cat not in buckets:
                warnings.append(f"news event {ev.get('event_id', i)} has unknown category "
                                f"{cat!r}; excluded from category tallies")
                continue
            if ev.get("is_duplicate_report"):
                buckets[cat]["dups"] += 1
            elif is_generic_headline(ev.get("headline")):
                buckets[cat]["generic"] += 1
                warnings.append(f"generic price-page style headline not treated as company "
                                f"event: {ev.get('headline')!r}")
            else:
                buckets[cat]["counted"].append(ev)
        facts: list[Fact] = []
        for cat in NEWS_CATEGORIES:
            b = buckets[cat]
            n = len(b["counted"])
            extra = ""
            if b["dups"]:
                extra += f"; {b['dups']} duplicate report(s) ignored"
            if b["generic"]:
                extra += f"; {b['generic']} generic price-page headline(s) flagged and excluded"
            if n == 0:
                facts.append(Fact(f"{cat}: no events{extra}"))
                continue
            sent: dict[str, int] = {}
            for ev in b["counted"]:
                k = str(ev.get("sentiment")).lower() if _present(ev.get("sentiment")) else "unknown"
                sent[k] = sent.get(k, 0) + 1
            sdesc = ", ".join(f"{k}={v}" for k, v in sorted(sent.items()))
            claims = []
            for ev in b["counted"]:
                im = num(ev.get("impact_magnitude"))
                if im is not None:
                    claims.append((f"impact_magnitude[{ev.get('event_id')}]", im,
                                   ev.get("evidence_id")))
            text = f"{cat}: {n} event(s) ({sdesc}){extra}"
            if cat == "competitor":
                text += ("; competitor-win/positive headlines describe the COMPETITOR and are "
                         "NOT positive for the target")
                warnings.append("competitor-category news: a competitor's win is NOT positive "
                                "for the target")
            facts.append(Fact(text, claims))
        q = num(sec.get("quarantined_count"))
        facts.append(Fact(f"quarantined items excluded: {q:g}" if q is not None
                          else "quarantined count unavailable"))
        return facts, warnings


# ----------------------------------------------------------------------------------------
# Promoter
# ----------------------------------------------------------------------------------------
class PromoterAnalystAgent(_DeterministicAnalyst):
    name = "promoter"
    FIELDS = ("promoters_pct", "promoter_change_qoq_pct", "pledged_pct", "fii_pct", "dii_pct",
              "insider_transactions", "management_commentary", "leadership_changes",
              "governance_flags")
    _NUMERIC = ("promoters_pct", "promoter_change_qoq_pct", "pledged_pct", "fii_pct", "dii_pct")

    def slots(self, pkg):
        sec = self.section(pkg) or {}
        return [(f"promoter.{f}",
                 (num(sec.get(f)) is not None) if f in self._NUMERIC else _present(sec.get(f)))
                for f in self.FIELDS]

    def facts(self, pkg):
        s = self.section(pkg)
        e = lambda f: self.eid(pkg, "promoter", f)  # noqa: E731
        out: list[Fact] = []
        p = num(s.get("promoters_pct"))
        if p is not None:
            out.append(Fact(f"promoter holding {p:g}%", [("promoters_pct", p, e("promoters_pct"))]))
        ch = num(s.get("promoter_change_qoq_pct"))
        if ch is not None:
            d = "increased" if ch > 0 else ("decreased" if ch < 0 else "unchanged")
            out.append(Fact(f"promoter holding {d} QoQ ({ch:g} pp)",
                            [("promoter_change_qoq_pct", ch, e("promoter_change_qoq_pct"))]))
        pl = num(s.get("pledged_pct"))
        if pl is not None:
            lvl = ("critical" if pl >= 50 else "high" if pl >= 25 else
                   "low" if pl > 0 else "none")
            out.append(Fact(f"pledged {pl:g}% of promoter holding: {lvl} pledge risk",
                            [("pledged_pct", pl, e("pledged_pct"))]))
        for f, lbl in (("fii_pct", "FII"), ("dii_pct", "DII")):
            v = num(s.get(f))
            if v is not None:
                out.append(Fact(f"{lbl} holding {v:g}%", [(f, v, e(f))]))
        it = s.get("insider_transactions")
        out.append(Fact("insider transactions: unavailable (no inference of insider activity)"
                        if it is None else
                        f"insider transactions: {len(it)} record(s) reported"
                        if isinstance(it, list) else "insider transactions: malformed input"))
        lc = s.get("leadership_changes")
        out.append(Fact("leadership changes: unavailable" if lc is None else
                        f"leadership changes: {len(lc)} reported" if isinstance(lc, list)
                        else "leadership changes: malformed input"))
        gf = s.get("governance_flags")
        out.append(Fact("governance flags: unavailable" if gf is None else
                        (f"governance flags: {len(gf)} raised" if gf else "governance flags: none raised")
                        if isinstance(gf, list) else "governance flags: malformed input"))
        out.append(Fact("management commentary: "
                        + ("unavailable" if s.get("management_commentary") is None else "provided")))
        return out, []


# ----------------------------------------------------------------------------------------
# Competitor
# ----------------------------------------------------------------------------------------
class CompetitorAnalystAgent(_DeterministicAnalyst):
    name = "competitor"
    FIELDS = ("competitors", "peer_metrics", "cross_stock_impacts", "industry_growth_pct")

    def slots(self, pkg):
        sec = self.section(pkg) or {}
        return [("competitor.competitors", isinstance(sec.get("competitors"), list)),
                ("competitor.peer_metrics", _present(sec.get("peer_metrics"))),
                ("competitor.cross_stock_impacts", isinstance(sec.get("cross_stock_impacts"), list)),
                ("competitor.industry_growth_pct", num(sec.get("industry_growth_pct")) is not None)]

    def has_data(self, pkg):
        s = self.section(pkg) or {}
        return any(slot for _, slot in self.slots(pkg)) and bool(s)

    def _impacts(self, pkg) -> list[dict]:
        v = (self.section(pkg) or {}).get("cross_stock_impacts")
        return [i for i in v if isinstance(i, dict)] if isinstance(v, list) else []

    def extra_ref_ids(self, pkg, by_id):
        return [i.get("evidence_id") for i in self._impacts(pkg) if i.get("evidence_id") in by_id]

    def facts(self, pkg):
        s = self.section(pkg)
        out: list[Fact] = []
        warnings: list[str] = []
        comps = s.get("competitors")
        if isinstance(comps, list):
            out.append(Fact(f"{len(comps)} competitor(s) listed: "
                            + (", ".join(str(c) for c in comps) if comps else "none")))
        pm = s.get("peer_metrics")
        out.append(Fact("peer metrics: unavailable" if pm is None else
                        f"peer metrics provided for {len(pm)} item(s)" if isinstance(pm, dict)
                        else "peer metrics: malformed input"))
        g = num(s.get("industry_growth_pct"))
        if g is not None:
            out.append(Fact(f"industry growth {g:g}%",
                            [("industry_growth_pct", g, self.eid(pkg, "competitor",
                                                                 "industry_growth_pct"))]))
        if isinstance(s.get("cross_stock_impacts"), list):
            imps = self._impacts(pkg)
            out.append(Fact(f"{len(imps)} cross-stock impact(s) reported"
                            if imps else "cross-stock impacts: none reported"))
            for i in imps:
                out.append(Fact(f"{i.get('source_symbol')} -> {i.get('target_symbol')}: "
                                f"{i.get('relation')}, direction {i.get('direction')}; "
                                f"{i.get('rationale')}"))
                if not i.get("evidence_id"):
                    warnings.append(f"cross-stock impact {i.get('source_symbol')}->"
                                    f"{i.get('target_symbol')} lacks evidence_id")
        return out, warnings


ALL_ANALYSTS = (FundamentalAnalystAgent, TechnicalAnalystAgent, NewsAnalystAgent,
                PromoterAnalystAgent, CompetitorAnalystAgent)
