"""Read-only, point-in-time evidence loader for the Phase 6 shadow research graph.

``load_evidence(symbol, as_of, repo_root=".", analysis_timestamp=None)`` maps existing
repository artifacts into the shared ``EvidencePackage.data`` contract::

    data = {'fundamental':{..}, 'technical':{..}, 'news':{..}, 'promoter':{..},
            'competitor':{..}, 'quantitative':{..}, 'market_regime':{..}, 'price_risk':{..},
            '_missing': [...], '_notes': [...], '_cutoff': {...}}

Rules
-----
* Data cutoff = ``min(end-of-day IST of as_of, analysis_timestamp)``. Anything whose
  publication time is after the cutoff is excluded and noted in ``_notes``.
* Every non-None field has an ``EvidenceRef`` (id ``<section>:<SYMBOL>:<field>``) carrying
  ``retrieved_at`` and the sha256 of the source file; ``<section>['evidence_ids']`` maps
  field -> id.  ``retrieved_at`` is ``min(wall-clock now, cutoff)`` because report
  validation forbids evidence retrieved after the cutoff.
* Missing / unavailable data is ``None`` (never a 0 sentinel) and listed in ``_missing``.
* The loader only ever *reads* files.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from backend.phase6.agents_base import EvidencePackage
from backend.phase6.contracts import EvidenceRef

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

NEWS_CATEGORIES = ("company_news", "promoter_management", "industry_sector", "competitor",
                   "regulatory_government", "global_macro")

FUNDAMENTAL_FIELDS = ("revenue_cr", "profit_cr", "revenue_growth_pct", "profit_growth_pct",
                      "opm_pct", "net_margin_pct", "operating_cash_flow_cr", "free_cash_flow_cr",
                      "debt_to_equity", "current_ratio", "interest_coverage", "pe", "pb",
                      "roe_pct", "roce_pct", "piotroski_score")
TECHNICAL_FIELDS = ("close", "rsi_14", "macd_hist", "sma_20", "sma_50", "sma_200", "atr_pct",
                    "return_1m_pct", "return_3m_pct", "volume_surge_ratio", "support",
                    "resistance", "adx", "trend")
PROMOTER_FIELDS = ("promoters_pct", "promoter_change_qoq_pct", "pledged_pct", "fii_pct", "dii_pct",
                   "insider_transactions", "management_commentary", "leadership_changes",
                   "governance_flags")
COMPETITOR_FIELDS = ("competitors", "peer_metrics", "cross_stock_impacts", "industry_growth_pct")
REGIME_FIELDS = ("regime", "india_vix_close", "nifty_50_day_change_pct", "observation_date")
PRICE_RISK_FIELDS = ("avg_daily_value_cr", "amihud", "volatility_20d_pct", "max_drawdown_1y_pct",
                     "beta", "position_weight_pct")
QUANT_FIELDS = ("prediction_id", "model_name", "model_version", "observation_date", "generated_at",
                "market_regime", "prob_next_day_up", "prob_5d_up", "prob_10d_up",
                "expected_next_day_return_pct", "expected_5d_return_pct",
                "expected_10d_return_pct", "net_expected_1d_return_pct",
                "net_expected_5d_return_pct", "net_expected_10d_return_pct",
                "confidence_score", "model_agreement_score", "recommended_action",
                "abstention_reason", "top_bullish_drivers", "top_bearish_drivers",
                "validation", "validation_generated_at", "validation_model_version")


# ----------------------------------------------------------------------------- helpers
def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    try:  # pandas NA / NaT / numpy nan
        import pandas as pd
        if not isinstance(v, (list, dict, tuple, str)) and pd.isna(v):
            return True
    except Exception:
        pass
    return False


def _clean(v: Any) -> Any:
    """JSON-safe deep conversion (numpy -> python, NaN -> None, ts -> iso)."""
    if _is_missing(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)) or (hasattr(v, "tolist") and not isinstance(v, (str, bytes))):
        seq = v.tolist() if hasattr(v, "tolist") else v
        if isinstance(seq, (list, tuple)):
            return [_clean(x) for x in seq]
        return _clean(seq)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return _clean(v.item())
        except Exception:
            pass
    if isinstance(v, float):
        return float(v)
    return v


def _parse_ts(s: Any, default_tz=IST) -> Optional[datetime]:
    if _is_missing(s):
        return None
    if isinstance(s, datetime):
        dt = s
    else:
        txt = str(s).strip()
        txt = re.sub(r"\s+IST$", "", txt)
        try:
            dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(txt)
            except Exception:
                for fmt in ("%a, %d %b %Y %H:%M", "%a, %d %b %Y %H:%M:%S"):
                    try:
                        dt = datetime.strptime(txt, fmt)
                        break
                    except ValueError:
                        dt = None
                if dt is None:
                    return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(UTC)


def _sha_cache_key(p: Path):
    st = p.stat()
    return (str(p), st.st_mtime_ns, st.st_size)


_SHA: dict = {}
_JSON: dict = {}
_PQ: dict = {}


def _sha256_file(p: Path) -> str:
    k = _sha_cache_key(p)
    if k not in _SHA:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA[k] = h.hexdigest()
    return _SHA[k]


def _read_json(p: Path) -> Any:
    if not p.is_file():
        return None
    try:
        k = _sha_cache_key(p)
        if k not in _JSON:
            _JSON[k] = json.loads(p.read_text(encoding="utf-8"))
        return copy.deepcopy(_JSON[k])
    except Exception:
        return None


def _read_parquet(p: Path):
    if not p.is_file():
        return None
    try:
        import pandas as pd
        k = _sha_cache_key(p)
        if k not in _PQ:
            _PQ[k] = pd.read_parquet(p)
        return _PQ[k]
    except Exception:
        return None


def _num(v: Any) -> Optional[float]:
    if _is_missing(v) or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


class _Builder:
    """Collects EvidenceRefs and section dicts for one (symbol, cutoff)."""

    def __init__(self, symbol: str, repo_root: Path, cutoff: datetime, retrieved_at: datetime):
        self.symbol, self.root, self.cutoff, self.retrieved_at = symbol, repo_root, cutoff, retrieved_at
        self.refs: dict[str, EvidenceRef] = {}
        self.missing: list[str] = []
        self.notes: list[str] = []
        self.data: dict[str, dict] = {}

    def section(self, name: str) -> dict:
        return self.data.setdefault(name, {"evidence_ids": {}})

    def ref(self, source_id: str, dataset: str, path: str, record_key: str,
            published_at: Optional[datetime], sha: Optional[str]) -> str:
        if published_at is not None and published_at > self.cutoff:
            raise ValueError(f"internal: evidence {source_id} published after cutoff")
        if source_id not in self.refs:
            self.refs[source_id] = EvidenceRef(
                source_id=source_id, dataset=dataset, path=path, record_key=record_key,
                published_at=published_at, retrieved_at=self.retrieved_at, sha256=sha)
        return source_id

    def put(self, section: str, field: str, value: Any, *, dataset: str, path: str,
            record_key: str, published_at: Optional[datetime], sha: Optional[str]) -> None:
        sec = self.section(section)
        if _is_missing(value):
            sec[field] = None
            self.missing.append(f"{section}.{field}")
            return
        sec[field] = _clean(value)
        sec["evidence_ids"][field] = self.ref(f"{section}:{self.symbol}:{field}", dataset, path,
                                              record_key, published_at, sha)

    def absent(self, section: str, fields, reason: str) -> None:
        sec = self.section(section)
        for f in fields:
            sec[f] = None
            self.missing.append(f"{section}.{f}")
        self.notes.append(f"{section}: {reason}")


# ----------------------------------------------------------------------------- loader
def _cutoff(as_of: str, analysis_timestamp: Optional[datetime]):
    d = date.fromisoformat(as_of)
    eod = datetime.combine(d, time(23, 59, 59), tzinfo=IST).astimezone(UTC)
    if analysis_timestamp is None:
        analysis = datetime.now(UTC)
    else:
        analysis = analysis_timestamp if analysis_timestamp.tzinfo else analysis_timestamp.replace(tzinfo=UTC)
        analysis = analysis.astimezone(UTC)
    return min(eod, analysis), analysis


def load_evidence(symbol: str, as_of: str, *, repo_root: str | Path = ".",
                  analysis_timestamp: Optional[datetime] = None) -> EvidencePackage:
    symbol = symbol.upper()
    root = Path(repo_root)
    cutoff, analysis = _cutoff(as_of, analysis_timestamp)
    now = datetime.now(UTC)
    b = _Builder(symbol, root, cutoff, min(now, cutoff))

    doc_path = root / "docs" / "data" / f"{symbol}.json"
    doc = _read_json(doc_path)
    doc_ok, doc_pub, doc_sha = False, None, None
    if not isinstance(doc, dict):
        b.notes.append(f"docs/data/{symbol}.json absent or unreadable")
    else:
        doc_sha = _sha256_file(doc_path)
        doc_pub = _parse_ts(doc.get("fetched_at"))
        if doc_pub is None:
            b.notes.append(f"docs/data/{symbol}.json has no parseable fetched_at; excluded")
        elif doc_pub > cutoff:
            b.notes.append(f"docs/data/{symbol}.json snapshot fetched_at={doc_pub.isoformat()} is "
                           f"after cutoff {cutoff.isoformat()}; excluded (fundamental/promoter/"
                           f"peer data)")
        else:
            doc_ok = True
    docinfo = dict(dataset="docs_symbol_snapshot", path=f"docs/data/{symbol}.json",
                   published_at=doc_pub, sha=doc_sha)

    _fundamental(b, doc if doc_ok else None, docinfo)
    _technical_and_price_risk(b, as_of)
    _news(b)
    _promoter(b, doc if doc_ok else None, docinfo)
    _competitor(b, doc if doc_ok else None, docinfo)
    quant_raw = _quantitative(b, as_of)
    _regime(b, as_of)

    data = dict(b.data)
    data["_missing"] = list(b.missing)
    data["_notes"] = list(b.notes)
    data["_cutoff"] = {"as_of": as_of, "data_cutoff_timestamp": cutoff.isoformat(),
                       "analysis_timestamp": analysis.isoformat()}
    return EvidencePackage(symbol=symbol, as_of=as_of, analysis_timestamp=analysis,
                           data_cutoff_timestamp=cutoff,
                           evidence=tuple(b.refs[k] for k in sorted(b.refs)),
                           data=data, quantitative_predictions=quant_raw)


# ----------------------------------------------------------------------------- fundamental
def _fundamental(b: _Builder, doc: Optional[dict], di: dict) -> None:
    if doc is None:
        b.absent("fundamental", FUNDAMENTAL_FIELDS, "no admissible docs snapshot")
        return
    ov = doc.get("overview") or {}
    qs = doc.get("last_5_quarterly_reports") or []
    q = qs[-1] if qs and isinstance(qs[-1], dict) else {}
    cb = doc.get("cash_and_balance_sheet") or {}
    period = q.get("period") or "n/a"
    rev, prof = _num(q.get("revenue_cr")), _num(q.get("net_profit_cr"))
    net_margin = (prof / rev * 100.0) if (rev and prof is not None and rev != 0) else None
    interest, op = _num(q.get("interest_cr")), _num(q.get("operating_profit_cr"))
    icov = (op / interest) if (op is not None and interest and interest > 0) else None
    price, bv = _num(ov.get("current_price")), _num(ov.get("book_value"))
    pb = (price / bv) if (price and bv and bv > 0) else None

    def nz(v):  # zero cash-flow is a scraper sentinel, never a real value
        v = _num(v)
        return None if (v is None or v == 0.0) else v

    vals = {
        "revenue_cr": (rev, f"last_5_quarterly_reports[{period}].revenue_cr"),
        "profit_cr": (prof, f"last_5_quarterly_reports[{period}].net_profit_cr"),
        "revenue_growth_pct": (_num(q.get("revenue_growth_pct")), f"quarter {period}.revenue_growth_pct"),
        "profit_growth_pct": (_num(q.get("net_profit_growth_pct")), f"quarter {period}.net_profit_growth_pct"),
        "opm_pct": (_num(q.get("opm_pct")), f"quarter {period}.opm_pct"),
        "net_margin_pct": (net_margin, f"computed net_profit/revenue quarter {period}"),
        "operating_cash_flow_cr": (nz(cb.get("latest_ocf_cr")), "cash_and_balance_sheet.latest_ocf_cr (0 treated as missing)"),
        "free_cash_flow_cr": (nz(cb.get("latest_fcf_cr")), "cash_and_balance_sheet.latest_fcf_cr (0 treated as missing)"),
        "debt_to_equity": (_num(cb.get("debt_to_equity")), "cash_and_balance_sheet.debt_to_equity"),
        "current_ratio": (None, "not provided by source"),
        "interest_coverage": (icov, f"computed operating_profit/interest quarter {period}"),
        "pe": (_num(ov.get("stock_pe")), "overview.stock_pe"),
        "pb": (pb, "computed overview.current_price/book_value"),
        "roe_pct": (_num(ov.get("roe_pct")), "overview.roe_pct"),
        "roce_pct": (_num(ov.get("roce_pct")), "overview.roce_pct"),
        "piotroski_score": (_num(ov.get("piotroski_score")), "overview.piotroski_score"),
    }
    for f in FUNDAMENTAL_FIELDS:
        v, key = vals[f]
        b.put("fundamental", f, v, dataset=di["dataset"], path=di["path"],
              record_key=key, published_at=di["published_at"], sha=di["sha"])


# ----------------------------------------------------------------------------- prices
def _price_frames(b: _Builder, as_of: str):
    """Return (equity df for symbol, nifty df, notes) filtered to <= as_of and <= cutoff."""
    import pandas as pd
    eq = _read_parquet(b.root / "data" / "processed" / "equities_daily.parquet")
    idx = _read_parquet(b.root / "data" / "processed" / "indices_daily.parquet")

    def filt(df, sel):
        if df is None:
            return None
        d = df[sel(df)].copy()
        d = d[d["observation_date"] <= as_of]
        pub = d["publication_timestamp"].map(_parse_ts)
        ok = pub.map(lambda t: t is not None and t <= b.cutoff)
        excluded = int((~ok).sum())
        if excluded:
            b.notes.append(f"prices: excluded {excluded} row(s) published after cutoff")
        d = d[ok].copy()
        d["_pub"] = pub[ok]
        return d.sort_values("observation_date").reset_index(drop=True)

    e = filt(eq, lambda d: (d["symbol"] == b.symbol) & (d.get("series", "EQ") == "EQ")) if eq is not None else None
    n = None
    if idx is not None:
        codes = idx["index_code"].astype(str)
        for cand in ("NIFTY_50", "NIFTY50", "NIFTY 50"):
            if (codes == cand).any():
                n = filt(idx, lambda d, c=cand: d["index_code"] == c)
                break
    return e, n


def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _technical_and_price_risk(b: _Builder, as_of: str) -> None:
    import numpy as np
    import pandas as pd
    eq_path = "data/processed/equities_daily.parquet"
    eq_file = b.root / eq_path
    sha = _sha256_file(eq_file) if eq_file.is_file() else None
    try:
        e, nifty = _price_frames(b, as_of)
    except Exception as ex:  # corrupt data => treat as missing
        e, nifty = None, None
        b.notes.append(f"prices: unreadable ({type(ex).__name__})")
    if e is None or len(e) == 0:
        b.absent("technical", TECHNICAL_FIELDS, "no price history <= cutoff")
        b.absent("price_risk", PRICE_RISK_FIELDS, "no price history <= cutoff")
        return
    last = e.iloc[-1]
    pub = last["_pub"].to_pydatetime()
    key = f"{b.symbol}:{last['observation_date']}"
    c, h, l, v = e["close"].astype(float), e["high"].astype(float), e["low"].astype(float), e["volume"].astype(float)
    n = len(e)

    def sma(w):
        return float(c.tail(w).mean()) if n >= w else None

    delta = c.diff()
    rsi = None
    if n > 14:
        ag, al = _wilder(delta.clip(lower=0), 14), _wilder((-delta).clip(lower=0), 14)
        al_last, ag_last = al.iloc[-1], ag.iloc[-1]
        if not (pd.isna(al_last) or pd.isna(ag_last)):
            rsi = 100.0 if al_last == 0 else float(100 - 100 / (1 + ag_last / al_last))
    macd_hist = None
    if n >= 35:
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        macd_hist = float((macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1])
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    atr_pct = float(atr.iloc[-1] / c.iloc[-1] * 100) if n > 14 and not pd.isna(atr.iloc[-1]) else None
    adx = None
    if n > 28:
        up, dn = h.diff(), -l.diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=e.index)
        ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=e.index)
        satr = _wilder(tr, 14)
        pdi, ndi = 100 * _wilder(pdm, 14) / satr, 100 * _wilder(ndm, 14) / satr
        dx = (100 * (pdi - ndi).abs() / (pdi + ndi)).replace([np.inf, -np.inf], np.nan)
        a = _wilder(dx.dropna(), 14)
        adx = float(a.iloc[-1]) if len(a) and not pd.isna(a.iloc[-1]) else None
    ret = lambda k: float((c.iloc[-1] / c.iloc[-1 - k] - 1) * 100) if n > k else None
    surge = None
    if n > 20 and v.iloc[-21:-1].mean() > 0:
        surge = float(v.iloc[-1] / v.iloc[-21:-1].mean())
    s50, s200 = sma(50), sma(200)
    trend = None
    if s50 is not None and s200 is not None:
        cl = float(c.iloc[-1])
        trend = "UPTREND" if cl > s50 > s200 else "DOWNTREND" if cl < s50 < s200 else "MIXED"
    tech = {
        "close": float(c.iloc[-1]), "rsi_14": rsi, "macd_hist": macd_hist, "sma_20": sma(20),
        "sma_50": s50, "sma_200": s200, "atr_pct": atr_pct, "return_1m_pct": ret(21),
        "return_3m_pct": ret(63), "volume_surge_ratio": surge,
        "support": float(l.tail(20).min()) if n >= 20 else None,
        "resistance": float(h.tail(20).max()) if n >= 20 else None, "adx": adx, "trend": trend,
    }
    for f in TECHNICAL_FIELDS:
        b.put("technical", f, tech[f], dataset="equities_daily", path=eq_path,
              record_key=f"{key} (computed from bars <= {last['observation_date']})",
              published_at=pub, sha=sha)

    # ---- price risk
    pr: dict[str, Any] = {f: None for f in PRICE_RISK_FIELDS}
    r = c.pct_change()
    if n >= 21:
        pr["volatility_20d_pct"] = float(r.tail(20).std(ddof=1) * math.sqrt(252) * 100)
    t20 = e["turnover_cr"].astype(float).tail(20)
    if n >= 20 and t20.notna().all():
        pr["avg_daily_value_cr"] = float(t20.mean())
        pos = (t20 > 0).values
        if pos.all():
            pr["amihud"] = float((r.tail(20).abs() * 100 / t20).mean())
    if n >= 60:
        w = c.tail(252)
        pr["max_drawdown_1y_pct"] = float(((w / w.cummax()) - 1).min() * 100)
    beta_sha = sha
    if nifty is not None and len(nifty) >= 60 and n >= 60:
        nr = nifty.set_index("observation_date")["close"].astype(float).pct_change()
        sr = e.set_index("observation_date")["close"].astype(float).pct_change()
        j = pd.concat([sr, nr], axis=1, keys=["s", "m"]).dropna().tail(252)
        if len(j) >= 60 and j["m"].var() > 0:
            pr["beta"] = float(j["s"].cov(j["m"]) / j["m"].var())
            ip = b.root / "data" / "processed" / "indices_daily.parquet"
            beta_sha = hashlib.sha256(((sha or "") + _sha256_file(ip)).encode()).hexdigest()
    for f in PRICE_RISK_FIELDS:
        if f == "position_weight_pct":
            b.put("price_risk", f, None, dataset="n/a", path="", record_key="portfolio not modelled",
                  published_at=None, sha=None)
            continue
        b.put("price_risk", f, pr[f], dataset="equities_daily" if f != "beta" else "equities_daily+indices_daily",
              path=eq_path, record_key=f"{key} (computed from bars <= {last['observation_date']})",
              published_at=pub, sha=beta_sha if f == "beta" else sha)


# ----------------------------------------------------------------------------- news
def _news(b: _Builder) -> None:
    p = "data/predictions/structured_events.parquet"
    sec = b.section("news")
    sec.update({"events": None, "quarantined_count": None, "duplicate_count": None,
                "excluded_post_cutoff_count": None})
    df = _read_parquet(b.root / p)
    if df is None:
        b.notes.append("news: structured_events.parquet absent")
        b.missing.append("news.events")
    else:
        sha = _sha256_file(b.root / p)
        d = df[df["ticker"].astype(str).str.upper() == b.symbol]
        events, dup, post = [], 0, 0
        latest_pub = None
        for _, row in d.iterrows():
            pub = _parse_ts(row.get("publication_timestamp"))
            if pub is None or pub > b.cutoff:
                post += 1
                continue
            if bool(row.get("is_duplicate_report")):
                dup += 1
                continue
            eid = str(row["event_id"])
            evid = b.ref(f"news:{b.symbol}:{eid}", "structured_events", p, eid, pub, sha)
            events.append({
                "event_id": eid, "headline": _clean(row.get("event_description")),
                "category": _clean(row.get("event_category")),
                "sentiment": _clean(row.get("sentiment")),
                "impact_magnitude": _clean(row.get("impact_magnitude")),
                "publication_timestamp": pub.isoformat(),
                "effective_trading_date": _clean(row.get("effective_trading_date")),
                "is_duplicate_report": False, "evidence_id": evid,
            })
            latest_pub = max(latest_pub, pub) if latest_pub else pub
        events.sort(key=lambda e: (e["publication_timestamp"], e["event_id"]), reverse=True)
        sec["events"], sec["duplicate_count"], sec["excluded_post_cutoff_count"] = events, dup, post
        if post:
            b.notes.append(f"news: excluded {post} event(s) published after cutoff")
        sec["evidence_ids"]["events"] = b.ref(f"news:{b.symbol}:events", "structured_events", p,
                                              f"{b.symbol}:{len(events)} events", latest_pub, sha)
        sec["evidence_ids"]["duplicate_count"] = sec["evidence_ids"]["events"]
        sec["evidence_ids"]["excluded_post_cutoff_count"] = sec["evidence_ids"]["events"]
        if not events:
            b.missing.append("news.events")
    qp = "data/predictions/quarantined_events.parquet"
    q = _read_parquet(b.root / qp)
    if q is None:
        b.missing.append("news.quarantined_count")
        b.notes.append("news: quarantined_events.parquet absent")
    else:
        qd = q[q["symbol"].astype(str).str.upper() == b.symbol]
        n_ok, newest = 0, None
        for _, row in qd.iterrows():
            ts = _parse_ts(row.get("collection_timestamp"))
            if ts is not None and ts <= b.cutoff:
                n_ok += 1
                newest = max(newest, ts) if newest else ts
        sec["quarantined_count"] = n_ok
        sec["evidence_ids"]["quarantined_count"] = b.ref(
            f"news:{b.symbol}:quarantined_count", "quarantined_events", qp,
            f"{b.symbol}:{n_ok} quarantined", newest, _sha256_file(b.root / qp))


# ----------------------------------------------------------------------------- promoter
def _promoter(b: _Builder, doc: Optional[dict], di: dict) -> None:
    if doc is None:
        b.absent("promoter", PROMOTER_FIELDS, "no admissible docs snapshot")
        return
    sh = doc.get("stockholders") or {}
    ci = doc.get("corporate_intelligence") or {}
    hist = sh.get("quarterly_history") or []
    qoq = _num(ci.get("promoter_chg_1q"))
    qoq_src = "corporate_intelligence.promoter_chg_1q"
    if qoq is None and len(hist) >= 2:
        a, c0 = _num(hist[-1].get("promoters_pct")), _num(hist[-2].get("promoters_pct"))
        if a is not None and c0 is not None:
            qoq, qoq_src = round(a - c0, 4), "computed stockholders.quarterly_history last two quarters"
    arts = (doc.get("promoter_and_competitor_intel") or {}).get("promoter_statements_and_articles") or []
    lines = []
    for a in arts:
        t = _parse_ts(a.get("published"))
        if t is not None and t <= b.cutoff and a.get("headline"):
            lines.append(f"[{t.date().isoformat()} {a.get('speaker_or_source') or 'n/a'}] {a['headline']}")
    vals = {
        "promoters_pct": (_num(sh.get("promoters_pct")), "stockholders.promoters_pct"),
        "promoter_change_qoq_pct": (qoq, qoq_src),
        "pledged_pct": (_num(ci.get("promoter_pledge_pct")), "corporate_intelligence.promoter_pledge_pct"),
        "fii_pct": (_num(sh.get("fii_pct")), "stockholders.fii_pct"),
        "dii_pct": (_num(sh.get("dii_pct")), "stockholders.dii_pct"),
        "insider_transactions": (None, "not provided by source"),
        "management_commentary": (" | ".join(lines) if lines else None,
                                  "promoter_and_competitor_intel.promoter_statements_and_articles"),
        "leadership_changes": (None, "not provided by source"),
        "governance_flags": (None, "not provided by source"),
    }
    for f in PROMOTER_FIELDS:
        v, key = vals[f]
        b.put("promoter", f, v, dataset=di["dataset"], path=di["path"], record_key=key,
              published_at=di["published_at"], sha=di["sha"])


# ----------------------------------------------------------------------------- competitor
def _competitor(b: _Builder, doc: Optional[dict], di: dict) -> None:
    sym = b.symbol
    # competitors from the static relationship map (code artifact)
    comps, rsha, newest_pub, sha_c, sha_e = None, None, None, "", ""
    try:
        from backend.news_intel.relationships import RELATIONSHIP_MAP
        entry = RELATIONSHIP_MAP.get(sym)
        if entry:
            comps = [c["symbol"] for c in entry.get("competitors", [])] or None
            rsha = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        comps = None
    b.put("competitor", "competitors", comps, dataset="relationship_map",
          path="backend/news_intel/relationships.py", record_key=f"RELATIONSHIP_MAP[{sym}].competitors",
          published_at=None, sha=rsha if comps else None)
    # peer metrics from docs snapshot
    peers = None
    if doc is not None:
        lst = (doc.get("promoter_and_competitor_intel") or {}).get("competitors") or []
        peers = {}
        for c in lst:
            if isinstance(c, dict) and c.get("symbol"):
                peers[c["symbol"]] = {k: _num(c.get(k)) for k in
                                      ("current_price", "day_change_pct", "return_1m_pct", "pe", "roe_pct")}
        peers = peers or None
    else:
        b.notes.append("competitor.peer_metrics: no admissible docs snapshot")
    b.put("competitor", "peer_metrics", peers, dataset=di["dataset"], path=di["path"],
          record_key="promoter_and_competitor_intel.competitors", published_at=di["published_at"],
          sha=di["sha"])
    # cross stock impacts (dated via originating event publication)
    cp, ep = "data/predictions/cross_stock_impacts.parquet", "data/predictions/structured_events.parquet"
    ci_df, ev_df = _read_parquet(b.root / cp), _read_parquet(b.root / ep)
    impacts = None
    if ci_df is None or ev_df is None:
        b.notes.append("competitor.cross_stock_impacts: source parquet absent")
    else:
        sha_c, sha_e = _sha256_file(b.root / cp), _sha256_file(b.root / ep)
        pubmap = {str(r.event_id): _parse_ts(r.publication_timestamp) for r in ev_df.itertuples()}
        impacts, skipped, newest = [], 0, None
        for r in ci_df[ci_df["affected_company"].astype(str).str.upper() == sym].itertuples():
            pub = pubmap.get(str(r.originating_event_id))
            if pub is None or pub > b.cutoff:
                skipped += 1
                continue
            sid = f"competitor:{sym}:cross_stock_impact:{r.originating_event_id}"
            eid = b.ref(sid, "cross_stock_impacts", cp, str(r.originating_event_id), pub,
                        hashlib.sha256((sha_c + sha_e).encode()).hexdigest())
            impacts.append({"source_symbol": _clean(r.originating_entity), "target_symbol": sym,
                            "relation": _clean(r.relationship_type),
                            "direction": _clean(r.expected_impact_direction),
                            "rationale": _clean(r.rationale), "evidence_id": eid})
            newest = max(newest, pub) if newest else pub
        if skipped:
            b.notes.append(f"competitor.cross_stock_impacts: excluded {skipped} impact(s) with no "
                           f"dated originating event or published after cutoff")
        impacts = sorted(impacts, key=lambda x: (x["source_symbol"] or "", x["evidence_id"])) or None
        newest_pub = newest
    b.put("competitor", "cross_stock_impacts", impacts, dataset="cross_stock_impacts", path=cp,
          record_key=f"affected_company={sym}", published_at=newest_pub if impacts else None,
          sha=hashlib.sha256((sha_c + sha_e).encode()).hexdigest() if impacts else None)
    b.put("competitor", "industry_growth_pct", None, dataset="n/a", path="", record_key="not provided",
          published_at=None, sha=None)


# ----------------------------------------------------------------------------- quantitative
def _quantitative(b: _Builder, as_of: str) -> dict:
    sym = b.symbol
    cands = ["data/predictions/latest_predictions.json", "docs/data/latest_predictions.json"]
    row, src, sha, gen = None, None, None, None
    for rel in cands:
        j = _read_json(b.root / rel)
        if not isinstance(j, dict):
            continue
        g = _parse_ts(j.get("generated_at"))
        for r in j.get("predictions", []):
            if isinstance(r, dict) and str(r.get("symbol", "")).upper() == sym:
                pg = _parse_ts(r.get("generated_at")) or g
                if pg is None or pg > b.cutoff or str(r.get("observation_date", "9999")) > as_of:
                    b.notes.append(f"quantitative: {rel} prediction generated after cutoff/as_of; excluded")
                    continue
                if row is None:
                    row, src, sha, gen = r, rel, _sha256_file(b.root / rel), pg
                else:
                    other = _clean(r)
                    if other != _clean(row):
                        b.notes.append(f"quantitative: {rel} differs from {src}; using {src}")
    sec_fields = QUANT_FIELDS
    if row is None:
        b.absent("quantitative", sec_fields, "no admissible prediction")
        return {}
    row = _clean(row)
    # validation (forward validation status)
    val, val_gen, val_ver, val_sha, vsrc = None, None, None, None, None
    for rel in ("data/predictions/forward_validation.json", "docs/data/forward_validation.json"):
        j = _read_json(b.root / rel)
        if not isinstance(j, dict):
            continue
        vg = _parse_ts(j.get("generated_at"))
        if vg is None or vg > b.cutoff:
            b.notes.append(f"quantitative: {rel} generated after cutoff; validation excluded")
            continue
        he = j.get("horizon_evaluations")
        if isinstance(he, dict):
            keep = ("status", "matured_count", "pending_count", "total_snapshots", "min_sample_threshold",
                    "directional_accuracy_pct", "brier_score", "roc_auc", "net_return_pct",
                    "gross_return_pct", "benchmark_accuracy_pct", "benchmark_net_return_pct")
            val = {h: {k: _clean(x.get(k)) for k in keep} for h, x in he.items() if isinstance(x, dict)}
            val_gen, val_ver, vsrc = vg.isoformat(), _clean(j.get("model_version")), rel
            val_sha = _sha256_file(b.root / rel)
            break
    nominal_ids = {}
    vals = {k: row.get(k) for k in sec_fields}
    vals["validation"], vals["validation_generated_at"], vals["validation_model_version"] = val, val_gen, val_ver
    for f in sec_fields:
        is_val = f.startswith("validation")
        b.put("quantitative", f, vals[f], dataset="forward_validation" if is_val else "latest_predictions",
              path=(vsrc if is_val else src) or "", record_key=(f"{sym}" if not is_val else "horizon_evaluations"),
              published_at=_parse_ts(vals.get("validation_generated_at")) if is_val else gen,
              sha=val_sha if is_val else sha)
    return copy.deepcopy(row)


# ----------------------------------------------------------------------------- regime
def _regime(b: _Builder, as_of: str) -> None:
    sym = b.symbol
    cand = []
    j = _read_json(b.root / "docs" / "data" / "regime.json")
    if isinstance(j, dict):
        cand.append((j, "docs/data/regime.json"))
    df = _read_parquet(b.root / "data" / "processed" / "market_regimes.parquet")
    if df is not None and len(df):
        d = df[df["observation_date"] <= as_of].sort_values("observation_date")
        if len(d):
            cand.append((_clean(d.iloc[-1].to_dict()), "data/processed/market_regimes.parquet"))
    chosen = None
    for r, rel in cand:
        od = str(r.get("observation_date") or "")
        pub = _parse_ts(f"{od}T18:00:00+05:30") if od else None  # NSE EOD publication convention
        if pub is not None and od <= as_of and pub <= b.cutoff:
            if chosen is None or od > str(chosen[0].get("observation_date")):
                chosen = (r, rel, pub)
    if chosen is None:
        b.absent("market_regime", REGIME_FIELDS, "no regime observation <= cutoff")
        return
    r, rel, pub = chosen
    sha = _sha256_file(b.root / rel)
    for f in REGIME_FIELDS:
        b.put("market_regime", f, r.get(f), dataset="market_regime", path=rel,
              record_key=f"observation_date={r.get('observation_date')}", published_at=pub, sha=sha)
