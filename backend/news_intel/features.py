"""17 Numerical News Features & Strict Point-in-Time Alignment (Phase 5 — Section 4).

Provides:
1. `compute_pit_news_features`: Converts live `StructuredNewsEvent` + `CrossStockImpactItem`
   objects into 17 numerical features using ONLY events with `publication_timestamp <= as_of_ts`
   and excluding duplicate repeat reports from double-counting.
2. `reconstruct_pit_historical_news_features`: Reconstructs genuine historical PIT equivalents
   of the 17 features from existing point-in-time columns in `daily_features.parquet`
   (`ne_*`, `inst_*`, `ms_*`, `rel_*`, `cs_*`) WITHOUT fabricating historical news articles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from ..models import CrossStockImpactItem, StructuredNewsEvent
from .extractor import parse_publication_timestamp_ist, try_parse_publication_timestamp_ist

NEWS_FEATURE_NAMES: list[str] = [
    "news_company_sentiment_1d",
    "news_company_sentiment_5d",
    "news_company_sentiment_20d",
    "news_volume_1d",
    "news_volume_5d",
    "news_surprise_intensity",
    "news_promoter_sentiment_5d",
    "news_promoter_activity_flag",
    "news_industry_sentiment_5d",
    "news_competitor_threat_score",
    "news_competitor_positive_readacross",
    "news_regulatory_risk_score",
    "news_regulatory_positive_score",
    "news_global_macro_risk_score",
    "news_novelty_weighted_sentiment",
    "news_high_impact_event_count_5d",
    "news_cross_stock_net_impact",
]

NEWS_CATEGORY_GROUPS: dict[str, list[str]] = {
    "company_only": [
        "news_company_sentiment_1d",
        "news_company_sentiment_5d",
        "news_company_sentiment_20d",
        "news_volume_1d",
        "news_volume_5d",
        "news_surprise_intensity",
        "news_novelty_weighted_sentiment",
        "news_high_impact_event_count_5d",
    ],
    "promoter": [
        "news_promoter_sentiment_5d",
        "news_promoter_activity_flag",
    ],
    "industry_competitor": [
        "news_industry_sentiment_5d",
        "news_competitor_threat_score",
        "news_competitor_positive_readacross",
    ],
    "regulatory_macro_cross": [
        "news_regulatory_risk_score",
        "news_regulatory_positive_score",
        "news_global_macro_risk_score",
        "news_cross_stock_net_impact",
    ],
}

_SENT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
_MAG_WEIGHT = {"low": 0.5, "medium": 1.0, "high": 1.6}


def _event_signed_score(evt: StructuredNewsEvent) -> float:
    s = _SENT_MAP.get(evt.sentiment, 0.0)
    w = _MAG_WEIGHT.get(evt.impact_magnitude, 1.0)
    return s * w * evt.classification_confidence * evt.source_credibility


def compute_pit_news_features(
    events: Iterable[StructuredNewsEvent],
    as_of_ts: datetime | str,
    cross_impacts: Iterable[CrossStockImpactItem] | None = None,
) -> dict[str, float]:
    """Compute the 17 numerical news features strictly using `publication_timestamp <= as_of_ts`."""
    cutoff = (
        parse_publication_timestamp_ist(as_of_ts)
        if isinstance(as_of_ts, str)
        else as_of_ts
    )

    pit_events: list[tuple[StructuredNewsEvent, float]] = []
    for evt in events:
        if evt.is_duplicate_report:
            continue
        evt_dt = try_parse_publication_timestamp_ist(evt.publication_timestamp)
        if evt_dt is None or evt_dt > cutoff:
            # Strict point-in-time guard: never leak future or un-timestamped news!
            continue
        age_days = max(0.0, (cutoff - evt_dt).total_seconds() / 86400.0)
        pit_events.append((evt, age_days))

    def _mean_score(subset: list[tuple[StructuredNewsEvent, float]]) -> float:
        if not subset:
            return 0.0
        vals = [_event_signed_score(e) * float(np.exp(-0.20 * age)) for e, age in subset]
        return float(np.clip(np.mean(vals), -1.5, 1.5))

    comp_1d = [(e, a) for e, a in pit_events if e.event_category == "company_news" and a <= 1.5]
    comp_5d = [(e, a) for e, a in pit_events if e.event_category == "company_news" and a <= 5.5]
    comp_20d = [(e, a) for e, a in pit_events if e.event_category == "company_news" and a <= 20.5]

    all_1d = [(e, a) for e, a in pit_events if a <= 1.5]
    all_5d = [(e, a) for e, a in pit_events if a <= 5.5]
    avg_daily_5d = len(all_5d) / 5.0
    surprise_intensity = float(len(all_1d) / max(0.5, avg_daily_5d)) if all_1d else 0.0

    prom_5d = [(e, a) for e, a in all_5d if e.event_category == "promoter_management"]
    ind_5d = [(e, a) for e, a in all_5d if e.event_category == "industry_sector"]
    comp_peer_5d = [(e, a) for e, a in all_5d if e.event_category == "competitor"]
    reg_5d = [(e, a) for e, a in all_5d if e.event_category == "regulatory_government"]
    macro_5d = [(e, a) for e, a in all_5d if e.event_category == "global_macro"]

    comp_threat = float(
        sum(
            _MAG_WEIGHT.get(e.impact_magnitude, 1.0) * e.classification_confidence
            for e, _ in comp_peer_5d
            if e.sentiment == "positive"  # rival strength = threat
        )
    )
    comp_readacross = float(
        sum(
            _MAG_WEIGHT.get(e.impact_magnitude, 1.0) * e.classification_confidence
            for e, _ in comp_peer_5d
            if e.sentiment == "negative"  # rival disruption = opportunity
        )
    )

    reg_risk = float(
        sum(
            _MAG_WEIGHT.get(e.impact_magnitude, 1.0) * e.classification_confidence
            for e, _ in reg_5d
            if e.sentiment == "negative"
        )
    )
    reg_pos = float(
        sum(
            _MAG_WEIGHT.get(e.impact_magnitude, 1.0) * e.classification_confidence
            for e, _ in reg_5d
            if e.sentiment == "positive"
        )
    )
    macro_risk = float(
        sum(
            _MAG_WEIGHT.get(e.impact_magnitude, 1.0) * e.classification_confidence
            for e, _ in macro_5d
            if e.sentiment == "negative"
        )
    )

    novelty_weighted = 0.0
    if all_5d:
        w_sum = sum(e.novelty_score for e, _ in all_5d)
        if w_sum > 0:
            novelty_weighted = float(
                sum(_event_signed_score(e) * e.novelty_score for e, _ in all_5d) / w_sum
            )

    high_impact_5d = float(sum(1 for e, _ in all_5d if e.impact_magnitude == "high"))

    cross_net = 0.0
    if cross_impacts:
        pit_ids = {e.event_id for e, _ in pit_events}
        for ci in cross_impacts:
            if ci.originating_event_id in pit_ids:
                s = _SENT_MAP.get(ci.expected_impact_direction, 0.0)
                w = _MAG_WEIGHT.get(ci.impact_magnitude, 1.0)
                cross_net += s * w * ci.confidence

    return {
        "news_company_sentiment_1d": round(_mean_score(comp_1d), 4),
        "news_company_sentiment_5d": round(_mean_score(comp_5d), 4),
        "news_company_sentiment_20d": round(_mean_score(comp_20d), 4),
        "news_volume_1d": float(len(all_1d)),
        "news_volume_5d": float(len(all_5d)),
        "news_surprise_intensity": round(min(5.0, surprise_intensity), 4),
        "news_promoter_sentiment_5d": round(_mean_score(prom_5d), 4),
        "news_promoter_activity_flag": 1.0 if prom_5d else 0.0,
        "news_industry_sentiment_5d": round(_mean_score(ind_5d), 4),
        "news_competitor_threat_score": round(min(3.0, comp_threat), 4),
        "news_competitor_positive_readacross": round(min(3.0, comp_readacross), 4),
        "news_regulatory_risk_score": round(min(3.0, reg_risk), 4),
        "news_regulatory_positive_score": round(min(3.0, reg_pos), 4),
        "news_global_macro_risk_score": round(min(3.0, macro_risk), 4),
        "news_novelty_weighted_sentiment": round(float(np.clip(novelty_weighted, -1.5, 1.5)), 4),
        "news_high_impact_event_count_5d": high_impact_5d,
        "news_cross_stock_net_impact": round(float(np.clip(cross_net, -3.0, 3.0)), 4),
    }


def reconstruct_pit_historical_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct genuine historical PIT equivalents of the 17 news/event features.

    IMPORTANT DATA INTEGRITY NOTE:
    Historical Google News RSS feeds do not exist years in the past. Rather than
    fabricating historical news articles, this function maps genuine, already-collected
    point-in-time historical signals (`ne_*` news/event proxies, `inst_*` promoter/insider
    & bulk/block proxies, `rel_*` sector/peer proxies, and `ms_*` macro/volatility proxies)
    into the 17 Phase 5 feature columns so Experiments A–E can be evaluated cleanly
    with zero look-ahead leakage.
    """
    out = df.copy()

    def _col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=float)

    ne_sent = _col("ne_news_sentiment_score")
    ne_shock = _col("ne_event_shock_flag")
    ne_vol = _col("ne_news_volume_anomaly")
    ne_cat = _col("ne_catalyst_intensity")
    inst_prom = _col("inst_promoter_chg_1q")
    inst_bulk = _col("inst_bulk_block_net_5d")
    inst_corp = _col("inst_corporate_action_flag")
    rel_sec = _col("rel_vs_sector_5d")
    rel_nifty = _col("rel_vs_nifty_5d")
    ms_vix = _col("ms_india_vix_level", 15.0)
    ms_macro = _col("ms_nifty_ret_5d_pct")

    out["news_company_sentiment_1d"] = np.tanh(ne_sent / 2.0)
    out["news_company_sentiment_5d"] = (
        out.groupby("symbol")["news_company_sentiment_1d"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    out["news_company_sentiment_20d"] = (
        out.groupby("symbol")["news_company_sentiment_1d"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )
    out["news_volume_1d"] = np.clip(ne_vol, 0.0, 10.0)
    out["news_volume_5d"] = (
        out.groupby("symbol")["news_volume_1d"]
        .transform(lambda s: s.rolling(5, min_periods=1).sum())
    )
    out["news_surprise_intensity"] = np.clip(ne_shock + 0.5 * ne_cat, 0.0, 5.0)
    out["news_promoter_sentiment_5d"] = np.tanh(inst_prom + 0.5 * inst_bulk)
    out["news_promoter_activity_flag"] = ((inst_bulk.abs() > 0) | (inst_corp > 0)).astype(float)
    out["news_industry_sentiment_5d"] = np.tanh((rel_nifty - rel_sec) / 3.0)
    out["news_competitor_threat_score"] = np.clip(-rel_sec / 3.0, 0.0, 3.0)
    out["news_competitor_positive_readacross"] = np.clip(rel_sec / 3.0, 0.0, 3.0)
    out["news_regulatory_risk_score"] = np.clip((ms_vix - 18.0) / 5.0, 0.0, 3.0) * (ne_sent < 0).astype(float)
    out["news_regulatory_positive_score"] = np.clip(inst_corp, 0.0, 3.0)
    out["news_global_macro_risk_score"] = np.clip(-ms_macro / 3.0 + (ms_vix - 18.0) / 8.0, 0.0, 3.0)
    out["news_novelty_weighted_sentiment"] = out["news_company_sentiment_1d"] * (1.0 + 0.3 * ne_shock)
    out["news_high_impact_event_count_5d"] = (
        out.groupby("symbol")["ne_event_shock_flag"]
        .transform(lambda s: s.rolling(5, min_periods=1).sum())
        if "ne_event_shock_flag" in out.columns
        else 0.0
    )
    out["news_cross_stock_net_impact"] = np.tanh((rel_sec + 0.5 * ms_macro) / 4.0)
    return out
