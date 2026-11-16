"""Phase 5 End-to-End Stock Intelligence Pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import BASE_DIR, IST_TZ, PREDICTIONS_DATA_DIR
from ..models import (
    CrossStockImpactItem,
    Phase5IntelligencePayload,
    StockIntelligenceItem,
    StructuredNewsEvent,
)
from ..storage.duckdb_store import DuckDBStore
from .collector import collect_all_pilot_news
from .combiner import build_multi_horizon_outlooks
from .cross_stock import compute_cross_stock_impacts
from .extractor import extract_structured_events_with_quarantine
from .features import compute_pit_news_features
from .monitoring import build_monitoring_snapshot
from .relationships import PILOT_UNIVERSE_5, get_pilot_relationship


def _load_latest_quant_map(predictions_dir: Path | None = None) -> tuple[dict[str, dict[str, Any]], str, str]:
    p_dir = predictions_dir or PREDICTIONS_DATA_DIR
    candidates = [
        p_dir / "latest_predictions.json",
        BASE_DIR / "docs" / "data" / "latest_predictions.json",
    ]
    latest_path = next((p for p in candidates if p.exists()), None)
    if latest_path is None:
        return {}, "Sideways", datetime.now(IST_TZ).isoformat()
    try:
        raw = json.loads(latest_path.read_text(encoding="utf-8"))
        regime = str(raw.get("market_regime") or "Sideways")
        gen_at = str(raw.get("generated_at") or datetime.now(IST_TZ).isoformat())
        items = raw.get("predictions") or raw.get("items") or [] if isinstance(raw, dict) else []
        by_sym = {str(it.get("symbol", "")).upper(): it for it in items if isinstance(it, dict) and it.get("symbol")}
        return by_sym, regime, gen_at
    except Exception:
        return {}, "Sideways", datetime.now(IST_TZ).isoformat()


def run_phase5_intelligence_pipeline(
    symbols: tuple[str, ...] | list[str] = PILOT_UNIVERSE_5,
    use_network: bool = True,
    persist: bool = True,
    store: DuckDBStore | None = None,
) -> Phase5IntelligencePayload:
    """Run the 6-category news collection, extraction, cross-stock & multi-horizon combiner."""
    st = store or DuckDBStore()
    now_ist = datetime.now(IST_TZ)
    now_iso = now_ist.isoformat()

    collected = collect_all_pilot_news(symbols=symbols, use_network=use_network)
    raw_by_sym = collected["by_symbol"]
    warnings: list[str] = list(collected.get("warnings", []))

    events_by_sym: dict[str, list[StructuredNewsEvent]] = {}
    all_events: list[StructuredNewsEvent] = []
    quarantined_total = 0
    for sym, raw_items in raw_by_sym.items():
        evts, quarantined = extract_structured_events_with_quarantine(raw_items)
        quarantined_total += len(quarantined)
        events_by_sym[sym] = evts
        all_events.extend(evts)
    if quarantined_total > 0:
        warnings.append(
            f"Quarantined {quarantined_total} news items due to missing/invalid publication timestamps or entity contamination."
        )

    cross_by_sym = compute_cross_stock_impacts(events_by_sym, universe=tuple(symbols))
    all_cross: list[CrossStockImpactItem] = []
    for clist in cross_by_sym.values():
        all_cross.extend(clist)

    quant_by_sym, regime_label, quant_updated_at = _load_latest_quant_map(st.predictions_dir)
    if not quant_by_sym:
        warnings.append("latest_predictions.json not found or empty; quantitative horizon models marked UNAVAILABLE.")
        data_freshness = "Unavailable"
    else:
        try:
            q_dt = datetime.fromisoformat(quant_updated_at.replace("Z", "+00:00"))
            age_hrs = max(0.0, (now_ist - q_dt.astimezone(IST_TZ)).total_seconds() / 3600.0)
            data_freshness = "Current" if age_hrs <= 36.0 else f"Stale ({age_hrs:.0f}h old)"
        except Exception:
            data_freshness = "Current"

    stocks_out: dict[str, StockIntelligenceItem] = {}
    for sym in [s.upper() for s in symbols]:
        rel = get_pilot_relationship(sym)
        evts = events_by_sym.get(sym, [])
        c_impacts = cross_by_sym.get(sym, [])
        n_feats = compute_pit_news_features(evts, as_of_ts=now_ist, cross_impacts=c_impacts)
        q_pred = quant_by_sym.get(sym)
        trade_eligible = bool(q_pred.get("recommended_action") != "NO TRADE") if q_pred else False

        outlooks = build_multi_horizon_outlooks(
            symbol=sym,
            quant_pred=q_pred,
            news_features=n_feats,
            cross_impacts=c_impacts,
            regime_label=regime_label,
            trade_eligible=trade_eligible,
            data_freshness=data_freshness,
        )

        prom_notes = [e.event_description for e in evts if e.event_category == "promoter_management"][:4]
        comp_notes = [e.event_description for e in evts if e.event_category == "competitor"][:4]
        reg_notes = [e.event_description for e in evts if e.event_category == "regulatory_government"][:4]
        macro_notes = [e.event_description for e in evts if e.event_category == "global_macro"][:4]
        ind_score = n_feats.get("news_industry_sentiment_5d", 0.0)
        ind_outlook = "Positive" if ind_score >= 0.08 else ("Cautious" if ind_score <= -0.08 else "Neutral")

        q_summary = (
            {
                "prob_next_day_up": q_pred.get("prob_next_day_up"),
                "prob_5d_up": q_pred.get("prob_5d_up"),
                "prob_10d_up": q_pred.get("prob_10d_up"),
                "expected_next_day_return_pct": q_pred.get("expected_next_day_return_pct"),
                "expected_5d_return_pct": q_pred.get("expected_5d_return_pct"),
                "expected_10d_return_pct": q_pred.get("expected_10d_return_pct"),
                "recommended_action": q_pred.get("recommended_action"),
                "confidence_score": q_pred.get("confidence_score"),
                "market_regime": regime_label,
            }
            if q_pred
            else {"recommended_action": "UNAVAILABLE", "market_regime": regime_label}
        )

        stocks_out[sym] = StockIntelligenceItem(
            symbol=sym,
            company_name=rel["company_name"],
            sector=rel["sector"],
            observation_date=now_ist.strftime("%Y-%m-%d"),
            quant_score_summary=q_summary,
            news_sentiment_score=round(float(n_feats.get("news_company_sentiment_5d", 0.0)), 4),
            news_features=n_feats,
            promoter_intelligence=prom_notes or ["No recent promoter/insider filings flagged in live window."],
            industry_outlook=ind_outlook,
            competitor_developments=comp_notes or [f"Tracking peers: {', '.join(c['symbol'] for c in rel.get('competitors', []))}"],
            regulatory_developments=reg_notes or rel.get("regulatory_exposures", [])[:2],
            global_exposure=macro_notes or rel.get("global_macro_links", [])[:2],
            events_timeline=sorted(evts, key=lambda x: x.publication_timestamp, reverse=True)[:12],
            cross_stock_impacts=c_impacts[:8],
            horizon_outlooks=outlooks,
        )

    consolidated_feed = sorted(
        [e for e in all_events if not e.is_duplicate_report],
        key=lambda x: x.publication_timestamp,
        reverse=True,
    )[:30]

    monitoring = build_monitoring_snapshot(store=st, data_quality_warnings=warnings)

    payload = Phase5IntelligencePayload(
        generated_at=now_iso,
        model_version="5.0.0-pilot",
        last_news_collection_at=str(collected["collected_at"]),
        last_quant_update_at=quant_updated_at,
        latest_prediction_at=now_iso,
        pilot_universe=[s.upper() for s in symbols],
        stocks=stocks_out,
        consolidated_feed=consolidated_feed,
        monitoring=monitoring,
        data_quality_warnings=warnings,
    )

    if persist:
        if all_events:
            st.upsert_structured_events(all_events)
        if all_cross:
            st.upsert_cross_stock_impacts(all_cross)
        out_json = st.predictions_dir / "phase5_intelligence.json"
        out_json.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    return payload
