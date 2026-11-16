"""Cross-Stock & Macro Relationship Impact Engine (Phase 5 — Section 5)."""

from __future__ import annotations

from ..models import CrossStockImpactItem, StructuredNewsEvent
from .relationships import PILOT_UNIVERSE_5, RELATIONSHIP_MAP

_DEMAND_READACROSS_TERMS = ("demand", "industry", "sector", "margin", "exports", "capex", "5g", "ftth", "bfsi", "rural")
_DISRUPTION_TERMS = ("disruption", "recall", "penalty", "fine", "delay", "shutdown", "outage", "loss", "miss", "probe")
_ZERO_SUM_TERMS = ("contract", "order", "win", "wins", "market share", "launch", "tariff", "deal")


def _asymmetric_competitor_direction(
    event: StructuredNewsEvent,
    asymmetry_mode: str,
) -> tuple[str, str]:
    """Determine asymmetric impact direction & rationale on a pilot stock from a competitor event."""
    desc = event.event_description.lower()
    sent = event.sentiment

    if any(t in desc for t in _DISRUPTION_TERMS) and sent == "negative":
        return "positive", "Competitor operational disruption or regulatory setback creates market-share opportunity."

    if asymmetry_mode == "competitive_zero_sum" and any(t in desc for t in _ZERO_SUM_TERMS) and sent == "positive":
        return "negative", "Direct rival contract win / share gain represents a competitive headwind."

    if any(t in desc for t in _DEMAND_READACROSS_TERMS):
        if sent == "positive":
            return "positive", "Peer demand/sector expansion signals constructive industry read-across."
        if sent == "negative":
            return "negative", "Peer demand slowdown signals cautious sector read-across."

    if asymmetry_mode == "demand_bellwether":
        return sent, "Global/tier-1 bellwether guidance read-across for sector order visibility."

    return "neutral", "Competitor update monitored with balanced immediate read-across."


def compute_cross_stock_impacts(
    events_by_symbol: dict[str, list[StructuredNewsEvent]],
    universe: tuple[str, ...] = PILOT_UNIVERSE_5,
) -> dict[str, list[CrossStockImpactItem]]:
    """Map structured events across the 5-stock pilot relationship graph."""
    impacts: dict[str, list[CrossStockImpactItem]] = {sym: [] for sym in universe}

    # 1. Direct competitor, regulatory & global macro events recorded under each pilot stock
    for sym in universe:
        rel = RELATIONSHIP_MAP.get(sym, {})
        comp_modes = {
            c.get("symbol", "").upper(): c.get("asymmetry", "demand_readacross_and_rival")
            for c in rel.get("competitors", [])
        }
        seen_ids: set[str] = set()

        for evt in events_by_symbol.get(sym, []):
            if evt.is_duplicate_report or evt.event_id in seen_ids:
                continue
            seen_ids.add(evt.event_id)

            if evt.event_category == "competitor":
                comp_entity = (evt.related_competitor_or_industry or "Competitor").upper()
                mode = comp_modes.get(comp_entity, "demand_readacross_and_rival")
                direction, rationale = _asymmetric_competitor_direction(evt, mode)
                impacts[sym].append(
                    CrossStockImpactItem(
                        affected_company=sym,
                        originating_event_id=evt.event_id,
                        originating_entity=evt.related_competitor_or_industry or "Sector Competitor",
                        relationship_type="competitor",
                        expected_impact_direction=direction,  # type: ignore[arg-type]
                        impact_magnitude=evt.impact_magnitude,
                        confidence=round(evt.classification_confidence * 0.85, 3),
                        expected_duration=evt.expected_impact_duration,
                        rationale=rationale,
                    )
                )
            elif evt.event_category == "regulatory_government":
                impacts[sym].append(
                    CrossStockImpactItem(
                        affected_company=sym,
                        originating_event_id=evt.event_id,
                        originating_entity=evt.original_source,
                        relationship_type="regulatory_exposure",
                        expected_impact_direction=evt.sentiment,
                        impact_magnitude=evt.impact_magnitude,
                        confidence=round(evt.classification_confidence * 0.90, 3),
                        expected_duration=evt.expected_impact_duration,
                        rationale=f"Regulatory / policy read-across: {evt.event_description[:110]}",
                    )
                )
            elif evt.event_category == "global_macro":
                impacts[sym].append(
                    CrossStockImpactItem(
                        affected_company=sym,
                        originating_event_id=evt.event_id,
                        originating_entity="Global Macro",
                        relationship_type="global_macro",
                        expected_impact_direction=evt.sentiment,
                        impact_magnitude=evt.impact_magnitude,
                        confidence=round(evt.classification_confidence * 0.80, 3),
                        expected_duration=evt.expected_impact_duration,
                        rationale=f"Macro / commodity link: {evt.event_description[:110]}",
                    )
                )

    # 2. Cross-pilot spillovers:
    #    - RELIANCE (Jio 5G/FTTH capex) -> BIRLACABLE (supplier_customer)
    #    - TCS (tier-1 IT demand) -> TATAELXSI (sector_peer)
    spillover_rules = [
        ("RELIANCE", "BIRLACABLE", "supplier_customer", ("jio", "5g", "ftth", "fiber", "telecom", "broadband", "capex")),
        ("TCS", "TATAELXSI", "sector_peer", ("it", "genai", "ai", "bfsi", "deal", "margin", "attrition", "guidance")),
    ]
    for src_sym, dst_sym, rel_type, keywords in spillover_rules:
        if dst_sym not in impacts:
            continue
        for evt in events_by_symbol.get(src_sym, []):
            if evt.is_duplicate_report or evt.event_category not in ("company_news", "industry_sector"):
                continue
            desc = evt.event_description.lower()
            if any(kw in desc for kw in keywords):
                impacts[dst_sym].append(
                    CrossStockImpactItem(
                        affected_company=dst_sym,
                        originating_event_id=evt.event_id,
                        originating_entity=src_sym,
                        relationship_type=rel_type,  # type: ignore[arg-type]
                        expected_impact_direction=evt.sentiment,
                        impact_magnitude="medium" if evt.impact_magnitude == "high" else "low",
                        confidence=round(evt.classification_confidence * 0.75, 3),
                        expected_duration="2-5 trading sessions",
                        rationale=f"{src_sym} {rel_type.replace('_', ' ')} spillover to {dst_sym}: {evt.event_description[:100]}",
                    )
                )

    return impacts
