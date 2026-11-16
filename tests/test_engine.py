"""End-to-end tests for BharatAlpha AI v3.0 — Indian Market Research & Next-Day Decision Agent."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agents import run_full_stock_research
from backend.main import app

client = TestClient(app)


def test_health_and_universe_5_watchlist() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    u = client.get("/api/universe")
    assert u.status_code == 200
    watchlist = u.json()["universe"]
    assert len(watchlist) == 5
    symbols = [x["symbol"] for x in watchlist]
    assert symbols == ["RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE"]


def test_full_research_pipeline_tataelxsi_and_birlacable() -> None:
    bundle = run_full_stock_research("TATAELXSI")
    assert bundle.overview.symbol == "TATAELXSI"
    assert bundle.technicals.current_price > 0
    assert len(bundle.last_5_annual_reports) == 5
    assert len(bundle.last_5_quarterly_reports) == 5
    assert bundle.cash_and_balance_sheet.total_cash_in_hand_cr > 0
    assert bundle.stockholders.promoters_pct > 0
    assert bundle.rd_and_innovation.latest_rd_spend_cr > 0
    assert len(bundle.agent_committee) == 5
    assert bundle.promoter_and_competitor_intel is not None
    assert len(bundle.promoter_and_competitor_intel.key_leaders) >= 2
    assert len(bundle.promoter_and_competitor_intel.promoter_statements_and_articles) >= 1
    assert len(bundle.promoter_and_competitor_intel.competitors) == 4


def test_api_research_and_movers_endpoints() -> None:
    r = client.get("/api/research/RELIANCE")
    assert r.status_code == 200
    data = r.json()
    assert data["overview"]["symbol"] == "RELIANCE"
    assert len(data["last_5_annual_reports"]) == 5
    assert len(data["agent_committee"]) == 5
    assert "action" in data["next_day_guidance"]
    assert data["promoter_and_competitor_intel"] is not None
    assert len(data["promoter_and_competitor_intel"]["competitors"]) == 4

    m = client.get("/api/movers")
    assert m.status_code == 200
    movers = m.json()
    assert len(movers["top_20_gainers"]) == 20
    assert len(movers["top_20_losers"]) == 20
    assert movers["top_20_gainers"][0]["reason_1"]
    assert movers["top_20_gainers"][0]["reason_2"]
    assert movers["top_20_losers"][0]["reason_1"]
    assert movers["top_20_losers"][0]["reason_2"]
