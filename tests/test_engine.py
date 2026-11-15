"""End-to-end tests for BharatAlpha AI — Indian Market Research & Next-Day Decision Agent."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agents import run_full_stock_research
from backend.main import app

client = TestClient(app)


def test_health_and_universe() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    u = client.get("/api/universe")
    assert u.status_code == 200
    assert len(u.json()["universe"]) >= 15


def test_full_research_pipeline_sunpharma() -> None:
    bundle = run_full_stock_research("SUNPHARMA")
    assert bundle.overview.symbol == "SUNPHARMA"
    assert bundle.technicals.current_price > 0
    assert len(bundle.last_5_annual_reports) == 5
    assert len(bundle.last_5_quarterly_reports) == 5
    assert bundle.cash_and_balance_sheet.total_cash_in_hand_cr > 0
    assert bundle.stockholders.promoters_pct > 0
    assert len(bundle.stockholders.named_stockholders) > 0
    assert bundle.rd_and_innovation.latest_rd_spend_cr > 0
    assert len(bundle.agent_committee) == 5
    assert bundle.next_day_guidance.entry_zone_low > 0
    assert bundle.next_day_guidance.target_1_next_day > 0
    assert bundle.next_day_guidance.stop_loss > 0


def test_api_research_endpoint() -> None:
    r = client.get("/api/research/RELIANCE")
    assert r.status_code == 200
    data = r.json()
    assert data["overview"]["symbol"] == "RELIANCE"
    assert len(data["last_5_annual_reports"]) == 5
    assert len(data["agent_committee"]) == 5
    assert "action" in data["next_day_guidance"]
