"""Contract and presence tests for the Intraday Trading Desk dashboard."""

import json
from pathlib import Path
import pytest

TRADER_DIR = Path("docs/trader")
DATA_FILE = TRADER_DIR / "data" / "trader_state.json"


def test_dashboard_file_structure():
    """Assert all required static assets for /trader/ are present."""
    assert (TRADER_DIR / "index.html").exists()
    assert (TRADER_DIR / "styles.css").exists()
    assert (TRADER_DIR / "app.js").exists()
    assert DATA_FILE.exists()


def test_dashboard_html_sections():
    """Assert index.html contains all 7 mandated terminal sections."""
    html = (TRADER_DIR / "index.html").read_text(encoding="utf-8")

    assert "Central Market Intelligence & Council Posture" in html
    assert "5-Minute Intraday Stock Scanner" in html
    assert "Central Agent Debate & Evidence Cross-Examination" in html
    assert "NSE ↔ BSE Cross-Exchange Executable Arbitrage" in html
    assert "Simulated Paper Portfolio & Double-Entry Ledger" in html
    assert "Strategy Research Library & Backtests" in html
    assert "AI Model Allocation & Token Budget" in html
    assert "HISTORICAL REPLAY & SIMULATION ONLY" in html
    assert "data-provenance-banner" in html


def test_trader_state_json_contract():
    """Assert trader_state.json contains all required schema keys."""
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    # Root keys
    for k in [
        "metadata",
        "data_provenance",
        "market_overview",
        "council_debate",
        "stock_scanner",
        "cross_exchange_arbitrage",
        "strategy_performance",
        "paper_portfolio",
        "cost_monitoring",
    ]:
        assert k in data, f"Missing key: {k}"

    # Data Provenance
    prov = data["data_provenance"]
    assert prov["source_mode"] in ["SYNTHETIC_TEST", "HISTORICAL_REAL"]
    assert prov["datasets_count"] >= 5
    assert len(prov["datasets"][0]["sha256_hash"]) == 64

    # Market Overview
    mo = data["market_overview"]
    assert "nifty_50" in mo
    assert "bank_nifty" in mo
    assert "india_vix" in mo
    assert "central_posture" in mo
    assert mo["central_posture"]["posture"] in ["BUY_BULLISH", "SELL_BEARISH", "NEUTRAL", "HOLD_HIGH_RISK"]

    # Stock Scanner
    assert len(data["stock_scanner"]) >= 5
    for stk in data["stock_scanner"]:
        assert "symbol" in stk
        assert "current_price" in stk
        assert "vwap" in stk
        assert "trading_agent_action" in stk

    # Cross-Exchange Arbitrage
    assert len(data["cross_exchange_arbitrage"]) > 0
    for arb in data["cross_exchange_arbitrage"]:
        assert "isin" in arb
        assert "gross_spread_inr" in arb
        assert "total_friction_inr" in arb
        assert "net_profit_inr" in arb
        assert "status" in arb

    # Paper Portfolio
    port = data["paper_portfolio"]
    assert port["initial_capital_inr"] == 1_000_000.0
    assert "cash_inr" in port
    assert "total_equity_inr" in port
    assert "max_drawdown_pct" in port

    # Cost Monitoring
    cm = data["cost_monitoring"]
    assert cm["daily_budget_usd"] == 0.50
    assert "model_routing_map" in cm
    assert "llm_calls_count" in cm
    assert "fallback_calls_count" in cm
