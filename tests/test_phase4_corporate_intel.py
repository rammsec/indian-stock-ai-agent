"""Phase 4 Automated Test Suite: Institutional, NSE Delivery, Bulk/Block Deals,
Corporate Actions & GitHub Pages Live Integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.collectors.nse_corporate_collector import (
    build_corporate_intelligence_for_symbol,
    extract_corporate_actions_from_equities,
    fetch_official_nse_bulk_and_block_deals,
)
from backend.storage.duckdb_store import DuckDBStore

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_official_nse_bulk_and_block_deals_archive():
    deals = fetch_official_nse_bulk_and_block_deals()
    assert len(deals) > 0
    top = deals[0]
    assert top.deal_type in ("BULK", "BLOCK")
    assert top.side in ("BUY", "SELL")
    assert top.quantity > 0
    assert top.price_inr > 0
    assert top.value_cr >= 0


def test_corporate_actions_extractor_reliance():
    store = DuckDBStore()
    eq_df = pd.read_parquet(store.equities_parquet)
    actions = extract_corporate_actions_from_equities(eq_df, "RELIANCE")
    assert len(actions) > 0
    assert any(a.action_type in ("DIVIDEND", "SPLIT") for a in actions)


def test_build_corporate_intelligence_for_symbol():
    store = DuckDBStore()
    eq_df = pd.read_parquet(store.equities_parquet)
    rel_json = json.loads((ROOT_DIR / "docs" / "data" / "RELIANCE.json").read_text())
    deals = fetch_official_nse_bulk_and_block_deals()

    summary = build_corporate_intelligence_for_symbol(
        symbol="RELIANCE",
        equities_df=eq_df,
        raw_company_bundle=rel_json,
        all_bulk_block_deals=deals,
    )
    assert summary.symbol == "RELIANCE"
    assert summary.promoter_holding_pct is not None and summary.promoter_holding_pct > 0
    assert summary.latest_delivery_pct is not None and summary.latest_delivery_pct > 0
    assert 0.0 <= summary.institutional_conviction_score <= 100.0
    assert summary.smart_money_verdict in (
        "STRONG ACCUMULATION",
        "MODERATE ACCUMULATION",
        "NEUTRAL",
        "DISTRIBUTION",
    )
    assert len(summary.key_signals) >= 2


def test_duckdb_phase4_parquet_views():
    store = DuckDBStore()
    assert store.bulk_block_parquet.exists()
    assert store.corporate_intel_parquet.exists()

    deals_df = store.query_df("SELECT COUNT(*) AS cnt FROM v_nse_bulk_block_deals")
    assert int(deals_df["cnt"].iloc[0]) >= 50

    ci_df = store.query_df("SELECT COUNT(*) AS cnt FROM v_corporate_intel")
    assert int(ci_df["cnt"].iloc[0]) >= 15


def test_github_pages_static_bundles_and_ui_enriched():
    docs_dir = ROOT_DIR / "docs"
    index_html = (docs_dir / "index.html").read_text()
    assert "page-quant" in index_html
    assert "sec-quant-intel" in index_html
    assert "/api/backtest" in index_html
    assert "/api/predictions" in index_html
    assert "/api/corporate-intel" in index_html

    for sym in ["RELIANCE", "TCS", "TATAELXSI", "ITC", "BIRLACABLE"]:
        bundle = json.loads((docs_dir / "data" / f"{sym}.json").read_text())
        assert "quantitative_prediction" in bundle and bundle["quantitative_prediction"] is not None
        assert "corporate_intelligence" in bundle and bundle["corporate_intelligence"] is not None
        assert "market_regime" in bundle and bundle["market_regime"] is not None
        assert "data_quality" in bundle and bundle["data_quality"] is not None
