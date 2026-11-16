"""Phase 4 Institutional & Corporate Intelligence Runner (Sections 5C & 5D).

Collects and normalizes:
1. Official NSE Bulk & Block Deals (`bulk.csv` & `block.csv` archives)
2. Corporate Actions (verified historical Dividends & Stock Splits)
3. Quarterly Promoter, FII & DII ownership changes (`1Q` & `4Q` deltas)
4. Official NSE Bhavcopy Delivery % vs 20D average & F&O positioning heuristics
5. Persists:
   - `/data/processed/nse_bulk_block_deals.parquet` (`v_nse_bulk_block_deals`)
   - `/data/processed/corporate_intelligence.parquet` (`v_corporate_intel`)
   - `docs/data/corporate_intel.json`
   - Enriches all `docs/data/{SYMBOL}.json` files with `corporate_intelligence`, `market_regime`, and `quantitative_prediction`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.collectors.nse_corporate_collector import (
    build_corporate_intelligence_for_symbol,
    fetch_official_nse_bulk_and_block_deals,
)
from backend.data_engine import fetch_raw_company_data
from backend.storage.duckdb_store import DuckDBStore

DOCS_DATA_DIR = ROOT_DIR / "docs" / "data"


def run_phase4_pipeline() -> dict[str, Any]:
    store = DuckDBStore()
    if not store.equities_parquet.exists():
        raise FileNotFoundError(
            "Phase 1 equities_daily.parquet not found. Run `scripts/run_phase1_foundation.py` first."
        )

    equities_df = pd.read_parquet(store.equities_parquet)
    excluded_stems = {
        "UNIVERSE",
        "SCANNER",
        "META",
        "MOVERS",
        "REGIME",
        "BACKTEST_SUMMARY",
        "LATEST_PREDICTIONS",
        "CORPORATE_INTEL",
    }
    json_syms = [
        p.stem.upper()
        for p in DOCS_DATA_DIR.glob("*.json")
        if p.stem.upper() not in excluded_stems
    ]
    symbols = sorted(set(equities_df["symbol"].unique().tolist()) | set(json_syms))

    print("=" * 84)
    print("BHARATALPHA AI — PHASE 4 INSTITUTIONAL, BULK/BLOCK & CORPORATE INTELLIGENCE")
    print("=" * 84)

    print("\n[1/3] Fetching Official NSE Bulk & Block Deal Archives...")
    deals = fetch_official_nse_bulk_and_block_deals()
    deals_cnt = store.upsert_bulk_block_deals(deals)
    top_deals = sorted(deals, key=lambda d: d.value_cr, reverse=True)[:25]
    print(f"  -> Parsed & stored {deals_cnt} official NSE Bulk/Block deals in {store.bulk_block_parquet.name}")
    for d in top_deals[:5]:
        print(
            f"     • [{d.deal_type}] {d.deal_date} | {d.symbol:<10} | {d.side:<4} | "
            f"₹{d.value_cr:>7.2f} Cr @ ₹{d.price_inr:.2f} | {d.client_name[:38]}"
        )

    regime_data = None
    regime_path = DOCS_DATA_DIR / "regime.json"
    if regime_path.exists():
        try:
            regime_data = json.loads(regime_path.read_text())
        except Exception:
            regime_data = None

    quality_by_sym: dict[str, dict[str, Any]] = {}
    if store.quality_parquet.exists():
        try:
            q_df = pd.read_parquet(store.quality_parquet)
            for _, row in q_df.iterrows():
                quality_by_sym[str(row["symbol"]).upper()] = {
                    "symbol": str(row["symbol"]).upper(),
                    "evaluated_at": str(row.get("evaluated_at", "")),
                    "quality_score": float(row.get("quality_score", 85.0)),
                    "trade_eligible": bool(row.get("trade_eligible", True)),
                    "has_minimum_price_history": bool(row.get("has_minimum_price_history", True)),
                    "price_bars_count": int(row.get("price_bars_count", 248)),
                    "missing_critical_fields": json.loads(str(row.get("missing_critical_fields_json", "[]"))),
                    "warnings": json.loads(str(row.get("warnings_json", "[]"))),
                    "critical_failures": json.loads(str(row.get("critical_failures_json", "[]"))),
                }
        except Exception:
            pass

    print(f"\n[2/3] Building Institutional, Delivery, Corporate Action & F&O Intelligence for {len(symbols)} stocks...")
    summaries = []
    for sym in symbols:
        sym_u = sym.upper()
        json_path = DOCS_DATA_DIR / f"{sym_u}.json"
        bundle: dict[str, Any] | None = None
        if json_path.exists():
            try:
                bundle = json.loads(json_path.read_text())
            except Exception:
                bundle = None
        if not bundle or "stockholders" not in bundle:
            try:
                bundle = fetch_raw_company_data(sym_u)
            except Exception:
                bundle = {}

        corp_sum = build_corporate_intelligence_for_symbol(
            symbol=sym_u,
            equities_df=equities_df,
            raw_company_bundle=bundle,
            all_bulk_block_deals=deals,
        )
        summaries.append(corp_sum)

        # Enrich existing `docs/data/{SYM}.json` if present
        if json_path.exists() and isinstance(bundle, dict) and "overview" in bundle:
            bundle["corporate_intelligence"] = corp_sum.model_dump()
            if regime_data:
                bundle["market_regime"] = regime_data
            if sym_u in quality_by_sym:
                bundle["data_quality"] = quality_by_sym[sym_u]
            json_path.write_text(json.dumps(bundle, indent=2))

        print(
            f"  • {sym_u:<11} | Score={corp_sum.institutional_conviction_score:>4.1f}/100 ({corp_sum.smart_money_verdict:<21}) | "
            f"Prom={str(corp_sum.promoter_holding_pct)+'%':<7} | 4Q Inst Flow={str(corp_sum.net_institutional_flow_4q)+'%':<7} | "
            f"Deliv={str(corp_sum.latest_delivery_pct)+'%':<7} | {corp_sum.fno_positioning_heuristic}"
        )

    corp_cnt = store.upsert_corporate_intelligence(summaries)
    print(f"\n[3/3] Saved {corp_cnt} corporate intelligence records to {store.corporate_intel_parquet.name}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_exchange_bulk_block_deals": deals_cnt,
        "top_exchange_bulk_block_deals": [d.model_dump() for d in top_deals],
        "stocks": {s.symbol: s.model_dump() for s in summaries},
    }
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / "corporate_intel.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run_phase4_pipeline()
