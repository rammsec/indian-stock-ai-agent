"""Phase 1 Data Foundation Pipeline — Historical Backfill, NSE Bhavcopy/Delivery,
Market Regime Classification, Provenance Tracking, and DuckDB/Parquet Verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.agents import run_full_stock_research  # noqa: E402
from backend.collectors import (  # noqa: E402
    backfill_equity_history_with_delivery,
    collect_and_classify_market_regime,
    collect_indices_history,
)
from backend.config import DEFAULT_WATCHLIST_5  # noqa: E402
from backend.data_engine import fetch_raw_company_data  # noqa: E402
from backend.models import (  # noqa: E402
    CashAndBalanceSheet,
    CompanyOverview,
    RDAndInnovation,
    ReportPeriod,
    StockholderAnalysis,
)
from backend.storage import (  # noqa: E402
    DuckDBStore,
    ensure_storage_directories,
    evaluate_stock_data_quality,
)
from backend.technical_engine import build_technical_snapshot  # noqa: E402

PHASE1_SYMBOLS = [item["symbol"] for item in DEFAULT_WATCHLIST_5] + [
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "SUNPHARMA",
    "BHARTIARTL",
    "LT",
    "SBIN",
    "TRENT",
    "KPITTECH",
    "HFCL",
]


def main() -> None:
    dirs = ensure_storage_directories()
    print(f"[Phase 1] Initialized 5 storage layers: {[k for k in dirs.keys()]}")

    store = DuckDBStore()

    # 1. Backfill Equity History + Official NSE Delivery Reports
    print(f"[Phase 1] Backfilling 2Y OHLCV + Official NSE Delivery reports for {len(PHASE1_SYMBOLS)} stocks...")
    equity_bars, latest_breadth = backfill_equity_history_with_delivery(
        PHASE1_SYMBOLS, period="2y", recent_bhavcopy_days=5
    )
    eq_count = store.upsert_equity_bars(equity_bars)
    print(
        f"[Phase 1] Upserted {eq_count:,} equity daily bars into {store.equities_parquet.name} "
        f"(Latest NSE Breadth A/D: {latest_breadth.get('advance_decline_ratio')})"
    )

    # 2. Collect NSE Broad, Volatility (India VIX) & Sector Indices
    print("[Phase 1] Collecting 2Y NSE Market, Volatility (India VIX) & Sector Indices...")
    index_bars = collect_indices_history(period="2y")
    idx_count = store.upsert_index_bars(index_bars)
    print(f"[Phase 1] Upserted {idx_count:,} index daily bars into {store.indices_parquet.name}")

    # 3. Classify Market Regime
    regime_snap = collect_and_classify_market_regime(
        index_bars=index_bars,
        equity_bars=equity_bars,
        breadth=latest_breadth,
    )
    store.upsert_market_regime(regime_snap)
    docs_data_dir = ROOT_DIR / "docs" / "data"
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    (docs_data_dir / "regime.json").write_text(
        json.dumps(regime_snap.model_dump(), indent=2)
    )
    print(
        f"[Phase 1] Market Regime ({regime_snap.observation_date}): {regime_snap.regime} | "
        f"Nifty 50: {regime_snap.nifty_50_close} ({regime_snap.nifty_50_day_change_pct:+.2f}%) | "
        f"India VIX: {regime_snap.india_vix_close} | Allow Aggressive Longs: {regime_snap.allow_aggressive_longs}"
    )

    # 4. Evaluate Data Quality & Provenance Records for Core Watchlist
    core_syms = [item["symbol"] for item in DEFAULT_WATCHLIST_5]
    all_prov = []
    for sym in core_syms:
        raw = fetch_raw_company_data(sym)
        ov = CompanyOverview(**raw["overview"])
        ann_5 = [ReportPeriod(**r) for r in raw["last_5_annual_reports"]]
        qtr_5 = [ReportPeriod(**r) for r in raw["last_5_quarterly_reports"]]
        cbs = CashAndBalanceSheet(**raw["cash_and_balance_sheet"])
        shp = StockholderAnalysis(**raw["stockholders"])
        rd = RDAndInnovation(**raw["rd_and_innovation"])
        tech = build_technical_snapshot(
            raw.get("price_rows", []),
            fallback_price=ov.current_price,
            fallback_high_52w=ov.high_52w,
            fallback_low_52w=ov.low_52w,
        )
        q_report, prov_records = evaluate_stock_data_quality(
            sym, ov, ann_5, qtr_5, cbs, shp, rd, tech, raw.get("price_rows", [])
        )
        store.upsert_data_quality(q_report)
        all_prov.extend(prov_records)
        print(
            f"[Phase 1] Quality Gate {sym}: Score={q_report.quality_score}/100 | "
            f"Eligible={q_report.trade_eligible} | Verified={q_report.verified_metrics_count} "
            f"Estimated={q_report.estimated_metrics_count} Missing={q_report.missing_metrics_count}"
        )

    prov_count = store.upsert_provenance_records(all_prov)
    print(f"[Phase 1] Upserted {prov_count} provenance records into {store.provenance_parquet.name}")

    # 5. Run DuckDB SQL Verification Query
    summary_df = store.query_df(
        """
        SELECT
            symbol,
            COUNT(*) AS total_bars,
            MIN(observation_date) AS first_date,
            MAX(observation_date) AS last_date,
            COUNT(delivery_pct) AS verified_delivery_bars,
            ROUND(AVG(delivery_pct), 2) AS avg_delivery_pct
        FROM v_equities_daily
        WHERE symbol IN ('RELIANCE', 'TCS', 'TATAELXSI', 'ITC', 'BIRLACABLE')
        GROUP BY symbol
        ORDER BY symbol
        """
    )
    print("\n[Phase 1] DuckDB Analytical View Verification (`v_equities_daily`):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
