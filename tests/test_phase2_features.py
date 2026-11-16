"""Automated Test Suite for Phase 2 — Feature Engineering, Point-in-Time Anti-Leakage & Selection."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from backend.features import (
    CANDIDATE_FEATURE_FAMILIES,
    TARGET_COLUMNS,
    assert_point_in_time_integrity,
    build_features_for_symbol,
    evaluate_and_select_features,
    get_candidate_feature_columns,
    verify_no_lookahead_leakage,
)
from backend.storage.duckdb_store import DuckDBStore


def test_phase2_feature_catalog_and_parquet_exist() -> None:
    """Verify `/data/features/daily_features.parquet` and `feature_catalog.json` exist and are queryable via DuckDB."""
    store = DuckDBStore()
    assert store.features_parquet.exists(), "daily_features.parquet must exist in /data/features/"
    assert store.feature_catalog_json.exists(), "feature_catalog.json must exist in /data/features/"

    catalog = json.loads(store.feature_catalog_json.read_text())
    cand_count = catalog["total_candidate_features"]
    assert 150 <= cand_count <= 250, f"Expected 150-250 candidate features, got {cand_count}"
    assert catalog["selected_features_count"] >= 90
    assert catalog["dropped_features_count"] > 0

    # Verify all 8 feature families are populated
    fam_summary = catalog["family_summary"]
    assert len(fam_summary) == 8
    for fam_name, stats in fam_summary.items():
        assert stats["candidate_count"] > 0, f"Family {fam_name} has 0 candidate features"

    # Query via DuckDB `v_daily_features` view
    q = store.query_df(
        "SELECT symbol, COUNT(*) AS rows_cnt, MIN(observation_date) AS min_dt, MAX(observation_date) AS max_dt "
        "FROM v_daily_features GROUP BY symbol ORDER BY symbol"
    )
    assert len(q) >= 15
    assert int(q["rows_cnt"].sum()) >= 7000


def test_point_in_time_sebi_lodr_publication_lag_no_backfill() -> None:
    """Verify a financial report for `Mar 2026` (published `2026-05-30`) is NaN on `2026-04-15` and visible on `2026-06-01`."""
    dates = pd.date_range("2026-03-01", "2026-06-15", freq="B").strftime("%Y-%m-%d")
    eq_df = pd.DataFrame(
        {
            "symbol": "TESTCO",
            "observation_date": dates,
            "open": np.linspace(100.0, 120.0, len(dates)),
            "high": np.linspace(102.0, 122.0, len(dates)),
            "low": np.linspace(99.0, 118.0, len(dates)),
            "close": np.linspace(101.0, 121.0, len(dates)),
            "volume": np.full(len(dates), 100000.0),
            "delivery_pct": np.nan,
            "deliverable_qty": np.nan,
            "vwap": np.nan,
            "turnover_cr": np.nan,
            "trades_count": np.nan,
        }
    )
    idx_df = pd.DataFrame(
        {
            "index_code": "NIFTY_50",
            "observation_date": dates,
            "open": np.linspace(22000.0, 23000.0, len(dates)),
            "high": np.linspace(22100.0, 23100.0, len(dates)),
            "low": np.linspace(21900.0, 22900.0, len(dates)),
            "close": np.linspace(22050.0, 23050.0, len(dates)),
        }
    )
    bundle = {
        "overview": {"sector": "Information Technology", "market_cap_cr": 50000.0},
        "last_5_annual_reports": [
            {
                "period": "Mar 2026",
                "revenue_cr": 10000.0,
                "operating_profit_cr": 2500.0,
                "net_profit_cr": 1800.0,
                "opm_pct": 25.0,
                "eps_inr": 45.0,
                "revenue_growth_pct": 14.0,
                "net_profit_growth_pct": 16.0,
            }
        ],
        "last_5_quarterly_reports": [
            {
                "period": "Mar 2026",
                "revenue_cr": 2600.0,
                "operating_profit_cr": 650.0,
                "net_profit_cr": 480.0,
                "opm_pct": 25.0,
                "eps_inr": 12.0,
                "revenue_growth_pct": 4.2,
                "net_profit_growth_pct": 5.1,
            }
        ],
        "stockholders": {
            "quarterly_history": [
                {
                    "period": "Mar 2026",
                    "promoters_pct": 68.5,
                    "fii_pct": 14.2,
                    "dii_pct": 10.3,
                    "public_pct": 7.0,
                }
            ]
        },
    }

    f_df = build_features_for_symbol(eq_df, idx_df, raw_company_bundle=bundle)

    # Before statutory publication date (May 30, 2026 for Mar 2026), annual/qtr/shp features MUST be NaN
    before_pub = f_df[f_df["observation_date"] == "2026-04-15"].iloc[0]
    assert np.isnan(before_pub["fq_opm_annual_pct"]), "Mar 2026 annual OPM leaked before May 30, 2026!"
    assert np.isnan(before_pub["eg_qtr_opm_pct"]), "Mar 2026 quarterly OPM leaked before May 30, 2026!"
    assert np.isnan(before_pub["inst_promoters_pct"]), "Mar 2026 promoter holding leaked before May 30, 2026!"
    assert np.isnan(before_pub["fq_dynamic_pe"]), "Mar 2026 EPS leaked into dynamic P/E before May 30, 2026!"

    # After statutory publication date (e.g. June 1, 2026), features MUST be populated
    after_pub = f_df[f_df["observation_date"] == "2026-06-01"].iloc[0]
    assert after_pub["fq_opm_annual_pct"] == 25.0
    assert after_pub["eg_qtr_opm_pct"] == 25.0
    assert after_pub["inst_promoters_pct"] == 68.5
    assert after_pub["fq_dynamic_pe"] > 0


def test_active_future_perturbation_zero_lookahead_leakage() -> None:
    """Verify perturbing future prices/volumes/reports after cutoff_date causes 0.0 change before cutoff_date."""
    store = DuckDBStore()
    eq_df = pd.read_parquet(store.equities_parquet)
    idx_df = pd.read_parquet(store.indices_parquet)
    rel_df = eq_df[eq_df["symbol"] == "RELIANCE"]

    res = verify_no_lookahead_leakage(
        equity_df=rel_df,
        indices_df=idx_df,
        raw_company_bundle=None,
        cutoff_ratio=0.75,
    )
    assert res["passed"] is True
    assert res["max_abs_diff_before_cutoff"] <= 1e-12
    assert len(res["violating_columns"]) == 0


def test_missing_delivery_and_fno_oi_never_fabricated_as_zero() -> None:
    """Verify missing historical delivery % and unreported exchange F&O OI remain NaN, never 0.0."""
    store = DuckDBStore()
    feat_df = pd.read_parquet(store.features_parquet)

    # Historical rows from 2025 (Yahoo Finance backfill) have no NSE Bhavcopy delivery -> must be NaN
    hist_2025 = feat_df[feat_df["observation_date"] < "2026-01-01"]
    assert hist_2025["vd_delivery_pct"].isna().all(), "Historical Yahoo bars without NSE Bhavcopy must keep vd_delivery_pct as NaN"
    assert (hist_2025["vd_has_verified_delivery"] == 0.0).all()

    # Recent NSE Bhavcopy bars must have verified non-null delivery_pct
    recent_verified = feat_df[feat_df["vd_has_verified_delivery"] == 1.0]
    assert len(recent_verified) >= 15, "Recent NSE Bhavcopy trading days must have verified delivery_pct"
    assert (recent_verified["vd_delivery_pct"] > 0).all()

    # Unreported historical exchange OI/PCR must remain NaN (never fabricated)
    assert feat_df["dv_futures_oi"].isna().all()
    assert feat_df["dv_options_pcr"].isna().all()


def test_target_columns_separated_from_candidate_features() -> None:
    """Ensure forward target columns (`target_*`) are completely excluded from candidate & selected features."""
    store = DuckDBStore()
    feat_df = pd.read_parquet(store.features_parquet)
    cand_cols = get_candidate_feature_columns(feat_df)
    integrity = assert_point_in_time_integrity(feat_df)

    assert integrity["passed"] is True
    for t_col in TARGET_COLUMNS:
        assert t_col in feat_df.columns
        assert t_col not in cand_cols
