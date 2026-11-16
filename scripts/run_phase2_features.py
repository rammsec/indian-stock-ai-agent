"""Phase 2 Feature Engineering Runner (Section 7).

Builds the complete point-in-time historical feature matrix across all 8 required
feature families, runs active perturbation anti-leakage verification, evaluates
feature coverage/variance/Spearman Rank IC & collinearity pruning, and saves:
- `/data/features/daily_features.parquet`
- `/data/features/feature_catalog.json`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.data_engine import fetch_raw_company_data
from backend.features.feature_engine import build_historical_feature_matrix
from backend.features.leakage_guard import (
    assert_point_in_time_integrity,
    verify_no_lookahead_leakage,
)
from backend.features.selector import evaluate_and_select_features
from backend.storage.duckdb_store import DuckDBStore

DOCS_DATA_DIR = ROOT_DIR / "docs" / "data"


def _load_company_bundles(symbols: list[str]) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        sym_u = sym.upper()
        json_path = DOCS_DATA_DIR / f"{sym_u}.json"
        if json_path.exists():
            try:
                bundles[sym_u] = json.loads(json_path.read_text())
                continue
            except Exception:
                pass
        try:
            bundles[sym_u] = fetch_raw_company_data(sym_u)
        except Exception as exc:
            print(f"  [WARN] Could not build raw company bundle for {sym_u}: {exc}")
    return bundles


def run_phase2_pipeline() -> dict[str, Any]:
    store = DuckDBStore()
    if not store.equities_parquet.exists() or not store.indices_parquet.exists():
        raise FileNotFoundError(
            "Phase 1 processed Parquet files not found. Run `scripts/run_phase1_foundation.py` first."
        )

    equities_df = pd.read_parquet(store.equities_parquet)
    indices_df = pd.read_parquet(store.indices_parquet)
    symbols = sorted(equities_df["symbol"].unique().tolist())

    print("=" * 76)
    print(f"BHARATALPHA AI — PHASE 2 FEATURE ENGINEERING ({len(symbols)} NSE STOCKS)")
    print("=" * 76)

    print("\n[1/4] Loading point-in-time company financial & shareholding bundles...")
    company_bundles = _load_company_bundles(symbols)
    print(f"  -> Loaded {len(company_bundles)}/{len(symbols)} company fundamental bundles")

    print("\n[2/4] Generating ~190 point-in-time features across 8 feature families...")
    feature_df = build_historical_feature_matrix(
        equities_df=equities_df,
        indices_df=indices_df,
        company_bundles=company_bundles,
        include_targets=True,
    )
    integrity_report = assert_point_in_time_integrity(feature_df)
    print(
        f"  -> Generated {integrity_report['total_rows']} rows x "
        f"{integrity_report['total_candidate_features']} candidate features across "
        f"{integrity_report['symbols_count']} stocks"
    )
    for fam, cnt in integrity_report["family_counts"].items():
        print(f"     • {fam:<40} : {cnt:>3} candidate features")

    print("\n[3/4] Running active future-perturbation anti-leakage checks...")
    leakage_checks: list[dict[str, Any]] = []
    for test_sym in ["RELIANCE", "TCS", "BIRLACABLE"]:
        if test_sym in symbols:
            sub_eq = equities_df[equities_df["symbol"] == test_sym]
            res = verify_no_lookahead_leakage(
                equity_df=sub_eq,
                indices_df=indices_df,
                raw_company_bundle=company_bundles.get(test_sym),
                cutoff_ratio=0.75,
            )
            leakage_checks.append(res)
            print(
                f"  -> {test_sym:<10} | cutoff={res['cutoff_date']} | "
                f"checked={res['checked_features_count']} features | "
                f"max_abs_diff={res['max_abs_diff_before_cutoff']:.2e} | PASSED={res['passed']}"
            )

    print("\n[4/4] Evaluating feature coverage, variance, Spearman Rank IC & collinearity...")
    selection_report = evaluate_and_select_features(
        feature_df=feature_df,
        min_coverage_pct=0.05,
        max_corr_threshold=0.96,
    )
    selection_report["integrity_report"] = integrity_report
    selection_report["leakage_perturbation_checks"] = leakage_checks

    saved_rows = store.upsert_feature_matrix(feature_df)
    store.feature_catalog_json.write_text(json.dumps(selection_report, indent=2))

    print(
        f"  -> Candidate features : {selection_report['total_candidate_features']}\n"
        f"  -> Selected features  : {selection_report['selected_features_count']}\n"
        f"  -> Pruned features    : {selection_report['dropped_features_count']} "
        f"(zero-variance, sparse, or |r| > {selection_report['max_corr_threshold']})\n"
        f"  -> Saved Parquet      : {store.features_parquet} ({saved_rows} rows)\n"
        f"  -> Saved Catalog JSON : {store.feature_catalog_json}"
    )

    print("\nTop 10 Predictive Features by Historical Spearman Rank IC (1D & 5D):")
    for idx, item in enumerate(selection_report["top_25_features_by_ic"][:10], start=1):
        print(
            f"  {idx:>2}. {item['feature']:<34} | {item['family']:<34} | "
            f"IC_1D={item['rank_ic_1d']:+.4f} | IC_5D={item['rank_ic_5d']:+.4f}"
        )

    return selection_report


if __name__ == "__main__":
    run_phase2_pipeline()
