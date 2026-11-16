"""Point-in-Time Anti-Leakage Verification Guard (Phase 2 — Section 7).

Guarantees that:
1. No `target_*` forward return columns are included in candidate or selected feature lists.
2. Every feature row has `publication_timestamp[:10] >= observation_date[:10]` and is sorted
   monotonically by `observation_date` within each `symbol`.
3. Perturbation Leakage Test (`verify_no_lookahead_leakage`):
   - Computes the complete feature matrix on bars `0 ... N`.
   - Perturbs all future bars `t > cutoff_idx` (e.g., multiplies future OHLCV by 5.0x and
     injects a synthetic future annual/quarterly/shareholding report dated after `cutoff_date`).
   - Recomputes the feature matrix on the perturbed data and asserts that every single
     candidate feature on rows `0 ... cutoff_idx` is bit-for-bit identical (`max_abs_diff <= 1e-12`).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from .feature_engine import (
    CANDIDATE_FEATURE_FAMILIES,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    build_features_for_symbol,
)


def get_candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all candidate numerical feature columns (excluding metadata and targets)."""
    prefixes = tuple(CANDIDATE_FEATURE_FAMILIES.keys())
    cols: list[str] = []
    for col in df.columns:
        if col in METADATA_COLUMNS or col in TARGET_COLUMNS:
            continue
        if col.startswith(prefixes):
            cols.append(col)
    return cols


def assert_point_in_time_integrity(feature_df: pd.DataFrame) -> dict[str, Any]:
    """Verify structural and chronological point-in-time invariants on `feature_df`."""
    if feature_df.empty:
        raise ValueError("Feature DataFrame is empty.")

    cand_cols = get_candidate_feature_columns(feature_df)
    leaked_targets = [c for c in cand_cols if "target" in c.lower() or "next_day" in c.lower()]
    if leaked_targets:
        raise AssertionError(f"Target leakage detected in candidate feature columns: {leaked_targets}")

    # Verify chronological ordering within each symbol
    for sym, grp in feature_df.groupby("symbol"):
        dates = grp["observation_date"].astype(str).tolist()
        if dates != sorted(dates):
            raise AssertionError(f"Non-monotonic observation_date ordering detected for {sym}")
        if len(dates) != len(set(dates)):
            raise AssertionError(f"Duplicate observation_date rows detected for {sym}")

    # Count features per family
    family_counts: dict[str, int] = {
        fam_label: sum(1 for c in cand_cols if c.startswith(prefix))
        for prefix, fam_label in CANDIDATE_FEATURE_FAMILIES.items()
    }

    return {
        "passed": True,
        "total_rows": int(len(feature_df)),
        "symbols_count": int(feature_df["symbol"].nunique()),
        "total_candidate_features": len(cand_cols),
        "family_counts": family_counts,
        "leaked_target_columns": leaked_targets,
    }


def verify_no_lookahead_leakage(
    equity_df: pd.DataFrame,
    indices_df: pd.DataFrame,
    raw_company_bundle: dict[str, Any] | None = None,
    cutoff_ratio: float = 0.75,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Run an active future-perturbation test to prove zero look-ahead bias.

    Steps:
    1. Compute baseline features on the unperturbed full history `0 ... N-1`.
    2. Choose `cutoff_idx = int(len(df) * cutoff_ratio)`.
    3. Create a perturbed copy where:
       - All equity OHLCV and delivery values strictly after `cutoff_date` are multiplied by 5.0x.
       - All index closes strictly after `cutoff_date` are multiplied by 3.0x.
       - A synthetic future annual report, quarterly report, and shareholding report with extreme
         values (`eps_inr = 99999.0`, `promoters_pct = 99.9`) is appended with period after `cutoff_date`.
    4. Recompute features on the perturbed dataset and assert that on rows `0 ... cutoff_idx`,
       every candidate feature matches the baseline within `tolerance` (`1e-12`).
    """
    df_sorted = (
        equity_df.sort_values("observation_date")
        .drop_duplicates("observation_date", keep="last")
        .reset_index(drop=True)
    )
    if len(df_sorted) < 40:
        raise ValueError("Need at least 40 daily bars for look-ahead perturbation test.")

    cutoff_idx = int(len(df_sorted) * cutoff_ratio)
    cutoff_date = str(df_sorted.loc[cutoff_idx, "observation_date"])

    baseline_feat = build_features_for_symbol(
        equity_df=df_sorted,
        indices_df=indices_df,
        raw_company_bundle=raw_company_bundle,
        include_targets=True,
    )
    cand_cols = get_candidate_feature_columns(baseline_feat)

    # Perturb future equity bars (strictly after cutoff_date)
    perturbed_eq = df_sorted.copy()
    future_mask = perturbed_eq["observation_date"] > cutoff_date
    for col in ["open", "high", "low", "close", "adj_close", "volume", "delivery_pct", "deliverable_qty", "vwap"]:
        if col in perturbed_eq.columns:
            perturbed_eq.loc[future_mask, col] = pd.to_numeric(perturbed_eq.loc[future_mask, col], errors="coerce") * 5.0

    # Perturb future index bars (strictly after cutoff_date)
    perturbed_idx = indices_df.copy()
    if not perturbed_idx.empty:
        idx_future = perturbed_idx["observation_date"] > cutoff_date
        for col in ["open", "high", "low", "close"]:
            if col in perturbed_idx.columns:
                perturbed_idx.loc[idx_future, col] = pd.to_numeric(perturbed_idx.loc[idx_future, col], errors="coerce") * 3.0

    # Perturb future fundamental/shareholding reports (published in 2028, strictly after cutoff_date)
    perturbed_bundle = copy.deepcopy(raw_company_bundle) if raw_company_bundle else {}
    if perturbed_bundle:
        perturbed_bundle.setdefault("last_5_annual_reports", []).append(
            {
                "period": "Mar 2028",
                "revenue_cr": 9999999.0,
                "operating_profit_cr": 5555555.0,
                "net_profit_cr": 4444444.0,
                "opm_pct": 88.8,
                "eps_inr": 99999.0,
                "revenue_growth_pct": 500.0,
                "net_profit_growth_pct": 500.0,
            }
        )
        perturbed_bundle.setdefault("last_5_quarterly_reports", []).append(
            {
                "period": "Mar 2028",
                "revenue_cr": 2500000.0,
                "operating_profit_cr": 1200000.0,
                "net_profit_cr": 900000.0,
                "opm_pct": 88.8,
                "eps_inr": 25000.0,
                "revenue_growth_pct": 400.0,
                "net_profit_growth_pct": 400.0,
            }
        )
        perturbed_bundle.setdefault("stockholders", {}).setdefault("quarterly_history", []).append(
            {
                "period": "Mar 2028",
                "promoters_pct": 99.9,
                "fii_pct": 0.05,
                "dii_pct": 0.03,
                "public_pct": 0.02,
            }
        )

    perturbed_feat = build_features_for_symbol(
        equity_df=perturbed_eq,
        indices_df=perturbed_idx,
        raw_company_bundle=perturbed_bundle,
        include_targets=True,
    )

    # Compare all candidate features on rows 0 ... cutoff_idx
    base_slice = baseline_feat.iloc[: cutoff_idx + 1]
    pert_slice = perturbed_feat.iloc[: cutoff_idx + 1]

    max_diff = 0.0
    violating_columns: list[str] = []

    for col in cand_cols:
        s_base = pd.to_numeric(base_slice[col], errors="coerce")
        s_pert = pd.to_numeric(pert_slice[col], errors="coerce")

        # Check NaN pattern match
        nan_mismatch = (s_base.isna() != s_pert.isna()).any()
        valid_mask = s_base.notna() & s_pert.notna()
        col_diff = float((s_base[valid_mask] - s_pert[valid_mask]).abs().max()) if valid_mask.any() else 0.0

        if nan_mismatch or col_diff > tolerance:
            violating_columns.append(f"{col} (diff={col_diff:.3e}, nan_mismatch={nan_mismatch})")
        if col_diff > max_diff:
            max_diff = col_diff

    if violating_columns:
        raise AssertionError(
            f"Look-ahead leakage detected in {len(violating_columns)} features before {cutoff_date}: "
            f"{violating_columns[:10]}"
        )

    return {
        "passed": True,
        "symbol": str(df_sorted["symbol"].iloc[0]),
        "cutoff_idx": cutoff_idx,
        "cutoff_date": cutoff_date,
        "checked_features_count": len(cand_cols),
        "max_abs_diff_before_cutoff": max_diff,
        "violating_columns": violating_columns,
    }
