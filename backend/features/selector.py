"""Feature Evaluation, IC Ranking & Collinearity Pruning (Phase 2 — Section 7).

Evaluates the ~185 candidate numerical features across all 8 feature families:
1. Coverage Ratio (`non_null_pct`)
2. Cross-sectional & temporal dispersion (`std`)
3. Spearman Rank Information Coefficient (`rank_ic_1d`, `rank_ic_5d`) against
   `target_next_day_ret` and `target_5d_ret` on historical bars.
4. Pairwise collinearity pruning (`|corr| > max_corr_threshold`), retaining the
   feature with stronger `|rank_ic_1d|` when two features in the same family are
   near-duplicates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .feature_engine import CANDIDATE_FEATURE_FAMILIES
from .leakage_guard import get_candidate_feature_columns


def _resolve_family_name(col: str) -> str:
    for prefix, label in CANDIDATE_FEATURE_FAMILIES.items():
        if col.startswith(prefix):
            return label
    return "Other"


def evaluate_and_select_features(
    feature_df: pd.DataFrame,
    min_coverage_pct: float = 0.05,
    max_corr_threshold: float = 0.96,
) -> dict[str, Any]:
    """Evaluate all candidate features and select the clean non-redundant feature set."""
    cand_cols = get_candidate_feature_columns(feature_df)
    total_rows = max(1, len(feature_df))

    # Historical subset where 1D and 5D forward returns are known
    valid_1d = feature_df["target_next_day_ret"].notna() if "target_next_day_ret" in feature_df.columns else pd.Series(False, index=feature_df.index)
    valid_5d = feature_df["target_5d_ret"].notna() if "target_5d_ret" in feature_df.columns else pd.Series(False, index=feature_df.index)

    y_1d = pd.to_numeric(feature_df.loc[valid_1d, "target_next_day_ret"], errors="coerce") if valid_1d.any() else pd.Series(dtype=float)
    y_5d = pd.to_numeric(feature_df.loc[valid_5d, "target_5d_ret"], errors="coerce") if valid_5d.any() else pd.Series(dtype=float)

    metrics_by_col: dict[str, dict[str, Any]] = {}
    stage1_survivors: list[str] = []
    dropped_features: dict[str, str] = {}

    for col in cand_cols:
        s = pd.to_numeric(feature_df[col], errors="coerce")
        non_null_count = int(s.notna().sum())
        coverage_pct = round(non_null_count / total_rows * 100.0, 2)
        std_val = float(s.std()) if non_null_count > 1 else 0.0
        if np.isnan(std_val):
            std_val = 0.0

        # Compute Spearman Rank IC on historical bars with at least 15 non-null pairs
        ic_1d = 0.0
        if valid_1d.any():
            s_1d = s.loc[valid_1d]
            both_1d = s_1d.notna() & y_1d.notna()
            if int(both_1d.sum()) >= 15 and float(s_1d[both_1d].std()) > 1e-12:
                corr_val = s_1d[both_1d].rank().corr(y_1d[both_1d].rank())
                if pd.notna(corr_val):
                    ic_1d = float(corr_val)

        ic_5d = 0.0
        if valid_5d.any():
            s_5d = s.loc[valid_5d]
            both_5d = s_5d.notna() & y_5d.notna()
            if int(both_5d.sum()) >= 15 and float(s_5d[both_5d].std()) > 1e-12:
                corr_val5 = s_5d[both_5d].rank().corr(y_5d[both_5d].rank())
                if pd.notna(corr_val5):
                    ic_5d = float(corr_val5)

        combined_ic_score = round(0.6 * abs(ic_1d) + 0.4 * abs(ic_5d), 5)

        metrics_by_col[col] = {
            "feature": col,
            "family": _resolve_family_name(col),
            "non_null_count": non_null_count,
            "coverage_pct": coverage_pct,
            "std": round(std_val, 6),
            "rank_ic_1d": round(ic_1d, 5),
            "rank_ic_5d": round(ic_5d, 5),
            "combined_abs_ic": combined_ic_score,
        }

        if non_null_count == 0 or coverage_pct < min_coverage_pct:
            dropped_features[col] = f"insufficient_coverage ({coverage_pct:.2f}% < {min_coverage_pct:.2f}%)"
        elif std_val <= 1e-10:
            dropped_features[col] = "zero_variance_constant"
        else:
            stage1_survivors.append(col)

    # Stage 2: Pairwise collinearity pruning within Stage 1 survivors
    # Sort survivors by `combined_abs_ic` descending so stronger predictors are kept first
    stage1_sorted = sorted(
        stage1_survivors,
        key=lambda c: metrics_by_col[c]["combined_abs_ic"],
        reverse=True,
    )

    corr_matrix = (
        feature_df[stage1_sorted]
        .apply(pd.to_numeric, errors="coerce")
        .corr()
        .abs()
    )

    selected_features: list[str] = []
    for col in stage1_sorted:
        is_redundant = False
        for kept in selected_features:
            c_val = corr_matrix.loc[col, kept] if (col in corr_matrix.index and kept in corr_matrix.columns) else 0.0
            if pd.notna(c_val) and float(c_val) > max_corr_threshold:
                dropped_features[col] = f"collinear_with_{kept} (|r|={float(c_val):.3f})"
                is_redundant = True
                break
        if not is_redundant:
            selected_features.append(col)

    # Restore logical family ordering for `selected_features`
    selected_features_ordered = [c for c in cand_cols if c in set(selected_features)]

    # Family summary counts
    family_summary: dict[str, dict[str, int]] = {}
    for prefix, label in CANDIDATE_FEATURE_FAMILIES.items():
        fam_cands = [c for c in cand_cols if c.startswith(prefix)]
        fam_sel = [c for c in selected_features_ordered if c.startswith(prefix)]
        family_summary[label] = {
            "candidate_count": len(fam_cands),
            "selected_count": len(fam_sel),
            "dropped_count": len(fam_cands) - len(fam_sel),
        }

    top_25_by_ic = sorted(
        [metrics_by_col[c] for c in selected_features_ordered],
        key=lambda m: m["combined_abs_ic"],
        reverse=True,
    )[:25]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_rows_evaluated": total_rows,
        "total_candidate_features": len(cand_cols),
        "selected_features_count": len(selected_features_ordered),
        "dropped_features_count": len(dropped_features),
        "max_corr_threshold": max_corr_threshold,
        "min_coverage_pct": min_coverage_pct,
        "family_summary": family_summary,
        "selected_features": selected_features_ordered,
        "top_25_features_by_ic": top_25_by_ic,
        "dropped_features": dropped_features,
        "all_feature_metrics": [metrics_by_col[c] for c in cand_cols],
    }
