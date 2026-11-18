"""Priority 2 — Reporting Source of Truth Contract & Builder.

Read-only module that normalizes published metrics into a canonical provenance
envelope containing:
- model_version
- horizon
- evaluation_period (start_date, end_date)
- data_population
- canonical_predictions_count
- matured_outcomes_count
- pending_outcomes_count
- observation_classification (PROSPECTIVE | HISTORICAL | MIXED)
- generated_at
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..config import IST_TZ
from ..news_intel.forward_validation import compute_forward_horizon_metrics
from ..storage.duckdb_store import DuckDBStore


class EvaluationPeriod(BaseModel):
    start_date: str | None = None
    end_date: str | None = None


class CanonicalMetricRecord(BaseModel):
    """Standardized provenance envelope required for every published metric."""

    metric_source: str
    model_version: str
    horizon: Literal["1D", "5D", "10D", "ALL"]
    evaluation_period: EvaluationPeriod
    data_population: str
    canonical_predictions_count: int
    matured_outcomes_count: int
    pending_outcomes_count: int
    observation_classification: Literal["PROSPECTIVE", "HISTORICAL", "MIXED"]
    generated_at: str
    is_authoritative: bool = True
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ReportingSourceOfTruth(BaseModel):
    """Unified read-only reporting truth across forward validation and legacy monitoring."""

    generated_at: str
    base_commit_sha: str = "8f328b8cdb8f7a1e508c1c47a648849945873013"
    authoritative_source: str = "forward_predictions.parquet + forward_outcomes.parquet (canonical)"
    horizon_records: dict[str, CanonicalMetricRecord] = Field(default_factory=dict)
    legacy_monitoring_record: CanonicalMetricRecord | None = None


def build_reporting_source_of_truth(
    store: DuckDBStore | None = None,
    baseline_start_date: str = "2026-11-16",
) -> ReportingSourceOfTruth:
    """Build the standardized reporting source of truth without mutating production stores."""
    st = store or DuckDBStore()
    now_iso = datetime.now(IST_TZ).isoformat()

    snaps_df = pd.read_parquet(st.forward_predictions_parquet) if st.forward_predictions_parquet.exists() else pd.DataFrame()
    outs_df = pd.read_parquet(st.forward_outcomes_parquet) if st.forward_outcomes_parquet.exists() else pd.DataFrame()

    if not snaps_df.empty and "is_duplicate_run" in snaps_df.columns:
        canon_snaps = snaps_df[~snaps_df["is_duplicate_run"].fillna(False).astype(bool)].copy()
    else:
        canon_snaps = snaps_df.copy()

    canon_ids = set(canon_snaps["prediction_id"].astype(str)) if not canon_snaps.empty else set()
    canon_outs = (
        outs_df[outs_df["prediction_id"].astype(str).isin(canon_ids)].copy()
        if (not outs_df.empty and canon_ids)
        else pd.DataFrame(columns=outs_df.columns) if not outs_df.empty else pd.DataFrame()
    )

    obs_dates = sorted(canon_snaps["observation_date"].astype(str).unique().tolist()) if not canon_snaps.empty else []
    eval_period = EvaluationPeriod(
        start_date=obs_dates[0] if obs_dates else None,
        end_date=obs_dates[-1] if obs_dates else None,
    )

    if not obs_dates:
        obs_class: Literal["PROSPECTIVE", "HISTORICAL", "MIXED"] = "PROSPECTIVE"
    elif all(d >= baseline_start_date for d in obs_dates):
        obs_class = "PROSPECTIVE"
    elif all(d < baseline_start_date for d in obs_dates):
        obs_class = "HISTORICAL"
    else:
        obs_class = "MIXED"

    model_ver = (
        str(canon_snaps["model_version"].dropna().iloc[-1])
        if (not canon_snaps.empty and "model_version" in canon_snaps.columns and canon_snaps["model_version"].notna().any())
        else "5.3.0-pilot"
    )

    horizon_records: dict[str, CanonicalMetricRecord] = {}
    for h in ("1D", "5D", "10D"):
        he = compute_forward_horizon_metrics(canon_snaps, canon_outs, h, model_version=model_ver)
        horizon_records[h] = CanonicalMetricRecord(
            metric_source="forward_validation",
            model_version=he.model_version,
            horizon=h,  # type: ignore[arg-type]
            evaluation_period=eval_period,
            data_population="pilot_5_canonical_snapshots",
            canonical_predictions_count=he.total_snapshots,
            matured_outcomes_count=he.matured_count,
            pending_outcomes_count=he.pending_count,
            observation_classification=obs_class,
            generated_at=now_iso,
            is_authoritative=True,
            metrics={
                "directional_accuracy_pct": he.directional_accuracy_pct,
                "roc_auc": he.roc_auc,
                "brier_score": he.brier_score,
                "rank_ic": he.rank_ic,
                "gross_return_pct": he.gross_return_pct,
                "net_return_pct": he.net_return_pct,
                "status": he.status,
            },
        )

    # Non-authoritative reconciliation record for legacy prediction_log.parquet / phase5_intelligence.json
    legacy_rec: CanonicalMetricRecord | None = None
    if st.predictions_parquet.exists():
        plog = pd.read_parquet(st.predictions_parquet)
        pl_dates = sorted(plog["observation_date"].astype(str).unique().tolist()) if not plog.empty else []
        pl_eval = plog.dropna(subset=["prob_next_day_up", "realized_next_day_return_pct"]) if not plog.empty else pd.DataFrame()
        pl_canon = plog.drop_duplicates(subset=["symbol", "observation_date"]) if not plog.empty else pd.DataFrame()
        versions = sorted(plog["model_version"].dropna().astype(str).unique().tolist()) if (not plog.empty and "model_version" in plog.columns) else []
        legacy_rec = CanonicalMetricRecord(
            metric_source="phase5_intelligence_monitoring (prediction_log.parquet)",
            model_version="+".join(versions) if versions else "unknown",
            horizon="1D",
            evaluation_period=EvaluationPeriod(
                start_date=pl_dates[0] if pl_dates else None,
                end_date=pl_dates[-1] if pl_dates else None,
            ),
            data_population=f"quant_{plog['symbol'].nunique() if not plog.empty else 0}_stocks_unfiltered_tail50",
            canonical_predictions_count=len(pl_canon),
            matured_outcomes_count=len(pl_eval),
            pending_outcomes_count=max(0, len(plog) - len(pl_eval)),
            observation_classification="MIXED" if len(versions) > 1 else "PROSPECTIVE",
            generated_at=now_iso,
            is_authoritative=False,
            notes=[
                "Non-authoritative for pilot forward validation: mixes 15-stock quant universe, multiple model versions, and un-deduplicated runs."
            ],
        )

    return ReportingSourceOfTruth(
        generated_at=now_iso,
        horizon_records=horizon_records,
        legacy_monitoring_record=legacy_rec,
    )
