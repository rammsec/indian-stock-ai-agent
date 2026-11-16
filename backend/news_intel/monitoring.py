"""Phase 5.2 Monitoring Snapshot — Multi-Horizon Accuracy, AUC, Brier, Rank IC, Coverage & Drift."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import BASE_DIR, IST_TZ
from ..models import HorizonMonitoringMetrics, MonitoringSnapshot
from ..storage.duckdb_store import DuckDBStore

_MIN_EVAL_SAMPLES = 10


def _compute_horizon_metrics(
    df: pd.DataFrame,
    horizon: str,
    prob_col: str,
    ret_col: str,
    coverage_pct: float | None,
    freshness: str,
    bench_prob_col: str | None = None,
) -> HorizonMonitoringMetrics:
    if df.empty or prob_col not in df.columns or ret_col not in df.columns:
        return HorizonMonitoringMetrics(
            horizon=horizon,  # type: ignore[arg-type]
            evaluated_outcomes_count=0,
            prediction_coverage_pct=coverage_pct,
            data_freshness=freshness,
            status="INSUFFICIENT_DATA",
        )

    eval_df = df.dropna(subset=[prob_col, ret_col]).copy()
    n_eval = len(eval_df)
    if n_eval < _MIN_EVAL_SAMPLES:
        return HorizonMonitoringMetrics(
            horizon=horizon,  # type: ignore[arg-type]
            evaluated_outcomes_count=n_eval,
            prediction_coverage_pct=coverage_pct,
            data_freshness=freshness,
            status="INSUFFICIENT_DATA",
        )

    recent = eval_df.tail(50)
    y_ret = pd.to_numeric(recent[ret_col], errors="coerce")
    y_true = (y_ret > 0).astype(int).values
    p_up = pd.to_numeric(recent[prob_col], errors="coerce").clip(0.01, 0.99).values
    pred_up = (p_up >= 0.50).astype(int)

    acc = round(float((pred_up == y_true).mean() * 100.0), 2)
    brier = round(float(brier_score_loss(y_true, p_up)), 4)
    try:
        auc = round(float(roc_auc_score(y_true, p_up)), 4) if len(set(y_true)) > 1 else None
    except Exception:
        auc = None
    ic_raw = pd.Series(p_up).rank().corr(y_ret.rank())
    rank_ic = round(float(ic_raw), 4) if pd.notna(ic_raw) else None

    bench_diff: float | None = None
    if bench_prob_col and bench_prob_col in recent.columns:
        b_sub = recent.dropna(subset=[bench_prob_col])
        if len(b_sub) >= _MIN_EVAL_SAMPLES:
            b_true = (pd.to_numeric(b_sub[ret_col], errors="coerce") > 0).astype(int).values
            b_pred = (pd.to_numeric(b_sub[bench_prob_col], errors="coerce") >= 0.50).astype(int).values
            b_acc = float((b_pred == b_true).mean() * 100.0)
            bench_diff = round(acc - b_acc, 2)

    status = "WARNING" if (acc < 48.0 or brier > 0.27) else "OK"
    return HorizonMonitoringMetrics(
        horizon=horizon,  # type: ignore[arg-type]
        evaluated_outcomes_count=n_eval,
        directional_accuracy_pct=acc,
        roc_auc=auc,
        brier_score=brier,
        rank_ic=rank_ic,
        prediction_coverage_pct=coverage_pct,
        data_freshness=freshness,
        model_vs_benchmark_accuracy_diff_pct=bench_diff,
        status=status,  # type: ignore[arg-type]
    )


def _live_coverage_and_freshness(store: DuckDBStore, now_ist: datetime) -> tuple[dict[str, float | None], str]:
    cov: dict[str, float | None] = {"1D": None, "5D": None, "10D": None}
    freshness = "Unavailable"
    for cand in (store.predictions_dir / "latest_predictions.json", BASE_DIR / "docs" / "data" / "latest_predictions.json"):
        if cand.exists():
            try:
                raw = json.loads(cand.read_text(encoding="utf-8"))
                gen_at = str(raw.get("generated_at") or "")
                if gen_at:
                    dt = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
                    age_hrs = max(0.0, (now_ist - dt.astimezone(IST_TZ)).total_seconds() / 3600.0)
                    freshness = "Current" if age_hrs <= 36.0 else f"Stale ({age_hrs:.0f}h old)"
                preds = raw.get("predictions") or []
                if preds:
                    n = len(preds)
                    cov["1D"] = round(sum(1 for p in preds if p.get("prob_next_day_up") is not None) / n * 100.0, 1)
                    cov["5D"] = round(sum(1 for p in preds if p.get("prob_5d_up") is not None) / n * 100.0, 1)
                    cov["10D"] = round(sum(1 for p in preds if p.get("prob_10d_up") is not None) / n * 100.0, 1)
                break
            except Exception:
                pass
    return cov, freshness


def build_monitoring_snapshot(
    store: DuckDBStore | None = None,
    data_quality_warnings: list[str] | None = None,
) -> MonitoringSnapshot:
    """Compute horizon-specific live monitoring metrics from `prediction_log.parquet`."""
    st = store or DuckDBStore()
    now_dt = datetime.now(IST_TZ)
    now_ist = now_dt.isoformat()
    dq_warnings = list(data_quality_warnings or [])
    failure_alerts: list[str] = []

    cov_map, freshness = _live_coverage_and_freshness(st, now_dt)

    total_logged = 0
    evaluated_count = 0
    rolling_acc: float | None = None
    brier_recent: float | None = None
    drift_status: str = "INSUFFICIENT_DATA"
    horizon_metrics: dict[str, HorizonMonitoringMetrics] = {}

    if st.predictions_parquet.exists():
        try:
            df = pd.read_parquet(st.predictions_parquet)
            total_logged = len(df)
            specs = (
                ("1D", "prob_next_day_up", "realized_next_day_return_pct", "momentum_prob_up"),
                ("5D", "prob_5d_up", "realized_5d_return_pct", None),
                ("10D", "prob_10d_up", "realized_10d_return_pct", None),
            )
            for h, p_col, r_col, b_col in specs:
                hm = _compute_horizon_metrics(df, h, p_col, r_col, cov_map.get(h), freshness, bench_prob_col=b_col)
                horizon_metrics[h] = hm
                if hm.status == "WARNING" and hm.evaluated_outcomes_count >= _MIN_EVAL_SAMPLES:
                    failure_alerts.append(
                        f"Model calibration/accuracy drift warning ({h}): accuracy={hm.directional_accuracy_pct}%, Brier={hm.brier_score}."
                    )

            hm_1d = horizon_metrics["1D"]
            evaluated_count = hm_1d.evaluated_outcomes_count
            rolling_acc = hm_1d.directional_accuracy_pct
            brier_recent = hm_1d.brier_score
            drift_status = hm_1d.status
            if evaluated_count < _MIN_EVAL_SAMPLES:
                dq_warnings.append(
                    f"Only {evaluated_count} realized 1D outcomes evaluated so far (minimum {_MIN_EVAL_SAMPLES} required for drift check)."
                )
        except Exception as exc:
            failure_alerts.append(f"Failed to read prediction_log.parquet: {exc}")
    else:
        for h in ("1D", "5D", "10D"):
            horizon_metrics[h] = HorizonMonitoringMetrics(
                horizon=h,  # type: ignore[arg-type]
                prediction_coverage_pct=cov_map.get(h),
                data_freshness=freshness,
                status="INSUFFICIENT_DATA",
            )
        dq_warnings.append("prediction_log.parquet not yet populated; live outcome tracking initialized.")

    return MonitoringSnapshot(
        evaluated_at=now_ist,
        total_logged_predictions=total_logged,
        evaluated_outcomes_count=evaluated_count,
        rolling_directional_accuracy_pct=rolling_acc,
        brier_score_recent=brier_recent,
        drift_status=drift_status,  # type: ignore[arg-type]
        horizon_metrics=horizon_metrics,
        data_quality_warnings=dq_warnings,
        failure_alerts=failure_alerts,
    )
