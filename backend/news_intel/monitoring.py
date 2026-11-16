"""Phase 5 Monitoring Snapshot — Accuracy, Drift, Data Quality & Alerts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from ..config import IST_TZ
from ..models import MonitoringSnapshot
from ..storage.duckdb_store import DuckDBStore


def build_monitoring_snapshot(
    store: DuckDBStore | None = None,
    data_quality_warnings: list[str] | None = None,
) -> MonitoringSnapshot:
    """Compute live monitoring metrics from immutable `prediction_log.parquet`."""
    st = store or DuckDBStore()
    now_ist = datetime.now(IST_TZ).isoformat()
    dq_warnings = list(data_quality_warnings or [])
    failure_alerts: list[str] = []

    total_logged = 0
    evaluated_count = 0
    rolling_acc: float | None = None
    brier_recent: float | None = None
    drift_status: str = "INSUFFICIENT_DATA"

    if st.predictions_parquet.exists():
        try:
            df = pd.read_parquet(st.predictions_parquet)
            total_logged = len(df)
            if "realized_next_day_return_pct" in df.columns and "prob_next_day_up" in df.columns:
                eval_df = df.dropna(subset=["realized_next_day_return_pct", "prob_next_day_up"]).copy()
                evaluated_count = len(eval_df)
                if evaluated_count >= 10:
                    recent = eval_df.tail(50)
                    y_true = (pd.to_numeric(recent["realized_next_day_return_pct"], errors="coerce") > 0).astype(int).values
                    p_up = pd.to_numeric(recent["prob_next_day_up"], errors="coerce").clip(0.01, 0.99).values
                    pred_up = (p_up >= 0.50).astype(int)
                    rolling_acc = round(float((pred_up == y_true).mean() * 100.0), 2)
                    brier_recent = round(float(brier_score_loss(y_true, p_up)), 4)

                    if rolling_acc < 48.0 or brier_recent > 0.27:
                        drift_status = "WARNING"
                        failure_alerts.append(
                            f"Model calibration/accuracy drift warning: recent 1D accuracy={rolling_acc}%, Brier={brier_recent}."
                        )
                    else:
                        drift_status = "OK"
                else:
                    dq_warnings.append(
                        f"Only {evaluated_count} realized 1D outcomes evaluated so far (minimum 10 required for drift check)."
                    )
        except Exception as exc:
            failure_alerts.append(f"Failed to read prediction_log.parquet: {exc}")
    else:
        dq_warnings.append("prediction_log.parquet not yet populated; live outcome tracking initialized.")

    return MonitoringSnapshot(
        evaluated_at=now_ist,
        total_logged_predictions=total_logged,
        evaluated_outcomes_count=evaluated_count,
        rolling_directional_accuracy_pct=rolling_acc,
        brier_score_recent=brier_recent,
        drift_status=drift_status,  # type: ignore[arg-type]
        data_quality_warnings=dq_warnings,
        failure_alerts=failure_alerts,
    )
