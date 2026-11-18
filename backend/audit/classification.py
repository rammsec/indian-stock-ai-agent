"""Read-only provenance and maturity classification for BharatAlpha AI predictions & outcomes.

Classifies every (prediction, horizon) record without modifying production files:
- provenance_class: PROSPECTIVE | HISTORICAL | DUPLICATE | INVALID
- maturity_class:   MATURED | PENDING | UNAVAILABLE
- reporting_class:  single label with precedence DUPLICATE > INVALID > HISTORICAL > MATURED/PENDING
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import IST_TZ, is_nse_trading_day

HORIZON_SESSIONS: dict[str, int] = {"1D": 1, "5D": 5, "10D": 10}
PROB_COLS: dict[str, str] = {"1D": "prob_1d_up", "5D": "prob_5d_up", "10D": "prob_10d_up"}
NSE_OPEN_TIME = time(9, 15, 0)
NSE_CLOSE_TIME = time(15, 30, 0)


class ClassifiedRecord(BaseModel):
    prediction_id: str
    symbol: str
    observation_date: str
    horizon: str
    nse_session: str | None = None
    prediction_timestamp: str | None = None
    target_entry_date: str | None = None
    target_exit_date: str | None = None
    provenance_class: str  # PROSPECTIVE | HISTORICAL | DUPLICATE | INVALID
    maturity_class: str  # MATURED | PENDING | UNAVAILABLE
    reporting_class: str  # DUPLICATE | INVALID | HISTORICAL | MATURED | PENDING
    reason: str = ""


class HorizonClassificationSummary(BaseModel):
    horizon: str
    total_stored: int = 0
    unique_canonical: int = 0
    prospective: int = 0
    historical: int = 0
    duplicate: int = 0
    invalid: int = 0
    pending: int = 0
    matured: int = 0


class ProspectiveHorizonMetrics(BaseModel):
    horizon: str
    n_matured_prospective: int = 0
    min_sample_threshold: int = 10
    statistically_sufficient: bool = False
    directional_accuracy_pct: float | None = None
    roc_auc: float | None = None
    brier_score: float | None = None
    mean_gross_return_pct: float | None = None
    mean_net_return_pct: float | None = None
    note: str = ""


class LegacyLogClassificationSummary(BaseModel):
    total_stored: int = 0
    unique_canonical_symbol_date: int = 0
    duplicate: int = 0
    evaluated: int = 0
    pending: int = 0
    model_versions: dict[str, int] = Field(default_factory=dict)
    deduped_1d_accuracy_pct: float | None = None


class ClassificationReport(BaseModel):
    horizon_summaries: dict[str, HorizonClassificationSummary] = Field(default_factory=dict)
    prospective_metrics: dict[str, ProspectiveHorizonMetrics] = Field(default_factory=dict)
    legacy_prediction_log: LegacyLogClassificationSummary | None = None
    records: list[ClassifiedRecord] = Field(default_factory=list)


def _next_nse_trading_day(obs_date_str: str) -> date | None:
    try:
        d = date.fromisoformat(str(obs_date_str).strip()[:10])
    except Exception:
        return None
    cur = d + timedelta(days=1)
    for _ in range(30):
        if is_nse_trading_day(cur):
            return cur
        cur += timedelta(days=1)
    return None


def _nth_nse_trading_day_from(start_date: date, n_sessions: int) -> date:
    """Return the n-th NSE trading session starting at start_date (n=1 -> start_date)."""
    if n_sessions <= 1:
        return start_date
    cur = start_date
    count = 1
    for _ in range(90):
        if count >= n_sessions:
            return cur
        cur += timedelta(days=1)
        if is_nse_trading_day(cur):
            count += 1
    return cur


def _parse_ist_timestamp(ts: Any) -> datetime | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    s = str(ts).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(IST_TZ) if dt.tzinfo else dt.replace(tzinfo=IST_TZ)
    except Exception:
        return None


def classify_forward_predictions(
    predictions_df: pd.DataFrame,
    outcomes_df: pd.DataFrame | None = None,
    horizons: tuple[str, ...] = ("1D", "5D", "10D"),
    min_samples: int = 10,
) -> tuple[list[ClassifiedRecord], dict[str, HorizonClassificationSummary], dict[str, ProspectiveHorizonMetrics]]:
    """Classify each (prediction, horizon) into provenance_class, maturity_class, and reporting_class."""
    if predictions_df is None or predictions_df.empty:
        empty_sum = {h: HorizonClassificationSummary(horizon=h) for h in horizons}
        empty_met = {
            h: ProspectiveHorizonMetrics(
                horizon=h, min_sample_threshold=min_samples, note="No forward predictions available."
            )
            for h in horizons
        }
        return [], empty_sum, empty_met

    preds = predictions_df.copy()
    if "is_duplicate_run" not in preds.columns:
        preds["is_duplicate_run"] = False
    preds["is_duplicate_run"] = preds["is_duplicate_run"].fillna(False).astype(bool)

    out_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if outcomes_df is not None and not outcomes_df.empty:
        for _, row in outcomes_df.iterrows():
            key = (str(row.get("prediction_id", "")), str(row.get("horizon", "")))
            out_lookup[key] = row.to_dict()

    all_records: list[ClassifiedRecord] = []
    summaries: dict[str, HorizonClassificationSummary] = {}
    metrics: dict[str, ProspectiveHorizonMetrics] = {}

    for h in horizons:
        n_sess = HORIZON_SESSIONS[h]
        candidates: list[dict[str, Any]] = []

        for _, row in preds.iterrows():
            pid = str(row.get("prediction_id", ""))
            sym = str(row.get("symbol", "")).upper().strip()
            obs = str(row.get("observation_date", "")).strip()
            sess = None if pd.isna(row.get("nse_session")) else str(row.get("nse_session"))
            ts_raw = None if pd.isna(row.get("prediction_timestamp")) else str(row.get("prediction_timestamp"))
            is_dup_run = bool(row.get("is_duplicate_run", False))

            entry_dt = _next_nse_trading_day(obs)
            exit_dt = _nth_nse_trading_day_from(entry_dt, n_sess) if entry_dt else None
            ts_ist = _parse_ist_timestamp(ts_raw)

            out_row = out_lookup.get((pid, h), {})
            raw_status = str(out_row.get("status", "PENDING")).upper().strip()
            maturity_class = raw_status if raw_status in ("MATURED", "PENDING", "UNAVAILABLE") else "PENDING"

            if is_dup_run:
                prov = "DUPLICATE"
                reason = "Flagged is_duplicate_run=True in forward_predictions."
            elif entry_dt is None or exit_dt is None or ts_ist is None:
                prov = "INVALID"
                reason = "Missing or unparseable observation_date or prediction_timestamp."
            else:
                entry_open = datetime.combine(entry_dt, NSE_OPEN_TIME, tzinfo=IST_TZ)
                exit_close = datetime.combine(exit_dt, NSE_CLOSE_TIME, tzinfo=IST_TZ)
                if ts_ist < entry_open:
                    prov = "PROSPECTIVE"
                    reason = f"Predicted before entry open ({entry_open.isoformat()})."
                elif ts_ist < exit_close:
                    prov = "INVALID"
                    reason = (
                        f"Look-ahead contamination: prediction_timestamp ({ts_ist.isoformat()}) "
                        f">= entry open ({entry_open.isoformat()}) and < exit close ({exit_close.isoformat()})."
                    )
                else:
                    prov = "HISTORICAL"
                    reason = (
                        f"Post-outcome snapshot: prediction_timestamp ({ts_ist.isoformat()}) "
                        f">= exit close ({exit_close.isoformat()})."
                    )

            candidates.append(
                {
                    "prediction_id": pid,
                    "symbol": sym,
                    "observation_date": obs,
                    "horizon": h,
                    "nse_session": sess,
                    "prediction_timestamp": ts_raw,
                    "target_entry_date": entry_dt.isoformat() if entry_dt else None,
                    "target_exit_date": exit_dt.isoformat() if exit_dt else None,
                    "provenance_class": prov,
                    "maturity_class": maturity_class,
                    "reason": reason,
                    "_ts_sort": ts_ist.isoformat() if ts_ist else "9999-99-99",
                    "_out_row": out_row,
                    "_pred_row": row,
                }
            )

        # Deduplicate PROSPECTIVE candidates targeting the same (symbol, target_entry_date, horizon)
        prospective_seen: dict[tuple[str, str, str], str] = {}
        prospective_indices = [i for i, c in enumerate(candidates) if c["provenance_class"] == "PROSPECTIVE"]
        prospective_indices.sort(key=lambda i: (candidates[i]["_ts_sort"], candidates[i]["prediction_id"]))

        for idx in prospective_indices:
            c = candidates[idx]
            tkey = (c["symbol"], c["target_entry_date"] or "", h)
            if tkey not in prospective_seen:
                prospective_seen[tkey] = c["prediction_id"]
            else:
                c["provenance_class"] = "DUPLICATE"
                c["reason"] = (
                    f"Duplicate target session ({c['target_entry_date']}) for {c['symbol']} {h}; "
                    f"earlier prospective snapshot {prospective_seen[tkey]} takes precedence."
                )

        # Assign final reporting_class (precedence: DUPLICATE > INVALID > HISTORICAL > MATURED/PENDING)
        h_records: list[ClassifiedRecord] = []
        matured_prospective_rows: list[dict[str, Any]] = []
        for c in candidates:
            prov = c["provenance_class"]
            mat = c["maturity_class"]
            if prov in ("DUPLICATE", "INVALID", "HISTORICAL"):
                rep = prov
            else:
                rep = mat

            rec = ClassifiedRecord(
                prediction_id=c["prediction_id"],
                symbol=c["symbol"],
                observation_date=c["observation_date"],
                horizon=h,
                nse_session=c["nse_session"],
                prediction_timestamp=c["prediction_timestamp"],
                target_entry_date=c["target_entry_date"],
                target_exit_date=c["target_exit_date"],
                provenance_class=prov,
                maturity_class=mat,
                reporting_class=rep,
                reason=c["reason"],
            )
            h_records.append(rec)
            all_records.append(rec)

            if rep == "MATURED":
                out_row = c["_out_row"]
                pred_row = c["_pred_row"]
                prob_col = PROB_COLS[h]
                matured_prospective_rows.append(
                    {
                        "prob_up": pred_row.get(prob_col),
                        "gross_return_pct": out_row.get("gross_return_pct"),
                        "net_return_pct": out_row.get("net_return_pct"),
                    }
                )

        unique_canon = int((~preds["is_duplicate_run"]).sum())
        summaries[h] = HorizonClassificationSummary(
            horizon=h,
            total_stored=len(preds),
            unique_canonical=unique_canon,
            prospective=sum(1 for r in h_records if r.provenance_class == "PROSPECTIVE"),
            historical=sum(1 for r in h_records if r.provenance_class == "HISTORICAL"),
            duplicate=sum(1 for r in h_records if r.provenance_class == "DUPLICATE"),
            invalid=sum(1 for r in h_records if r.provenance_class == "INVALID"),
            pending=sum(1 for r in h_records if r.reporting_class == "PENDING"),
            matured=sum(1 for r in h_records if r.reporting_class == "MATURED"),
        )

        metrics[h] = _compute_prospective_metrics(h, matured_prospective_rows, min_samples=min_samples)

    return all_records, summaries, metrics


def _compute_prospective_metrics(
    horizon: str,
    rows: list[dict[str, Any]],
    min_samples: int = 10,
) -> ProspectiveHorizonMetrics:
    if not rows:
        return ProspectiveHorizonMetrics(
            horizon=horizon,
            n_matured_prospective=0,
            min_sample_threshold=min_samples,
            statistically_sufficient=False,
            note="No matured genuinely prospective observations.",
        )

    df = pd.DataFrame(rows).dropna(subset=["prob_up", "gross_return_pct", "net_return_pct"])
    n = len(df)
    if n == 0:
        return ProspectiveHorizonMetrics(
            horizon=horizon,
            n_matured_prospective=0,
            min_sample_threshold=min_samples,
            statistically_sufficient=False,
            note="No valid matured prospective observations.",
        )

    y_gross = pd.to_numeric(df["gross_return_pct"], errors="coerce").values
    y_net = pd.to_numeric(df["net_return_pct"], errors="coerce").values
    y_up = (y_gross > 0).astype(int)
    p_up = pd.to_numeric(df["prob_up"], errors="coerce").clip(0.01, 0.99).values
    pred_up = (p_up >= 0.50).astype(int)

    acc = round(float((pred_up == y_up).mean() * 100.0), 2)
    brier = round(float(brier_score_loss(y_up, p_up)), 4)
    auc = round(float(roc_auc_score(y_up, p_up)), 4) if len(set(y_up)) > 1 else None
    sufficient = n >= min_samples
    note = (
        f"Statistically sufficient (n={n} >= {min_samples})."
        if sufficient
        else f"Statistically insufficient for validation inference (n={n} < {min_samples}); reported for audit transparency only."
    )

    return ProspectiveHorizonMetrics(
        horizon=horizon,
        n_matured_prospective=n,
        min_sample_threshold=min_samples,
        statistically_sufficient=sufficient,
        directional_accuracy_pct=acc,
        roc_auc=auc,
        brier_score=brier,
        mean_gross_return_pct=round(float(np.mean(y_gross)), 4),
        mean_net_return_pct=round(float(np.mean(y_net)), 4),
        note=note,
    )


def classify_legacy_prediction_log(pred_log_df: pd.DataFrame | None) -> LegacyLogClassificationSummary:
    """Classify legacy prediction_log.parquet rows by (symbol, observation_date) deduplication."""
    if pred_log_df is None or pred_log_df.empty:
        return LegacyLogClassificationSummary()

    df = pred_log_df.copy()
    sort_cols = [c for c in ("symbol", "observation_date", "generated_at", "prediction_id") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    seen_keys: set[tuple[str, str]] = set()
    is_dup: list[bool] = []
    for _, row in df.iterrows():
        key = (str(row.get("symbol", "")).upper().strip(), str(row.get("observation_date", "")).strip())
        if key in seen_keys:
            is_dup.append(True)
        else:
            seen_keys.add(key)
            is_dup.append(False)

    df["_is_dup"] = is_dup
    eval_mask = df["realized_next_day_return_pct"].notna() if "realized_next_day_return_pct" in df.columns else pd.Series(False, index=df.index)
    versions = (
        df["model_version"].astype(str).value_counts().to_dict()
        if "model_version" in df.columns
        else {}
    )

    dedup_eval = df[(~df["_is_dup"]) & eval_mask]
    dedup_acc: float | None = None
    if not dedup_eval.empty and "prob_next_day_up" in dedup_eval.columns:
        p = pd.to_numeric(dedup_eval["prob_next_day_up"], errors="coerce")
        r = pd.to_numeric(dedup_eval["realized_next_day_return_pct"], errors="coerce")
        valid = pd.DataFrame({"p": p, "r": r}).dropna()
        if not valid.empty:
            dedup_acc = round(float(((valid["p"] >= 0.50) == (valid["r"] > 0)).mean() * 100.0), 2)

    return LegacyLogClassificationSummary(
        total_stored=len(df),
        unique_canonical_symbol_date=int((~df["_is_dup"]).sum()),
        duplicate=int(df["_is_dup"].sum()),
        evaluated=int(eval_mask.sum()),
        pending=int((~eval_mask).sum()),
        model_versions={str(k): int(v) for k, v in versions.items()},
        deduped_1d_accuracy_pct=dedup_acc,
    )
