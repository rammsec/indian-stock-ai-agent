"""Reproducible Read-Only Data Integrity & Audit Engine (BharatAlpha AI).

Enforces non-destructive derived classifications:
- Cohort: `PROSPECTIVE` (observation_date >= cutoff and valid collection/prediction timing)
  vs `HISTORICAL` (pre-cutoff / retroactive baseline).
- Identity: `DUPLICATE` vs `CANONICAL`.
- Execution timing: `INVALID_LOOKAHEAD` when `prediction_timestamp > entry_date 09:15:00+05:30`
  (or `entry_date < prediction_timestamp.date()`), `INVALID` for missing/non-positive prices
  or chronological violations, `PENDING` vs `MATURED` for valid horizons.

Never modifies original Parquet/JSON records or model/feature/prediction logic.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST_TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_PROSPECTIVE_CUTOFF = "2026-11-17"


def _parse_ts_ist(val: Any) -> datetime | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST_TZ)
        return dt.astimezone(IST_TZ)
    except Exception:
        return None


def _entry_open_ts_ist(entry_date: Any) -> datetime | None:
    if entry_date is None or (isinstance(entry_date, float) and pd.isna(entry_date)):
        return None
    s = str(entry_date).strip()[:10]
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    try:
        return datetime.fromisoformat(f"{s}T09:15:00+05:30")
    except Exception:
        return None


def classify_forward_snapshot_record(
    observation_date: str,
    prediction_timestamp: str | None,
    is_duplicate_run: bool,
    outcome_status: str | None,
    entry_date: str | None,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
) -> dict[str, Any]:
    """Return derived non-destructive audit labels for a prediction/outcome pair."""
    obs = str(observation_date).strip()[:10]
    cohort = "PROSPECTIVE" if obs >= prospective_cutoff else "HISTORICAL"
    identity_status = "DUPLICATE" if bool(is_duplicate_run) else "CANONICAL"

    pred_dt = _parse_ts_ist(prediction_timestamp)
    entry_open_dt = _entry_open_ts_ist(entry_date)
    lookahead_violation = False
    lookahead_delta_minutes: float | None = None

    if pred_dt is not None and entry_open_dt is not None:
        if pred_dt > entry_open_dt:
            lookahead_violation = True
            lookahead_delta_minutes = round((pred_dt - entry_open_dt).total_seconds() / 60.0, 2)
    elif pred_dt is not None and outcome_status == "PENDING":
        # Pending snapshot whose latest possible next-day open after obs_date is already <= pred_dt.date()
        if pred_dt.date().isoformat() > obs and (pred_dt.hour * 60 + pred_dt.minute) >= (9 * 60 + 15):
            # e.g. obs=2026-11-18, pred_ts=2026-11-19T17:28 IST -> next bar after obs is 2026-11-19 09:15 IST
            next_day_open = _entry_open_ts_ist(pred_dt.date().isoformat())
            if next_day_open is not None and pred_dt > next_day_open:
                lookahead_violation = True
                lookahead_delta_minutes = round((pred_dt - next_day_open).total_seconds() / 60.0, 2)

    norm_status = str(outcome_status or "PENDING").upper()
    if lookahead_violation:
        audit_status = "INVALID_LOOKAHEAD"
    elif norm_status in ("INVALID", "UNAVAILABLE"):
        audit_status = "INVALID"
    elif norm_status == "MATURED":
        audit_status = "MATURED"
    else:
        audit_status = "PENDING"

    return {
        "cohort": cohort,
        "identity_status": identity_status,
        "audit_status": audit_status,
        "lookahead_violation": lookahead_violation,
        "lookahead_delta_minutes": lookahead_delta_minutes,
    }


def audit_metadata_version_drift(root_dir: Path) -> dict[str, Any]:
    """Check 5.0.0-pilot vs 5.3.0-pilot drift across JSON artifacts and Parquet snapshots."""
    targets = {
        "data_phase5_intelligence": root_dir / "data" / "predictions" / "phase5_intelligence.json",
        "docs_phase5_intelligence": root_dir / "docs" / "data" / "phase5_intelligence.json",
        "data_forward_validation": root_dir / "data" / "predictions" / "forward_validation.json",
        "docs_forward_validation": root_dir / "docs" / "data" / "forward_validation.json",
    }
    versions: dict[str, str | None] = {}
    generated_at: dict[str, str | None] = {}
    for key, path in targets.items():
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            versions[key] = raw.get("model_version")
            generated_at[key] = raw.get("generated_at")
        else:
            versions[key] = None
            generated_at[key] = None

    fp_path = root_dir / "data" / "predictions" / "forward_predictions.parquet"
    fp_versions: list[str] = []
    if fp_path.exists():
        fp = pd.read_parquet(fp_path)
        if "model_version" in fp.columns:
            fp_versions = sorted(fp["model_version"].dropna().astype(str).unique().tolist())

    unique_versions = sorted({v for v in list(versions.values()) + fp_versions if v is not None})
    has_drift = len(unique_versions) > 1
    return {
        "has_version_drift": has_drift,
        "unique_versions_detected": unique_versions,
        "json_versions": versions,
        "json_generated_at": generated_at,
        "forward_predictions_parquet_versions": fp_versions,
        "severity": "HIGH" if has_drift else "OK",
    }


def audit_forward_predictions_and_outcomes(
    root_dir: Path,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
) -> dict[str, Any]:
    """Audit forward_predictions.parquet and forward_outcomes.parquet with derived classifications."""
    fp_path = root_dir / "data" / "predictions" / "forward_predictions.parquet"
    fo_path = root_dir / "data" / "predictions" / "forward_outcomes.parquet"
    if not fp_path.exists() or not fo_path.exists():
        return {"status": "MISSING_FILES", "fp_exists": fp_path.exists(), "fo_exists": fo_path.exists()}

    fp = pd.read_parquet(fp_path)
    fo = pd.read_parquet(fo_path)

    fp_dup_ids = int(fp["prediction_id"].duplicated().sum()) if "prediction_id" in fp.columns else 0
    is_dup_series = fp["is_duplicate_run"].fillna(False).astype(bool) if "is_duplicate_run" in fp.columns else pd.Series(False, index=fp.index)
    canonical_count = int((~is_dup_series).sum())
    duplicate_count = int(is_dup_series.sum())

    prospective_snaps = int((fp["observation_date"].astype(str) >= prospective_cutoff).sum())
    historical_snaps = int((fp["observation_date"].astype(str) < prospective_cutoff).sum())
    prospective_canonical = int(((fp["observation_date"].astype(str) >= prospective_cutoff) & (~is_dup_series)).sum())
    historical_canonical = int(((fp["observation_date"].astype(str) < prospective_cutoff) & (~is_dup_series)).sum())

    # Join outcomes with predictions to classify every (prediction_id, horizon) record
    merged = fo.merge(
        fp[["prediction_id", "prediction_timestamp", "nse_session", "prediction_run_id", "is_duplicate_run"]],
        on="prediction_id",
        how="left",
    )

    classifications: list[dict[str, Any]] = []
    lookahead_runs: dict[str, dict[str, Any]] = {}
    for _, row in merged.iterrows():
        cls = classify_forward_snapshot_record(
            observation_date=str(row["observation_date"]),
            prediction_timestamp=row.get("prediction_timestamp"),
            is_duplicate_run=bool(row.get("is_duplicate_run") if pd.notna(row.get("is_duplicate_run")) else False),
            outcome_status=str(row.get("status") or "PENDING"),
            entry_date=row.get("entry_date"),
            prospective_cutoff=prospective_cutoff,
        )
        classifications.append(cls)
        if cls["lookahead_violation"] and cls["identity_status"] == "CANONICAL" and str(row["horizon"]) == "1D":
            ts_key = str(row.get("prediction_timestamp"))
            if ts_key not in lookahead_runs:
                lookahead_runs[ts_key] = {
                    "prediction_timestamp": ts_key,
                    "observation_date": str(row["observation_date"]),
                    "nse_session": str(row.get("nse_session")),
                    "entry_date": str(row.get("entry_date")) if pd.notna(row.get("entry_date")) else None,
                    "outcome_status": str(row.get("status")),
                    "lookahead_delta_minutes": cls["lookahead_delta_minutes"],
                    "affected_symbols": 0,
                }
            lookahead_runs[ts_key]["affected_symbols"] += 1

    cls_df = pd.DataFrame(classifications)
    merged_cls = pd.concat([merged.reset_index(drop=True), cls_df], axis=1)

    canonical_1d = merged_cls[(merged_cls["identity_status"] == "CANONICAL") & (merged_cls["horizon"] == "1D")]
    audit_status_counts_all = cls_df["audit_status"].value_counts().to_dict()
    audit_status_counts_canonical_1d = canonical_1d["audit_status"].value_counts().to_dict()

    raw_horizon_status = fo.groupby(["horizon", "status"]).size().to_dict()
    raw_horizon_status_str = {f"{k[0]}_{k[1]}": int(v) for k, v in raw_horizon_status.items()}

    benchmark_non_null = int(fo["benchmark_return_pct"].notna().sum()) if "benchmark_return_pct" in fo.columns else 0

    return {
        "total_snapshots": len(fp),
        "duplicate_prediction_ids": fp_dup_ids,
        "canonical_snapshots": canonical_count,
        "duplicate_run_snapshots": duplicate_count,
        "prospective_cutoff": prospective_cutoff,
        "prospective_snapshots": prospective_snaps,
        "historical_snapshots": historical_snaps,
        "prospective_canonical_snapshots": prospective_canonical,
        "historical_canonical_snapshots": historical_canonical,
        "total_outcomes": len(fo),
        "raw_horizon_status_counts": raw_horizon_status_str,
        "derived_audit_status_counts_all": {str(k): int(v) for k, v in audit_status_counts_all.items()},
        "derived_audit_status_counts_canonical_1d": {str(k): int(v) for k, v in audit_status_counts_canonical_1d.items()},
        "canonical_lookahead_runs": list(lookahead_runs.values()),
        "benchmark_return_pct_non_null": benchmark_non_null,
        "severity": "CRITICAL" if lookahead_runs or benchmark_non_null == 0 else "OK",
    }


def audit_prediction_log(
    root_dir: Path,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
) -> dict[str, Any]:
    """Audit prediction_log.parquet for duplicate (symbol, observation_date) contamination and monitoring inflation."""
    pl_path = root_dir / "data" / "predictions" / "prediction_log.parquet"
    if not pl_path.exists():
        return {"status": "MISSING_FILE"}

    pl = pd.read_parquet(pl_path)
    dup_sym_obs = int(pl.duplicated(subset=["symbol", "observation_date"]).sum())
    dup_pid = int(pl["prediction_id"].duplicated().sum())
    eval_1d = int(pl["realized_next_day_return_pct"].notna().sum()) if "realized_next_day_return_pct" in pl.columns else 0
    eval_5d = int(pl["realized_5d_return_pct"].notna().sum()) if "realized_5d_return_pct" in pl.columns else 0
    eval_10d = int(pl["realized_10d_return_pct"].notna().sum()) if "realized_10d_return_pct" in pl.columns else 0

    by_date = (
        pl.groupby("observation_date")
        .agg(
            rows=("prediction_id", "count"),
            unique_symbols=("symbol", "nunique"),
            evaluated_1d=("realized_next_day_return_pct", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    gen_tz_utc = int(pl["generated_at"].astype(str).str.endswith("+00:00").sum()) if "generated_at" in pl.columns else 0

    return {
        "total_rows": len(pl),
        "unique_symbols": int(pl["symbol"].nunique()),
        "duplicate_prediction_ids": dup_pid,
        "duplicate_symbol_observation_date": dup_sym_obs,
        "prospective_rows": int((pl["observation_date"].astype(str) >= prospective_cutoff).sum()),
        "historical_rows": int((pl["observation_date"].astype(str) < prospective_cutoff).sum()),
        "evaluated_1d_count": eval_1d,
        "evaluated_5d_count": eval_5d,
        "evaluated_10d_count": eval_10d,
        "generated_at_utc_count": gen_tz_utc,
        "by_observation_date": by_date,
        "severity": "HIGH" if dup_sym_obs > 0 else "OK",
    }


def audit_equities_and_indices(root_dir: Path) -> dict[str, Any]:
    """Audit equities_daily.parquet and indices_daily.parquet for price, volume, CA, delivery, and naming integrity."""
    eq_path = root_dir / "data" / "processed" / "equities_daily.parquet"
    idx_path = root_dir / "data" / "processed" / "indices_daily.parquet"
    if not eq_path.exists() or not idx_path.exists():
        return {"status": "MISSING_FILES"}

    eq = pd.read_parquet(eq_path)
    idx = pd.read_parquet(idx_path)

    eq_max_dt = str(eq["observation_date"].max())
    stale_eq_syms = eq.groupby("symbol")["observation_date"].max()
    stale_eq = {str(k): str(v) for k, v in stale_eq_syms[stale_eq_syms != eq_max_dt].to_dict().items()}

    ohlc_violations = int(
        ((eq["high"] < eq["low"]) | (eq["close"] > eq["high"]) | (eq["close"] < eq["low"]) | (eq["open"] > eq["high"]) | (eq["open"] < eq["low"])).sum()
    )
    nonpos_prices = {
        col: int((eq[col].isna() | (eq[col] <= 0)).sum())
        for col in ("open", "high", "low", "close", "adj_close")
    }
    zero_vol_rows = eq[eq["volume"] <= 0]
    zero_vol_by_date = {str(k): int(v) for k, v in zero_vol_rows.groupby("observation_date")["symbol"].count().to_dict().items()}

    deliv_null = int(eq["delivery_pct"].isna().sum()) if "delivery_pct" in eq.columns else len(eq)
    deliv_non_null = len(eq) - deliv_null

    idx_codes = sorted(idx["index_code"].astype(str).unique().tolist())
    has_nifty_underscore = "NIFTY_50" in idx_codes
    has_nifty_space = "NIFTY 50" in idx_codes
    idx_nonpos_ohlc = {
        col: int((idx[col].isna() | (idx[col] <= 0)).sum())
        for col in ("open", "high", "low", "close")
    }
    idx_vol_null_by_code = {str(k): int(v) for k, v in idx[idx["volume"].isna()]["index_code"].value_counts().to_dict().items()}

    return {
        "equities": {
            "rows": len(eq),
            "symbols": int(eq["symbol"].nunique()),
            "date_min": str(eq["observation_date"].min()),
            "date_max": eq_max_dt,
            "duplicate_symbol_date": int(eq.duplicated(subset=["symbol", "observation_date"]).sum()),
            "non_positive_or_null_prices": nonpos_prices,
            "ohlc_envelope_violations": ohlc_violations,
            "zero_volume_rows": int(len(zero_vol_rows)),
            "zero_volume_by_date": zero_vol_by_date,
            "delivery_pct_missing_rows": deliv_null,
            "delivery_pct_present_rows": deliv_non_null,
            "delivery_pct_coverage_pct": round(deliv_non_null / max(1, len(eq)) * 100.0, 2),
            "dividends_positive_rows": int((eq["dividends"].fillna(0) > 0).sum()),
            "stock_splits_positive_rows": int((eq["stock_splits"].fillna(0) > 0).sum()),
            "adj_close_differs_from_close_rows": int((eq["adj_close"] != eq["close"]).sum()),
            "stale_symbols": stale_eq,
        },
        "indices": {
            "rows": len(idx),
            "index_count": int(idx["index_code"].nunique()),
            "date_min": str(idx["observation_date"].min()),
            "date_max": str(idx["observation_date"].max()),
            "duplicate_index_date": int(idx.duplicated(subset=["index_code", "observation_date"]).sum()),
            "non_positive_or_null_prices": idx_nonpos_ohlc,
            "volume_null_by_index": idx_vol_null_by_code,
            "has_NIFTY_50_underscore": has_nifty_underscore,
            "has_NIFTY_50_space": has_nifty_space,
            "nifty_naming_mismatch_bug": bool(has_nifty_underscore and not has_nifty_space),
        },
        "severity": "HIGH" if (has_nifty_underscore and not has_nifty_space) or len(zero_vol_rows) > 0 else "OK",
    }


def audit_structured_events(
    root_dir: Path,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
) -> dict[str, Any]:
    """Audit structured_events.parquet and quarantined_events.parquet for provenance & timestamp integrity."""
    se_path = root_dir / "data" / "predictions" / "structured_events.parquet"
    qe_path = root_dir / "data" / "predictions" / "quarantined_events.parquet"
    if not se_path.exists():
        return {"status": "MISSING_FILE"}

    se = pd.read_parquet(se_path)
    qe = pd.read_parquet(qe_path) if qe_path.exists() else pd.DataFrame()

    null_coll = int(se["collection_timestamp"].isna().sum())
    non_null_coll = int(se["collection_timestamp"].notna().sum())

    pub_utc = pd.to_datetime(se["publication_timestamp"], utc=True, errors="coerce")
    coll_utc = pd.to_datetime(se["collection_timestamp"], utc=True, errors="coerce")
    pub_after_coll = int(((pub_utc > coll_utc) & coll_utc.notna()).sum())

    pub_ist_hours = pd.to_datetime(se["publication_timestamp"], errors="coerce").dt.hour.value_counts().to_dict()
    at_0700_ist = int(pub_ist_hours.get(7, 0))

    prospective_by_pub = int((se["publication_timestamp"].astype(str) >= prospective_cutoff).sum())
    prospective_by_coll = int((se["collection_timestamp"].dropna().astype(str) >= prospective_cutoff).sum())

    return {
        "valid_events": len(se),
        "quarantined_events": len(qe),
        "null_collection_timestamp_rows": null_coll,
        "non_null_collection_timestamp_rows": non_null_coll,
        "is_genuinely_forward_collected": bool(null_coll == 0 and len(se) > 0),
        "duplicate_reports_count": int(se["is_duplicate_report"].fillna(False).sum()),
        "publication_after_collection_count": pub_after_coll,
        "events_at_0700_ist_count": at_0700_ist,
        "prospective_by_publication_date": prospective_by_pub,
        "prospective_by_collection_date": prospective_by_coll,
        "severity": "HIGH" if null_coll > 0 else "OK",
    }


def audit_paper_trades(root_dir: Path) -> dict[str, Any]:
    """Audit paper_trades.parquet for legacy action naming, look-ahead entries, and exit_date coverage."""
    pt_path = root_dir / "data" / "predictions" / "paper_trades.parquet"
    if not pt_path.exists():
        return {"status": "MISSING_FILE"}

    pt = pd.read_parquet(pt_path)
    action_counts = {str(k): int(v) for k, v in pt["action"].value_counts(dropna=False).to_dict().items()}
    legacy_rejected = int((pt["action"] == "REJECTED").sum())
    entry_executed = pt[pt["action"] == "ENTRY_EXECUTED"]
    lookahead_entries = int(
        entry_executed["prediction_id"].astype(str).str.contains("20261117T102529").sum()
    )

    return {
        "total_rows": len(pt),
        "action_counts": action_counts,
        "legacy_rejected_action_rows": legacy_rejected,
        "entry_executed_count": int(len(entry_executed)),
        "lookahead_entry_executed_count": lookahead_entries,
        "non_null_entry_date_rows": int(pt["entry_date"].notna().sum()),
        "non_null_exit_date_rows": int(pt["exit_date"].notna().sum()),
        "severity": "HIGH" if legacy_rejected > 0 or lookahead_entries > 0 else "OK",
    }


def run_full_data_integrity_audit(
    root_dir: Path | str,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
    output_json_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run all read-only data integrity audits and optionally write DATA_QUALITY_REPORT.json."""
    root = Path(root_dir).resolve()
    now_ist = datetime.now(IST_TZ).isoformat()

    report: dict[str, Any] = {
        "audit_generated_at": now_ist,
        "prospective_cutoff": prospective_cutoff,
        "read_only_guarantee": True,
        "sections": {
            "metadata_version_drift": audit_metadata_version_drift(root),
            "forward_predictions_and_outcomes": audit_forward_predictions_and_outcomes(root, prospective_cutoff),
            "prediction_log": audit_prediction_log(root, prospective_cutoff),
            "equities_and_indices": audit_equities_and_indices(root),
            "structured_events": audit_structured_events(root, prospective_cutoff),
            "paper_trades": audit_paper_trades(root),
        },
    }

    severities = [sec.get("severity", "OK") for sec in report["sections"].values()]
    if "CRITICAL" in severities:
        overall = "CRITICAL"
    elif "HIGH" in severities:
        overall = "HIGH"
    else:
        overall = "OK"
    report["overall_status"] = overall

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    out_json = repo_root / "DATA_QUALITY_REPORT.json"
    res = run_full_data_integrity_audit(repo_root, output_json_path=out_json)
    print(f"Audit complete: overall_status={res['overall_status']} -> {out_json}")
