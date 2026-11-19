"""Formal Prediction & Event Identity Contract and Shadow Validation Gate.

Workstream 1 (Data Integrity) remediation module for BharatAlpha AI.
Provides:
1. Deterministic canonical identity keys:
   - `canonical_prediction_id(symbol, observation_date, nse_session)`
   - `daily_canonical_key(symbol, target_entry_date)`
2. Mandatory provenance schema enforcement (`MANDATORY_PROVENANCE_FIELDS`).
3. Non-destructive validation gates (`validate_prediction_records`,
   `validate_event_records`) that preserve raw observations and classify every
   record into `PROSPECTIVE`, `HISTORICAL`, `DUPLICATE`, `QUARANTINED`, or
   `INVALID_LOOKAHEAD` with explicit `rejection_reason`.
4. Derived shadow layer builder (`build_remediated_shadow_layer`) under
   `data/shadow/v1_remediated/` that resolves `5.0.0-pilot` vs `5.3.0-pilot`
   metadata drift and `prediction_log.parquet` duplicate `(symbol, observation_date)`
   contamination without mutating any original Parquet, JSON, or model files.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST_TZ = ZoneInfo("Asia/Kolkata")
SCHEMA_VERSION = "1.0.0"
REMEDIATION_VERSION = "rem-1.0.0"
DEFAULT_PROSPECTIVE_CUTOFF = "2026-11-17"
CANONICAL_MODEL_VERSION = "5.3.0-pilot"
CANONICAL_FEATURE_SET_VERSION = "237+17"

VALID_NSE_SESSIONS = ("PRE_MARKET", "INTRADAY", "POST_MARKET")
VALID_CLASSIFICATIONS = (
    "PROSPECTIVE",
    "HISTORICAL",
    "DUPLICATE",
    "QUARANTINED",
    "INVALID_LOOKAHEAD",
)

MANDATORY_PROVENANCE_FIELDS = (
    "schema_version",
    "remediation_version",
    "source_dataset",
    "model_version",
    "feature_set_version",
    "collection_timestamp",
    "prediction_timestamp",
    "validation_status",
    "rejection_reason",
)


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in ("nan", "nat", "none", "<na>")


def _parse_ts_ist(val: Any) -> datetime | None:
    if _is_missing(val):
        return None
    s = str(val).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST_TZ)
        return dt.astimezone(IST_TZ)
    except Exception:
        return None


def _entry_open_ts_ist(entry_date: Any) -> datetime | None:
    if _is_missing(entry_date):
        return None
    s = str(entry_date).strip()[:10]
    try:
        return datetime.fromisoformat(f"{s}T09:15:00+05:30")
    except Exception:
        return None


def _infer_nse_session(ts_ist: datetime | None) -> str:
    if ts_ist is None:
        return "UNKNOWN"
    minutes = ts_ist.hour * 60 + ts_ist.minute
    if minutes < 9 * 60 + 15:
        return "PRE_MARKET"
    if minutes < 15 * 60 + 30:
        return "INTRADAY"
    return "POST_MARKET"


def _next_trading_date_str(obs_date_str: str) -> str:
    dt = datetime.fromisoformat(obs_date_str[:10]).date() + timedelta(days=1)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.isoformat()


def canonical_prediction_id(
    symbol: str,
    observation_date: str,
    nse_session: str,
) -> str:
    """Return a deterministic canonical prediction identifier.

    Format: `cpid_{YYYY-MM-DD}_{SYMBOL}_{NSE_SESSION}`
    """
    if _is_missing(symbol):
        raise ValueError("symbol must be a non-empty string")
    if _is_missing(observation_date):
        raise ValueError("observation_date must be a non-empty YYYY-MM-DD string")
    if _is_missing(nse_session):
        raise ValueError("nse_session must be a non-empty string")

    norm_symbol = str(symbol).strip().upper()
    norm_date = str(observation_date).strip()[:10]
    try:
        datetime.fromisoformat(norm_date)
    except ValueError as exc:
        raise ValueError(f"Invalid observation_date: {observation_date!r}") from exc

    norm_session = str(nse_session).strip().upper()
    if norm_session not in VALID_NSE_SESSIONS:
        raise ValueError(
            f"Invalid nse_session {nse_session!r}; expected one of {VALID_NSE_SESSIONS}"
        )

    return f"cpid_{norm_date}_{norm_symbol}_{norm_session}"


def daily_canonical_key(symbol: str, target_entry_date: str) -> str:
    """Return a deterministic daily canonical key for execution/deduplication.

    Format: `dck_{YYYY-MM-DD}_{SYMBOL}`
    """
    if _is_missing(symbol):
        raise ValueError("symbol must be a non-empty string")
    if _is_missing(target_entry_date):
        raise ValueError("target_entry_date must be a non-empty YYYY-MM-DD string")

    norm_symbol = str(symbol).strip().upper()
    norm_date = str(target_entry_date).strip()[:10]
    try:
        datetime.fromisoformat(norm_date)
    except ValueError as exc:
        raise ValueError(f"Invalid target_entry_date: {target_entry_date!r}") from exc

    return f"dck_{norm_date}_{norm_symbol}"


def enforce_provenance(
    df: pd.DataFrame,
    source_dataset: str,
    default_model_version: str | None = None,
    default_feature_set_version: str | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` with all mandatory provenance columns populated."""
    out = df.copy()
    out["schema_version"] = SCHEMA_VERSION
    out["remediation_version"] = REMEDIATION_VERSION
    out["source_dataset"] = source_dataset

    if "model_version" not in out.columns:
        out["model_version"] = default_model_version
    elif default_model_version is not None:
        out["model_version"] = out["model_version"].where(
            out["model_version"].notna(), default_model_version
        )

    if "feature_set_version" not in out.columns:
        out["feature_set_version"] = default_feature_set_version
    elif default_feature_set_version is not None:
        out["feature_set_version"] = out["feature_set_version"].where(
            out["feature_set_version"].notna(), default_feature_set_version
        )

    if "collection_timestamp" not in out.columns:
        out["collection_timestamp"] = None
    if "prediction_timestamp" not in out.columns:
        out["prediction_timestamp"] = None
    if "validation_status" not in out.columns:
        out["validation_status"] = "QUARANTINED"
    if "rejection_reason" not in out.columns:
        out["rejection_reason"] = None

    return out


def check_provenance_completeness(df: pd.DataFrame) -> dict[str, Any]:
    """Verify mandatory provenance fields exist and report null counts."""
    missing_columns = [c for c in MANDATORY_PROVENANCE_FIELDS if c not in df.columns]
    null_counts: dict[str, int] = {}
    for col in MANDATORY_PROVENANCE_FIELDS:
        if col in df.columns:
            null_counts[col] = int(df[col].apply(_is_missing).sum())

    invalid_statuses: list[str] = []
    if "validation_status" in df.columns:
        observed = {str(v) for v in df["validation_status"].dropna().unique().tolist()}
        invalid_statuses = sorted(observed - set(VALID_CLASSIFICATIONS))

    required_non_null = (
        "schema_version",
        "remediation_version",
        "source_dataset",
        "validation_status",
    )
    has_missing_required = any(null_counts.get(c, 0) > 0 for c in required_non_null)

    return {
        "is_complete": len(missing_columns) == 0
        and not has_missing_required
        and len(invalid_statuses) == 0,
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "invalid_validation_statuses": invalid_statuses,
        "row_count": len(df),
    }


def validate_prediction_records(
    df: pd.DataFrame,
    *,
    source_dataset: str = "forward_predictions.parquet",
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
    outcomes_df: pd.DataFrame | None = None,
    dedupe_by_symbol_obs_date: bool = False,
    default_model_version: str | None = None,
    default_feature_set_version: str | None = None,
) -> pd.DataFrame:
    """Validate and classify prediction records non-destructively.

    Preserves all raw rows in the returned DataFrame while adding:
    - `canonical_prediction_id` (deterministic)
    - `daily_canonical_key` (deterministic)
    - `cohort` (`PROSPECTIVE` or `HISTORICAL`)
    - `validation_status` (`PROSPECTIVE`, `HISTORICAL`, `DUPLICATE`, `QUARANTINED`, `INVALID_LOOKAHEAD`)
    - `rejection_reason`
    - All `MANDATORY_PROVENANCE_FIELDS`
    """
    out = df.copy()
    has_explicit_collection_col = "collection_timestamp" in out.columns

    # Map prediction_log's generated_at if prediction_timestamp is absent
    if "prediction_timestamp" not in out.columns and "generated_at" in out.columns:
        out["prediction_timestamp"] = out["generated_at"]

    out = enforce_provenance(
        out,
        source_dataset=source_dataset,
        default_model_version=default_model_version,
        default_feature_set_version=default_feature_set_version,
    )

    if not has_explicit_collection_col:
        out["collection_timestamp"] = out["prediction_timestamp"]

    # Join 1D entry_date from outcomes_df if provided
    entry_date_map: dict[str, str] = {}
    outcome_status_map: dict[str, str] = {}
    if outcomes_df is not None and not outcomes_df.empty and "prediction_id" in outcomes_df.columns:
        fo_1d = outcomes_df
        if "horizon" in fo_1d.columns:
            fo_1d = fo_1d[fo_1d["horizon"] == "1D"]
        for _, r in fo_1d.iterrows():
            pid = str(r["prediction_id"])
            if not _is_missing(r.get("entry_date")):
                entry_date_map[pid] = str(r["entry_date"]).strip()[:10]
            if not _is_missing(r.get("status")):
                outcome_status_map[pid] = str(r["status"]).strip().upper()

    # Pre-compute parsed timestamps and deterministic IDs
    parsed_ts: list[datetime | None] = []
    cpids: list[str | None] = []
    dcks: list[str | None] = []
    target_entry_dates: list[str | None] = []
    cohorts: list[str] = []
    quarantine_reasons: list[str | None] = []

    for idx, row in out.iterrows():
        sym = row.get("symbol")
        obs = row.get("observation_date")
        pred_ts_raw = row.get("prediction_timestamp")
        coll_ts_raw = row.get("collection_timestamp")
        pred_dt = _parse_ts_ist(pred_ts_raw)
        parsed_ts.append(pred_dt)

        q_reason: str | None = None
        if _is_missing(sym):
            q_reason = "MISSING_SYMBOL"
        elif _is_missing(obs):
            q_reason = "MISSING_OBSERVATION_DATE"
        elif pred_dt is None:
            q_reason = "MISSING_OR_INVALID_PREDICTION_TIMESTAMP"
        elif has_explicit_collection_col and _is_missing(coll_ts_raw):
            q_reason = "MISSING_COLLECTION_TIMESTAMP"
        elif _is_missing(row.get("model_version")):
            q_reason = "MISSING_MODEL_VERSION"
        elif _is_missing(row.get("feature_set_version")):
            q_reason = "MISSING_FEATURE_SET_VERSION"

        obs_str = str(obs).strip()[:10] if not _is_missing(obs) else ""
        cohorts.append("PROSPECTIVE" if obs_str >= prospective_cutoff else "HISTORICAL")

        # Determine nse_session
        raw_sess = row.get("nse_session")
        sess = str(raw_sess).strip().upper() if not _is_missing(raw_sess) else _infer_nse_session(pred_dt)

        # Compute deterministic canonical_prediction_id
        cpid: str | None = None
        if q_reason is None and sess in VALID_NSE_SESSIONS:
            try:
                cpid = canonical_prediction_id(str(sym), obs_str, sess)
            except ValueError as exc:
                q_reason = f"INVALID_IDENTITY:{exc}"
        elif q_reason is None:
            q_reason = f"INVALID_NSE_SESSION:{sess}"

        # Determine target_entry_date & daily_canonical_key
        pid_str = str(row.get("prediction_id")) if not _is_missing(row.get("prediction_id")) else ""
        t_entry = entry_date_map.get(pid_str)
        if t_entry is None and not _is_missing(row.get("entry_date")):
            t_entry = str(row.get("entry_date")).strip()[:10]
        if t_entry is None and not _is_missing(row.get("target_session_date")):
            t_entry = str(row.get("target_session_date")).strip()[:10]
        if t_entry is None and obs_str:
            try:
                t_entry = _next_trading_date_str(obs_str)
            except Exception:
                t_entry = None

        dck: str | None = None
        if not _is_missing(sym) and t_entry is not None:
            try:
                dck = daily_canonical_key(str(sym), t_entry)
            except ValueError:
                dck = None

        cpids.append(cpid)
        dcks.append(dck)
        target_entry_dates.append(t_entry)
        quarantine_reasons.append(q_reason)

    out["canonical_prediction_id"] = cpids
    out["daily_canonical_key"] = dcks
    out["target_entry_date"] = target_entry_dates
    out["cohort"] = cohorts
    out["_parsed_pred_ts"] = parsed_ts
    out["_quarantine_reason"] = quarantine_reasons

    # Determine canonical vs duplicate ordering deterministically
    sort_cols = ["_parsed_pred_ts"]
    if "prediction_id" in out.columns:
        sort_cols.append("prediction_id")
    ordered_indices = out.sort_values(by=sort_cols, kind="mergesort", na_position="last").index

    seen_cpid: set[str] = set()
    seen_sym_obs: set[tuple[str, str]] = set()
    statuses: dict[Any, str] = {}
    reasons: dict[Any, str | None] = {}

    for idx in ordered_indices:
        row = out.loc[idx]
        q_reason = row["_quarantine_reason"]
        if not _is_missing(q_reason):
            statuses[idx] = "QUARANTINED"
            reasons[idx] = str(q_reason)
            continue

        cpid = str(row["canonical_prediction_id"])
        sym_obs = (str(row["symbol"]).strip().upper(), str(row["observation_date"]).strip()[:10])

        if cpid in seen_cpid:
            statuses[idx] = "DUPLICATE"
            reasons[idx] = "DUPLICATE_CANONICAL_PREDICTION_ID"
            continue
        if dedupe_by_symbol_obs_date and sym_obs in seen_sym_obs:
            statuses[idx] = "DUPLICATE"
            reasons[idx] = "DUPLICATE_SYMBOL_OBSERVATION_DATE"
            continue

        seen_cpid.add(cpid)
        seen_sym_obs.add(sym_obs)

        # Look-ahead check
        pred_dt = row["_parsed_pred_ts"]
        pid_str = str(row.get("prediction_id")) if not _is_missing(row.get("prediction_id")) else ""
        explicit_entry = entry_date_map.get(pid_str)
        if explicit_entry is None and not _is_missing(row.get("entry_date")):
            explicit_entry = str(row.get("entry_date")).strip()[:10]
        outcome_status = outcome_status_map.get(pid_str, "PENDING")

        lookahead_violation = False
        lookahead_delta: float | None = None
        entry_open_dt = _entry_open_ts_ist(explicit_entry) if explicit_entry else None
        if pred_dt is not None and entry_open_dt is not None:
            if pred_dt > entry_open_dt:
                lookahead_violation = True
                lookahead_delta = round((pred_dt - entry_open_dt).total_seconds() / 60.0, 2)
        elif pred_dt is not None and outcome_status == "PENDING":
            obs_str = str(row["observation_date"]).strip()[:10]
            if pred_dt.date().isoformat() > obs_str and (pred_dt.hour * 60 + pred_dt.minute) >= (9 * 60 + 15):
                next_open = _entry_open_ts_ist(pred_dt.date().isoformat())
                if next_open is not None and pred_dt > next_open:
                    lookahead_violation = True
                    lookahead_delta = round((pred_dt - next_open).total_seconds() / 60.0, 2)

        if lookahead_violation:
            statuses[idx] = "INVALID_LOOKAHEAD"
            reasons[idx] = f"LOOKAHEAD_VIOLATION_DELTA_{lookahead_delta}_MIN"
        else:
            statuses[idx] = str(row["cohort"])
            reasons[idx] = None

    out["validation_status"] = [statuses[i] for i in out.index]
    out["rejection_reason"] = [reasons[i] for i in out.index]
    out = out.drop(columns=["_parsed_pred_ts", "_quarantine_reason"])
    return out


def validate_event_records(
    df: pd.DataFrame,
    *,
    source_dataset: str = "structured_events.parquet",
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
    default_model_version: str = CANONICAL_MODEL_VERSION,
    default_feature_set_version: str = CANONICAL_FEATURE_SET_VERSION,
) -> pd.DataFrame:
    """Validate and classify news/event records non-destructively.

    Enforces:
    - Missing `collection_timestamp` -> `QUARANTINED` (`MISSING_COLLECTION_TIMESTAMP`)
    - `publication_timestamp > collection_timestamp` -> `INVALID_LOOKAHEAD` (`PUBLICATION_AFTER_COLLECTION`)
    - Duplicate report (`is_duplicate_report == True` or duplicate `event_id`) -> `DUPLICATE`
    - Valid records -> `PROSPECTIVE` (`>= prospective_cutoff`) or `HISTORICAL` (`< prospective_cutoff`)
    """
    out = enforce_provenance(
        df,
        source_dataset=source_dataset,
        default_model_version=default_model_version,
        default_feature_set_version=default_feature_set_version,
    )

    seen_event_ids: set[str] = set()
    statuses: list[str] = []
    reasons: list[str | None] = []
    cohorts: list[str] = []

    for _, row in out.iterrows():
        coll_raw = row.get("collection_timestamp")
        pub_raw = row.get("publication_timestamp")
        coll_dt = _parse_ts_ist(coll_raw)
        pub_dt = _parse_ts_ist(pub_raw)

        # Cohort based on collection date if present, else publication date
        ref_date = (
            coll_dt.date().isoformat()
            if coll_dt is not None
            else (pub_dt.date().isoformat() if pub_dt is not None else "")
        )
        cohort = "PROSPECTIVE" if ref_date >= prospective_cutoff else "HISTORICAL"
        cohorts.append(cohort)

        if _is_missing(coll_raw) or coll_dt is None:
            statuses.append("QUARANTINED")
            reasons.append("MISSING_COLLECTION_TIMESTAMP")
            continue

        if _is_missing(pub_raw) or pub_dt is None:
            statuses.append("QUARANTINED")
            reasons.append("MISSING_PUBLICATION_TIMESTAMP")
            continue

        if pub_dt > coll_dt:
            statuses.append("INVALID_LOOKAHEAD")
            reasons.append("PUBLICATION_AFTER_COLLECTION")
            continue

        is_dup_flag = bool(row.get("is_duplicate_report")) if not _is_missing(row.get("is_duplicate_report")) else False
        eid = str(row.get("event_id")).strip() if not _is_missing(row.get("event_id")) else ""
        if is_dup_flag or (eid and eid in seen_event_ids):
            statuses.append("DUPLICATE")
            reasons.append("DUPLICATE_EVENT_REPORT")
            continue

        if eid:
            seen_event_ids.add(eid)

        statuses.append(cohort)
        reasons.append(None)

    out["cohort"] = cohorts
    out["validation_status"] = statuses
    out["rejection_reason"] = reasons
    return out


def resolve_metadata_version_drift(
    root_dir: Path | str,
    canonical_model_version: str = CANONICAL_MODEL_VERSION,
    canonical_feature_set_version: str = CANONICAL_FEATURE_SET_VERSION,
) -> dict[str, Any]:
    """Detect and resolve metadata version drift (e.g. 5.0.0-pilot vs 5.3.0-pilot) for the shadow layer."""
    root = Path(root_dir).resolve()
    targets = {
        "data_phase5_intelligence": root / "data" / "predictions" / "phase5_intelligence.json",
        "docs_phase5_intelligence": root / "docs" / "data" / "phase5_intelligence.json",
        "data_forward_validation": root / "data" / "predictions" / "forward_validation.json",
        "docs_forward_validation": root / "docs" / "data" / "forward_validation.json",
    }
    observed_versions: dict[str, str | None] = {}
    for key, path in targets.items():
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            observed_versions[key] = raw.get("model_version")
        else:
            observed_versions[key] = None

    fp_path = root / "data" / "predictions" / "forward_predictions.parquet"
    fp_versions: list[str] = []
    if fp_path.exists():
        fp = pd.read_parquet(fp_path)
        if "model_version" in fp.columns:
            fp_versions = sorted(fp["model_version"].dropna().astype(str).unique().tolist())

    unique_observed = sorted(
        {v for v in list(observed_versions.values()) + fp_versions if v is not None}
    )
    drift_detected = len(unique_observed) > 1

    return {
        "schema_version": SCHEMA_VERSION,
        "remediation_version": REMEDIATION_VERSION,
        "canonical_model_version": canonical_model_version,
        "canonical_feature_set_version": canonical_feature_set_version,
        "drift_detected_in_raw": drift_detected,
        "raw_unique_versions": unique_observed,
        "raw_json_versions": observed_versions,
        "raw_forward_predictions_versions": fp_versions,
        "shadow_resolved_versions": {k: canonical_model_version for k in targets},
        "drift_resolved": True,
    }


def build_remediated_shadow_layer(
    root_dir: Path | str,
    output_dir: Path | str | None = None,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
) -> dict[str, Any]:
    """Build versioned derived datasets under `data/shadow/v1_remediated/` without mutating originals."""
    root = Path(root_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else root / "data" / "shadow" / "v1_remediated"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "remediation_version": REMEDIATION_VERSION,
        "prospective_cutoff": prospective_cutoff,
        "output_dir": str(out_dir),
        "datasets": {},
    }

    # 1. forward_predictions + forward_outcomes
    fp_path = root / "data" / "predictions" / "forward_predictions.parquet"
    fo_path = root / "data" / "predictions" / "forward_outcomes.parquet"
    if fp_path.exists():
        fp = pd.read_parquet(fp_path)
        fo = pd.read_parquet(fo_path) if fo_path.exists() else None
        fp_rem = validate_prediction_records(
            fp,
            source_dataset="data/predictions/forward_predictions.parquet",
            prospective_cutoff=prospective_cutoff,
            outcomes_df=fo,
            dedupe_by_symbol_obs_date=False,
        )
        fp_out = out_dir / "forward_predictions_remediated.parquet"
        fp_rem.to_parquet(fp_out, index=False)
        summary["datasets"]["forward_predictions"] = {
            "raw_rows": len(fp),
            "shadow_rows": len(fp_rem),
            "status_counts": {str(k): int(v) for k, v in fp_rem["validation_status"].value_counts().to_dict().items()},
            "provenance": check_provenance_completeness(fp_rem),
            "output_path": str(fp_out),
        }

    # 2. prediction_log (deduplicate by symbol, observation_date)
    pl_path = root / "data" / "predictions" / "prediction_log.parquet"
    if pl_path.exists():
        pl = pd.read_parquet(pl_path)
        pl_rem = validate_prediction_records(
            pl,
            source_dataset="data/predictions/prediction_log.parquet",
            prospective_cutoff=prospective_cutoff,
            dedupe_by_symbol_obs_date=True,
            default_feature_set_version=CANONICAL_FEATURE_SET_VERSION,
        )
        pl_dedup = pl_rem[pl_rem["validation_status"] != "DUPLICATE"].copy()
        pl_out = out_dir / "prediction_log_remediated.parquet"
        pl_dedup_out = out_dir / "prediction_log_deduplicated.parquet"
        pl_rem.to_parquet(pl_out, index=False)
        pl_dedup.to_parquet(pl_dedup_out, index=False)
        summary["datasets"]["prediction_log"] = {
            "raw_rows": len(pl),
            "shadow_rows": len(pl_rem),
            "deduplicated_rows": len(pl_dedup),
            "duplicate_rows_filtered": int((pl_rem["validation_status"] == "DUPLICATE").sum()),
            "status_counts": {str(k): int(v) for k, v in pl_rem["validation_status"].value_counts().to_dict().items()},
            "provenance": check_provenance_completeness(pl_rem),
            "output_path": str(pl_out),
            "deduplicated_output_path": str(pl_dedup_out),
        }

    # 3. structured_events
    se_path = root / "data" / "predictions" / "structured_events.parquet"
    if se_path.exists():
        se = pd.read_parquet(se_path)
        se_rem = validate_event_records(
            se,
            source_dataset="data/predictions/structured_events.parquet",
            prospective_cutoff=prospective_cutoff,
        )
        se_out = out_dir / "structured_events_remediated.parquet"
        se_rem.to_parquet(se_out, index=False)
        summary["datasets"]["structured_events"] = {
            "raw_rows": len(se),
            "shadow_rows": len(se_rem),
            "status_counts": {str(k): int(v) for k, v in se_rem["validation_status"].value_counts().to_dict().items()},
            "provenance": check_provenance_completeness(se_rem),
            "output_path": str(se_out),
        }

    # 4. Metadata version drift manifest
    manifest = resolve_metadata_version_drift(root)
    manifest_out = out_dir / "metadata_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary["metadata_manifest"] = manifest

    summary_out = out_dir / "remediation_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
