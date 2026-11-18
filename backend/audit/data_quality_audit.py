"""Independent Data Quality Monitoring & Audit Module (Workstream B).

Read-only audit across:
1. Prediction integrity (forward_predictions & prediction_log)
2. Outcome integrity (forward_outcomes vs equities_daily)
3. News integrity (structured_events & quarantined_events)
4. Benchmark integrity (Nifty 50 & horizon-matched naive momentum)
5. Cross-report consistency (phase5_intelligence.json vs forward_validation.json)
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..config import IST_TZ, is_nse_trading_day
from ..news_intel.extractor import _COMPANY_ALIASES, _FALSE_POSITIVE_ENTITY_PATTERNS
from ..news_intel.forward_validation import _compute_nifty_window_return
from ..news_intel.relationships import PILOT_UNIVERSE_5, RELATIONSHIP_MAP
from ..storage.duckdb_store import DuckDBStore

VALID_NSE_SESSIONS = {"PRE_MARKET", "INTRADAY", "POST_MARKET"}
VALID_HORIZONS = {"1D", "5D", "10D"}
HORIZON_HOLD_DAYS = {"1D": 1, "5D": 5, "10D": 10}
VALID_OUTCOME_STATUSES = {"MATURED", "PENDING", "INVALID", "UNAVAILABLE"}


class DomainAuditSummary(BaseModel):
    domain: str
    records_inspected: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    missing_data: int = 0
    quarantined_records: int = 0
    warnings: list[str] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class DataQualityAuditReport(BaseModel):
    generated_at: str
    overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    total_records_inspected: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    missing_data: int = 0
    quarantined_records: int = 0
    warnings: list[str] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)
    prediction_integrity: DomainAuditSummary
    outcome_integrity: DomainAuditSummary
    news_integrity: DomainAuditSummary
    benchmark_integrity: DomainAuditSummary
    cross_report_consistency: dict[str, Any] = Field(default_factory=dict)


def _parse_iso_dt(val: Any) -> datetime | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "nat", "none", "null"}:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(IST_TZ) if dt.tzinfo else dt.replace(tzinfo=IST_TZ)
    except Exception:
        return None


def _parse_iso_date(val: Any) -> date | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "nat", "none", "null"}:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def audit_prediction_integrity(
    snaps_df: pd.DataFrame,
    prediction_log_df: pd.DataFrame | None = None,
) -> DomainAuditSummary:
    """Check prediction IDs, timestamps, NSE sessions, run IDs, duplicates, and canonical links."""
    summary = DomainAuditSummary(domain="prediction_integrity")
    if snaps_df is None or snaps_df.empty:
        summary.warnings.append("forward_predictions store is empty; 0 snapshots inspected.")
        return summary

    n = len(snaps_df)
    summary.records_inspected = n

    invalid_ids: set[int] = set()
    missing_cnt = 0
    dup_cnt = 0

    # 1. Unique prediction_id check
    if "prediction_id" not in snaps_df.columns:
        summary.critical_failures.append("Missing required column 'prediction_id' in forward_predictions.")
        summary.invalid_records = n
        return summary

    pid_series = snaps_df["prediction_id"].astype(str)
    dup_pid_mask = pid_series.duplicated(keep="first") | pid_series.str.strip().eq("") | snaps_df["prediction_id"].isna()
    if dup_pid_mask.any():
        bad_pids = pid_series[dup_pid_mask].tolist()
        summary.critical_failures.append(f"Duplicate or empty prediction_id detected ({len(bad_pids)} rows): {bad_pids[:5]}")
        invalid_ids.update(snaps_df.index[dup_pid_mask].tolist())

    # 2. Observation date & NSE trading day check + prediction_timestamp validity
    now_ist = datetime.now(IST_TZ)
    non_trading_dates: list[str] = []
    invalid_ts_rows = 0
    for idx, row in snaps_df.iterrows():
        obs_d = _parse_iso_date(row.get("observation_date"))
        pred_ts = _parse_iso_dt(row.get("prediction_timestamp"))
        if obs_d is None:
            invalid_ts_rows += 1
            missing_cnt += 1
            invalid_ids.add(idx)
        elif not is_nse_trading_day(obs_d):
            non_trading_dates.append(f"{row.get('symbol')}:{obs_d.isoformat()}")
            invalid_ids.add(idx)

        if pred_ts is None:
            invalid_ts_rows += 1
            missing_cnt += 1
            invalid_ids.add(idx)
        elif pred_ts > now_ist:
            # Allow slight clock skew up to 5 minutes, flag beyond that
            if (pred_ts - now_ist).total_seconds() > 300:
                summary.warnings.append(f"Future prediction_timestamp on {row.get('prediction_id')}: {pred_ts.isoformat()}")

    if invalid_ts_rows > 0:
        summary.critical_failures.append(f"{invalid_ts_rows} prediction snapshot(s) have missing/invalid timestamps or dates.")
    if non_trading_dates:
        summary.critical_failures.append(
            f"{len(non_trading_dates)} prediction snapshot(s) fall on NSE holidays/weekends: {non_trading_dates[:5]}"
        )

    # 3. NSE session & prediction_run_id
    if "nse_session" not in snaps_df.columns:
        summary.warnings.append("Column 'nse_session' missing from forward_predictions.")
        missing_cnt += 1
    else:
        bad_sess = snaps_df[~snaps_df["nse_session"].astype(str).isin(VALID_NSE_SESSIONS)]
        if not bad_sess.empty:
            summary.critical_failures.append(f"{len(bad_sess)} snapshot(s) have invalid nse_session.")
            invalid_ids.update(bad_sess.index.tolist())

    if "prediction_run_id" not in snaps_df.columns:
        summary.warnings.append("Column 'prediction_run_id' missing from forward_predictions.")
        missing_cnt += 1
    else:
        missing_rid = snaps_df["prediction_run_id"].isna() | (snaps_df["prediction_run_id"].astype(str).str.strip() == "")
        if missing_rid.any():
            missing_cnt += int(missing_rid.sum())
            summary.warnings.append(f"{int(missing_rid.sum())} snapshot(s) lack prediction_run_id.")

    # 4. Duplicate & canonical consistency
    if "is_duplicate_run" in snaps_df.columns and "canonical_prediction_id" in snaps_df.columns:
        all_pids = set(snaps_df["prediction_id"].astype(str))
        canon_pids = set(
            snaps_df.loc[~snaps_df["is_duplicate_run"].fillna(False).astype(bool), "prediction_id"].astype(str)
        )
        dup_mask = snaps_df["is_duplicate_run"].fillna(False).astype(bool)
        dup_cnt = int(dup_mask.sum())

        # Check canonical records have canonical_prediction_id == prediction_id
        bad_canon_self = snaps_df[~dup_mask & (snaps_df["canonical_prediction_id"].astype(str) != snaps_df["prediction_id"].astype(str))]
        if not bad_canon_self.empty:
            summary.critical_failures.append(
                f"{len(bad_canon_self)} canonical snapshot(s) have canonical_prediction_id != prediction_id."
            )
            invalid_ids.update(bad_canon_self.index.tolist())

        # Check duplicate records point to an existing canonical_prediction_id
        dangling_canon = snaps_df[dup_mask & ~snaps_df["canonical_prediction_id"].astype(str).isin(canon_pids)]
        if not dangling_canon.empty:
            summary.critical_failures.append(
                f"{len(dangling_canon)} duplicate snapshot(s) reference non-canonical or missing canonical_prediction_id."
            )
            invalid_ids.update(dangling_canon.index.tolist())

        # Check if multiple non-duplicate rows exist for same (symbol, observation_date, nse_session)
        if "nse_session" in snaps_df.columns:
            canon_df = snaps_df[~dup_mask]
            multi_canon = canon_df.duplicated(subset=["symbol", "observation_date", "nse_session"], keep=False)
            if multi_canon.any():
                summary.critical_failures.append(
                    f"{int(multi_canon.sum())} canonical snapshot(s) collide on (symbol, observation_date, nse_session)."
                )
                invalid_ids.update(canon_df.index[multi_canon].tolist())

        # Check orphaned canonical_prediction_id not in all_pids
        bad_ref = snaps_df[~snaps_df["canonical_prediction_id"].astype(str).isin(all_pids)]
        if not bad_ref.empty:
            summary.critical_failures.append(f"{len(bad_ref)} row(s) have dangling canonical_prediction_id.")
            invalid_ids.update(bad_ref.index.tolist())
    else:
        summary.warnings.append("Canonical deduplication columns (is_duplicate_run / canonical_prediction_id) missing.")

    # 5. Also inspect prediction_log.parquet if provided (cross-check)
    plog_dups = 0
    if prediction_log_df is not None and not prediction_log_df.empty:
        plog_dups = int(prediction_log_df.duplicated(subset=["symbol", "observation_date"]).sum())
        if plog_dups > 0:
            summary.warnings.append(
                f"Legacy prediction_log.parquet contains {plog_dups} duplicate (symbol, observation_date) rows "
                f"out of {len(prediction_log_df)} total; un-deduplicated readers will overcount."
            )

    summary.invalid_records = len(invalid_ids)
    summary.duplicate_records = dup_cnt
    summary.missing_data = missing_cnt
    summary.valid_records = max(0, n - summary.invalid_records - summary.duplicate_records)
    summary.details = {
        "total_snapshots": n,
        "canonical_snapshots": n - dup_cnt,
        "duplicate_snapshots": dup_cnt,
        "legacy_prediction_log_duplicates": plog_dups,
    }
    return summary


def audit_outcome_integrity(
    snaps_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    equities_df: pd.DataFrame | None = None,
) -> DomainAuditSummary:
    """Check horizon assignment, entry/exit prices, maturity, premature finalisation, and status classification."""
    summary = DomainAuditSummary(domain="outcome_integrity")
    if outcomes_df is None or outcomes_df.empty:
        summary.warnings.append("forward_outcomes store is empty; 0 outcomes inspected.")
        return summary

    n = len(outcomes_df)
    summary.records_inspected = n
    invalid_ids: set[int] = set()
    missing_cnt = 0

    # Build symbol -> sorted trading dates map if equities_df provided
    sym_dates: dict[str, list[str]] = {}
    global_max_dt = ""
    if equities_df is not None and not equities_df.empty:
        eq = equities_df[["symbol", "observation_date"]].copy()
        eq["symbol"] = eq["symbol"].astype(str).str.upper()
        eq["observation_date"] = eq["observation_date"].astype(str)
        eq = eq.sort_values(["symbol", "observation_date"]).drop_duplicates()
        global_max_dt = str(eq["observation_date"].max())
        sym_dates = {str(s): grp["observation_date"].tolist() for s, grp in eq.groupby("symbol", sort=False)}

    # Check duplicate (prediction_id, horizon)
    dup_out = outcomes_df.duplicated(subset=["prediction_id", "horizon"], keep="first")
    dup_cnt = int(dup_out.sum())
    if dup_cnt > 0:
        summary.critical_failures.append(f"{dup_cnt} duplicate (prediction_id, horizon) outcome record(s) found.")
        invalid_ids.update(outcomes_df.index[dup_out].tolist())

    # Check every snapshot has all 3 horizons
    if snaps_df is not None and not snaps_df.empty:
        snap_pids = set(snaps_df["prediction_id"].astype(str))
        out_by_pid = outcomes_df.groupby("prediction_id")["horizon"].apply(set).to_dict()
        missing_horizons = [pid for pid in snap_pids if set(out_by_pid.get(pid, set())) != VALID_HORIZONS]
        if missing_horizons:
            summary.warnings.append(
                f"{len(missing_horizons)} snapshot(s) missing one or more of 1D/5D/10D outcome rows."
            )
            missing_cnt += len(missing_horizons)

    matured_cnt = 0
    pending_cnt = 0
    invalid_status_cnt = 0
    unavailable_cnt = 0

    for idx, row in outcomes_df.iterrows():
        horizon = str(row.get("horizon", ""))
        status = str(row.get("status", ""))
        sym = str(row.get("symbol", "")).upper()
        obs_dt = str(row.get("observation_date", ""))

        if horizon not in VALID_HORIZONS:
            summary.critical_failures.append(f"Invalid horizon '{horizon}' on {row.get('prediction_id')}.")
            invalid_ids.add(idx)
            continue

        if status not in VALID_OUTCOME_STATUSES:
            summary.critical_failures.append(f"Invalid outcome status '{status}' on {row.get('prediction_id')}:{horizon}.")
            invalid_ids.add(idx)
            continue

        if status == "MATURED":
            matured_cnt += 1
            ep = row.get("entry_price")
            xp = row.get("exit_price")
            ed = row.get("entry_date")
            xd = row.get("exit_date")
            if pd.isna(ep) or pd.isna(xp) or float(ep) <= 0 or float(xp) <= 0:
                summary.critical_failures.append(
                    f"MATURED outcome {row.get('prediction_id')}:{horizon} has missing or non-positive price (entry={ep}, exit={xp})."
                )
                invalid_ids.add(idx)
                missing_cnt += 1
            if pd.isna(ed) or pd.isna(xd) or str(ed) <= obs_dt or str(xd) < str(ed):
                summary.critical_failures.append(
                    f"MATURED outcome {row.get('prediction_id')}:{horizon} has invalid window ({obs_dt} -> {ed} -> {xd})."
                )
                invalid_ids.add(idx)
            if global_max_dt and pd.notna(xd) and str(xd) > global_max_dt:
                summary.critical_failures.append(
                    f"Prematurely finalised outcome {row.get('prediction_id')}:{horizon}: exit_date {xd} > market max {global_max_dt}."
                )
                invalid_ids.add(idx)
            # Verify exact trading-session maturity count if equities_df available
            dates_list = sym_dates.get(sym, [])
            if dates_list and pd.notna(ed) and pd.notna(xd):
                after = [d for d in dates_list if d > obs_dt]
                hold_n = HORIZON_HOLD_DAYS[horizon]
                if len(after) < hold_n:
                    summary.critical_failures.append(
                        f"Prematurely MATURED outcome {row.get('prediction_id')}:{horizon}: only {len(after)} sessions after {obs_dt}, need {hold_n}."
                    )
                    invalid_ids.add(idx)
        elif status == "PENDING":
            pending_cnt += 1
            # Check if a PENDING outcome should actually have matured already
            dates_list = sym_dates.get(sym, [])
            if dates_list:
                after = [d for d in dates_list if d > obs_dt]
                if len(after) >= HORIZON_HOLD_DAYS[horizon]:
                    summary.warnings.append(
                        f"Outcome {row.get('prediction_id')}:{horizon} is still PENDING despite {len(after)} sessions available after {obs_dt}."
                    )
        elif status == "INVALID":
            invalid_status_cnt += 1
            if not str(row.get("status_reason") or "").strip():
                summary.warnings.append(f"INVALID outcome {row.get('prediction_id')}:{horizon} lacks status_reason.")
        elif status == "UNAVAILABLE":
            unavailable_cnt += 1
            if not str(row.get("status_reason") or "").strip():
                summary.warnings.append(f"UNAVAILABLE outcome {row.get('prediction_id')}:{horizon} lacks status_reason.")

    summary.invalid_records = len(invalid_ids)
    summary.duplicate_records = dup_cnt
    summary.missing_data = missing_cnt
    summary.valid_records = max(0, n - summary.invalid_records - summary.duplicate_records)
    summary.details = {
        "matured": matured_cnt,
        "pending": pending_cnt,
        "invalid_status": invalid_status_cnt,
        "unavailable_status": unavailable_cnt,
    }
    return summary


def audit_news_integrity(
    events_df: pd.DataFrame,
    quarantined_df: pd.DataFrame | None = None,
    baseline_start_date: str = "2026-11-16",
) -> DomainAuditSummary:
    """Check publication/collection timestamps, entity attribution, quarantine tracking, and prospective separation."""
    summary = DomainAuditSummary(domain="news_integrity")
    q_cnt = len(quarantined_df) if (quarantined_df is not None and not quarantined_df.empty) else 0
    summary.quarantined_records = q_cnt

    if events_df is None or events_df.empty:
        summary.records_inspected = q_cnt
        summary.warnings.append("structured_events store is empty.")
        return summary

    n = len(events_df)
    summary.records_inspected = n + q_cnt
    invalid_ids: set[int] = set()
    missing_cnt = 0
    dup_cnt = int(events_df["is_duplicate_report"].fillna(False).sum()) if "is_duplicate_report" in events_df.columns else 0

    now_ist = datetime.now(IST_TZ)
    prospective_cnt = 0
    historical_cnt = 0
    entity_contamination_cnt = 0

    for idx, row in events_df.iterrows():
        pub_dt = _parse_iso_dt(row.get("publication_timestamp"))
        coll_dt = _parse_iso_dt(row.get("collection_timestamp"))
        ticker = str(row.get("ticker", "")).upper().strip()
        category = str(row.get("event_category", "")).strip()
        desc = str(row.get("event_description", "")).lower()

        if pub_dt is None:
            summary.critical_failures.append(f"Event {row.get('event_id')} missing valid publication_timestamp.")
            invalid_ids.add(idx)
            missing_cnt += 1
        elif pub_dt > now_ist and (pub_dt - now_ist).total_seconds() > 300:
            summary.critical_failures.append(f"Event {row.get('event_id')} has future publication_timestamp: {pub_dt.isoformat()}.")
            invalid_ids.add(idx)

        if coll_dt is None:
            missing_cnt += 1
            historical_cnt += 1
        else:
            if pub_dt is not None and pub_dt > coll_dt and (pub_dt - coll_dt).total_seconds() > 300:
                summary.critical_failures.append(
                    f"Event {row.get('event_id')} has publication_timestamp ({pub_dt.isoformat()}) > collection_timestamp ({coll_dt.isoformat()})."
                )
                invalid_ids.add(idx)
            if pub_dt is not None and pub_dt.strftime("%Y-%m-%d") < baseline_start_date:
                historical_cnt += 1
            else:
                prospective_cnt += 1

        # Company attribution & false-positive entity contamination check
        if ticker and ticker not in RELATIONSHIP_MAP and ticker not in PILOT_UNIVERSE_5:
            summary.warnings.append(f"Event {row.get('event_id')} attributed to unknown ticker '{ticker}'.")

        fp_patterns = _FALSE_POSITIVE_ENTITY_PATTERNS.get(ticker, ())
        if any(re.search(pat, desc) for pat in fp_patterns):
            entity_contamination_cnt += 1
            invalid_ids.add(idx)
            summary.critical_failures.append(
                f"Unrelated entity contamination on {ticker} event {row.get('event_id')}: '{row.get('event_description')}'"
            )

        if category == "company_news" and ticker in _COMPANY_ALIASES:
            aliases = _COMPANY_ALIASES[ticker]
            if not any(re.search(rf"\b{re.escape(a)}\b", desc) for a in aliases):
                # Check if it mentions a different pilot company instead
                other_match = [
                    other_t for other_t, other_als in _COMPANY_ALIASES.items()
                    if other_t != ticker and any(re.search(rf"\b{re.escape(a)}\b", desc) for a in other_als)
                ]
                if other_match:
                    entity_contamination_cnt += 1
                    invalid_ids.add(idx)
                    summary.critical_failures.append(
                        f"Misattributed company_news event {row.get('event_id')}: tagged {ticker} but mentions {other_match}."
                    )

    if historical_cnt > 0:
        summary.warnings.append(
            f"{historical_cnt} pre-baseline/historical news event(s) (missing collection_timestamp or pre-{baseline_start_date}) "
            f"coexist with {prospective_cnt} prospective event(s) in structured_events."
        )

    if quarantined_df is not None and not quarantined_df.empty:
        no_reason = quarantined_df["quarantine_reason"].isna() | (quarantined_df["quarantine_reason"].astype(str).str.strip() == "")
        if no_reason.any():
            summary.warnings.append(f"{int(no_reason.sum())} quarantined news record(s) lack quarantine_reason.")

    summary.invalid_records = len(invalid_ids)
    summary.duplicate_records = dup_cnt
    summary.missing_data = missing_cnt
    summary.valid_records = max(0, n - summary.invalid_records - summary.duplicate_records)
    summary.details = {
        "structured_events_total": n,
        "prospective_events": prospective_cnt,
        "historical_events": historical_cnt,
        "quarantined_events": q_cnt,
        "entity_contamination_count": entity_contamination_cnt,
    }
    return summary


def audit_benchmark_integrity(
    snaps_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    indices_df: pd.DataFrame | None,
) -> DomainAuditSummary:
    """Check Nifty 50 availability, matching windows, benchmark return math, and horizon-matched assumptions."""
    summary = DomainAuditSummary(domain="benchmark_integrity")
    if indices_df is None or indices_df.empty or "index_code" not in indices_df.columns:
        summary.critical_failures.append("indices_daily store is missing or empty; Nifty 50 benchmark unavailable.")
        summary.missing_data = 1
        return summary

    codes_upper = indices_df["index_code"].astype(str).str.upper()
    n50 = indices_df[codes_upper.isin({"NIFTY 50", "NIFTY_50"})].copy()
    if n50.empty:
        summary.critical_failures.append("No 'NIFTY 50' or 'NIFTY_50' rows found in indices_daily.")
        summary.missing_data = 1
        return summary
    if "NIFTY 50" not in set(codes_upper) and "NIFTY_50" in set(codes_upper):
        summary.warnings.append(
            "indices_daily uses index_code='NIFTY_50' (underscore), whereas forward_validation.py queries 'NIFTY 50' (space), causing null benchmark returns."
        )

    n50_map: dict[str, tuple[float | None, float | None]] = {}
    for _, r in n50.iterrows():
        dt_s = str(r["observation_date"])
        op = float(r["open"]) if "open" in n50.columns and pd.notna(r.get("open")) else None
        cl = float(r["close"]) if "close" in n50.columns and pd.notna(r.get("close")) else None
        n50_map[dt_s] = (op, cl)

    invalid_cnt = 0
    missing_cnt = 0
    inspected = 0

    if outcomes_df is not None and not outcomes_df.empty:
        matured = outcomes_df[outcomes_df["status"] == "MATURED"]
        inspected += len(matured)
        for _, row in matured.iterrows():
            ed = str(row.get("entry_date") or "")
            xd = str(row.get("exit_date") or "")
            stored_bench = row.get("benchmark_return_pct")
            recomputed = _compute_nifty_window_return(n50_map, ed, xd)
            if recomputed is None:
                missing_cnt += 1
                summary.warnings.append(
                    f"Nifty 50 data missing for matured outcome window [{ed}, {xd}] ({row.get('prediction_id')}:{row.get('horizon')})."
                )
            elif pd.isna(stored_bench):
                missing_cnt += 1
                summary.warnings.append(
                    f"benchmark_return_pct is null on {row.get('prediction_id')}:{row.get('horizon')} despite Nifty 50 availability."
                )
            elif abs(float(stored_bench) - float(recomputed)) > 0.05:
                invalid_cnt += 1
                summary.critical_failures.append(
                    f"Benchmark return mismatch on {row.get('prediction_id')}:{row.get('horizon')}: "
                    f"stored={stored_bench}%, recomputed={recomputed}%."
                )

    # Check naive momentum benchmark horizon consistency on snapshots
    if snaps_df is not None and not snaps_df.empty:
        inspected += len(snaps_df)
        for col in ("benchmark_prob_1d_up", "benchmark_prob_5d_up", "benchmark_prob_10d_up"):
            if col not in snaps_df.columns:
                summary.warnings.append(f"Snapshot missing horizon-matched benchmark column '{col}'.")
                missing_cnt += 1

    summary.records_inspected = inspected
    summary.invalid_records = invalid_cnt
    summary.missing_data = missing_cnt
    summary.valid_records = max(0, inspected - invalid_cnt - missing_cnt)
    summary.details = {
        "nifty50_bars_available": len(n50_map),
        "nifty50_min_date": min(n50_map.keys()) if n50_map else None,
        "nifty50_max_date": max(n50_map.keys()) if n50_map else None,
    }
    return summary


def _check_cross_report_consistency(predictions_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Compare phase5_intelligence.json and forward_validation.json for version/count consistency."""
    warnings: list[str] = []
    info: dict[str, Any] = {}
    p5_path = predictions_dir / "phase5_intelligence.json"
    fv_path = predictions_dir / "forward_validation.json"
    if not (p5_path.exists() and fv_path.exists()):
        return info, warnings

    try:
        p5 = json.loads(p5_path.read_text(encoding="utf-8"))
        fv = json.loads(fv_path.read_text(encoding="utf-8"))
        p5_ver = p5.get("model_version")
        fv_ver = fv.get("model_version")
        p5_eval = (p5.get("monitoring") or {}).get("evaluated_outcomes_count")
        fv_matured = fv.get("matured_outcomes_total")
        info = {
            "phase5_model_version": p5_ver,
            "forward_validation_model_version": fv_ver,
            "phase5_monitoring_evaluated_outcomes": p5_eval,
            "forward_validation_matured_outcomes": fv_matured,
        }
        if p5_ver != fv_ver:
            warnings.append(
                f"Model version mismatch across published reports: phase5_intelligence.json='{p5_ver}' vs forward_validation.json='{fv_ver}'."
            )
        if p5_eval != fv_matured:
            warnings.append(
                f"Evaluated outcome count mismatch across published reports: phase5_intelligence.json monitoring={p5_eval} vs forward_validation.json matured={fv_matured}."
            )
    except Exception as exc:
        warnings.append(f"Failed to compare published JSON reports: {exc}")
    return info, warnings


def run_full_data_quality_audit(store: DuckDBStore | None = None) -> DataQualityAuditReport:
    """Run the complete read-only data quality audit across all domains and return a structured report."""
    st = store or DuckDBStore()

    snaps_df = pd.read_parquet(st.forward_predictions_parquet) if st.forward_predictions_parquet.exists() else pd.DataFrame()
    plog_df = pd.read_parquet(st.predictions_parquet) if st.predictions_parquet.exists() else pd.DataFrame()
    outcomes_df = pd.read_parquet(st.forward_outcomes_parquet) if st.forward_outcomes_parquet.exists() else pd.DataFrame()
    equities_df = pd.read_parquet(st.equities_parquet) if st.equities_parquet.exists() else pd.DataFrame()
    indices_df = pd.read_parquet(st.indices_parquet) if st.indices_parquet.exists() else pd.DataFrame()
    events_df = pd.read_parquet(st.events_parquet) if st.events_parquet.exists() else pd.DataFrame()
    quarantined_df = pd.read_parquet(st.quarantined_events_parquet) if st.quarantined_events_parquet.exists() else pd.DataFrame()

    pred_audit = audit_prediction_integrity(snaps_df, prediction_log_df=plog_df)
    out_audit = audit_outcome_integrity(snaps_df, outcomes_df, equities_df=equities_df)
    news_audit = audit_news_integrity(events_df, quarantined_df=quarantined_df)
    bench_audit = audit_benchmark_integrity(snaps_df, outcomes_df, indices_df=indices_df)
    cross_info, cross_warnings = _check_cross_report_consistency(st.predictions_dir)

    domains = (pred_audit, out_audit, news_audit, bench_audit)
    all_warnings = [w for d in domains for w in d.warnings] + cross_warnings
    all_critical = [c for d in domains for c in d.critical_failures]

    overall: Literal["PASS", "WARN", "FAIL"] = "FAIL" if all_critical else ("WARN" if all_warnings else "PASS")

    return DataQualityAuditReport(
        generated_at=datetime.now(IST_TZ).isoformat(),
        overall_status=overall,
        total_records_inspected=sum(d.records_inspected for d in domains),
        valid_records=sum(d.valid_records for d in domains),
        invalid_records=sum(d.invalid_records for d in domains),
        duplicate_records=sum(d.duplicate_records for d in domains),
        missing_data=sum(d.missing_data for d in domains),
        quarantined_records=sum(d.quarantined_records for d in domains),
        warnings=all_warnings,
        critical_failures=all_critical,
        prediction_integrity=pred_audit,
        outcome_integrity=out_audit,
        news_integrity=news_audit,
        benchmark_integrity=bench_audit,
        cross_report_consistency=cross_info,
    )
