"""Deterministic Data Integrity Audit Engine for BharatAlpha AI.

Read-only audit across all datasets:
- A: Predictions (forward_predictions + legacy prediction_log)
- B: Outcomes (forward_outcomes + equities_daily + optional reference snapshot)
- C: News (structured_events + quarantined_events + phase5_intelligence timelines)
- D: Benchmark (indices_daily + forward_validation benchmarks)
- E: Published Reports (docs/data vs data/predictions consistency)
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..config import IST_TZ, is_nse_trading_day
from ..news_intel.extractor import _COMPANY_ALIASES, _FALSE_POSITIVE_ENTITY_PATTERNS
from ..news_intel.forward_validation import _compute_nifty_window_return
from ..storage.duckdb_store import DuckDBStore
from .classification import (
    ClassificationReport,
    NSE_OPEN_TIME,
    _next_nse_trading_day,
    _parse_ist_timestamp,
    classify_forward_predictions,
    classify_legacy_prediction_log,
)

Severity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
OverallStatus = Literal["PASS", "WARN", "FAIL"]


class CheckResult(BaseModel):
    check_id: str
    dataset: str
    description: str
    passed: bool
    severity: Severity
    affected_records: int = 0
    examples: list[str] = Field(default_factory=list)


class DatasetAuditResult(BaseModel):
    dataset: str
    records_inspected: int = 0
    checks: list[CheckResult] = Field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: OverallStatus = "PASS"


class DataIntegrityReport(BaseModel):
    audit_timestamp: str
    base_commit_sha: str = "8f328b8cdb8f7a1e508c1c47a648849945873013"
    datasets: dict[str, DatasetAuditResult] = Field(default_factory=dict)
    classification: ClassificationReport = Field(default_factory=ClassificationReport)
    overall_status: OverallStatus = "PASS"


def _summarize_dataset(dataset: str, records_inspected: int, checks: list[CheckResult]) -> DatasetAuditResult:
    passed = sum(1 for c in checks if c.passed)
    failed = len(checks) - passed
    errors: list[str] = []
    warnings: list[str] = []
    for c in checks:
        if not c.passed:
            msg = f"[{c.check_id}] ({c.severity}) {c.description} [affected={c.affected_records}]"
            if c.severity in ("ERROR", "CRITICAL"):
                errors.append(msg)
            elif c.severity == "WARNING":
                warnings.append(msg)
    status: OverallStatus = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return DatasetAuditResult(
        dataset=dataset,
        records_inspected=records_inspected,
        checks=checks,
        passed_checks=passed,
        failed_checks=failed,
        errors=errors,
        warnings=warnings,
        status=status,
    )


def _is_non_empty(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    return str(val).strip() not in ("", "nan", "None", "NaT")


def _parse_json_dict(val: Any) -> dict[str, Any]:
    if isinstance(val, dict):
        return val
    if not _is_non_empty(val):
        return {}
    try:
        parsed = json.loads(str(val))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Dataset A1: forward_predictions
# ---------------------------------------------------------------------------


def check_forward_predictions(
    predictions_df: pd.DataFrame | None,
    outcomes_df: pd.DataFrame | None = None,
) -> DatasetAuditResult:
    dataset = "forward_predictions"
    if predictions_df is None or predictions_df.empty:
        return _summarize_dataset(
            dataset,
            0,
            [
                CheckResult(
                    check_id="PRED_EMPTY",
                    dataset=dataset,
                    description="forward_predictions dataset is empty or missing.",
                    passed=False,
                    severity="WARNING",
                )
            ],
        )

    df = predictions_df.copy()
    checks: list[CheckResult] = []

    # 1. Missing prediction_id
    missing_id = df[~df["prediction_id"].apply(_is_non_empty)] if "prediction_id" in df.columns else df
    checks.append(
        CheckResult(
            check_id="PRED_MISSING_ID",
            dataset=dataset,
            description="Every forward prediction must have a non-empty prediction_id.",
            passed=len(missing_id) == 0,
            severity="CRITICAL",
            affected_records=len(missing_id),
            examples=[str(i) for i in missing_id.index[:5]],
        )
    )

    # 2. Duplicate prediction_id
    dup_id = df[df["prediction_id"].duplicated(keep=False)] if "prediction_id" in df.columns else pd.DataFrame()
    checks.append(
        CheckResult(
            check_id="PRED_DUPLICATE_ID",
            dataset=dataset,
            description="prediction_id must be unique across forward_predictions.",
            passed=len(dup_id) == 0,
            severity="CRITICAL",
            affected_records=len(dup_id),
            examples=dup_id["prediction_id"].astype(str).head(5).tolist() if not dup_id.empty else [],
        )
    )

    # 3. Unflagged duplicate (symbol, observation_date, nse_session)
    is_dup = df["is_duplicate_run"].fillna(False).astype(bool) if "is_duplicate_run" in df.columns else pd.Series(False, index=df.index)
    canon = df[~is_dup].copy()
    unflagged_dups: list[str] = []
    if not canon.empty and {"symbol", "observation_date", "nse_session"}.issubset(canon.columns):
        grp = canon.groupby(["symbol", "observation_date", "nse_session"]).size()
        bad_keys = grp[grp > 1]
        unflagged_dups = [f"{k[0]}|{k[1]}|{k[2]} (n={v})" for k, v in bad_keys.items()]
    checks.append(
        CheckResult(
            check_id="PRED_UNFLAGGED_DUP_SESSION",
            dataset=dataset,
            description="At most one canonical (is_duplicate_run=False) snapshot per (symbol, observation_date, nse_session).",
            passed=len(unflagged_dups) == 0,
            severity="ERROR",
            affected_records=len(unflagged_dups),
            examples=unflagged_dups[:5],
        )
    )

    # 4. Duplicate target entry session across different nse_sessions
    dup_targets: list[str] = []
    if not canon.empty and {"symbol", "observation_date"}.issubset(canon.columns):
        canon["_target_entry"] = canon["observation_date"].apply(lambda d: (_next_nse_trading_day(str(d)) or "").isoformat() if _next_nse_trading_day(str(d)) else "")
        tgrp = canon[canon["_target_entry"] != ""].groupby(["symbol", "_target_entry"]).size()
        bad_t = tgrp[tgrp > 1]
        dup_targets = [f"{k[0]}|entry={k[1]} (n={v})" for k, v in bad_t.items()]
    checks.append(
        CheckResult(
            check_id="PRED_DUPLICATE_TARGET_SESSION",
            dataset=dataset,
            description="Canonical snapshots across sessions should not target the same (symbol, entry_date) without deduplication.",
            passed=len(dup_targets) == 0,
            severity="WARNING",
            affected_records=len(dup_targets),
            examples=dup_targets[:5],
        )
    )

    # 5. Missing model_version / feature_set_version
    miss_mv = df[~df["model_version"].apply(_is_non_empty)] if "model_version" in df.columns else df
    checks.append(
        CheckResult(
            check_id="PRED_MISSING_MODEL_VERSION",
            dataset=dataset,
            description="Every prediction must record a non-empty model_version.",
            passed=len(miss_mv) == 0,
            severity="ERROR",
            affected_records=len(miss_mv),
            examples=miss_mv["prediction_id"].astype(str).head(5).tolist() if "prediction_id" in miss_mv else [],
        )
    )
    miss_fv = df[~df["feature_set_version"].apply(_is_non_empty)] if "feature_set_version" in df.columns else df
    checks.append(
        CheckResult(
            check_id="PRED_MISSING_FEATURE_VERSION",
            dataset=dataset,
            description="Every prediction must record a non-empty feature_set_version.",
            passed=len(miss_fv) == 0,
            severity="WARNING",
            affected_records=len(miss_fv),
            examples=miss_fv["prediction_id"].astype(str).head(5).tolist() if "prediction_id" in miss_fv else [],
        )
    )

    # 6. Missing/invalid prediction_timestamp & non-trading observation_date
    bad_ts: list[str] = []
    non_trading_obs: list[str] = []
    post_event_canon: list[str] = []
    for _, row in df.iterrows():
        pid = str(row.get("prediction_id", ""))
        ts_ist = _parse_ist_timestamp(row.get("prediction_timestamp"))
        if ts_ist is None:
            bad_ts.append(pid)
        obs_str = str(row.get("observation_date", "")).strip()[:10]
        try:
            obs_d = date.fromisoformat(obs_str)
            if not is_nse_trading_day(obs_d):
                non_trading_obs.append(f"{pid} ({obs_str})")
        except Exception:
            non_trading_obs.append(f"{pid} ({obs_str})")

        if not bool(row.get("is_duplicate_run", False)) and ts_ist is not None:
            entry_d = _next_nse_trading_day(obs_str)
            if entry_d is not None:
                entry_open = datetime.combine(entry_d, NSE_OPEN_TIME, tzinfo=IST_TZ)
                if ts_ist >= entry_open:
                    post_event_canon.append(f"{pid} (ts={ts_ist.isoformat()} >= entry={entry_open.isoformat()})")

    checks.append(
        CheckResult(
            check_id="PRED_INVALID_TIMESTAMP",
            dataset=dataset,
            description="Every prediction must have a valid ISO prediction_timestamp.",
            passed=len(bad_ts) == 0,
            severity="CRITICAL",
            affected_records=len(bad_ts),
            examples=bad_ts[:5],
        )
    )
    checks.append(
        CheckResult(
            check_id="PRED_NON_TRADING_OBS_DATE",
            dataset=dataset,
            description="observation_date must be a valid NSE trading day.",
            passed=len(non_trading_obs) == 0,
            severity="ERROR",
            affected_records=len(non_trading_obs),
            examples=non_trading_obs[:5],
        )
    )
    checks.append(
        CheckResult(
            check_id="PRED_POST_EVENT_TIMING",
            dataset=dataset,
            description="Canonical prospective predictions must be generated strictly before entry open (09:15 IST on next trading day).",
            passed=len(post_event_canon) == 0,
            severity="ERROR",
            affected_records=len(post_event_canon),
            examples=post_event_canon[:5],
        )
    )

    # 7. Missing horizon probabilities vs model_availability & probability bounds [0, 1]
    avail_mismatch: list[str] = []
    prob_out_of_range: list[str] = []
    prob_cols = [
        "prob_1d_up",
        "prob_5d_up",
        "prob_10d_up",
        "benchmark_prob_up",
        "benchmark_prob_1d_up",
        "benchmark_prob_5d_up",
        "benchmark_prob_10d_up",
    ]
    for _, row in df.iterrows():
        pid = str(row.get("prediction_id", ""))
        avail = _parse_json_dict(row.get("model_availability"))
        for h, pcol in (("1D", "prob_1d_up"), ("5D", "prob_5d_up"), ("10D", "prob_10d_up")):
            if avail.get(h) is True:
                val = row.get(pcol)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    avail_mismatch.append(f"{pid}:{pcol}")
        for col in prob_cols:
            if col not in df.columns:
                continue
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                if col in ("prob_1d_up", "prob_5d_up", "prob_10d_up") and not avail:
                    prob_out_of_range.append(f"{pid}:{col}=NaN")
                continue
            try:
                fval = float(val)
                if math.isnan(fval) or fval < 0.0 or fval > 1.0:
                    prob_out_of_range.append(f"{pid}:{col}={fval}")
            except Exception:
                prob_out_of_range.append(f"{pid}:{col}={val}")

    checks.append(
        CheckResult(
            check_id="PRED_HORIZON_AVAILABILITY_MISMATCH",
            dataset=dataset,
            description="Horizons marked available in model_availability must have non-null probabilities.",
            passed=len(avail_mismatch) == 0,
            severity="ERROR",
            affected_records=len(avail_mismatch),
            examples=avail_mismatch[:5],
        )
    )
    checks.append(
        CheckResult(
            check_id="PRED_PROB_RANGE",
            dataset=dataset,
            description="All model and benchmark probabilities must lie in [0.0, 1.0].",
            passed=len(prob_out_of_range) == 0,
            severity="CRITICAL",
            affected_records=len(prob_out_of_range),
            examples=prob_out_of_range[:5],
        )
    )

    # 8. INFO: historical vs prospective summary
    _, h_sums, _ = classify_forward_predictions(df, outcomes_df, horizons=("1D",))
    s1 = h_sums["1D"]
    checks.append(
        CheckResult(
            check_id="PRED_PROSPECTIVE_VS_HISTORICAL",
            dataset=dataset,
            description=(
                f"1D classification counts: total={s1.total_stored}, canonical={s1.unique_canonical}, "
                f"prospective={s1.prospective} (matured={s1.matured}, pending={s1.pending}), "
                f"historical={s1.historical}, invalid={s1.invalid}, duplicate={s1.duplicate}."
            ),
            passed=True,
            severity="INFO",
            affected_records=s1.total_stored,
        )
    )

    return _summarize_dataset(dataset, len(df), checks)


# ---------------------------------------------------------------------------
# Dataset A2: Legacy prediction_log
# ---------------------------------------------------------------------------


def check_legacy_prediction_log(pred_log_df: pd.DataFrame | None) -> DatasetAuditResult:
    dataset = "prediction_log"
    if pred_log_df is None or pred_log_df.empty:
        return _summarize_dataset(
            dataset,
            0,
            [
                CheckResult(
                    check_id="LOG_EMPTY",
                    dataset=dataset,
                    description="prediction_log dataset is empty or missing.",
                    passed=False,
                    severity="WARNING",
                )
            ],
        )

    df = pred_log_df.copy()
    checks: list[CheckResult] = []

    miss_id = df[~df["prediction_id"].apply(_is_non_empty)] if "prediction_id" in df.columns else df
    checks.append(
        CheckResult(
            check_id="LOG_MISSING_ID",
            dataset=dataset,
            description="Every prediction_log row must have a non-empty prediction_id.",
            passed=len(miss_id) == 0,
            severity="CRITICAL",
            affected_records=len(miss_id),
        )
    )

    dup_id = df[df["prediction_id"].duplicated(keep=False)] if "prediction_id" in df.columns else pd.DataFrame()
    checks.append(
        CheckResult(
            check_id="LOG_DUPLICATE_ID",
            dataset=dataset,
            description="prediction_id must be unique in prediction_log.",
            passed=len(dup_id) == 0,
            severity="CRITICAL",
            affected_records=len(dup_id),
        )
    )

    summary = classify_legacy_prediction_log(df)
    checks.append(
        CheckResult(
            check_id="LOG_DUPLICATE_SYMBOL_DATE",
            dataset=dataset,
            description="prediction_log contains repeated runs for the same (symbol, observation_date) counted independently by monitoring.",
            passed=summary.duplicate == 0,
            severity="WARNING",
            affected_records=summary.duplicate,
            examples=[f"total={summary.total_stored}, unique={summary.unique_canonical_symbol_date}, duplicates={summary.duplicate}"],
        )
    )

    checks.append(
        CheckResult(
            check_id="LOG_MIXED_MODEL_VERSIONS",
            dataset=dataset,
            description="prediction_log mixes multiple model_version cohorts in a single rolling metric.",
            passed=len(summary.model_versions) <= 1,
            severity="WARNING",
            affected_records=len(summary.model_versions),
            examples=[f"{k}: {v}" for k, v in summary.model_versions.items()],
        )
    )

    bad_probs: list[str] = []
    for col in ("prob_next_day_up", "prob_5d_up", "prob_10d_up"):
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            fval = float(val)
            if fval < 0.0 or fval > 1.0:
                bad_probs.append(f"{df.loc[idx, 'prediction_id']}:{col}={fval}")
    checks.append(
        CheckResult(
            check_id="LOG_PROB_RANGE",
            dataset=dataset,
            description="All prediction_log probabilities must lie in [0.0, 1.0].",
            passed=len(bad_probs) == 0,
            severity="CRITICAL",
            affected_records=len(bad_probs),
            examples=bad_probs[:5],
        )
    )

    checks.append(
        CheckResult(
            check_id="LOG_OUTCOME_DEFINITION_NOTE",
            dataset=dataset,
            description="prediction_log uses close[t]->close[t+1] returns, which differs from forward_outcomes open[t+1]->close[t+H].",
            passed=True,
            severity="INFO",
            affected_records=summary.evaluated,
        )
    )

    return _summarize_dataset(dataset, len(df), checks)


# ---------------------------------------------------------------------------
# Dataset B: forward_outcomes
# ---------------------------------------------------------------------------


def check_forward_outcomes(
    outcomes_df: pd.DataFrame | None,
    predictions_df: pd.DataFrame | None,
    equities_df: pd.DataFrame | None,
    reference_outcomes_df: pd.DataFrame | None = None,
) -> DatasetAuditResult:
    dataset = "forward_outcomes"
    if outcomes_df is None or outcomes_df.empty:
        return _summarize_dataset(
            dataset,
            0,
            [
                CheckResult(
                    check_id="OUT_EMPTY",
                    dataset=dataset,
                    description="forward_outcomes dataset is empty or missing.",
                    passed=False,
                    severity="WARNING",
                )
            ],
        )

    out = outcomes_df.copy()
    checks: list[CheckResult] = []

    pred_ids = set(predictions_df["prediction_id"].astype(str)) if predictions_df is not None and not predictions_df.empty else set()
    orphans = out[~out["prediction_id"].astype(str).isin(pred_ids)] if pred_ids else out
    checks.append(
        CheckResult(
            check_id="OUT_ORPHAN_OUTCOME",
            dataset=dataset,
            description="Every outcome must reference an existing prediction_id in forward_predictions.",
            passed=len(orphans) == 0,
            severity="ERROR",
            affected_records=len(orphans),
            examples=orphans["prediction_id"].astype(str).head(5).tolist() if not orphans.empty else [],
        )
    )

    # Missing horizon rows for canonical predictions
    missing_h: list[str] = []
    if predictions_df is not None and not predictions_df.empty:
        existing_pairs = set(zip(out["prediction_id"].astype(str), out["horizon"].astype(str)))
        is_dup = predictions_df["is_duplicate_run"].fillna(False).astype(bool) if "is_duplicate_run" in predictions_df.columns else pd.Series(False, index=predictions_df.index)
        for pid in predictions_df.loc[~is_dup, "prediction_id"].astype(str):
            for h in ("1D", "5D", "10D"):
                if (pid, h) not in existing_pairs:
                    missing_h.append(f"{pid}:{h}")
    checks.append(
        CheckResult(
            check_id="OUT_MISSING_HORIZON_ROW",
            dataset=dataset,
            description="Every canonical prediction should have 1D, 5D, and 10D rows in forward_outcomes.",
            passed=len(missing_h) == 0,
            severity="WARNING",
            affected_records=len(missing_h),
            examples=missing_h[:5],
        )
    )

    # Duplicate (prediction_id, horizon)
    dup_ph = out[out.duplicated(subset=["prediction_id", "horizon"], keep=False)]
    checks.append(
        CheckResult(
            check_id="OUT_DUPLICATE_PRED_HORIZON",
            dataset=dataset,
            description="(prediction_id, horizon) must be unique in forward_outcomes.",
            passed=len(dup_ph) == 0,
            severity="CRITICAL",
            affected_records=len(dup_ph),
            examples=[f"{r['prediction_id']}:{r['horizon']}" for _, r in dup_ph.head(5).iterrows()],
        )
    )

    # Build equities lookup per symbol
    eq_by_sym: dict[str, list[tuple[str, float, float]]] = {}
    eq_max_date = ""
    if equities_df is not None and not equities_df.empty:
        eq = equities_df.copy()
        eq["observation_date"] = eq["observation_date"].astype(str).str[:10]
        eq_max_date = str(eq["observation_date"].max())
        eq = eq.sort_values(["symbol", "observation_date"])
        for sym, grp in eq.groupby("symbol"):
            eq_by_sym[str(sym).upper()] = [
                (str(r["observation_date"]), float(r["open"]), float(r["close"]))
                for _, r in grp.iterrows()
            ]

    horizon_n = {"1D": 1, "5D": 5, "10D": 10}
    premature: list[str] = []
    invalid_px: list[str] = []
    date_align_err: list[str] = []
    px_mismatch: list[str] = []
    bench_missing: list[str] = []
    math_err: list[str] = []
    unexplained: list[str] = []
    overdue_pending: list[str] = []

    for _, row in out.iterrows():
        pid = str(row.get("prediction_id", ""))
        sym = str(row.get("symbol", "")).upper().strip()
        obs = str(row.get("observation_date", "")).strip()[:10]
        h = str(row.get("horizon", "")).strip()
        status = str(row.get("status", "")).upper().strip()
        n_sess = horizon_n.get(h, 1)

        bars = eq_by_sym.get(sym, [])
        future_bars = [b for b in bars if b[0] > obs]

        if status == "MATURED":
            if len(future_bars) < n_sess:
                premature.append(f"{pid}:{h} (have {len(future_bars)}<{n_sess})")
            e_dt = str(row.get("entry_date", ""))[:10]
            x_dt = str(row.get("exit_date", ""))[:10]
            if eq_max_date and x_dt > eq_max_date:
                premature.append(f"{pid}:{h} exit {x_dt} > max {eq_max_date}")

            ep = row.get("entry_price")
            xp = row.get("exit_price")
            if pd.isna(ep) or pd.isna(xp) or float(ep) <= 0 or float(xp) <= 0:
                invalid_px.append(f"{pid}:{h} (ep={ep}, xp={xp})")
            elif len(future_bars) >= n_sess:
                exp_entry = future_bars[0]
                exp_exit = future_bars[n_sess - 1]
                if e_dt != exp_entry[0] or x_dt != exp_exit[0]:
                    date_align_err.append(f"{pid}:{h} got ({e_dt},{x_dt}) expected ({exp_entry[0]},{exp_exit[0]})")
                if abs(float(ep) - exp_entry[1]) / max(exp_entry[1], 1e-6) > 2e-4 or abs(float(xp) - exp_exit[2]) / max(exp_exit[2], 1e-6) > 2e-4:
                    px_mismatch.append(f"{pid}:{h} got ({ep},{xp}) eq ({exp_entry[1]},{exp_exit[2]})")

            if pd.isna(row.get("benchmark_return_pct")):
                bench_missing.append(f"{pid}:{h}")

            gross = row.get("gross_return_pct")
            cost = row.get("transaction_cost_pct")
            net = row.get("net_return_pct")
            ca_adj = bool(row.get("corporate_action_adjusted", False)) if pd.notna(row.get("corporate_action_adjusted")) else False
            if pd.notna(ep) and pd.notna(xp) and float(ep) > 0 and pd.notna(gross) and pd.notna(net) and pd.notna(cost):
                if not ca_adj:
                    exp_gross = ((float(xp) / float(ep)) - 1.0) * 100.0
                    if abs(float(gross) - exp_gross) > 0.02:
                        math_err.append(f"{pid}:{h} gross={gross} != {exp_gross:.4f}")
                exp_net = float(gross) - float(cost)
                if abs(float(net) - exp_net) > 0.02:
                    math_err.append(f"{pid}:{h} net={net} != {exp_net:.4f}")

        elif status in ("INVALID", "UNAVAILABLE"):
            if not _is_non_empty(row.get("status_reason")):
                unexplained.append(f"{pid}:{h} ({status})")
        elif status == "PENDING":
            if len(future_bars) >= n_sess:
                overdue_pending.append(f"{pid}:{h} (target exit {future_bars[n_sess - 1][0]} <= {eq_max_date})")

    # Compare matured rows with reference snapshot if provided
    modified_finalised: list[str] = []
    if reference_outcomes_df is not None and not reference_outcomes_df.empty:
        ref = reference_outcomes_df[reference_outcomes_df["status"] == "MATURED"].copy()
        if not ref.empty:
            merged = ref.merge(out, on=["prediction_id", "horizon"], suffixes=("_ref", "_cur"), how="left")
            for _, r in merged.iterrows():
                pid_h = f"{r['prediction_id']}:{r['horizon']}"
                for col in ("status", "entry_date", "entry_price", "exit_date", "exit_price", "gross_return_pct", "net_return_pct"):
                    v_ref = r.get(f"{col}_ref")
                    v_cur = r.get(f"{col}_cur")
                    if str(v_ref) != str(v_cur):
                        modified_finalised.append(f"{pid_h}:{col} ({v_ref} -> {v_cur})")
                        break

    checks.extend(
        [
            CheckResult(
                check_id="OUT_PREMATURE_MATURITY",
                dataset=dataset,
                description="MATURED outcomes must have at least H future NSE trading sessions in equities_daily.",
                passed=len(premature) == 0,
                severity="CRITICAL",
                affected_records=len(premature),
                examples=premature[:5],
            ),
            CheckResult(
                check_id="OUT_INVALID_PRICES",
                dataset=dataset,
                description="MATURED outcomes must have positive entry_price and exit_price.",
                passed=len(invalid_px) == 0,
                severity="CRITICAL",
                affected_records=len(invalid_px),
                examples=invalid_px[:5],
            ),
            CheckResult(
                check_id="OUT_ENTRY_EXIT_DATE_ALIGNMENT",
                dataset=dataset,
                description="entry_date must equal first trading day after obs_date and exit_date must equal entry + (H-1) sessions.",
                passed=len(date_align_err) == 0,
                severity="ERROR",
                affected_records=len(date_align_err),
                examples=date_align_err[:5],
            ),
            CheckResult(
                check_id="OUT_PRICE_MISMATCH_EQUITIES",
                dataset=dataset,
                description="MATURED entry_price/exit_price must match equities_daily open/close.",
                passed=len(px_mismatch) == 0,
                severity="ERROR",
                affected_records=len(px_mismatch),
                examples=px_mismatch[:5],
            ),
            CheckResult(
                check_id="OUT_MODIFIED_AFTER_FINALISATION",
                dataset=dataset,
                description="Previously MATURED outcomes must remain immutable relative to reference snapshot.",
                passed=len(modified_finalised) == 0,
                severity="CRITICAL",
                affected_records=len(modified_finalised),
                examples=modified_finalised[:5],
            ),
            CheckResult(
                check_id="OUT_BENCHMARK_MISSING",
                dataset=dataset,
                description="MATURED outcomes should have a populated benchmark_return_pct.",
                passed=len(bench_missing) == 0,
                severity="WARNING",
                affected_records=len(bench_missing),
                examples=bench_missing[:5],
            ),
            CheckResult(
                check_id="OUT_RETURN_MATH",
                dataset=dataset,
                description="gross_return_pct and net_return_pct must match price ratio and transaction cost arithmetic.",
                passed=len(math_err) == 0,
                severity="ERROR",
                affected_records=len(math_err),
                examples=math_err[:5],
            ),
            CheckResult(
                check_id="OUT_UNEXPLAINED_STATUS",
                dataset=dataset,
                description="INVALID or UNAVAILABLE outcomes must include a non-empty status_reason.",
                passed=len(unexplained) == 0,
                severity="WARNING",
                affected_records=len(unexplained),
                examples=unexplained[:5],
            ),
            CheckResult(
                check_id="OUT_OVERDUE_PENDING",
                dataset=dataset,
                description="PENDING outcomes whose target exit session is already in equities_daily should be evaluated.",
                passed=len(overdue_pending) == 0,
                severity="WARNING",
                affected_records=len(overdue_pending),
                examples=overdue_pending[:5],
            ),
        ]
    )

    return _summarize_dataset(dataset, len(out), checks)


# ---------------------------------------------------------------------------
# Dataset C: News (structured_events + quarantined_events + phase5 timelines)
# ---------------------------------------------------------------------------


def check_news_integrity(
    events_df: pd.DataFrame | None,
    quarantined_df: pd.DataFrame | None = None,
    phase5_payload: dict[str, Any] | None = None,
    predictions_df: pd.DataFrame | None = None,
    audit_now: datetime | None = None,
) -> DatasetAuditResult:
    dataset = "news_events"
    if events_df is None or events_df.empty:
        return _summarize_dataset(
            dataset,
            0,
            [
                CheckResult(
                    check_id="NEWS_EMPTY",
                    dataset=dataset,
                    description="structured_events dataset is empty or missing.",
                    passed=False,
                    severity="WARNING",
                )
            ],
        )

    df = events_df.copy()
    now_ist = (audit_now or datetime.now(IST_TZ)).astimezone(IST_TZ)
    checks: list[CheckResult] = []

    miss_pub: list[str] = []
    miss_coll: list[str] = []
    coll_before_pub: list[str] = []
    future_pub: list[str] = []
    miss_ticker: list[str] = []
    misattrib: list[str] = []
    bad_sent: list[str] = []
    bad_scores: list[str] = []

    for idx, row in df.iterrows():
        eid = str(row.get("event_id", idx))
        pub_ist = _parse_ist_timestamp(row.get("publication_timestamp"))
        coll_ist = _parse_ist_timestamp(row.get("collection_timestamp"))
        if pub_ist is None:
            miss_pub.append(eid)
        else:
            if pub_ist > now_ist + timedelta(minutes=5):
                future_pub.append(f"{eid} ({pub_ist.isoformat()})")
        if coll_ist is None:
            miss_coll.append(eid)
        elif pub_ist is not None and coll_ist < pub_ist - timedelta(minutes=5):
            coll_before_pub.append(f"{eid} (coll={coll_ist.isoformat()} < pub={pub_ist.isoformat()})")

        ticker = str(row.get("ticker", "")).upper().strip() if _is_non_empty(row.get("ticker")) else ""
        if not ticker:
            miss_ticker.append(eid)
        else:
            text = f"{row.get('event_description', '')} {row.get('facts_summary', '')}".lower()
            for pat in _FALSE_POSITIVE_ENTITY_PATTERNS.get(ticker, ()):
                if pat in text:
                    misattrib.append(f"{eid}:{ticker} (fp={pat})")
                    break
            else:
                if str(row.get("event_category", "")) == "company_news" and ticker in _COMPANY_ALIASES:
                    aliases = _COMPANY_ALIASES[ticker]
                    if not any(a in text for a in aliases):
                        for other_t, other_als in _COMPANY_ALIASES.items():
                            if other_t != ticker and any(oa in text for oa in other_als):
                                misattrib.append(f"{eid}:{ticker} (mentions {other_t})")
                                break

        sent = str(row.get("sentiment", "")).lower().strip()
        if sent not in ("positive", "negative", "neutral"):
            bad_sent.append(f"{eid}:{sent}")

        for scol in ("classification_confidence", "novelty_score", "source_credibility"):
            if scol in df.columns and pd.notna(row.get(scol)):
                val = float(row[scol])
                if val < 0.0 or val > 1.0:
                    bad_scores.append(f"{eid}:{scol}={val}")

    dup_eid = df[df["event_id"].duplicated(keep=False)] if "event_id" in df.columns else pd.DataFrame()
    unflagged_url: list[str] = []
    if {"ticker", "source_url"}.issubset(df.columns):
        is_dup_rep = df["is_duplicate_report"].fillna(False).astype(bool) if "is_duplicate_report" in df.columns else pd.Series(False, index=df.index)
        for (t, u), grp in df.groupby(["ticker", "source_url"]):
            if len(grp) > 1 and not bool(is_dup_rep.loc[grp.index].iloc[1:].all()):
                unflagged_url.append(f"{t}|{str(u)[:60]} (n={len(grp)})")

    # Quarantine leak check
    q_pairs: set[tuple[str, str]] = set()
    if quarantined_df is not None and not quarantined_df.empty:
        for _, qrow in quarantined_df.iterrows():
            q_pairs.add((str(qrow.get("symbol", "")).upper().strip(), str(qrow.get("url", "")).strip()))

    leak_pairs: list[str] = []
    if q_pairs and {"ticker", "source_url"}.issubset(df.columns):
        for _, row in df.iterrows():
            pair = (str(row.get("ticker", "")).upper().strip(), str(row.get("source_url", "")).strip())
            if pair in q_pairs:
                leak_pairs.append(f"{pair[0]}|{row.get('event_id', '')}")

    # Phase 5 timeline checks: quarantine leak + collection_timestamp > latest_prediction_at
    coll_after_pred_warn: list[str] = []
    coll_after_pred_err: list[str] = []
    timeline_leaks: list[str] = []
    if phase5_payload:
        pred_ts = _parse_ist_timestamp(phase5_payload.get("latest_prediction_at") or phase5_payload.get("generated_at"))
        stocks = phase5_payload.get("stocks") or {}
        stock_iter = stocks.values() if isinstance(stocks, dict) else stocks
        for st in stock_iter:
            if not isinstance(st, dict):
                continue
            for ev in st.get("events_timeline") or []:
                pair = (str(ev.get("ticker", "")).upper().strip(), str(ev.get("source_url", "")).strip())
                if pair in q_pairs:
                    timeline_leaks.append(f"{pair[0]}|{ev.get('event_id', '')}")
                if pred_ts is not None:
                    cts = _parse_ist_timestamp(ev.get("collection_timestamp"))
                    if cts is not None and cts > pred_ts:
                        delta_s = (cts - pred_ts).total_seconds()
                        msg = f"{ev.get('event_id', '')} (+{delta_s:.1f}s)"
                        if delta_s > 900:
                            coll_after_pred_err.append(msg)
                        else:
                            coll_after_pred_warn.append(msg)

    # Late-collected events with effective_trading_date <= latest snapshot observation_date
    late_effective: list[str] = []
    if predictions_df is not None and not predictions_df.empty and "effective_trading_date" in df.columns:
        latest_snap = predictions_df.sort_values("prediction_timestamp").iloc[-1]
        snap_ts = _parse_ist_timestamp(latest_snap.get("prediction_timestamp"))
        snap_obs = str(latest_snap.get("observation_date", ""))[:10]
        if snap_ts is not None and snap_obs:
            for _, row in df.iterrows():
                cts = _parse_ist_timestamp(row.get("collection_timestamp"))
                eff = str(row.get("effective_trading_date", ""))[:10]
                if cts is not None and cts > snap_ts and eff and eff <= snap_obs:
                    late_effective.append(f"{row.get('event_id', '')} (eff={eff} <= obs={snap_obs})")

    checks.extend(
        [
            CheckResult(
                check_id="NEWS_MISSING_PUB_TS",
                dataset=dataset,
                description="All structured news events must have a valid publication_timestamp.",
                passed=len(miss_pub) == 0,
                severity="CRITICAL",
                affected_records=len(miss_pub),
                examples=miss_pub[:5],
            ),
            CheckResult(
                check_id="NEWS_MISSING_COLL_TS",
                dataset=dataset,
                description="All structured news events should record a collection_timestamp.",
                passed=len(miss_coll) == 0,
                severity="WARNING",
                affected_records=len(miss_coll),
                examples=miss_coll[:5],
            ),
            CheckResult(
                check_id="NEWS_COLL_BEFORE_PUB",
                dataset=dataset,
                description="collection_timestamp must not precede publication_timestamp by more than 5 minutes.",
                passed=len(coll_before_pub) == 0,
                severity="ERROR",
                affected_records=len(coll_before_pub),
                examples=coll_before_pub[:5],
            ),
            CheckResult(
                check_id="NEWS_FUTURE_PUB",
                dataset=dataset,
                description="publication_timestamp must not be in the future relative to audit time.",
                passed=len(future_pub) == 0,
                severity="WARNING",
                affected_records=len(future_pub),
                examples=future_pub[:5],
            ),
            CheckResult(
                check_id="NEWS_MISSING_TICKER",
                dataset=dataset,
                description="Every news event must be attributed to a non-empty ticker.",
                passed=len(miss_ticker) == 0,
                severity="ERROR",
                affected_records=len(miss_ticker),
                examples=miss_ticker[:5],
            ),
            CheckResult(
                check_id="NEWS_ENTITY_MISATTRIBUTION",
                dataset=dataset,
                description="News events must not match false-positive entity patterns or other-company aliases.",
                passed=len(misattrib) == 0,
                severity="WARNING",
                affected_records=len(misattrib),
                examples=misattrib[:5],
            ),
            CheckResult(
                check_id="NEWS_DUPLICATE_EVENT_ID",
                dataset=dataset,
                description="event_id must be unique in structured_events.",
                passed=len(dup_eid) == 0,
                severity="CRITICAL",
                affected_records=len(dup_eid),
            ),
            CheckResult(
                check_id="NEWS_UNFLAGGED_DUP_URL",
                dataset=dataset,
                description="Repeated (ticker, source_url) rows should be flagged is_duplicate_report=True.",
                passed=len(unflagged_url) == 0,
                severity="WARNING",
                affected_records=len(unflagged_url),
                examples=unflagged_url[:5],
            ),
            CheckResult(
                check_id="NEWS_INVALID_SENTIMENT",
                dataset=dataset,
                description="sentiment must be one of {positive, negative, neutral}.",
                passed=len(bad_sent) == 0,
                severity="ERROR",
                affected_records=len(bad_sent),
                examples=bad_sent[:5],
            ),
            CheckResult(
                check_id="NEWS_SCORE_RANGE",
                dataset=dataset,
                description="classification_confidence, novelty_score, and source_credibility must lie in [0, 1].",
                passed=len(bad_scores) == 0,
                severity="ERROR",
                affected_records=len(bad_scores),
                examples=bad_scores[:5],
            ),
            CheckResult(
                check_id="NEWS_QUARANTINE_LEAK",
                dataset=dataset,
                description="Quarantined (ticker, url) items must not appear in structured_events or published timelines.",
                passed=(len(leak_pairs) + len(timeline_leaks)) == 0,
                severity="ERROR",
                affected_records=len(set(leak_pairs + timeline_leaks)),
                examples=list(dict.fromkeys(leak_pairs + timeline_leaks))[:5],
            ),
            CheckResult(
                check_id="NEWS_COLL_AFTER_PRED_WARN",
                dataset=dataset,
                description="Published timeline events with collection_timestamp > latest_prediction_at (<= 15 min; pipeline timestamp ordering).",
                passed=len(coll_after_pred_warn) == 0,
                severity="WARNING",
                affected_records=len(coll_after_pred_warn),
                examples=coll_after_pred_warn[:5],
            ),
            CheckResult(
                check_id="NEWS_COLL_AFTER_PRED_ERR",
                dataset=dataset,
                description="Published timeline events with collection_timestamp > latest_prediction_at by > 15 minutes.",
                passed=len(coll_after_pred_err) == 0,
                severity="ERROR",
                affected_records=len(coll_after_pred_err),
                examples=coll_after_pred_err[:5],
            ),
            CheckResult(
                check_id="NEWS_LATE_EFFECTIVE_DATE",
                dataset=dataset,
                description="Events collected after snapshot timestamp should not carry effective_trading_date <= snapshot observation_date.",
                passed=len(late_effective) == 0,
                severity="WARNING",
                affected_records=len(late_effective),
                examples=late_effective[:5],
            ),
        ]
    )

    return _summarize_dataset(dataset, len(df), checks)


# ---------------------------------------------------------------------------
# Dataset D: Benchmark (indices_daily + forward_outcomes + forward_validation)
# ---------------------------------------------------------------------------


def check_benchmark_integrity(
    indices_df: pd.DataFrame | None,
    outcomes_df: pd.DataFrame | None = None,
    forward_validation_payload: dict[str, Any] | None = None,
) -> DatasetAuditResult:
    dataset = "benchmark"
    checks: list[CheckResult] = []
    if indices_df is None or indices_df.empty:
        return _summarize_dataset(
            dataset,
            0,
            [
                CheckResult(
                    check_id="BENCH_NIFTY_MISSING",
                    dataset=dataset,
                    description="indices_daily dataset is empty or missing.",
                    passed=False,
                    severity="CRITICAL",
                )
            ],
        )

    idx = indices_df.copy()
    has_space = idx[idx["index_code"] == "NIFTY 50"]
    has_under = idx[idx["index_code"] == "NIFTY_50"]
    n50 = has_space if not has_space.empty else has_under

    checks.append(
        CheckResult(
            check_id="BENCH_NIFTY_MISSING",
            dataset=dataset,
            description="Nifty 50 series must exist in indices_daily.",
            passed=not n50.empty,
            severity="CRITICAL",
            affected_records=0 if not n50.empty else 1,
        )
    )
    checks.append(
        CheckResult(
            check_id="BENCH_KEY_MISMATCH",
            dataset=dataset,
            description="indices_daily uses index_code='NIFTY_50' while forward_validation.py queries 'NIFTY 50'.",
            passed=not has_space.empty,
            severity="ERROR",
            affected_records=len(has_under) if has_space.empty else 0,
            examples=[f"NIFTY_50 rows={len(has_under)}, NIFTY 50 rows={len(has_space)}"],
        )
    )

    bad_days: list[str] = []
    bad_px: list[str] = []
    n50_map: dict[str, tuple[float | None, float | None]] = {}
    if not n50.empty:
        n50 = n50.copy()
        n50["observation_date"] = n50["observation_date"].astype(str).str[:10]
        for _, r in n50.iterrows():
            dstr = r["observation_date"]
            try:
                if not is_nse_trading_day(date.fromisoformat(dstr)):
                    bad_days.append(dstr)
            except Exception:
                bad_days.append(dstr)
            op = r.get("open")
            cl = r.get("close")
            if pd.isna(cl) or float(cl) <= 0 or (pd.notna(op) and float(op) <= 0):
                bad_px.append(f"{dstr} (open={op}, close={cl})")
            n50_map[dstr] = (float(op) if pd.notna(op) else None, float(cl) if pd.notna(cl) else None)

    checks.append(
        CheckResult(
            check_id="BENCH_NON_TRADING_DAYS",
            dataset=dataset,
            description="Nifty 50 index bars must fall on valid NSE trading days.",
            passed=len(bad_days) == 0,
            severity="WARNING",
            affected_records=len(bad_days),
            examples=bad_days[:5],
        )
    )
    checks.append(
        CheckResult(
            check_id="BENCH_INVALID_PRICES",
            dataset=dataset,
            description="Nifty 50 open/close prices must be positive and non-null.",
            passed=len(bad_px) == 0,
            severity="CRITICAL",
            affected_records=len(bad_px),
            examples=bad_px[:5],
        )
    )

    # Window return verification & stock==benchmark check
    window_err: list[str] = []
    eq_stock: list[str] = []
    if outcomes_df is not None and not outcomes_df.empty and n50_map:
        matured = outcomes_df[outcomes_df["status"] == "MATURED"]
        for _, r in matured.iterrows():
            b_ret = r.get("benchmark_return_pct")
            if pd.isna(b_ret):
                continue
            exp_b = _compute_nifty_window_return(n50_map, str(r["entry_date"])[:10], str(r["exit_date"])[:10])
            if exp_b is not None and abs(float(b_ret) - exp_b) > 0.02:
                window_err.append(f"{r['prediction_id']}:{r['horizon']} got {b_ret} != {exp_b}")
            g_ret = r.get("gross_return_pct")
            if pd.notna(g_ret) and abs(float(b_ret) - float(g_ret)) < 1e-6:
                eq_stock.append(f"{r['prediction_id']}:{r['horizon']}")

    checks.append(
        CheckResult(
            check_id="BENCH_WINDOW_RETURN_MATCH",
            dataset=dataset,
            description="Populated benchmark_return_pct on MATURED outcomes must match Nifty 50 window return.",
            passed=len(window_err) == 0,
            severity="ERROR",
            affected_records=len(window_err),
            examples=window_err[:5],
        )
    )
    checks.append(
        CheckResult(
            check_id="BENCH_EQUALS_STOCK_RETURN",
            dataset=dataset,
            description="benchmark_return_pct must not trivially equal stock gross_return_pct.",
            passed=len(eq_stock) == 0,
            severity="WARNING",
            affected_records=len(eq_stock),
            examples=eq_stock[:5],
        )
    )

    if forward_validation_payload:
        benchmarks = (forward_validation_payload.get("paper_portfolio") or {}).get("benchmarks") or {}
        req_keys = {"naive_momentum_1d", "naive_momentum_5d", "naive_momentum_10d", "nifty_50_buy_hold", "cash_no_trade"}
        missing_keys = sorted(req_keys - set(benchmarks.keys()))
        has_alias = "naive_momentum" in benchmarks
        checks.append(
            CheckResult(
                check_id="BENCH_HORIZON_KEYS_AND_ALIAS",
                dataset=dataset,
                description="Horizon-specific benchmark keys must be present; unlabelled 'naive_momentum' alias should be avoided.",
                passed=(len(missing_keys) == 0 and not has_alias),
                severity="WARNING",
                affected_records=len(missing_keys) + (1 if has_alias else 0),
                examples=([f"missing={missing_keys}"] if missing_keys else []) + (["unlabelled alias: naive_momentum"] if has_alias else []),
            )
        )
        unexpl_b: list[str] = []
        for k, bdict in benchmarks.items():
            if isinstance(bdict, dict) and str(bdict.get("status", "")).upper() in ("UNAVAILABLE", "INSUFFICIENT_DATA"):
                if not any(_is_non_empty(bdict.get(f)) for f in ("reason", "explanation", "status_reason", "note")):
                    unexpl_b.append(f"{k} (status={bdict.get('status')})")
        checks.append(
            CheckResult(
                check_id="BENCH_UNAVAILABLE_NO_REASON",
                dataset=dataset,
                description="UNAVAILABLE or INSUFFICIENT_DATA benchmarks should include an explanatory reason.",
                passed=len(unexpl_b) == 0,
                severity="WARNING",
                affected_records=len(unexpl_b),
                examples=unexpl_b,
            )
        )

    return _summarize_dataset(dataset, len(n50), checks)


# ---------------------------------------------------------------------------
# Dataset E: Published Reports Consistency
# ---------------------------------------------------------------------------


def check_published_reports(
    fv_docs: dict[str, Any] | None,
    fv_data: dict[str, Any] | None,
    p5_docs: dict[str, Any] | None,
    p5_data: dict[str, Any] | None,
    latest_preds: dict[str, Any] | None = None,
    predictions_df: pd.DataFrame | None = None,
    equities_df: pd.DataFrame | None = None,
) -> DatasetAuditResult:
    dataset = "published_reports"
    checks: list[CheckResult] = []
    inspected = sum(1 for x in (fv_docs, p5_docs, latest_preds) if x is not None)

    # 1. docs/data vs data/predictions sync
    sync_diffs: list[str] = []
    if fv_docs is not None and fv_data is not None and fv_docs != fv_data:
        sync_diffs.append("forward_validation.json differs between docs/data and data/predictions")
    if p5_docs is not None and p5_data is not None and p5_docs != p5_data:
        sync_diffs.append("phase5_intelligence.json differs between docs/data and data/predictions")
    checks.append(
        CheckResult(
            check_id="REP_DOCS_DATA_SYNC",
            dataset=dataset,
            description="Published docs/data JSONs must stay in sync with data/predictions JSONs.",
            passed=len(sync_diffs) == 0,
            severity="ERROR",
            affected_records=len(sync_diffs),
            examples=sync_diffs,
        )
    )

    fv = fv_docs or fv_data
    p5 = p5_docs or p5_data

    # 2. Model version mismatch
    if fv and p5:
        mv_fv = str(fv.get("model_version", ""))
        mv_p5 = str(p5.get("model_version", ""))
        checks.append(
            CheckResult(
                check_id="REP_MODEL_VERSION_MISMATCH",
                dataset=dataset,
                description="forward_validation.json and phase5_intelligence.json report different subsystem model_version labels.",
                passed=(mv_fv == mv_p5),
                severity="WARNING",
                affected_records=0 if mv_fv == mv_p5 else 2,
                examples=[f"forward_validation={mv_fv}", f"phase5_intelligence={mv_p5}"],
            )
        )

        # 3. Population mismatch (60 vs 10)
        mon = p5.get("monitoring") or {}
        n_mon = mon.get("evaluated_outcomes_count")
        n_fv = fv.get("matured_outcomes_total")
        checks.append(
            CheckResult(
                check_id="REP_OUTCOME_POPULATION_MISMATCH",
                dataset=dataset,
                description="phase5_intelligence.monitoring.evaluated_outcomes_count and forward_validation.matured_outcomes_total measure different populations.",
                passed=(n_mon == n_fv),
                severity="WARNING",
                affected_records=0 if n_mon == n_fv else 2,
                examples=[f"phase5_monitoring={n_mon}", f"forward_validation={n_fv}"],
            )
        )

        # 4. Silent tail(50) truncation in monitoring
        truncated = isinstance(n_mon, int) and n_mon > 50
        checks.append(
            CheckResult(
                check_id="REP_MONITORING_TRUNCATION",
                dataset=dataset,
                description="monitoring.py computes metrics on eval_df.tail(50) while reporting evaluated_outcomes_count > 50.",
                passed=not truncated,
                severity="WARNING",
                affected_records=(n_mon - 50) if truncated else 0,
                examples=[f"evaluated_outcomes_count={n_mon}, metric_window=50"] if truncated else [],
            )
        )

    # 5. phase5 stock observation_date vs latest forward_predictions observation_date
    if p5 and predictions_df is not None and not predictions_df.empty:
        max_pred_obs = str(predictions_df["observation_date"].astype(str).max())[:10]
        stocks = p5.get("stocks") or {}
        stock_iter = stocks.values() if isinstance(stocks, dict) else stocks
        obs_mismatches = [
            f"{st.get('symbol')}:{st.get('observation_date')} != {max_pred_obs}"
            for st in stock_iter
            if isinstance(st, dict) and str(st.get("observation_date", ""))[:10] != max_pred_obs
        ]
        checks.append(
            CheckResult(
                check_id="REP_OBS_DATE_MISMATCH",
                dataset=dataset,
                description="phase5_intelligence stock observation_date should match the underlying prediction snapshot observation_date.",
                passed=len(obs_mismatches) == 0,
                severity="WARNING",
                affected_records=len(obs_mismatches),
                examples=obs_mismatches[:5],
            )
        )

    # 6. latest_predictions staleness (>12h before phase5 generated_at)
    if p5 and latest_preds:
        p5_ts = _parse_ist_timestamp(p5.get("generated_at"))
        lp_ts = _parse_ist_timestamp(latest_preds.get("generated_at"))
        stale = False
        delta_h = 0.0
        if p5_ts and lp_ts:
            delta_h = (p5_ts - lp_ts).total_seconds() / 3600.0
            stale = delta_h > 12.0
        checks.append(
            CheckResult(
                check_id="REP_LATEST_PRED_STALE",
                dataset=dataset,
                description="latest_predictions.json was generated >12h before phase5_intelligence.json (stale quant inputs reused in pre-market run).",
                passed=not stale,
                severity="WARNING",
                affected_records=1 if stale else 0,
                examples=[f"age_hours={delta_h:.2f}h (latest={lp_ts.isoformat() if lp_ts else None})"] if stale else [],
            )
        )

    # 7. Session label vs equities max date heuristic
    if predictions_df is not None and not predictions_df.empty and equities_df is not None and not equities_df.empty:
        eq_max = str(equities_df["observation_date"].astype(str).max())[:10]
        sess_anomalies: list[str] = []
        is_dup = predictions_df["is_duplicate_run"].fillna(False).astype(bool) if "is_duplicate_run" in predictions_df.columns else pd.Series(False, index=predictions_df.index)
        for _, r in predictions_df.loc[~is_dup].iterrows():
            ts_ist = _parse_ist_timestamp(r.get("prediction_timestamp"))
            sess = str(r.get("nse_session", ""))
            if ts_ist and sess == "POST_MARKET" and ts_ist.date().isoformat() > eq_max:
                sess_anomalies.append(f"{r['prediction_id']} (POST_MARKET at {ts_ist.isoformat()} while equities_max={eq_max})")
        checks.append(
            CheckResult(
                check_id="REP_SESSION_WORKFLOW_CONSISTENCY",
                dataset=dataset,
                description="POST_MARKET session label when equities_daily has not updated for that date indicates delayed pre-market cron mislabelled by wall-clock time.",
                passed=len(sess_anomalies) == 0,
                severity="WARNING",
                affected_records=len(sess_anomalies),
                examples=sess_anomalies[:5],
            )
        )

    return _summarize_dataset(dataset, inspected, checks)


# ---------------------------------------------------------------------------
# Full Orchestrator
# ---------------------------------------------------------------------------


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_data_integrity_audit(
    store: DuckDBStore | None = None,
    docs_data_dir: Path | None = None,
    reference_outcomes_df: pd.DataFrame | None = None,
    base_commit_sha: str = "8f328b8cdb8f7a1e508c1c47a648849945873013",
) -> DataIntegrityReport:
    """Run the complete deterministic, read-only data integrity audit."""
    store = store or DuckDBStore()
    docs_dir = docs_data_dir or Path("docs/data")
    pred_dir = store.predictions_dir

    fwd_preds = _read_parquet_safe(store.forward_predictions_parquet)
    fwd_outs = _read_parquet_safe(store.forward_outcomes_parquet)
    pred_log = _read_parquet_safe(store.predictions_parquet)
    equities = _read_parquet_safe(store.equities_parquet)
    indices = _read_parquet_safe(store.indices_parquet)
    events = _read_parquet_safe(store.events_parquet)
    quarantined = _read_parquet_safe(store.quarantined_events_parquet)

    fv_docs = _read_json_safe(docs_dir / "forward_validation.json")
    fv_data = _read_json_safe(pred_dir / "forward_validation.json")
    p5_docs = _read_json_safe(docs_dir / "phase5_intelligence.json")
    p5_data = _read_json_safe(pred_dir / "phase5_intelligence.json")
    latest_preds = _read_json_safe(pred_dir / "latest_predictions.json")

    ds_results = {
        "forward_predictions": check_forward_predictions(fwd_preds, fwd_outs),
        "prediction_log": check_legacy_prediction_log(pred_log),
        "forward_outcomes": check_forward_outcomes(fwd_outs, fwd_preds, equities, reference_outcomes_df),
        "news_events": check_news_integrity(events, quarantined, p5_docs or p5_data, fwd_preds),
        "benchmark": check_benchmark_integrity(indices, fwd_outs, fv_docs or fv_data),
        "published_reports": check_published_reports(fv_docs, fv_data, p5_docs, p5_data, latest_preds, fwd_preds, equities),
    }

    records, h_sums, p_metrics = classify_forward_predictions(fwd_preds, fwd_outs)
    legacy_sum = classify_legacy_prediction_log(pred_log)
    classification = ClassificationReport(
        horizon_summaries=h_sums,
        prospective_metrics=p_metrics,
        legacy_prediction_log=legacy_sum,
        records=records,
    )

    statuses = {d.status for d in ds_results.values()}
    overall: OverallStatus = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    return DataIntegrityReport(
        audit_timestamp=datetime.now(IST_TZ).isoformat(),
        base_commit_sha=base_commit_sha,
        datasets=ds_results,
        classification=classification,
        overall_status=overall,
    )
