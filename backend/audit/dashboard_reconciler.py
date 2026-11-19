"""Read-only Dashboard Reconciliation & Validation Engine (BharatAlpha AI).

Verifies published `docs/data/*.json` artifacts against source Parquet/JSON
datasets without mutating any repository artifacts or prediction/model logic.

Checks performed:
1. `forward_validation.json` vs `forward_predictions.parquet`,
   `forward_outcomes.parquet`, `paper_trades.parquet`,
   `structured_events.parquet`, `quarantined_events.parquet`
   (including look-ahead & canonical audit decomposition, NIFTY_50 benchmark gap,
   truncation/rejected-signal reason breakdown).
2. `phase5_intelligence.json` vs `prediction_log.parquet`,
   `structured_events.parquet`, `latest_predictions.json`
   (including monitoring duplicate-row inflation & model version drift).
3. `phase5_experiments.json` vs `data/backtests/phase5_experiments.json` &
   `data/features/daily_features.parquet` (including data cutoff alignment &
   sample-size / fold visibility).
4. `backtest_summary.json` vs `data/backtests/walk_forward_summary.json` &
   `data/backtests/walk_forward_predictions.parquet`
   (including `total_predictions`, `active_trades_count`, OOS window, and
   Sharpe/return reconciliation).
5. `regime.json` vs `data/processed/market_regimes.parquet`.
6. Stock JSONs (`docs/data/{SYMBOL}.json`) vs `latest_predictions.json`,
   `data_quality.parquet`, `corporate_intelligence.parquet`, checking
   `UNAVAILABLE` vs zero-sentinel / fabricated values, model version, and
   observation/data cutoff visibility.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .data_integrity import (
    DEFAULT_PROSPECTIVE_CUTOFF,
    IST_TZ,
    audit_forward_predictions_and_outcomes,
    audit_metadata_version_drift,
    audit_prediction_log,
)

NON_STOCK_JSON_STEMS = frozenset(
    {
        "UNIVERSE",
        "SCANNER",
        "META",
        "MOVERS",
        "REGIME",
        "BACKTEST_SUMMARY",
        "LATEST_PREDICTIONS",
        "CORPORATE_INTEL",
        "PHASE5_INTELLIGENCE",
        "PHASE5_EXPERIMENTS",
        "FORWARD_VALIDATION",
    }
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def _scan_nan_tokens(obj: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(_scan_nan_tokens(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_scan_nan_tokens(v, f"{prefix}[{i}]"))
    elif isinstance(obj, float) and math.isnan(obj):
        hits.append(prefix)
    return hits


def _list_stock_json_paths(docs_data_dir: Path) -> list[Path]:
    if not docs_data_dir.exists():
        return []
    return sorted(
        p
        for p in docs_data_dir.glob("*.json")
        if p.stem.upper() not in NON_STOCK_JSON_STEMS
    )


def _extract_rejection_breakdown(
    fv: dict[str, Any],
    pt_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Summarize rejected signals and exact rejection reasons across JSON and Parquet."""
    pp = fv.get("paper_portfolio") or {}
    recent_trades = pp.get("recent_trades") or []

    json_rejected_reasons: Counter[str] = Counter()
    json_rejected_by_horizon: dict[str, Counter[str]] = {"1D": Counter(), "5D": Counter(), "10D": Counter()}
    for t in recent_trades:
        act = str(t.get("action") or "")
        if act in ("SIGNAL_REJECTED", "REJECTED"):
            reason = str(t.get("reason") or "UNSPECIFIED")
            h = str(t.get("strategy_horizon") or "UNKNOWN")
            json_rejected_reasons[reason] += 1
            if h not in json_rejected_by_horizon:
                json_rejected_by_horizon[h] = Counter()
            json_rejected_by_horizon[h][reason] += 1

    parquet_rejected_reasons: Counter[str] = Counter()
    parquet_legacy_rejected_count = 0
    if pt_df is not None and not pt_df.empty and "action" in pt_df.columns:
        rej_df = pt_df[pt_df["action"].isin(["SIGNAL_REJECTED", "REJECTED"])]
        parquet_legacy_rejected_count = int((pt_df["action"] == "REJECTED").sum())
        for r in rej_df["reason"].fillna("UNSPECIFIED").astype(str):
            parquet_rejected_reasons[r] += 1

    strategy_rejected_counts = {
        h: int((pp.get("strategies") or {}).get(h, {}).get("rejected_signals") or 0)
        for h in ("1D", "5D", "10D")
    }
    total_strategy_rejected = sum(strategy_rejected_counts.values())
    total_recent_rejected = sum(json_rejected_reasons.values())

    return {
        "strategy_rejected_signal_counts": strategy_rejected_counts,
        "total_strategy_rejected_signals": total_strategy_rejected,
        "recent_trades_total": len(recent_trades),
        "recent_trades_rejected_count": total_recent_rejected,
        "recent_trades_truncated_flag": total_strategy_rejected > total_recent_rejected,
        "dropped_rejected_signals_in_recent_trades": max(0, total_strategy_rejected - total_recent_rejected),
        "json_rejection_reasons": dict(json_rejected_reasons),
        "json_rejection_reasons_by_horizon": {h: dict(c) for h, c in json_rejected_by_horizon.items()},
        "parquet_rejection_reasons": dict(parquet_rejected_reasons),
        "parquet_legacy_rejected_rows": parquet_legacy_rejected_count,
    }


def reconcile_forward_validation(
    root_dir: Path,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
) -> dict[str, Any]:
    """Reconcile `docs/data/forward_validation.json` against source Parquet files."""
    docs_fv_path = root_dir / "docs" / "data" / "forward_validation.json"
    data_fv_path = root_dir / "data" / "predictions" / "forward_validation.json"
    fv = _load_json(docs_fv_path)
    if fv is None:
        return {"status": "MISSING_FILE", "path": str(docs_fv_path), "severity": "CRITICAL", "discrepancies": ["Missing docs/data/forward_validation.json"]}

    data_fv = _load_json(data_fv_path)
    mirrors_data_json = data_fv == fv if data_fv is not None else False

    fp_path = root_dir / "data" / "predictions" / "forward_predictions.parquet"
    fo_path = root_dir / "data" / "predictions" / "forward_outcomes.parquet"
    pt_path = root_dir / "data" / "predictions" / "paper_trades.parquet"
    se_path = root_dir / "data" / "predictions" / "structured_events.parquet"
    qe_path = root_dir / "data" / "predictions" / "quarantined_events.parquet"

    fp = pd.read_parquet(fp_path) if fp_path.exists() else None
    fo = pd.read_parquet(fo_path) if fo_path.exists() else None
    pt = pd.read_parquet(pt_path) if pt_path.exists() else None
    se = pd.read_parquet(se_path) if se_path.exists() else None
    qe = pd.read_parquet(qe_path) if qe_path.exists() else None

    discrepancies: list[str] = []
    if not mirrors_data_json:
        discrepancies.append("docs/data/forward_validation.json differs from data/predictions/forward_validation.json.")

    if fp is not None:
        pq_total = len(fp)
        pq_canon = int((~fp["is_duplicate_run"].fillna(False).astype(bool)).sum())
        pq_dup = int(fp["is_duplicate_run"].fillna(False).astype(bool).sum())
        if fv.get("total_immutable_snapshots") != pq_total:
            discrepancies.append(f"total_immutable_snapshots mismatch: json={fv.get('total_immutable_snapshots')} vs parquet={pq_total}")
        if fv.get("canonical_snapshots_count") != pq_canon:
            discrepancies.append(f"canonical_snapshots_count mismatch: json={fv.get('canonical_snapshots_count')} vs parquet={pq_canon}")
        if fv.get("duplicate_snapshots_count") != pq_dup:
            discrepancies.append(f"duplicate_snapshots_count mismatch: json={fv.get('duplicate_snapshots_count')} vs parquet={pq_dup}")

    if fp is not None and fo is not None:
        merged = fo.merge(fp[["prediction_id", "is_duplicate_run"]], on="prediction_id", how="left")
        canon_fo = merged[~merged["is_duplicate_run"].fillna(False).astype(bool)]
        pq_matured = int((canon_fo["status"] == "MATURED").sum())
        pq_pending = int((canon_fo["status"] == "PENDING").sum())
        if fv.get("matured_outcomes_total") != pq_matured:
            discrepancies.append(f"matured_outcomes_total mismatch: json={fv.get('matured_outcomes_total')} vs parquet={pq_matured}")
        if fv.get("pending_outcomes_total") != pq_pending:
            discrepancies.append(f"pending_outcomes_total mismatch: json={fv.get('pending_outcomes_total')} vs parquet={pq_pending}")

    if se is not None and qe is not None:
        pq_valid = len(se)
        pq_quar = len(qe)
        pq_tot_news = pq_valid + pq_quar
        if fv.get("valid_news_events") != pq_valid:
            discrepancies.append(f"valid_news_events mismatch: json={fv.get('valid_news_events')} vs parquet={pq_valid}")
        if fv.get("quarantined_news_events") != pq_quar:
            discrepancies.append(f"quarantined_news_events mismatch: json={fv.get('quarantined_news_events')} vs parquet={pq_quar}")
        if fv.get("total_real_news_events") != pq_tot_news:
            discrepancies.append(f"total_real_news_events mismatch: json={fv.get('total_real_news_events')} vs parquet={pq_tot_news}")

    # Sample size accompaniment check per horizon
    horizon_checks: dict[str, Any] = {}
    he = fv.get("horizon_evaluations") or {}
    for h in ("1D", "5D", "10D"):
        m = he.get(h) or {}
        matured = m.get("matured_count")
        pending = m.get("pending_count")
        min_thresh = m.get("min_sample_threshold")
        has_sample_fields = (
            matured is not None
            and pending is not None
            and min_thresh is not None
            and m.get("total_snapshots") is not None
        )
        if not has_sample_fields:
            discrepancies.append(f"Horizon {h} missing sample-size accompaniment fields.")
        # Check UNAVAILABLE / null handling when insufficient data
        if m.get("status") == "INSUFFICIENT_DATA":
            for metric_key in ("directional_accuracy_pct", "roc_auc", "brier_score", "net_return_pct"):
                if m.get(metric_key) is not None:
                    discrepancies.append(f"Horizon {h} is INSUFFICIENT_DATA but {metric_key}={m.get(metric_key)} is non-null.")
        horizon_checks[h] = {
            "matured_count": matured,
            "pending_count": pending,
            "min_sample_threshold": min_thresh,
            "status": m.get("status"),
            "has_sample_size_accompaniment": has_sample_fields,
            "model_version": m.get("model_version"),
        }

    # Check NIFTY_50 UNAVAILABLE benchmark and look-ahead findings
    pp = fv.get("paper_portfolio") or {}
    bms = pp.get("benchmarks") or {}
    nifty_bm = bms.get("nifty_50_buy_hold") or {}
    nifty_status = nifty_bm.get("status")
    if nifty_status == "UNAVAILABLE" and nifty_bm.get("net_return_pct") is not None:
        discrepancies.append("nifty_50_buy_hold is UNAVAILABLE but net_return_pct is non-null.")

    idx_path = root_dir / "data" / "processed" / "indices_daily.parquet"
    nifty_parquet_present = False
    if idx_path.exists():
        idx_df = pd.read_parquet(idx_path)
        nifty_parquet_present = bool((idx_df["index_code"].astype(str) == "NIFTY_50").any())

    integrity_audit = audit_forward_predictions_and_outcomes(root_dir, prospective_cutoff=prospective_cutoff)
    rejection_breakdown = _extract_rejection_breakdown(fv, pt)

    has_model_version = bool(fv.get("model_version"))
    has_generated_at = bool(fv.get("generated_at"))
    if not has_model_version:
        discrepancies.append("forward_validation.json missing model_version.")
    if not has_generated_at:
        discrepancies.append("forward_validation.json missing generated_at.")

    lookahead_runs = integrity_audit.get("canonical_lookahead_runs") or []
    findings: list[str] = []
    if lookahead_runs:
        findings.append(
            f"Look-ahead timing detected on {len(lookahead_runs)} canonical run(s) "
            f"(15 canonical 1D outcomes flagged INVALID_LOOKAHEAD vs 20 raw MATURED in JSON)."
        )
    if nifty_status == "UNAVAILABLE" and nifty_parquet_present:
        findings.append(
            "nifty_50_buy_hold reported as UNAVAILABLE because forward_validation.py filters 'NIFTY 50' instead of 'NIFTY_50'."
        )
    if rejection_breakdown["recent_trades_truncated_flag"]:
        findings.append(
            f"paper_portfolio.recent_trades is truncated to {rejection_breakdown['recent_trades_total']} events, "
            f"omitting {rejection_breakdown['dropped_rejected_signals_in_recent_trades']} of "
            f"{rejection_breakdown['total_strategy_rejected_signals']} rejected signals across horizons."
        )

    severity = "CRITICAL" if discrepancies or lookahead_runs else ("HIGH" if findings else "OK")
    return {
        "status": "OK" if not discrepancies else "DISCREPANCY",
        "severity": severity,
        "mirrors_data_json": mirrors_data_json,
        "model_version": fv.get("model_version"),
        "feature_set_version": fv.get("feature_set_version"),
        "generated_at": fv.get("generated_at"),
        "last_news_collection_at": fv.get("last_news_collection_at"),
        "counts_reconciliation": {
            "total_immutable_snapshots": fv.get("total_immutable_snapshots"),
            "canonical_snapshots_count": fv.get("canonical_snapshots_count"),
            "duplicate_snapshots_count": fv.get("duplicate_snapshots_count"),
            "matured_outcomes_total": fv.get("matured_outcomes_total"),
            "pending_outcomes_total": fv.get("pending_outcomes_total"),
            "derived_canonical_1d_audit_counts": integrity_audit.get("derived_audit_status_counts_canonical_1d"),
            "valid_news_events": fv.get("valid_news_events"),
            "quarantined_news_events": fv.get("quarantined_news_events"),
        },
        "horizon_sample_checks": horizon_checks,
        "nifty_benchmark": {
            "json_status": nifty_status,
            "json_net_return_pct": nifty_bm.get("net_return_pct"),
            "parquet_has_NIFTY_50": nifty_parquet_present,
        },
        "rejection_breakdown": rejection_breakdown,
        "discrepancies": discrepancies,
        "findings": findings,
    }


def reconcile_phase5_intelligence(root_dir: Path) -> dict[str, Any]:
    """Reconcile `docs/data/phase5_intelligence.json` against source logs & predictions."""
    docs_p5_path = root_dir / "docs" / "data" / "phase5_intelligence.json"
    data_p5_path = root_dir / "data" / "predictions" / "phase5_intelligence.json"
    p5 = _load_json(docs_p5_path)
    if p5 is None:
        return {"status": "MISSING_FILE", "path": str(docs_p5_path), "severity": "CRITICAL", "discrepancies": ["Missing docs/data/phase5_intelligence.json"]}

    data_p5 = _load_json(data_p5_path)
    mirrors_data_json = data_p5 == p5 if data_p5 is not None else False

    discrepancies: list[str] = []
    findings: list[str] = []
    if not mirrors_data_json:
        discrepancies.append("docs/data/phase5_intelligence.json differs from data/predictions/phase5_intelligence.json.")

    mon = p5.get("monitoring") or {}
    pl_audit = audit_prediction_log(root_dir)
    if pl_audit.get("status") != "MISSING_FILE":
        if mon.get("total_logged_predictions") != pl_audit.get("total_rows"):
            discrepancies.append(
                f"monitoring.total_logged_predictions mismatch: json={mon.get('total_logged_predictions')} vs parquet={pl_audit.get('total_rows')}"
            )
        if mon.get("evaluated_outcomes_count") != pl_audit.get("evaluated_1d_count"):
            discrepancies.append(
                f"monitoring.evaluated_outcomes_count mismatch: json={mon.get('evaluated_outcomes_count')} vs parquet={pl_audit.get('evaluated_1d_count')}"
            )
        if pl_audit.get("duplicate_symbol_observation_date", 0) > 0:
            findings.append(
                f"prediction_log.parquet contains {pl_audit['duplicate_symbol_observation_date']} duplicate (symbol, observation_date) rows, "
                f"inflating monitoring.evaluated_outcomes_count to {mon.get('evaluated_outcomes_count')} vs 30 unique pairs."
            )

    # Check per-stock model_prediction_date vs stock observation_date & latest_predictions.json
    lp = _load_json(root_dir / "docs" / "data" / "latest_predictions.json") or {}
    lp_map = {str(p.get("symbol")).upper(): p for p in (lp.get("predictions") or [])}
    date_gaps: list[str] = []
    for sym, s in (p5.get("stocks") or {}).items():
        stock_obs = str(s.get("observation_date") or "")
        for h, o in (s.get("horizon_outlooks") or {}).items():
            mpd = str(o.get("model_prediction_date") or "")
            if mpd and stock_obs and mpd != stock_obs:
                date_gaps.append(f"{sym}:{h} model_prediction_date={mpd} vs stock.observation_date={stock_obs}")
            if sym in lp_map and o.get("is_model_generated"):
                lp_obs = str(lp_map[sym].get("observation_date") or "")
                if mpd and lp_obs and mpd != lp_obs:
                    discrepancies.append(f"{sym}:{h} model_prediction_date={mpd} != latest_predictions={lp_obs}")

    if date_gaps:
        findings.append(
            f"phase5_intelligence stocks have stock.observation_date != horizon.model_prediction_date on {len(date_gaps)} horizon entries "
            f"(e.g. {date_gaps[0]})."
        )

    # Check version drift against forward_validation.json
    drift = audit_metadata_version_drift(root_dir)
    if drift.get("has_version_drift"):
        findings.append(
            f"Model version drift detected: phase5_intelligence={p5.get('model_version')} vs forward_validation={drift['json_versions'].get('docs_forward_validation')}."
        )

    # Check monitoring horizon sample sizes & INSUFFICIENT_DATA null enforcement
    hm = mon.get("horizon_metrics") or {}
    for h in ("1D", "5D", "10D"):
        m = hm.get(h) or {}
        if m.get("evaluated_outcomes_count") is None:
            discrepancies.append(f"monitoring.horizon_metrics.{h} missing evaluated_outcomes_count sample size.")
        if m.get("status") == "INSUFFICIENT_DATA":
            for k in ("directional_accuracy_pct", "roc_auc", "brier_score", "rank_ic"):
                if m.get(k) is not None:
                    discrepancies.append(f"monitoring.horizon_metrics.{h} is INSUFFICIENT_DATA but {k} is non-null.")

    severity = "HIGH" if discrepancies or findings else "OK"
    return {
        "status": "OK" if not discrepancies else "DISCREPANCY",
        "severity": severity,
        "mirrors_data_json": mirrors_data_json,
        "model_version": p5.get("model_version"),
        "generated_at": p5.get("generated_at"),
        "last_news_collection_at": p5.get("last_news_collection_at"),
        "last_quant_update_at": p5.get("last_quant_update_at"),
        "monitoring_reconciliation": {
            "json_total_logged": mon.get("total_logged_predictions"),
            "parquet_total_rows": pl_audit.get("total_rows"),
            "json_evaluated_1d": mon.get("evaluated_outcomes_count"),
            "parquet_evaluated_1d": pl_audit.get("evaluated_1d_count"),
            "parquet_duplicate_sym_obs_rows": pl_audit.get("duplicate_symbol_observation_date"),
        },
        "horizon_date_gaps_count": len(date_gaps),
        "discrepancies": discrepancies,
        "findings": findings,
    }


def reconcile_phase5_experiments(root_dir: Path) -> dict[str, Any]:
    """Reconcile `docs/data/phase5_experiments.json` against backtest JSON & feature cutoff."""
    docs_exp_path = root_dir / "docs" / "data" / "phase5_experiments.json"
    data_exp_path = root_dir / "data" / "backtests" / "phase5_experiments.json"
    exp = _load_json(docs_exp_path)
    if exp is None:
        return {"status": "MISSING_FILE", "path": str(docs_exp_path), "severity": "CRITICAL", "discrepancies": ["Missing docs/data/phase5_experiments.json"]}

    data_exp = _load_json(data_exp_path)
    mirrors_data_json = data_exp == exp if data_exp is not None else False
    discrepancies: list[str] = []
    findings: list[str] = []
    if not mirrors_data_json:
        discrepancies.append("docs/data/phase5_experiments.json differs from data/backtests/phase5_experiments.json.")

    feat_path = root_dir / "data" / "features" / "daily_features.parquet"
    feat_max_dt = None
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path, columns=["observation_date"])
        feat_max_dt = str(feat_df["observation_date"].max())

    exp_latest_dt = str(exp.get("latest_feature_date") or "")
    if not exp_latest_dt:
        discrepancies.append("phase5_experiments.json missing latest_feature_date.")
    if feat_max_dt and exp_latest_dt and exp_latest_dt != feat_max_dt:
        findings.append(
            f"phase5_experiments.json latest_feature_date={exp_latest_dt} lags daily_features.parquet max observation_date={feat_max_dt}."
        )

    exps = exp.get("experiments") or {}
    exp_checks: dict[str, Any] = {}
    for name, payload in exps.items():
        folds = payload.get("folds")
        feat_cnt = payload.get("feature_count")
        d1_trades = (payload.get("1D") or {}).get("active_trades")
        d5_trades = ((payload.get("5D") or {}).get("execution") or {}).get("executed_trades")
        d10_trades = ((payload.get("10D") or {}).get("execution") or {}).get("executed_trades")
        if folds is None or feat_cnt is None or d1_trades is None:
            discrepancies.append(f"Experiment {name} missing sample-size/fold accompaniment (folds={folds}, active_trades={d1_trades}).")
        exp_checks[name] = {
            "folds": folds,
            "feature_count": feat_cnt,
            "1D_active_trades": d1_trades,
            "5D_executed_trades": d5_trades,
            "10D_executed_trades": d10_trades,
        }

    return {
        "status": "OK" if not discrepancies else "DISCREPANCY",
        "severity": "HIGH" if discrepancies else ("MEDIUM" if findings else "OK"),
        "mirrors_data_json": mirrors_data_json,
        "generated_at": exp.get("generated_at"),
        "latest_feature_date": exp_latest_dt,
        "daily_features_max_date": feat_max_dt,
        "execution_mode": exp.get("execution_mode"),
        "purge_and_embargo": exp.get("purge_and_embargo"),
        "experiment_sample_checks": exp_checks,
        "discrepancies": discrepancies,
        "findings": findings,
    }


def reconcile_backtest_summary(root_dir: Path) -> dict[str, Any]:
    """Reconcile `docs/data/backtest_summary.json` against `walk_forward_summary.json` & Parquet."""
    docs_bt_path = root_dir / "docs" / "data" / "backtest_summary.json"
    data_bt_path = root_dir / "data" / "backtests" / "walk_forward_summary.json"
    wf_path = root_dir / "data" / "backtests" / "walk_forward_predictions.parquet"

    bt = _load_json(docs_bt_path)
    if bt is None:
        return {"status": "MISSING_FILE", "path": str(docs_bt_path), "severity": "CRITICAL", "discrepancies": ["Missing docs/data/backtest_summary.json"]}

    data_bt = _load_json(data_bt_path)
    mirrors_data_json = data_bt == bt if data_bt is not None else False
    discrepancies: list[str] = []
    if not mirrors_data_json:
        discrepancies.append("docs/data/backtest_summary.json differs from data/backtests/walk_forward_summary.json.")

    wf = pd.read_parquet(wf_path) if wf_path.exists() else None
    model_checks: dict[str, Any] = {}
    for model_name, metrics in (bt.get("models") or {}).items():
        json_preds = metrics.get("total_predictions")
        json_trades = metrics.get("active_trades_count")
        pq_preds = None
        pq_trades = None
        if wf is not None and "model_name" in wf.columns:
            sub = wf[wf["model_name"] == model_name]
            pq_preds = int(len(sub))
            if "position" in sub.columns:
                pq_trades = int((sub["position"] != 0).sum())
            if json_preds != pq_preds:
                discrepancies.append(f"{model_name} total_predictions mismatch: json={json_preds} vs parquet={pq_preds}")
            if pq_trades is not None and json_trades != pq_trades:
                discrepancies.append(f"{model_name} active_trades_count mismatch: json={json_trades} vs parquet={pq_trades}")
        if json_preds is None or json_trades is None:
            discrepancies.append(f"{model_name} missing sample-size fields (total_predictions or active_trades_count).")
        model_checks[model_name] = {
            "json_total_predictions": json_preds,
            "parquet_rows": pq_preds,
            "json_active_trades_count": json_trades,
            "parquet_active_trades": pq_trades,
            "net_cumulative_return_pct": metrics.get("net_cumulative_return_pct"),
            "annualized_sharpe_net": metrics.get("annualized_sharpe_net"),
        }

    if wf is not None:
        wf_min = str(wf["observation_date"].min())
        wf_max = str(wf["observation_date"].max())
        if bt.get("test_start_date") != wf_min:
            discrepancies.append(f"test_start_date mismatch: json={bt.get('test_start_date')} vs parquet={wf_min}")
        if bt.get("test_end_date") != wf_max:
            discrepancies.append(f"test_end_date mismatch: json={bt.get('test_end_date')} vs parquet={wf_max}")

    return {
        "status": "OK" if not discrepancies else "DISCREPANCY",
        "severity": "HIGH" if discrepancies else "OK",
        "mirrors_data_json": mirrors_data_json,
        "generated_at": bt.get("generated_at"),
        "test_start_date": bt.get("test_start_date"),
        "test_end_date": bt.get("test_end_date"),
        "total_walk_forward_folds": bt.get("total_walk_forward_folds"),
        "purge_gap_days": bt.get("purge_gap_days"),
        "embargo_sessions": bt.get("embargo_sessions"),
        "execution_mode": bt.get("execution_mode"),
        "nifty_50_buy_hold_return_pct": bt.get("nifty_50_buy_hold_return_pct"),
        "model_checks": model_checks,
        "discrepancies": discrepancies,
    }


def reconcile_regime(root_dir: Path) -> dict[str, Any]:
    """Reconcile `docs/data/regime.json` against `data/processed/market_regimes.parquet`."""
    reg_path = root_dir / "docs" / "data" / "regime.json"
    mr_path = root_dir / "data" / "processed" / "market_regimes.parquet"
    reg = _load_json(reg_path)
    if reg is None:
        return {"status": "MISSING_FILE", "path": str(reg_path), "severity": "CRITICAL", "discrepancies": ["Missing docs/data/regime.json"]}

    discrepancies: list[str] = []
    pq_latest: dict[str, Any] | None = None
    if mr_path.exists():
        mr = pd.read_parquet(mr_path).sort_values("observation_date")
        if not mr.empty:
            row = mr.iloc[-1].to_dict()
            pq_latest = {
                "observation_date": str(row.get("observation_date")),
                "regime": str(row.get("regime")),
                "nifty_50_close": float(row.get("nifty_50_close")),
                "india_vix_close": float(row.get("india_vix_close")),
                "advance_count": int(row.get("advance_count")),
                "decline_count": int(row.get("decline_count")),
                "advance_decline_ratio": float(row.get("advance_decline_ratio")),
            }
            for k, v in pq_latest.items():
                jv = reg.get(k)
                if isinstance(v, float):
                    if jv is None or abs(float(jv) - v) > 1e-4:
                        discrepancies.append(f"regime.{k} mismatch: json={jv} vs parquet={v}")
                elif jv != v:
                    discrepancies.append(f"regime.{k} mismatch: json={jv} vs parquet={v}")

    return {
        "status": "OK" if not discrepancies else "DISCREPANCY",
        "severity": "HIGH" if discrepancies else "OK",
        "observation_date": reg.get("observation_date"),
        "regime": reg.get("regime"),
        "parquet_latest": pq_latest,
        "discrepancies": discrepancies,
    }


def reconcile_stock_jsons(root_dir: Path) -> dict[str, Any]:
    """Reconcile `docs/data/{SYMBOL}.json` against predictions, data quality, and corporate intel."""
    docs_data = root_dir / "docs" / "data"
    lp = _load_json(docs_data / "latest_predictions.json") or {}
    lp_map = {str(p.get("symbol")).upper(): p for p in (lp.get("predictions") or [])}

    dq_path = root_dir / "data" / "processed" / "data_quality.parquet"
    dq_map: dict[str, dict[str, Any]] = {}
    if dq_path.exists():
        dq_df = pd.read_parquet(dq_path)
        dq_map = dq_df.set_index("symbol").to_dict(orient="index")

    ci_path = root_dir / "data" / "processed" / "corporate_intelligence.parquet"
    ci_syms: set[str] = set()
    if ci_path.exists():
        ci_df = pd.read_parquet(ci_path)
        ci_syms = set(ci_df["symbol"].astype(str).str.upper())

    stock_paths = _list_stock_json_paths(docs_data)
    discrepancies: list[str] = []
    findings: list[str] = []
    zero_cf_sentinel_symbols: list[str] = []
    missing_qp_symbols: list[str] = []
    missing_dq_symbols: list[str] = []
    stale_bundle_symbols: list[str] = []
    model_versions: Counter[str] = Counter()
    obs_dates: Counter[str] = Counter()

    for sp in stock_paths:
        sym = sp.stem.upper()
        bundle = _load_json(sp)
        if bundle is None:
            discrepancies.append(f"{sym}: unreadable JSON")
            continue

        qp = bundle.get("quantitative_prediction")
        dq = bundle.get("data_quality")
        ci = bundle.get("corporate_intelligence")
        cb = bundle.get("cash_and_balance_sheet") or {}

        # Zero-sentinel check on cash flow
        if cb.get("latest_ocf_cr") == 0.0 and cb.get("latest_fcf_cr") == 0.0:
            zero_cf_sentinel_symbols.append(sym)

        # Latest predictions reconciliation
        in_lp = sym in lp_map
        if in_lp and not qp:
            discrepancies.append(f"{sym}: present in latest_predictions.json but quantitative_prediction is null in {sp.name}.")
        elif not in_lp and qp:
            discrepancies.append(f"{sym}: has quantitative_prediction in {sp.name} but missing from latest_predictions.json.")
        elif qp and in_lp:
            lp_rec = lp_map[sym]
            if qp.get("prob_next_day_up") != lp_rec.get("prob_next_day_up"):
                discrepancies.append(
                    f"{sym}: prob_next_day_up mismatch (stock={qp.get('prob_next_day_up')} vs latest_predictions={lp_rec.get('prob_next_day_up')})."
                )
            if qp.get("observation_date") != lp_rec.get("observation_date"):
                discrepancies.append(
                    f"{sym}: observation_date mismatch (stock={qp.get('observation_date')} vs latest_predictions={lp_rec.get('observation_date')})."
                )
            mv = str(qp.get("model_version") or "UNAVAILABLE")
            model_versions[mv] += 1
            obs_dates[str(qp.get("observation_date") or "UNAVAILABLE")] += 1
        else:
            missing_qp_symbols.append(sym)

        # Data quality reconciliation
        in_dq = sym in dq_map
        if in_dq and dq is None:
            discrepancies.append(f"{sym}: present in data_quality.parquet but data_quality is null in {sp.name}.")
        elif dq is not None and in_dq:
            pq_score = round(float(dq_map[sym].get("quality_score", 0.0)), 2)
            j_score = round(float(dq.get("quality_score", 0.0)), 2)
            if pq_score != j_score:
                discrepancies.append(f"{sym}: data_quality.quality_score mismatch (stock={j_score} vs parquet={pq_score}).")
        elif dq is None:
            missing_dq_symbols.append(sym)

        # Corporate intelligence check
        if sym in ci_syms and ci is None:
            discrepancies.append(f"{sym}: present in corporate_intelligence.parquet but corporate_intelligence is null in {sp.name}.")

        # Bundle fetch freshness gap check
        fetched = str(bundle.get("fetched_at") or "")
        if fetched.startswith("2026-11-16") and qp and str(qp.get("observation_date") or "") > "2026-11-16":
            stale_bundle_symbols.append(sym)

    if zero_cf_sentinel_symbols:
        findings.append(
            f"{len(zero_cf_sentinel_symbols)}/{len(stock_paths)} stock JSONs report latest_ocf_cr=0.0 & latest_fcf_cr=0.0 sentinel "
            f"(rendered as UNAVAILABLE in report.js)."
        )
    if stale_bundle_symbols:
        findings.append(
            f"{len(stale_bundle_symbols)} stock JSONs ({', '.join(stale_bundle_symbols)}) have fetched_at=2026-11-16 while embedding quantitative_prediction observation_date=2026-11-18."
        )

    return {
        "status": "OK" if not discrepancies else "DISCREPANCY",
        "severity": "HIGH" if discrepancies else ("MEDIUM" if findings else "OK"),
        "total_stock_jsons": len(stock_paths),
        "with_quantitative_prediction": len(stock_paths) - len(missing_qp_symbols),
        "without_quantitative_prediction": missing_qp_symbols,
        "with_data_quality": len(stock_paths) - len(missing_dq_symbols),
        "without_data_quality_count": len(missing_dq_symbols),
        "zero_cashflow_sentinel_count": len(zero_cf_sentinel_symbols),
        "zero_cashflow_sentinel_symbols": zero_cf_sentinel_symbols,
        "stale_fetched_at_with_newer_qp_symbols": stale_bundle_symbols,
        "qp_model_versions": dict(model_versions),
        "qp_observation_dates": dict(obs_dates),
        "discrepancies": discrepancies,
        "findings": findings,
    }


def reconcile_dashboard_artifacts(
    root_dir: Path | str,
    prospective_cutoff: str = DEFAULT_PROSPECTIVE_CUTOFF,
    output_json_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run full read-only reconciliation across all `docs/data/*.json` dashboard artifacts."""
    root = Path(root_dir).resolve()
    docs_data = root / "docs" / "data"

    # Check raw NaN tokens across all docs/data/*.json
    nan_files: dict[str, list[str]] = {}
    if docs_data.exists():
        for jp in sorted(docs_data.glob("*.json")):
            payload = _load_json(jp)
            if payload is not None:
                nans = _scan_nan_tokens(payload)
                if nans:
                    nan_files[jp.name] = nans[:10]

    sections = {
        "forward_validation": reconcile_forward_validation(root, prospective_cutoff=prospective_cutoff),
        "phase5_intelligence": reconcile_phase5_intelligence(root),
        "phase5_experiments": reconcile_phase5_experiments(root),
        "backtest_summary": reconcile_backtest_summary(root),
        "regime": reconcile_regime(root),
        "stock_jsons": reconcile_stock_jsons(root),
        "model_version_drift": audit_metadata_version_drift(root),
    }

    total_discrepancies = sum(len(s.get("discrepancies", [])) for s in sections.values() if isinstance(s, dict))
    severities = [s.get("severity", "OK") for s in sections.values() if isinstance(s, dict)]
    if "CRITICAL" in severities or nan_files:
        overall = "CRITICAL"
    elif "HIGH" in severities or total_discrepancies > 0:
        overall = "HIGH"
    elif "MEDIUM" in severities:
        overall = "MEDIUM"
    else:
        overall = "OK"

    report: dict[str, Any] = {
        "reconciled_at": datetime.now(IST_TZ).isoformat(),
        "read_only_guarantee": True,
        "prospective_cutoff": prospective_cutoff,
        "overall_status": overall,
        "total_discrepancies": total_discrepancies,
        "nan_token_files": nan_files,
        "sections": sections,
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    res = reconcile_dashboard_artifacts(repo_root)
    print(
        f"Dashboard reconciliation complete: overall_status={res['overall_status']} "
        f"total_discrepancies={res['total_discrepancies']} nan_files={len(res['nan_token_files'])}"
    )
