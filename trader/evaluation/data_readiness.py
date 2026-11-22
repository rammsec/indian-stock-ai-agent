"""CYCLE-008 Phase 1 — Historical Data Readiness & Provenance Auditor.

Audits all committed historical market data (`data/intraday/*_1h.parquet`, 15m/5m
availability), Stage 1 historical archives (`prediction_log.parquet`,
`forward_predictions.parquet`, `market_regimes.parquet`, `structured_events.parquet`),
and `data/lab/sealed_holdout_registry.json` without modifying any files, making
network calls, fabricating candles, or upgrading Yahoo Finance provenance to VERIFIED.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.provenance_1h import VALIDATOR_VERSION, validate_1h_provenance
from trader.paper.cycle006b import DEFAULT_CACHE_DIR, sha256_file
from trader.paper.cycle006c_snapshot import (
    DEFAULT_FORWARD_PREDICTIONS_PATH,
    DEFAULT_MARKET_REGIMES_PATH,
    DEFAULT_PREDICTION_LOG_PATH,
    DEFAULT_STRUCTURED_EVENTS_PATH,
)

DEFAULT_HOLDOUT_REGISTRY_PATH = Path("data/lab/sealed_holdout_registry.json")
REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _audit_single_1h_symbol(
    symbol: str,
    cache_dir: Path,
    corporate_action_jump_threshold_pct: float,
) -> tuple[dict[str, Any], set[date]]:
    """Audit one symbol's committed 1h parquet file and return metadata + trading dates."""
    sym = symbol.strip().upper()
    pq_path = cache_dir / f"{sym}_1h.parquet"
    if not pq_path.exists():
        return (
            {
                "symbol": sym,
                "exists": False,
                "file": str(pq_path),
                "sha256": None,
                "row_count": 0,
                "trading_days": 0,
                "date_min": None,
                "date_max": None,
                "duplicate_timestamps": 0,
                "nan_count": 0,
                "ohlc_invariant_violations": 0,
                "timezone": None,
                "timezone_valid": False,
                "corporate_action_jump_count": 0,
                "corporate_action_jump_flags": [],
                "complete_sessions_ge_6_bars": 0,
            },
            set(),
        )

    file_sha = sha256_file(pq_path)
    df = pd.read_parquet(pq_path)
    missing_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing_cols or df.empty:
        return (
            {
                "symbol": sym,
                "exists": True,
                "file": str(pq_path),
                "sha256": file_sha,
                "row_count": int(len(df)),
                "trading_days": 0,
                "date_min": None,
                "date_max": None,
                "duplicate_timestamps": 0,
                "nan_count": int(df.isna().sum().sum()) if not df.empty else 0,
                "ohlc_invariant_violations": int(len(df)),
                "timezone": None,
                "timezone_valid": False,
                "missing_columns": missing_cols,
                "corporate_action_jump_count": 0,
                "corporate_action_jump_flags": [],
                "complete_sessions_ge_6_bars": 0,
            },
            set(),
        )

    ts_raw = pd.to_datetime(df["timestamp"])
    tz_obj = getattr(ts_raw.dt, "tz", None)
    tz_str = str(tz_obj) if tz_obj is not None else None
    tz_valid = tz_str is not None and ("Kolkata" in tz_str or "+05:30" in tz_str)

    if tz_obj is not None:
        ts_ist = ts_raw.dt.tz_convert(IST_TZ)
    else:
        ts_ist = ts_raw.dt.tz_localize(IST_TZ)

    dup_count = int(ts_ist.duplicated().sum())
    nan_count = int(df[list(REQUIRED_OHLCV_COLUMNS)].isna().sum().sum())

    ohlc_bad_mask = (
        df[["open", "high", "low", "close"]].isna().any(axis=1)
        | ~np.isfinite(df[["open", "high", "low", "close"]].to_numpy(dtype=float)).all(axis=1)
        | (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | df["volume"].isna()
        | (df["volume"] < 0)
    )
    ohlc_violations = int(ohlc_bad_mask.sum())

    work = df.copy()
    work["timestamp_ist"] = ts_ist
    work = work.sort_values("timestamp_ist").reset_index(drop=True)
    work["session_date"] = work["timestamp_ist"].dt.date

    trading_dates = set(work["session_date"].unique())
    bars_per_day = work.groupby("session_date").size()
    complete_sessions = int((bars_per_day >= 6).sum())

    # Corporate-action / split jump detection: >15% single-bar close jump
    pct_jumps = work["close"].pct_change().abs() * 100.0
    jump_mask = pct_jumps > corporate_action_jump_threshold_pct
    jump_flags: list[dict[str, Any]] = []
    for idx in work.index[jump_mask]:
        prev_close = float(work.loc[idx - 1, "close"])
        curr_close = float(work.loc[idx, "close"])
        jump_pct = ((curr_close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        jump_flags.append(
            {
                "timestamp_ist": work.loc[idx, "timestamp_ist"].isoformat(),
                "prev_close": round(prev_close, 4),
                "close": round(curr_close, 4),
                "jump_pct": round(jump_pct, 4),
            }
        )

    return (
        {
            "symbol": sym,
            "exists": True,
            "file": str(pq_path),
            "sha256": file_sha,
            "row_count": int(len(work)),
            "trading_days": int(len(trading_dates)),
            "date_min": min(trading_dates).isoformat() if trading_dates else None,
            "date_max": max(trading_dates).isoformat() if trading_dates else None,
            "duplicate_timestamps": dup_count,
            "nan_count": nan_count,
            "ohlc_invariant_violations": ohlc_violations,
            "timezone": tz_str,
            "timezone_valid": tz_valid,
            "corporate_action_jump_count": len(jump_flags),
            "corporate_action_jump_flags": jump_flags,
            "bars_per_day_min": int(bars_per_day.min()) if not bars_per_day.empty else 0,
            "bars_per_day_max": int(bars_per_day.max()) if not bars_per_day.empty else 0,
            "complete_sessions_ge_6_bars": complete_sessions,
        },
        trading_dates,
    )


def _audit_missing_timeframe(
    timeframe: str,
    symbols: tuple[str, ...],
    cache_dir: Path,
) -> dict[str, Any]:
    """Audit committed 15m or 5m archives; never fabricates or substitutes timeframes."""
    found: list[str] = []
    missing: list[str] = []
    for sym in symbols:
        sym_u = sym.strip().upper()
        p = cache_dir / f"{sym_u}_{timeframe}.parquet"
        if p.exists():
            found.append(sym_u)
        else:
            missing.append(sym_u)

    if not found:
        classification = "UNAVAILABLE"
    elif missing:
        classification = "PARTIALLY_AVAILABLE"
    else:
        classification = "AVAILABLE_UNVERIFIED"

    return {
        "timeframe": timeframe,
        "classification": classification,
        "available_symbols": found,
        "missing_symbols": missing,
        "cache_dir": str(cache_dir),
        "fabricated_or_substituted": False,
        "note": (
            f"Committed {timeframe} parquet archives are UNAVAILABLE in {cache_dir}; "
            "CYCLE-008 never fabricates candles or substitutes timeframes."
            if classification == "UNAVAILABLE"
            else f"Found {len(found)}/{len(symbols)} {timeframe} parquet files in {cache_dir}."
        ),
    }


def _audit_stage1_archives(
    prediction_log_path: Path,
    forward_predictions_path: Path,
    market_regimes_path: Path,
    structured_events_path: Path,
) -> dict[str, Any]:
    """Audit Stage 1 historical archives (exact observation dates and event timestamp completeness)."""
    result: dict[str, Any] = {}

    for key, path in (
        ("prediction_log", prediction_log_path),
        ("forward_predictions", forward_predictions_path),
        ("market_regimes", market_regimes_path),
    ):
        if not path.exists():
            result[key] = {
                "exists": False,
                "path": str(path),
                "sha256": None,
                "row_count": 0,
                "observation_date_count": 0,
                "observation_dates": [],
            }
            continue

        df = pd.read_parquet(path)
        obs_dates: list[str] = []
        if "observation_date" in df.columns and not df.empty:
            obs_dates = sorted(
                {
                    str(v)[:10]
                    for v in df["observation_date"].dropna().tolist()
                    if str(v).strip()
                }
            )
        entry: dict[str, Any] = {
            "exists": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": int(len(df)),
            "observation_date_count": len(obs_dates),
            "observation_dates": obs_dates,
        }
        if "symbol" in df.columns and not df.empty:
            entry["symbols_present"] = sorted(
                df["symbol"].astype(str).str.strip().str.upper().unique().tolist()
            )
        if key == "market_regimes" and "regime" in df.columns and not df.empty:
            entry["regimes_by_date"] = {
                str(r["observation_date"])[:10]: str(r["regime"]).strip()
                for _, r in df.iterrows()
            }
        result[key] = entry

    # Structured events audit (publication & collection timestamp completeness)
    if not structured_events_path.exists():
        result["structured_events"] = {
            "exists": False,
            "path": str(structured_events_path),
            "sha256": None,
            "row_count": 0,
            "publication_timestamp_present_count": 0,
            "publication_timestamp_missing_count": 0,
            "collection_timestamp_present_count": 0,
            "collection_timestamp_missing_count": 0,
            "collection_timestamp_completeness_pct": 0.0,
        }
    else:
        se_df = pd.read_parquet(structured_events_path)
        total_rows = int(len(se_df))
        if total_rows > 0 and "publication_timestamp" in se_df.columns:
            pub_present = int(
                se_df["publication_timestamp"]
                .apply(lambda v: pd.notna(v) and bool(str(v).strip()))
                .sum()
            )
        else:
            pub_present = 0
        if total_rows > 0 and "collection_timestamp" in se_df.columns:
            col_present = int(
                se_df["collection_timestamp"]
                .apply(lambda v: pd.notna(v) and bool(str(v).strip()))
                .sum()
            )
        else:
            col_present = 0

        completeness_pct = round((col_present / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        result["structured_events"] = {
            "exists": True,
            "path": str(structured_events_path),
            "sha256": sha256_file(structured_events_path),
            "row_count": total_rows,
            "publication_timestamp_present_count": pub_present,
            "publication_timestamp_missing_count": total_rows - pub_present,
            "collection_timestamp_present_count": col_present,
            "collection_timestamp_missing_count": total_rows - col_present,
            "collection_timestamp_completeness_pct": completeness_pct,
        }

    return result


def _audit_holdout_registry(
    holdout_registry_path: Path,
    common_1h_dates: list[date],
) -> dict[str, Any]:
    """Audit sealed_holdout_registry.json and partition common 1h dates cleanly."""
    common_iso = sorted(d.isoformat() for d in common_1h_dates)

    if not holdout_registry_path.exists():
        return {
            "exists": False,
            "path": str(holdout_registry_path),
            "sha256": None,
            "registry_version": None,
            "exposed_count": 0,
            "available_count": 0,
            "sealed_count": 0,
            "retired_count": 0,
            "total_registered_dates": 0,
            "common_1h_trading_days_count": len(common_iso),
            "non_holdout_count": len(common_iso),
            "non_holdout_date_count": len(common_iso),
            "non_holdout_date_min": common_iso[0] if common_iso else None,
            "non_holdout_date_max": common_iso[-1] if common_iso else None,
            "non_holdout_dates": common_iso,
            "exposed_holdout_date_count": 0,
            "exposed_holdout_date_min": None,
            "exposed_holdout_date_max": None,
            "exposed_holdout_dates": [],
            "excluded_straddle_count": 0,
            "excluded_straddle_dates": [],
            "forbidden_for_unseen_claims": True,
        }

    reg_sha = sha256_file(holdout_registry_path)
    reg_data = json.loads(holdout_registry_path.read_text(encoding="utf-8"))
    records = reg_data.get("records", {})
    sealed_dates_list = reg_data.get("sealed_dates", [])

    state_counts = {"EXPOSED": 0, "AVAILABLE": 0, "SEALED": 0, "RETIRED": 0}
    registered_dates: set[str] = set()
    if isinstance(records, dict):
        for d_key, rec in records.items():
            d_norm = str(d_key)[:10]
            registered_dates.add(d_norm)
            st = str(rec.get("state", "UNKNOWN")).upper() if isinstance(rec, dict) else "UNKNOWN"
            state_counts[st] = state_counts.get(st, 0) + 1
    for d_item in sealed_dates_list:
        registered_dates.add(str(d_item)[:10])

    exposed_holdout_dates = [d for d in common_iso if d in registered_dates]
    if exposed_holdout_dates:
        earliest_exposed = min(exposed_holdout_dates)
        non_holdout_dates = [d for d in common_iso if d < earliest_exposed]
        excluded_straddle_dates = [
            d for d in common_iso if d not in registered_dates and d >= earliest_exposed
        ]
    else:
        non_holdout_dates = [d for d in common_iso if d not in registered_dates]
        excluded_straddle_dates = []

    return {
        "exists": True,
        "path": str(holdout_registry_path),
        "sha256": reg_sha,
        "registry_version": reg_data.get("registry_version"),
        "exposed_count": int(state_counts.get("EXPOSED", 0)),
        "available_count": int(state_counts.get("AVAILABLE", 0)),
        "sealed_count": int(state_counts.get("SEALED", 0)),
        "retired_count": int(state_counts.get("RETIRED", 0)),
        "total_registered_dates": len(registered_dates),
        "common_1h_trading_days_count": len(common_iso),
        "non_holdout_count": len(non_holdout_dates),
        "non_holdout_date_count": len(non_holdout_dates),
        "non_holdout_date_min": non_holdout_dates[0] if non_holdout_dates else None,
        "non_holdout_date_max": non_holdout_dates[-1] if non_holdout_dates else None,
        "non_holdout_dates": non_holdout_dates,
        "exposed_holdout_date_count": len(exposed_holdout_dates),
        "exposed_holdout_date_min": exposed_holdout_dates[0] if exposed_holdout_dates else None,
        "exposed_holdout_date_max": exposed_holdout_dates[-1] if exposed_holdout_dates else None,
        "exposed_holdout_dates": exposed_holdout_dates,
        "excluded_straddle_count": len(excluded_straddle_dates),
        "excluded_straddle_dates": excluded_straddle_dates,
        "forbidden_for_unseen_claims": True,
    }


def audit_historical_data_readiness(
    *,
    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE,
    cache_dir_1h: Path = DEFAULT_CACHE_DIR,
    prediction_log_path: Path = DEFAULT_PREDICTION_LOG_PATH,
    forward_predictions_path: Path = DEFAULT_FORWARD_PREDICTIONS_PATH,
    market_regimes_path: Path = DEFAULT_MARKET_REGIMES_PATH,
    structured_events_path: Path = DEFAULT_STRUCTURED_EVENTS_PATH,
    holdout_registry_path: Path = DEFAULT_HOLDOUT_REGISTRY_PATH,
    corporate_action_jump_threshold_pct: float = 15.0,
) -> dict[str, Any]:
    """Run Phase 1 Historical Data Readiness & Provenance Audit across all committed archives."""
    per_symbol_1h: dict[str, Any] = {}
    common_dates: set[date] | None = None

    for sym in symbols:
        sym_meta, sym_dates = _audit_single_1h_symbol(
            symbol=sym,
            cache_dir=cache_dir_1h,
            corporate_action_jump_threshold_pct=corporate_action_jump_threshold_pct,
        )
        per_symbol_1h[sym.strip().upper()] = sym_meta
        if common_dates is None:
            common_dates = set(sym_dates)
        else:
            common_dates &= sym_dates

    common_dates_sorted = sorted(common_dates) if common_dates else []

    prov_verified, prov_report = validate_1h_provenance(
        symbols=symbols,
        cache_dir=cache_dir_1h,
    )
    # Hard rule: Yahoo Finance 1h provenance is NEVER upgraded to VERIFIED
    prov_status = str(prov_report.get("final_provenance_status", "FAILED"))

    timeframe_15m = _audit_missing_timeframe("15m", symbols, cache_dir_1h)
    timeframe_5m = _audit_missing_timeframe("5m", symbols, cache_dir_1h)

    stage1_archives = _audit_stage1_archives(
        prediction_log_path=prediction_log_path,
        forward_predictions_path=forward_predictions_path,
        market_regimes_path=market_regimes_path,
        structured_events_path=structured_events_path,
    )

    holdout_partition = _audit_holdout_registry(
        holdout_registry_path=holdout_registry_path,
        common_1h_dates=common_dates_sorted,
    )

    limitations: list[str] = []
    if not prov_verified:
        limitations.append(
            f"1h market data provenance is {prov_status} (verified=False; Yahoo Finance archive)."
        )
    if timeframe_15m["classification"] == "UNAVAILABLE":
        limitations.append("Committed 15m candle archives are UNAVAILABLE.")
    if timeframe_5m["classification"] == "UNAVAILABLE":
        limitations.append("Committed 5m candle archives are UNAVAILABLE.")
    if stage1_archives["prediction_log"]["observation_date_count"] < 20:
        limitations.append(
            "Stage 1 historical archives cover only "
            f"{stage1_archives['prediction_log']['observation_date_count']} observation dates "
            f"({stage1_archives['prediction_log']['observation_dates']}); insufficient for "
            "statistically significant multi-month OOS validation of Stage 1 or combined pipeline."
        )
    if holdout_partition["available_count"] == 0 and holdout_partition["sealed_count"] == 0:
        limitations.append(
            f"Sealed holdout registry has {holdout_partition['exposed_count']} EXPOSED, "
            "0 AVAILABLE, 0 SEALED dates; unseen holdout evaluation is NOT_EVALUATED."
        )

    return {
        "universe": [s.strip().upper() for s in symbols],
        "intraday_1h_audit": {
            "cache_dir": str(cache_dir_1h),
            "provenance_validator_version": VALIDATOR_VERSION,
            "provenance_status": prov_status,
            "verified": bool(prov_verified),
            "dataset_fingerprint": prov_report.get("dataset_fingerprint"),
            "common_trading_days_count": len(common_dates_sorted),
            "common_date_min": common_dates_sorted[0].isoformat() if common_dates_sorted else None,
            "common_date_max": common_dates_sorted[-1].isoformat() if common_dates_sorted else None,
            "corporate_action_jump_threshold_pct": corporate_action_jump_threshold_pct,
            "total_duplicate_timestamps": sum(
                int(m["duplicate_timestamps"]) for m in per_symbol_1h.values()
            ),
            "total_nan_count": sum(int(m["nan_count"]) for m in per_symbol_1h.values()),
            "total_ohlc_invariant_violations": sum(
                int(m["ohlc_invariant_violations"]) for m in per_symbol_1h.values()
            ),
            "total_corporate_action_jump_flags": sum(
                int(m["corporate_action_jump_count"]) for m in per_symbol_1h.values()
            ),
            "per_symbol": per_symbol_1h,
        },
        "intraday_15m_audit": timeframe_15m,
        "intraday_5m_audit": timeframe_5m,
        "stage1_historical_archives": stage1_archives,
        "holdout_registry_audit": holdout_partition,
        "limitations": limitations,
    }
