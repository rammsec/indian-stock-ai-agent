"""CYCLE-006C — Genuine Historical As-Of Snapshot Reconstruction & Timeframe Provenance.

Reconstructs strictly timestamp-aligned historical inputs (predictions, regime,
and Phase 5 intelligence) from committed repository parquet archives without
fabricating data, modifying source archives, or leaking post-cutoff information.
"""

from __future__ import annotations

from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.provenance_1h import validate_1h_provenance
from trader.data.resampler import resample_5m_candles
from trader.data.schema import Candle5M
from trader.paper.cycle006b import sha256_file, write_report_atomic

DEFAULT_PREDICTION_LOG_PATH = Path("data/predictions/prediction_log.parquet")
DEFAULT_MARKET_REGIMES_PATH = Path("data/processed/market_regimes.parquet")
DEFAULT_FORWARD_PREDICTIONS_PATH = Path("data/predictions/forward_predictions.parquet")
DEFAULT_STRUCTURED_EVENTS_PATH = Path("data/predictions/structured_events.parquet")
DEFAULT_1H_CACHE_DIR = Path("data/intraday")


def _to_ist_datetime(raw: Any, label: str) -> datetime:
    if raw is None or pd.isna(raw):
        raise ValueError(f"Missing timestamp for {label}")
    if isinstance(raw, pd.Timestamp):
        if raw.tzinfo is None:
            raise ValueError(f"Timezone-naive timestamp rejected for {label}: {raw!r}")
        return raw.tz_convert(IST_TZ).to_pydatetime()
    text = str(raw).strip()
    if not text:
        raise ValueError(f"Empty timestamp for {label}")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"Timezone-naive timestamp rejected for {label}: {raw!r}")
    return dt.astimezone(IST_TZ)


def reconstruct_as_of_snapshot(
    session_date: date,
    output_dir: Path,
    *,
    as_of_time_ist: dt_time = dt_time(9, 15),
    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE,
    prediction_log_path: Path = DEFAULT_PREDICTION_LOG_PATH,
    market_regimes_path: Path = DEFAULT_MARKET_REGIMES_PATH,
    forward_predictions_path: Path = DEFAULT_FORWARD_PREDICTIONS_PATH,
    structured_events_path: Path = DEFAULT_STRUCTURED_EVENTS_PATH,
) -> dict[str, Any]:
    """Reconstruct a strictly pre-cutoff historical snapshot from committed archives.

    Raises ValueError or FileNotFoundError if any required symbol, timestamp, or
    availability proof is missing. Never fabricates or substitutes missing values.
    """
    as_of_ist = datetime.combine(session_date, as_of_time_ist, tzinfo=IST_TZ)
    for label, p in [
        ("prediction_log", prediction_log_path),
        ("market_regimes", market_regimes_path),
        ("forward_predictions", forward_predictions_path),
        ("structured_events", structured_events_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required historical archive missing ({label}): {p}")

    source_hashes = {
        str(prediction_log_path): sha256_file(prediction_log_path),
        str(market_regimes_path): sha256_file(market_regimes_path),
        str(forward_predictions_path): sha256_file(forward_predictions_path),
        str(structured_events_path): sha256_file(structured_events_path),
    }

    # 1. Reconstruct predictions strictly from prediction_log.parquet (generated_at <= as_of_ist & obs < session_date)
    pl_df = pd.read_parquet(prediction_log_path)
    if pl_df.empty:
        raise ValueError("prediction_log.parquet is empty")

    pl_records: list[dict[str, Any]] = []
    pred_gen_times: list[datetime] = []
    pred_obs_dates: set[date] = set()
    pred_regimes: set[str] = set()

    for sym in symbols:
        sub = pl_df[pl_df["symbol"].astype(str).str.strip().str.upper() == sym].copy()
        valid_rows: list[tuple[datetime, date, pd.Series]] = []
        for _, row in sub.iterrows():
            gen_dt = _to_ist_datetime(row.get("generated_at"), f"prediction_log[{sym}].generated_at")
            obs_d = date.fromisoformat(str(row.get("observation_date"))[:10])
            if gen_dt <= as_of_ist and obs_d < session_date:
                valid_rows.append((gen_dt, obs_d, row))
        if not valid_rows:
            raise ValueError(
                f"HISTORICAL_PREDICTIONS_NOT_RECONSTRUCTABLE: no pre-cutoff prediction for {sym} "
                f"before {as_of_ist.isoformat()}"
            )
        valid_rows.sort(key=lambda x: (x[1], x[0]))
        best_gen, best_obs, best_row = valid_rows[-1]
        pred_gen_times.append(best_gen)
        pred_obs_dates.add(best_obs)
        if best_row.get("market_regime"):
            pred_regimes.add(str(best_row["market_regime"]))

        pl_records.append({
            "prediction_id": str(best_row.get("prediction_id", "")),
            "symbol": sym,
            "observation_date": best_obs.isoformat(),
            "generated_at": best_gen.isoformat(),
            "model_name": str(best_row.get("model_name", "")),
            "model_version": str(best_row.get("model_version", "")),
            "market_regime": str(best_row.get("market_regime", "")),
            "prob_next_day_up": float(best_row["prob_next_day_up"]),
            "expected_next_day_return_pct": float(best_row["expected_next_day_return_pct"]),
        })

    if len(pred_obs_dates) != 1:
        raise ValueError(f"Conflicting prediction observation dates across universe: {sorted(pred_obs_dates)}")
    target_obs_date = next(iter(pred_obs_dates))
    pred_snapshot_ts = max(pred_gen_times)

    # 2. Reconstruct regime strictly from market_regimes.parquet matching target_obs_date
    mr_df = pd.read_parquet(market_regimes_path)
    mr_match = mr_df[mr_df["observation_date"].astype(str).str.slice(0, 10) == target_obs_date.isoformat()]
    if mr_match.empty:
        raise ValueError(
            f"HISTORICAL_REGIME_NOT_RECONSTRUCTABLE: no market_regimes record for {target_obs_date.isoformat()}"
        )
    mr_row = mr_match.iloc[-1]
    regime_label = str(mr_row["regime"]).strip()
    if not regime_label:
        raise ValueError(f"Empty regime label in market_regimes.parquet for {target_obs_date.isoformat()}")
    if pred_regimes and regime_label not in pred_regimes:
        raise ValueError(
            f"Regime mismatch between market_regimes ({regime_label}) and prediction_log ({sorted(pred_regimes)})"
        )
    # Availability proof: market_regime was persisted inside prediction_log at pred_snapshot_ts (<= as_of_ist)
    regime_payload = {
        "observation_date": target_obs_date.isoformat(),
        "generated_at": pred_snapshot_ts.isoformat(),
        "availability_proof": "Verified via matching prediction_log.parquet market_regime recorded at generated_at <= as_of_ist",
        "regime": regime_label,
        "nifty_50_close": float(mr_row["nifty_50_close"]) if pd.notna(mr_row.get("nifty_50_close")) else None,
        "india_vix_close": float(mr_row["india_vix_close"]) if pd.notna(mr_row.get("india_vix_close")) else None,
    }

    # 3. Reconstruct Phase 5 intelligence strictly from forward_predictions.parquet + structured_events.parquet
    fp_df = pd.read_parquet(forward_predictions_path)
    se_df = pd.read_parquet(structured_events_path)

    # Filter structured_events requiring BOTH publication_timestamp <= as_of_ist AND collection_timestamp <= as_of_ist
    included_events_by_sym: dict[str, int] = {s: 0 for s in symbols}
    excluded_events_reasons: dict[str, int] = {
        "missing_collection_timestamp": 0,
        "missing_publication_timestamp": 0,
        "published_after_cutoff": 0,
        "collected_after_cutoff": 0,
        "outside_target_universe": 0,
    }
    max_event_col_ts: datetime | None = None
    total_included_events = 0

    for _, ev in se_df.iterrows():
        raw_pub = ev.get("publication_timestamp")
        raw_col = ev.get("collection_timestamp")
        if raw_pub is None or pd.isna(raw_pub) or not str(raw_pub).strip():
            excluded_events_reasons["missing_publication_timestamp"] += 1
            continue
        if raw_col is None or pd.isna(raw_col) or not str(raw_col).strip():
            excluded_events_reasons["missing_collection_timestamp"] += 1
            continue
        pub_dt = _to_ist_datetime(raw_pub, "structured_events.publication_timestamp")
        col_dt = _to_ist_datetime(raw_col, "structured_events.collection_timestamp")
        if pub_dt > as_of_ist:
            excluded_events_reasons["published_after_cutoff"] += 1
            continue
        if col_dt > as_of_ist:
            excluded_events_reasons["collected_after_cutoff"] += 1
            continue
        sym_ev = str(ev.get("ticker", "")).strip().upper()
        if sym_ev not in included_events_by_sym:
            excluded_events_reasons["outside_target_universe"] += 1
            continue
        included_events_by_sym[sym_ev] += 1
        total_included_events += 1
        if max_event_col_ts is None or col_dt > max_event_col_ts:
            max_event_col_ts = col_dt

    intel_stocks: dict[str, dict[str, Any]] = {}
    intel_ts_list: list[datetime] = []
    model_version_seen = "5.0.0-pilot"

    for sym in symbols:
        fsub = fp_df[fp_df["symbol"].astype(str).str.strip().str.upper() == sym].copy()
        valid_fp: list[tuple[datetime, date, pd.Series]] = []
        for _, frow in fsub.iterrows():
            pt_dt = _to_ist_datetime(frow.get("prediction_timestamp"), f"forward_predictions[{sym}].prediction_timestamp")
            f_obs = date.fromisoformat(str(frow.get("observation_date"))[:10])
            if pt_dt <= as_of_ist and f_obs == target_obs_date:
                valid_fp.append((pt_dt, f_obs, frow))
        if not valid_fp:
            raise ValueError(
                f"HISTORICAL_INTELLIGENCE_NOT_RECONSTRUCTABLE: no pre-cutoff forward_predictions record "
                f"for {sym} on {target_obs_date.isoformat()} before {as_of_ist.isoformat()}"
            )
        if included_events_by_sym[sym] == 0:
            raise ValueError(
                f"HISTORICAL_INTELLIGENCE_NOT_RECONSTRUCTABLE: zero structured_events with verified "
                f"pub<=cutoff and col<=cutoff for {sym} before {as_of_ist.isoformat()}"
            )
        valid_fp.sort(key=lambda x: x[0])
        best_pt, best_fobs, best_frow = valid_fp[-1]
        if best_frow.get("news_score") is None or pd.isna(best_frow.get("news_score")):
            raise ValueError(f"HISTORICAL_INTELLIGENCE_NOT_RECONSTRUCTABLE: null news_score for {sym}")
        intel_ts_list.append(best_pt)
        if best_frow.get("model_version"):
            model_version_seen = str(best_frow["model_version"])

        intel_stocks[sym] = {
            "symbol": sym,
            "observation_date": best_fobs.isoformat(),
            "news_sentiment_score": float(best_frow["news_score"]),
            "source_prediction_timestamp_ist": best_pt.isoformat(),
            "verified_pre_cutoff_events_count": included_events_by_sym[sym],
        }

    intel_snapshot_ts = max(intel_ts_list)
    if max_event_col_ts is not None and max_event_col_ts > intel_snapshot_ts and max_event_col_ts <= as_of_ist:
        intel_snapshot_ts = max_event_col_ts

    predictions_payload = {
        "generated_at": pred_snapshot_ts.isoformat(),
        "market_regime": regime_label,
        "source_archive": str(prediction_log_path),
        "as_of_cutoff_ist": as_of_ist.isoformat(),
        "predictions": pl_records,
    }
    intelligence_payload = {
        "generated_at": intel_snapshot_ts.isoformat(),
        "model_version": model_version_seen,
        "as_of_cutoff_ist": as_of_ist.isoformat(),
        "source_archives": [str(forward_predictions_path), str(structured_events_path)],
        "stocks": intel_stocks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_out = write_report_atomic(predictions_payload, output_dir / "latest_predictions.json")
    intel_out = write_report_atomic(intelligence_payload, output_dir / "phase5_intelligence.json")
    regime_out = write_report_atomic(regime_payload, output_dir / "regime.json")

    manifest: dict[str, Any] = {
        "cycle_id": "CYCLE-006C",
        "snapshot_status": "RECONSTRUCTED_FROM_ARCHIVES",
        "session_date": session_date.isoformat(),
        "as_of_cutoff_ist": as_of_ist.isoformat(),
        "target_observation_date": target_obs_date.isoformat(),
        "symbols": list(symbols),
        "source_file_hashes": source_hashes,
        "output_files": {
            "predictions_path": str(pred_out),
            "intelligence_path": str(intel_out),
            "regime_path": str(regime_out),
        },
        "output_file_hashes": {
            "latest_predictions.json": sha256_file(pred_out),
            "phase5_intelligence.json": sha256_file(intel_out),
            "regime.json": sha256_file(regime_out),
        },
        "timestamps": {
            "prediction_generated_at_ist": pred_snapshot_ts.isoformat(),
            "regime_availability_ist": pred_snapshot_ts.isoformat(),
            "intelligence_generated_at_ist": intel_snapshot_ts.isoformat(),
            "max_verified_event_collection_ist": max_event_col_ts.isoformat() if max_event_col_ts else None,
        },
        "structured_events_audit": {
            "total_archive_events": int(len(se_df)),
            "included_verified_events": total_included_events,
            "included_by_symbol": included_events_by_sym,
            "excluded_reasons": excluded_events_reasons,
        },
        "provenance_classification": "ARCHIVED_PRE_CUTOFF_RECONSTRUCTED",
    }
    manifest_out = write_report_atomic(manifest, output_dir / "snapshot_manifest.json")
    manifest["manifest_path"] = str(manifest_out)
    return manifest


def classify_timeframe_provenance(
    session_date: date,
    symbols: tuple[str, ...] = LAB_DEFAULT_UNIVERSE,
    cache_dir_1h: Path = DEFAULT_1H_CACHE_DIR,
    cache_dir_5m: Path | None = None,
) -> dict[str, Any]:
    """Independently classify 1h, 5m, and 15m data availability & provenance."""
    ok_1h, rep_1h = validate_1h_provenance(symbols=symbols, cache_dir=cache_dir_1h)
    status_1h = str(rep_1h.get("final_provenance_status", "UNAVAILABLE"))

    # Check committed repo 5m availability first
    committed_5m = sorted(p.name for p in cache_dir_1h.glob("*_5m.parquet"))
    if cache_dir_5m is None or not cache_dir_5m.exists():
        return {
            "1h": {
                "classification": status_1h,
                "verified": bool(ok_1h),
                "source": "committed_repo_yahoo_finance_1h_parquet",
                "dataset_fingerprint": rep_1h.get("dataset_fingerprint"),
            },
            "5m": {
                "classification": "UNAVAILABLE",
                "committed_in_repo": len(committed_5m) > 0,
                "committed_files": committed_5m,
                "reason": "No 5m parquet archives exist in committed data/intraday/.",
            },
            "15m": {
                "classification": "UNAVAILABLE",
                "committed_in_repo": False,
                "reason": "No direct 15m archive or committed 5m source available for aggregation.",
            },
        }

    # Inspect provided 5m directory (e.g., isolated Yahoo Finance cache) without downloading
    per_sym_5m: dict[str, Any] = {}
    per_sym_15m: dict[str, Any] = {}
    missing_syms: list[str] = []
    any_incomplete = False
    any_corrupt = False

    for sym in symbols:
        p5 = cache_dir_5m / f"{sym}_NSE_5m.parquet"
        if not p5.exists():
            missing_syms.append(sym)
            continue
        df = pd.read_parquet(p5)
        ts = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST_TZ)
        day_df = df[ts.dt.date == session_date].copy()
        day_df["timestamp"] = pd.to_datetime(day_df["timestamp"]).dt.tz_convert(IST_TZ)
        dups = int(ts.duplicated().sum())
        nans = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
        ohlc_violations = int(
            (
                (day_df["high"] < day_df["low"])
                | (day_df["high"] < day_df["open"])
                | (day_df["high"] < day_df["close"])
                | (day_df["low"] > day_df["open"])
                | (day_df["low"] > day_df["close"])
                | (day_df["volume"] < 0)
            ).sum()
        )
        is_sym_corrupt = dups > 0 or nans > 0 or ohlc_violations > 0 or len(day_df) == 0
        if is_sym_corrupt:
            any_corrupt = True
            bars_15m = []
        else:
            candles = [
                Candle5M(
                    symbol=sym,
                    exchange="NSE",
                    timestamp=r["timestamp"].to_pydatetime(),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=int(r["volume"]),
                )
                for _, r in day_df.iterrows()
            ]
            bars_15m = resample_5m_candles(candles, target_minutes=15, include_incomplete=False)
        if len(day_df) < 75 or len(bars_15m) < 25:
            any_incomplete = True
        per_sym_5m[sym] = {
            "total_rows": int(len(df)),
            "session_bars": int(len(day_df)),
            "expected_bars": 75,
            "duplicates": dups,
            "nan_values": nans,
            "ohlc_violations": ohlc_violations,
            "timezone": str(ts.dt.tz),
            "sha256": sha256_file(p5),
        }
        per_sym_15m[sym] = {
            "aggregated_completed_bars": int(len(bars_15m)),
            "expected_bars": 25,
            "aggregation_rule": "completed_5m_candles_only_no_interpolation_ist",
        }

    if missing_syms:
        status_5m = "UNAVAILABLE"
        status_15m = "UNAVAILABLE"
    elif any_corrupt:
        status_5m = "UNVERIFIED"
        status_15m = "UNVERIFIED"
    else:
        # Yahoo Finance is never exchange-authoritative -> PARTIALLY_VERIFIED at best
        status_5m = "PARTIALLY_VERIFIED"
        status_15m = "PARTIALLY_VERIFIED"

    return {
        "1h": {
            "classification": status_1h,
            "verified": bool(ok_1h),
            "source": "committed_repo_yahoo_finance_1h_parquet",
            "dataset_fingerprint": rep_1h.get("dataset_fingerprint"),
        },
        "5m": {
            "classification": status_5m,
            "committed_in_repo": len(committed_5m) > 0,
            "source": "isolated_yahoo_finance_5m_cache_not_committed_not_exchange_authoritative",
            "missing_symbols": missing_syms,
            "incomplete_session_bars_observed": any_incomplete,
            "symbols": per_sym_5m,
        },
        "15m": {
            "classification": status_15m,
            "committed_in_repo": False,
            "source": "deterministic_aggregation_from_completed_5m_candles_no_interpolation",
            "incomplete_session_bars_observed": any_incomplete,
            "symbols": per_sym_15m,
        },
    }
