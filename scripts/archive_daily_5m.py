"""Daily 5-minute NSE candle archiver with completeness protection and health reporting."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.loader import CACHE_DIR, DataUnavailableError, load_historical_5m_candles
from trader.data.schema import DataSourceMode

HEALTH_REPORT_PATH = Path("data/lab/daily_archive_health.json")


def merge_and_archive_symbol(symbol: str, days: int = 10) -> dict[str, Any]:
    """Fetch recent `days` of 5m candles and merge into Parquet without overwriting more-complete sessions."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_NSE_5m.parquet"

    existing_df = pd.DataFrame()
    if cache_file.exists():
        existing_df = pd.read_parquet(cache_file)
        if not existing_df.empty:
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True).dt.tz_convert(IST_TZ)

    try:
        new_candles, prov = load_historical_5m_candles(
            symbol=symbol,
            exchange="NSE",
            days=days,
            mode=DataSourceMode.HISTORICAL_REAL,
            use_cache=False,
        )
    except DataUnavailableError as exc:
        print(f"Skipping {symbol}: {exc}")
        return {
            "symbol": symbol,
            "status": "FAILED",
            "error": str(exc),
            "total_bars": len(existing_df),
            "net_added": 0,
            "duplicates_removed": 0,
        }

    new_df = pd.DataFrame([
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in new_candles
    ])
    if not new_df.empty:
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True).dt.tz_convert(IST_TZ)

    if not existing_df.empty and not new_df.empty:
        # Per-date completeness protection: never overwrite a more-complete session with an incomplete fetch
        existing_counts = existing_df.groupby(existing_df["timestamp"].dt.date).size().to_dict()
        new_counts = new_df.groupby(new_df["timestamp"].dt.date).size().to_dict()
        protected_dates = {d for d, cnt in existing_counts.items() if cnt >= new_counts.get(d, 0)}

        filtered_new = new_df[~new_df["timestamp"].dt.date.isin(protected_dates)]
        raw_total = len(existing_df) + len(filtered_new)
        combined = pd.concat([existing_df, filtered_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
        dups_removed = raw_total - len(combined)
    elif not existing_df.empty:
        combined = existing_df
        dups_removed = 0
    else:
        combined = new_df
        dups_removed = 0

    combined.to_parquet(cache_file, index=False)
    added = max(0, len(combined) - len(existing_df))
    existing_sessions = int(existing_df["timestamp"].dt.date.nunique()) if not existing_df.empty else 0
    combined_sessions = int(combined["timestamp"].dt.date.nunique()) if not combined.empty else 0
    new_sessions_added = max(0, combined_sessions - existing_sessions)

    missing_intervals = 0
    latest_session_bars = 0
    if not combined.empty:
        per_day_counts = combined.groupby(combined["timestamp"].dt.date).size()
        missing_intervals = int(sum(max(0, 75 - int(c)) for c in per_day_counts))
        latest_session_bars = int(per_day_counts.iloc[-1])

    latest_ts = combined["timestamp"].max().isoformat() if not combined.empty else None
    print(f"Archived {symbol}: total={len(combined)} bars (net added={added}, new_sessions={new_sessions_added}, dups_removed={dups_removed})")
    return {
        "symbol": symbol,
        "status": "OK",
        "provider": prov.source_name,
        "retrieved_at_ist": prov.ingested_at.isoformat(),
        "total_bars": len(combined),
        "sessions_count": combined_sessions,
        "new_sessions_added": new_sessions_added,
        "net_added": added,
        "missing_interval_count": missing_intervals,
        "latest_session_bars": latest_session_bars,
        "duplicates_removed": dups_removed,
        "latest_bar_ist": latest_ts,
    }


def classify_archive_status(symbol_reports: list[dict[str, Any]], total_added: int) -> str:
    """Classify archive run into SUCCESS_NEW_DATA, SUCCESS_NO_NEW_DATA, PARTIAL_RETRIEVAL, or FAILED_RETRIEVAL."""
    ok_count = sum(1 for r in symbol_reports if r.get("status") == "OK")
    if ok_count == 0:
        return "FAILED_RETRIEVAL"
    if ok_count < len(symbol_reports):
        return "PARTIAL_RETRIEVAL"
    if total_added > 0:
        return "SUCCESS_NEW_DATA"
    return "SUCCESS_NO_NEW_DATA"


def main() -> dict[str, Any]:
    symbol_reports = []
    failed_symbols = []
    total_added = 0
    total_new_sessions = 0
    total_missing_intervals = 0

    for sym in LAB_DEFAULT_UNIVERSE:
        rep = merge_and_archive_symbol(sym, days=10)
        symbol_reports.append(rep)
        total_added += rep.get("net_added", 0)
        total_new_sessions += rep.get("new_sessions_added", 0)
        total_missing_intervals += rep.get("missing_interval_count", 0)
        if rep["status"] != "OK":
            failed_symbols.append(sym)

    archive_status = classify_archive_status(symbol_reports, total_added)
    prev_health = json.loads(HEALTH_REPORT_PATH.read_text(encoding="utf-8")) if HEALTH_REPORT_PATH.exists() else {}
    now_ist = datetime.now(tz=IST_TZ)
    latest_dates = [r.get("latest_bar_ist") for r in symbol_reports if r.get("latest_bar_ist")]
    latest_ts_str = max(latest_dates) if latest_dates else None

    freshness_status = "UNKNOWN"
    if latest_ts_str:
        latest_dt = datetime.fromisoformat(latest_ts_str)
        age_days = (now_ist.date() - latest_dt.date()).days
        freshness_status = "FRESH" if age_days <= 3 else ("STALE" if age_days <= 7 else "OUTDATED")

    per_symbol_coverage = {
        r["symbol"]: {
            "sessions": r.get("sessions_count", 0),
            "total_bars": r.get("total_bars", 0),
            "missing_intervals": r.get("missing_interval_count", 0),
            "latest_session_bars": r.get("latest_session_bars", 0),
            "status": r.get("status", "UNKNOWN"),
        }
        for r in symbol_reports
    }

    health = {
        "last_run_ist": now_ist.isoformat(),
        "last_successful_run_ist": (
            now_ist.isoformat() if archive_status != "FAILED_RETRIEVAL" else prev_health.get("last_successful_run_ist", "N/A")
        ),
        "archive_status": archive_status,
        "archive_freshness_status": freshness_status,
        "latest_trading_timestamp_ist": latest_ts_str,
        "total_new_bars_added": total_added,
        "new_sessions_added": total_new_sessions,
        "missing_interval_count": total_missing_intervals,
        "failed_symbols": failed_symbols,
        "provider_status": "HEALTHY" if not failed_symbols else ("DEGRADED" if len(failed_symbols) < len(LAB_DEFAULT_UNIVERSE) else "DOWN"),
        "per_symbol_coverage": per_symbol_coverage,
        "symbols": symbol_reports,
    }
    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_REPORT_PATH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    print(f"Daily 5m archive complete [{archive_status}]. Total new bars={total_added}, new sessions={total_new_sessions}.")
    return health


if __name__ == "__main__":
    main()
