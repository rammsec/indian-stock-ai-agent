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


def get_expected_latest_nse_session(now_ist: datetime) -> date:
    """Return the most recent NSE trading session expected to be complete as of now_ist."""
    from datetime import timedelta
    from trader.config import is_nse_trading_day

    d = now_ist.date()
    # If today is a trading day but market hasn't closed + settled (15:35 IST), latest completed is previous trading day
    if is_nse_trading_day(d) and (now_ist.hour, now_ist.minute) >= (15, 35):
        return d
    curr = d - timedelta(days=1)
    while not is_nse_trading_day(curr):
        curr -= timedelta(days=1)
    return curr


def classify_archive_status(symbol_reports: list[dict[str, Any]], total_added: int) -> str:
    """Legacy 4-state helper retained for backwards compatibility."""
    ok_count = sum(1 for r in symbol_reports if r.get("status") == "OK")
    if ok_count == 0:
        return "FAILED_RETRIEVAL"
    if ok_count < len(symbol_reports):
        return "PARTIAL_RETRIEVAL"
    if total_added > 0:
        return "SUCCESS_NEW_DATA"
    return "SUCCESS_NO_NEW_DATA"


def evaluate_archive_health(
    symbol_reports: list[dict[str, Any]],
    total_added: int,
    now_ist: datetime,
    prev_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Phase 11.2 archive classifier supporting all 7 required statuses & NSE calendar checks."""
    from datetime import timedelta
    from trader.config import is_nse_trading_day

    prev_health = prev_health or {}
    ok_count = sum(1 for r in symbol_reports if r.get("status") == "OK")
    validation_errors = [r["symbol"] for r in symbol_reports if r.get("status") == "VALIDATION_FAILED"]
    failed_symbols = [r["symbol"] for r in symbol_reports if r.get("status") != "OK"]

    latest_dates = [r.get("latest_bar_ist") for r in symbol_reports if r.get("latest_bar_ist")]
    latest_ts_str = max(latest_dates) if latest_dates else prev_health.get("latest_trading_timestamp_ist")
    expected_session = get_expected_latest_nse_session(now_ist)

    latest_market_date: date | None = None
    if latest_ts_str:
        latest_market_date = datetime.fromisoformat(latest_ts_str).date()

    # Count missed NSE trading sessions between latest_market_date and expected_session
    missed_trading_days = 0
    if latest_market_date is not None and latest_market_date < expected_session:
        curr = latest_market_date + timedelta(days=1)
        while curr <= expected_session:
            if is_nse_trading_day(curr):
                missed_trading_days += 1
            curr += timedelta(days=1)

    market_closed_now = (not is_nse_trading_day(now_ist.date())) or (now_ist.hour, now_ist.minute) < (15, 35)

    if validation_errors:
        archive_status = "FAILED_VALIDATION"
        freshness_status = "INVALID"
    elif ok_count == 0:
        archive_status = "FAILED_PROVIDER"
        freshness_status = "STALE" if missed_trading_days > 2 else prev_health.get("archive_freshness_status", "UNKNOWN")
    elif ok_count < len(symbol_reports):
        archive_status = "WARNING_EXPECTED_DATA_MISSING"
        freshness_status = "PARTIAL"
    elif latest_market_date is None or missed_trading_days > 2:
        archive_status = "STALE_ARCHIVE"
        freshness_status = "STALE"
    elif missed_trading_days > 0:
        archive_status = "WARNING_EXPECTED_DATA_MISSING"
        freshness_status = "INCOMPLETE"
    elif total_added > 0:
        archive_status = "SUCCESS_NEW_DATA"
        freshness_status = "FRESH"
    elif market_closed_now:
        archive_status = "SUCCESS_MARKET_CLOSED"
        freshness_status = "FRESH"
    else:
        archive_status = "SUCCESS_NO_NEW_DATA"
        freshness_status = "FRESH"

    total_new_sessions = sum(r.get("new_sessions_added", 0) for r in symbol_reports)
    total_missing_intervals = sum(r.get("missing_interval_count", 0) for r in symbol_reports)

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

    is_success = archive_status in ("SUCCESS_NEW_DATA", "SUCCESS_MARKET_CLOSED", "SUCCESS_NO_NEW_DATA")
    return {
        "last_run_ist": now_ist.isoformat(),
        "last_successful_run_ist": (
            now_ist.isoformat() if is_success else prev_health.get("last_successful_run_ist", "N/A")
        ),
        "archive_status": archive_status,
        "archive_freshness_status": freshness_status,
        "expected_latest_session": expected_session.isoformat(),
        "missed_trading_days": missed_trading_days,
        "latest_trading_timestamp_ist": latest_ts_str,
        "total_new_bars_added": total_added,
        "new_sessions_added": total_new_sessions,
        "missing_interval_count": total_missing_intervals,
        "failed_symbols": failed_symbols,
        "provider_status": "HEALTHY" if not failed_symbols else ("DEGRADED" if ok_count > 0 else "DOWN"),
        "per_symbol_coverage": per_symbol_coverage,
        "symbols": symbol_reports,
    }


def main() -> dict[str, Any]:
    symbol_reports = []
    total_added = 0

    for sym in LAB_DEFAULT_UNIVERSE:
        rep = merge_and_archive_symbol(sym, days=10)
        symbol_reports.append(rep)
        total_added += rep.get("net_added", 0)

    prev_health = json.loads(HEALTH_REPORT_PATH.read_text(encoding="utf-8")) if HEALTH_REPORT_PATH.exists() else {}
    now_ist = datetime.now(tz=IST_TZ)
    health = evaluate_archive_health(symbol_reports, total_added, now_ist, prev_health=prev_health)

    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_REPORT_PATH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    print(f"Daily 5m archive complete [{health['archive_status']}]. Total new bars={total_added}, missed_days={health['missed_trading_days']}.")
    return health


if __name__ == "__main__":
    main()
