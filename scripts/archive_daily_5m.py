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
    latest_ts = combined["timestamp"].max().isoformat() if not combined.empty else None
    print(f"Archived {symbol}: total={len(combined)} bars (net added={added}, dups_removed={dups_removed})")
    return {
        "symbol": symbol,
        "status": "OK",
        "provider": prov.source_name,
        "retrieved_at_ist": prov.ingested_at.isoformat(),
        "total_bars": len(combined),
        "net_added": added,
        "duplicates_removed": dups_removed,
        "latest_bar_ist": latest_ts,
    }


def main() -> dict[str, Any]:
    symbol_reports = []
    failed_symbols = []
    total_added = 0

    for sym in LAB_DEFAULT_UNIVERSE:
        rep = merge_and_archive_symbol(sym, days=10)
        symbol_reports.append(rep)
        total_added += rep["net_added"]
        if rep["status"] != "OK":
            failed_symbols.append(sym)

    latest_dates = [r.get("latest_bar_ist") for r in symbol_reports if r.get("latest_bar_ist")]
    health = {
        "last_successful_run_ist": datetime.now(tz=IST_TZ).isoformat(),
        "latest_trading_timestamp_ist": max(latest_dates) if latest_dates else None,
        "total_new_bars_added": total_added,
        "failed_symbols": failed_symbols,
        "provider_status": "HEALTHY" if not failed_symbols else "DEGRADED",
        "symbols": symbol_reports,
    }
    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_REPORT_PATH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    print(f"Daily 5m archive complete. Total new bars added: {total_added}. Health report saved to {HEALTH_REPORT_PATH}")
    return health


if __name__ == "__main__":
    main()
