"""Daily 5-minute NSE candle archiver: merges new Yahoo Finance bars into Parquet without losing older history."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.loader import CACHE_DIR, DataUnavailableError, load_historical_5m_candles
from trader.data.schema import DataSourceMode


def merge_and_archive_symbol(symbol: str, days: int = 10) -> int:
    """Fetch recent `days` of 5m candles and merge/deduplicate into data/intraday/{symbol}_NSE_5m.parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_NSE_5m.parquet"

    existing_df = pd.DataFrame()
    if cache_file.exists():
        existing_df = pd.read_parquet(cache_file)

    try:
        new_candles, _ = load_historical_5m_candles(
            symbol=symbol,
            exchange="NSE",
            days=days,
            mode=DataSourceMode.HISTORICAL_REAL,
            use_cache=False,
        )
    except DataUnavailableError as exc:
        print(f"Skipping {symbol}: {exc}")
        return 0

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

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True).dt.tz_convert(IST_TZ)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(cache_file, index=False)
    added = len(combined) - len(existing_df)
    print(f"Archived {symbol}: total={len(combined)} bars (net added={added})")
    return max(0, added)


def main() -> None:
    total_added = 0
    for sym in LAB_DEFAULT_UNIVERSE:
        total_added += merge_and_archive_symbol(sym, days=10)
    print(f"Daily 5m archive complete. Total new bars added: {total_added}")


if __name__ == "__main__":
    main()
