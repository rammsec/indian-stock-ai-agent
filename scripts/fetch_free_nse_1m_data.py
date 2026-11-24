#!/usr/bin/env python3
"""CYCLE-011: Free Genuine 1-Minute NSE Data Fetcher (Hugging Face + Yahoo Finance).

Downloads genuine 1-minute NSE OHLCV candles with zero API keys and zero synthetic fallback:
1. Historical 1m candles from Hugging Face `xxparthparekhxx/indian-stock-market-minute-data`
   (MIT license, Upstox NSE historical candles) via DuckDB HTTP range requests.
2. Latest rolling 5 trading days of 1m candles from Yahoo Finance (`v8/finance/chart`).

Saves clean, IST-normalized Parquet files under `data/intraday_1m/{SYM}_1m.parquet`
plus SHA-256 `data/intraday_1m/provenance_1m.json`.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from trader.config import IST_TZ
from trader.data.loader_1m import (
    DEFAULT_1M_DIR,
    DEFAULT_SYMBOLS_1M,
    normalize_1m_dataframe,
    sha256_file_1m,
    write_1m_provenance,
)

HF_MINUTE_GLOB = "hf://datasets/xxparthparekhxx/indian-stock-market-minute-data/minute/*.parquet"


def fetch_hf_1m_for_symbols(
    symbols: list[str],
    start_date: str = "2026-08-01",
) -> dict[str, pd.DataFrame]:
    """Stream genuine 1m NSE candles from Hugging Face via DuckDB Parquet row-group pruning."""
    con = duckdb.connect()
    con.sql("SET TimeZone='Asia/Kolkata'")
    sym_sql = ", ".join(f"'{s}'" for s in symbols)
    query = f"""
        SELECT
            symbol,
            timestamp AS timestamp_ist,
            CAST(open AS DOUBLE) AS open,
            CAST(high AS DOUBLE) AS high,
            CAST(low AS DOUBLE) AS low,
            CAST(close AS DOUBLE) AS close,
            CAST(volume AS DOUBLE) AS volume
        FROM '{HF_MINUTE_GLOB}'
        WHERE symbol IN ({sym_sql})
          AND timestamp >= TIMESTAMPTZ '{start_date} 09:00:00+05:30'
        ORDER BY symbol, timestamp
    """
    df_all = con.sql(query).df()
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sub = df_all.loc[df_all["symbol"] == sym].copy()
        sub["source"] = "huggingface_upstox_1m"
        out[sym] = sub
    return out


def fetch_yahoo_1m_for_symbol(symbol: str, range_str: str = "5d") -> pd.DataFrame:
    """Fetch latest rolling 5-7 days of genuine 1m NSE candles from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1m&range={range_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    rows: list[dict[str, Any]] = []
    for i, ts_sec in enumerate(timestamps):
        if i >= len(opens) or opens[i] is None or closes[i] is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "timestamp_ist": datetime.fromtimestamp(int(ts_sec), tz=IST_TZ),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i] or 0.0),
                "source": "yahoo_finance_1m",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "timestamp_ist", "open", "high", "low", "close", "volume", "source"])
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch free genuine 1m NSE candles (Hugging Face + Yahoo Finance).")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS_1M))
    parser.add_argument("--start-date", default="2026-08-01", help="Start date (YYYY-MM-DD) for Hugging Face history.")
    parser.add_argument("--output-dir", default=str(DEFAULT_1M_DIR))
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face history and fetch Yahoo 5d only.")
    parser.add_argument("--skip-yahoo", action="store_true", help="Skip Yahoo latest 5d and fetch Hugging Face only.")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_map: dict[str, pd.DataFrame] = {}
    if not args.skip_hf:
        print(f"[fetch_1m] Querying Hugging Face 1m Parquet for {symbols} from {args.start_date}...")
        hf_map = fetch_hf_1m_for_symbols(symbols, start_date=args.start_date)

    symbol_stats: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        frames: list[pd.DataFrame] = []
        if sym in hf_map and not hf_map[sym].empty:
            frames.append(hf_map[sym])
        if not args.skip_yahoo:
            try:
                ydf = fetch_yahoo_1m_for_symbol(sym, range_str="5d")
                if not ydf.empty:
                    frames.append(ydf)
            except Exception as exc:
                print(f"[fetch_1m] WARN Yahoo 1m failed for {sym}: {exc}")

        if not frames:
            print(f"[fetch_1m] WARN No 1m data fetched for {sym}")
            continue

        merged = pd.concat(frames, ignore_index=True)
        clean = normalize_1m_dataframe(merged, sym)
        out_path = out_dir / f"{sym}_1m.parquet"
        clean.to_parquet(out_path, index=False)

        unique_days = sorted(clean["timestamp_ist"].dt.date.unique())
        symbol_stats[sym] = {
            "parquet_file": out_path.name,
            "sha256": sha256_file_1m(out_path),
            "bars_count": int(len(clean)),
            "trading_days": int(len(unique_days)),
            "start_ist": clean["timestamp_ist"].iloc[0].isoformat(),
            "end_ist": clean["timestamp_ist"].iloc[-1].isoformat(),
            "sources": sorted(clean["source"].unique().tolist()),
        }
        print(
            f"[fetch_1m] {sym}: {len(clean):,} bars across {len(unique_days)} days "
            f"({unique_days[0]} -> {unique_days[-1]}), {out_path.stat().st_size:,} bytes"
        )

    prov_path = write_1m_provenance(out_dir, symbol_stats)
    print(f"[fetch_1m] Wrote provenance manifest: {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
