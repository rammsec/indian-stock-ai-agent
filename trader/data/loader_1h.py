"""1-Hour (60m) Historical NSE OHLCV Data Loader & Parquet Cache (Up to 1–2 Years)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from ..config import IST_TZ, LAB_DEFAULT_UNIVERSE, is_nse_trading_day
from ..intelligence.parsing import compute_candle_series_hash
from .loader import CACHE_DIR
from .loader_1m import parse_tz_aware_ist
from .schema import Candle5M, DataProvenance, DataSourceMode


def fetch_and_cache_1h_universe(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    period: str = "1y",
) -> dict[str, Any]:
    """Fetch 1h NSE candles from Yahoo Finance and merge safely into {SYM}_1h.parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for sym in symbols:
        yf_sym = sym if sym.endswith((".NS", ".BO")) else f"{sym}.NS"
        cache_file = CACHE_DIR / f"{sym}_1h.parquet"
        existing_df = pd.read_parquet(cache_file) if cache_file.exists() else pd.DataFrame()
        if not existing_df.empty:
            existing_df["timestamp"] = parse_tz_aware_ist(existing_df["timestamp"], label="timestamp")

        df = yf.Ticker(yf_sym).history(period=period, interval="1h", auto_adjust=False)
        if df is not None and not df.empty:
            df = df.reset_index()
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["timestamp"] = parse_tz_aware_ist(df[ts_col], label=ts_col)
            new_df = pd.DataFrame({
                "timestamp": df["timestamp"],
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": df["Volume"].fillna(0).astype(int),
            })
            # Keep only regular NSE market hours (09:15..15:15 IST)
            hhmm = new_df["timestamp"].dt.strftime("%H:%M")
            new_df = new_df[(hhmm >= "09:15") & (hhmm <= "15:15")].copy()
        else:
            new_df = pd.DataFrame()

        if not existing_df.empty and not new_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
        elif not existing_df.empty:
            combined = existing_df
        else:
            combined = new_df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)

        if not combined.empty:
            combined.to_parquet(cache_file, index=False)

        sessions = int(combined["timestamp"].dt.date.nunique()) if not combined.empty else 0
        summary[sym] = {
            "total_bars": len(combined),
            "sessions": sessions,
            "start_date": combined["timestamp"].dt.date.min().isoformat() if not combined.empty else None,
            "end_date": combined["timestamp"].dt.date.max().isoformat() if not combined.empty else None,
            "cache_file": str(cache_file),
        }

    return summary


def get_available_1h_dates(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    min_bars_per_stock: int = 6,
) -> tuple[list[date], list[dict[str, Any]]]:
    """Return sorted eligible dates (all symbols >= min_bars_per_stock) and excluded dates."""
    date_counts: dict[date, dict[str, int]] = {}
    all_dates: set[date] = set()

    for sym in symbols:
        cache_file = CACHE_DIR / f"{sym}_1h.parquet"
        if not cache_file.exists():
            continue
        df = pd.read_parquet(cache_file, columns=["timestamp"])
        df["timestamp"] = parse_tz_aware_ist(df["timestamp"], label="timestamp")
        counts = df.groupby(df["timestamp"].dt.date).size().to_dict()
        for d, cnt in counts.items():
            if is_nse_trading_day(d):
                all_dates.add(d)
                date_counts.setdefault(d, {})[sym] = int(cnt)

    eligible: list[date] = []
    excluded: list[dict[str, Any]] = []
    for d in sorted(all_dates):
        sym_map = date_counts.get(d, {})
        missing_syms = [s for s in symbols if sym_map.get(s, 0) < min_bars_per_stock]
        if not missing_syms:
            eligible.append(d)
        else:
            excluded.append({
                "date": d.isoformat(),
                "reason": "; ".join(f"{s}: {sym_map.get(s, 0)}/7 1h bars" for s in missing_syms),
            })
    return eligible, excluded


def load_1h_session_with_warmup(
    symbol: str,
    target_date: date,
    warmup_days: int = 10,
) -> tuple[list[Candle5M], list[Candle5M], DataProvenance]:
    """Load cached 1h candles for target_date and prior warmup_days (strictly <= target_date)."""
    cache_file = CACHE_DIR / f"{symbol}_1h.parquet"
    if not cache_file.exists():
        now_ts = datetime.now(tz=IST_TZ)
        return [], [], DataProvenance(
            source_mode=DataSourceMode.HISTORICAL_REAL,
            source_name="UNAVAILABLE",
            symbol=symbol,
            exchange="NSE",
            ingested_at=now_ts,
            bars_count=0,
            start_timestamp=now_ts,
            end_timestamp=now_ts,
            sha256_hash="UNAVAILABLE",
        )

    df = pd.read_parquet(cache_file)
    df["timestamp"] = parse_tz_aware_ist(df["timestamp"], label="timestamp")
    df["d"] = df["timestamp"].dt.date
    warmup_start = target_date - timedelta(days=warmup_days * 2)
    df_win = df[(df["d"] >= warmup_start) & (df["d"] <= target_date)].sort_values("timestamp")

    w_bars: list[Candle5M] = []
    s_bars: list[Candle5M] = []
    for row in df_win.itertuples(index=False):
        c = Candle5M(
            symbol=symbol,
            exchange="NSE",
            timestamp=row.timestamp.to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume),
        )
        if row.d < target_date:
            w_bars.append(c)
        elif row.d == target_date:
            s_bars.append(c)

    # Keep at most warmup_days unique dates
    if w_bars:
        unique_w_dates = sorted({b.timestamp.date() for b in w_bars})[-warmup_days:]
        w_set = set(unique_w_dates)
        w_bars = [b for b in w_bars if b.timestamp.date() in w_set]

    sha = compute_candle_series_hash(s_bars) if s_bars else "EMPTY"
    now_ts = datetime.now(tz=IST_TZ)
    prov = DataProvenance(
        source_mode=DataSourceMode.HISTORICAL_REAL,
        source_name="yahoo_finance_1h",
        symbol=symbol,
        exchange="NSE",
        ingested_at=now_ts,
        bars_count=len(s_bars),
        start_timestamp=s_bars[0].timestamp if s_bars else now_ts,
        end_timestamp=s_bars[-1].timestamp if s_bars else now_ts,
        sha256_hash=sha,
    )
    return w_bars, s_bars, prov
