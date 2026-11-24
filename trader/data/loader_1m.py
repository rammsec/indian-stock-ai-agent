"""CYCLE-011: Verified 1-Minute NSE Historical Data Loader & Provenance Schema.

Loads genuine 1-minute NSE equity candles archived from free public sources:
1. Hugging Face `xxparthparekhxx/indian-stock-market-minute-data` (MIT, Upstox NSE 1m candles)
2. Yahoo Finance `1m` API (rolling latest 5-7 NSE trading days)

HARD INVARIANTS:
- Strictly timezone-aware Asia/Kolkata (IST) timestamps.
- Strictly NSE regular trading hours [09:15:00, 15:30:00) IST, Monday-Friday.
- Zero synthetic bar fabrication or forward-fill gap hallucination.
- Deterministic OHLC sanity validation (`high >= max(open, close)`, `low <= min(open, close)`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ..config import IST_TZ
from .schema import Candle5M

DEFAULT_1M_DIR: Final[Path] = Path("data/intraday_1m")
DEFAULT_SYMBOLS_1M: Final[tuple[str, ...]] = ("TCS", "RELIANCE", "ITC", "TATAELXSI", "BIRLACABLE")
NSE_OPEN_TIME: Final[time] = time(9, 15, 0)
NSE_CLOSE_TIME: Final[time] = time(15, 30, 0)


@dataclass(frozen=True)
class OneMinuteSessionBundle:
    """One NSE trading day of genuine 1m bars plus prior causal warmup bars."""

    symbol: str
    trading_date: date
    warmup_bars: list[Candle5M]
    session_bars: list[Candle5M]
    source_providers: tuple[str, ...]


def sha256_file_1m(path: Path) -> str:
    """Compute SHA-256 hex digest for a 1m Parquet archive."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_1m_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Clean, validate, deduplicate, and IST-normalize a 1m OHLCV DataFrame.

    Drops out-of-session timestamps, weekends, non-positive prices, and invalid OHLC rows.
    Never interpolates or fabricates missing minutes.
    """
    required = {"timestamp_ist", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"1m DataFrame missing required columns: {sorted(missing)}")
    if df.empty:
        return pd.DataFrame(
            columns=["symbol", "timestamp_ist", "ts_close_ist", "open", "high", "low", "close", "volume", "source"]
        )

    out = df.copy()
    ts = pd.to_datetime(out["timestamp_ist"], utc=True).dt.tz_convert(IST_TZ)
    out["timestamp_ist"] = ts
    out["symbol"] = symbol.strip().upper()
    if "source" not in out.columns:
        out["source"] = "unknown"

    # Strict NSE regular session filter [09:15, 15:30) IST, weekdays only
    t_col = out["timestamp_ist"].dt.time
    wd_col = out["timestamp_ist"].dt.day_of_week
    valid_session = (wd_col < 5) & (t_col >= NSE_OPEN_TIME) & (t_col < NSE_CLOSE_TIME)
    out = out.loc[valid_session].copy()

    # Numeric & OHLC invariants
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    valid_ohlc = (
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["volume"] >= 0)
        & (out["high"] >= out[["open", "close"]].max(axis=1))
        & (out["low"] <= out[["open", "close"]].min(axis=1))
    )
    out = out.loc[valid_ohlc].copy()

    # Deduplicate by timestamp (preferring latest source entry) and sort chronologically
    out = out.sort_values("timestamp_ist").drop_duplicates(subset=["timestamp_ist"], keep="last").reset_index(drop=True)
    out["ts_close_ist"] = out["timestamp_ist"] + pd.Timedelta(minutes=1)
    return out[["symbol", "timestamp_ist", "ts_close_ist", "open", "high", "low", "close", "volume", "source"]]


def load_1m_parquet(symbol: str, base_dir: str | Path = DEFAULT_1M_DIR) -> pd.DataFrame:
    """Load and validate a symbol's 1m Parquet archive."""
    sym = symbol.strip().upper()
    path = Path(base_dir) / f"{sym}_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"1m Parquet archive not found for {sym}: {path}")
    raw = pd.read_parquet(path)
    return normalize_1m_dataframe(raw, sym)


def load_1m_sessions(
    symbol: str,
    *,
    base_dir: str | Path = DEFAULT_1M_DIR,
    min_bars_per_day: int = 300,
    warmup_bars_count: int = 50,
    max_days: int | None = None,
) -> list[OneMinuteSessionBundle]:
    """Split a symbol's 1m archive into chronological trading sessions with causal warmup."""
    df = load_1m_parquet(symbol, base_dir=base_dir)
    if df.empty:
        return []

    sym = symbol.strip().upper()
    df["trade_date"] = df["timestamp_ist"].dt.date
    all_dates = sorted(df["trade_date"].unique())
    if max_days is not None and max_days > 0:
        selected_dates = set(all_dates[-max_days:])
    else:
        selected_dates = set(all_dates)

    # Convert all rows once to Candle5M-compatible objects (generic OHLCV container)
    candles_by_date: dict[date, list[Candle5M]] = {}
    sources_by_date: dict[date, set[str]] = {}
    for row in df.to_dict(orient="records"):
        ts_open = pd.Timestamp(row["timestamp_ist"]).to_pydatetime().astimezone(IST_TZ)
        d = ts_open.date()
        candles_by_date.setdefault(d, []).append(
            Candle5M(
                symbol=sym,
                exchange="NSE",
                timestamp=ts_open,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
        sources_by_date.setdefault(d, set()).add(str(row.get("source", "unknown")))

    bundles: list[OneMinuteSessionBundle] = []
    prior_bars: list[Candle5M] = []
    for d in all_dates:
        day_bars = candles_by_date.get(d, [])
        if d in selected_dates and len(day_bars) >= min_bars_per_day:
            bundles.append(
                OneMinuteSessionBundle(
                    symbol=sym,
                    trading_date=d,
                    warmup_bars=list(prior_bars[-warmup_bars_count:]) if warmup_bars_count > 0 else [],
                    session_bars=day_bars,
                    source_providers=tuple(sorted(sources_by_date.get(d, set()))),
                )
            )
        prior_bars.extend(day_bars)
        if len(prior_bars) > max(warmup_bars_count * 4, 400):
            prior_bars = prior_bars[-max(warmup_bars_count * 2, 200) :]

    return bundles


def write_1m_provenance(
    output_dir: Path,
    symbol_stats: dict[str, dict[str, Any]],
) -> Path:
    """Write SHA-256 provenance manifest for all 1m Parquet files."""
    manifest = {
        "schema_version": "cycle011_1m_provenance_v1",
        "generated_at_ist": datetime.now(IST_TZ).isoformat(),
        "live_trading_enabled": False,
        "synthetic_substitution": False,
        "sources": [
            "huggingface:xxparthparekhxx/indian-stock-market-minute-data (MIT, Upstox NSE 1m)",
            "yahoo_finance:v8/chart?interval=1m&range=5d",
        ],
        "symbols": symbol_stats,
    }
    prov_path = output_dir / "provenance_1m.json"
    prov_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prov_path
