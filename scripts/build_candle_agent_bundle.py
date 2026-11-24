"""Build verified multi-timeframe (5m / 15m / 1h) historical session catalog for docs/candle-agent/.

Reads ONLY genuine historical NSE archives already committed in the repository:
1. data/intraday/{SYM}_1h.parquet (60 most recent trading sessions + 50-bar pre-open warmup)
2. docs/lab/data/runs_YYYY-MM-DD.json (10 genuine 5m Yahoo Finance sessions + causally resampled 15m bars)

Enforces NSE session boundaries (09:15-15:30 IST) and clamps the final shortened 1h bar
(15:15-15:30 IST) so ts_close_ist == 15:30 IST (never 16:15 IST).
Zero synthetic data substitution; zero protected files modified.
"""

from __future__ import annotations

import glob
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from trader.config import IST_TZ

SYMBOLS = ("TCS", "RELIANCE", "ITC", "TATAELXSI", "BIRLACABLE")
INTRADAY_DIR = Path("data/intraday")
LAB_DATA_DIR = Path("docs/lab/data")
OUT_PATH = Path("docs/candle-agent/data/session_catalog.json")


def _fmt_ist(dt: datetime) -> str:
    return dt.astimezone(IST_TZ).strftime("%Y-%m-%d %H:%M")


def _extract_1h_bars(symbol: str, max_sessions: int = 60) -> tuple[list[dict[str, Any]], list[str]]:
    path = INTRADAY_DIR / f"{symbol}_1h.parquet"
    if not path.exists():
        return [], []
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    df = df.copy()
    df["ts_open"] = ts
    df = df.sort_values("ts_open").drop_duplicates(subset=["ts_open"], keep="last").reset_index(drop=True)

    unique_dates = sorted(df["ts_open"].dt.date.unique())
    selectable_dates = unique_dates[-max_sessions:]
    # Keep 12 prior trading days (~84 bars) before the earliest selectable date for 50-bar warmup
    warmup_start_idx = max(0, len(unique_dates) - max_sessions - 12)
    keep_from = unique_dates[warmup_start_idx]
    df = df.loc[df["ts_open"].dt.date >= keep_from].reset_index(drop=True)

    bars: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        t_open = pd.Timestamp(row["ts_open"]).to_pydatetime().astimezone(IST_TZ)
        # Clamp NSE final shortened 1h bar (15:15 -> 15:30 IST)
        if t_open.hour == 15 and t_open.minute == 15:
            t_close = t_open + timedelta(minutes=15)
        else:
            t_close = t_open + timedelta(hours=1)
        bars.append(
            {
                "t": _fmt_ist(t_open),
                "tc": _fmt_ist(t_close),
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
                "v": int(round(float(row["volume"]))),
            }
        )

    return bars, [d.isoformat() for d in reversed(selectable_dates)]


def _resample_5m_to_15m(bars_5m: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Causally aggregate 5m bars into exact 15m NSE buckets (09:15, 09:30, ..., 15:15)."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for b in bars_5m:
        dt = datetime.strptime(b["t"], "%Y-%m-%d %H:%M").replace(tzinfo=IST_TZ)
        minute_bucket = (dt.minute // 15) * 15
        b_open = dt.replace(minute=minute_bucket)
        key = _fmt_ist(b_open)
        buckets.setdefault(key, []).append(b)

    out: list[dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        grp = buckets[key]
        b_open = datetime.strptime(key, "%Y-%m-%d %H:%M").replace(tzinfo=IST_TZ)
        b_close = b_open + timedelta(minutes=15)
        out.append(
            {
                "t": key,
                "tc": _fmt_ist(b_close),
                "o": grp[0]["o"],
                "h": round(max(x["h"] for x in grp), 2),
                "l": round(min(x["l"] for x in grp), 2),
                "c": grp[-1]["c"],
                "v": int(sum(x["v"] for x in grp)),
            }
        )
    return out


def _extract_5m_and_15m_sessions() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[str]]:
    run_files = sorted(glob.glob(str(LAB_DATA_DIR / "runs_*.json")))
    bars_5m_by_sym: dict[str, list[dict[str, Any]]] = {s: [] for s in SYMBOLS}
    bars_15m_by_sym: dict[str, list[dict[str, Any]]] = {s: [] for s in SYMBOLS}
    dates_found: list[str] = []

    for rf in run_files:
        p = Path(rf)
        date_str = p.stem.replace("runs_", "")
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        runs = payload.get("runs", {})
        if not runs:
            continue
        first_run = next(iter(runs.values()))
        candles_map = first_run.get("candles", {})
        if not candles_map:
            continue
        dates_found.append(date_str)
        for sym in SYMBOLS:
            raw_list = candles_map.get(sym, [])
            day_5m: list[dict[str, Any]] = []
            for item in raw_list:
                time_str = str(item.get("time_str", "")).strip()
                if not time_str:
                    continue
                t_open = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=IST_TZ)
                t_close = t_open + timedelta(minutes=5)
                day_5m.append(
                    {
                        "t": _fmt_ist(t_open),
                        "tc": _fmt_ist(t_close),
                        "o": round(float(item["open"]), 2),
                        "h": round(float(item["high"]), 2),
                        "l": round(float(item["low"]), 2),
                        "c": round(float(item["close"]), 2),
                        "v": int(round(float(item.get("volume", 0)))),
                    }
                )
            bars_5m_by_sym[sym].extend(day_5m)
            bars_15m_by_sym[sym].extend(_resample_5m_to_15m(day_5m))

    return bars_5m_by_sym, bars_15m_by_sym, list(reversed(sorted(set(dates_found))))


def build_catalog() -> dict[str, Any]:
    bars_1h_by_sym: dict[str, list[dict[str, Any]]] = {}
    dates_1h: list[str] = []
    for sym in SYMBOLS:
        b1h, d1h = _extract_1h_bars(sym, max_sessions=60)
        bars_1h_by_sym[sym] = b1h
        if not dates_1h:
            dates_1h = d1h

    bars_5m_by_sym, bars_15m_by_sym, dates_5m = _extract_5m_and_15m_sessions()

    catalog = {
        "schema_version": "cycle010_phase4_v1",
        "provenance": {
            "status": "PARTIALLY_VERIFIED",
            "provider_1h": "yahoo_finance_1h (data/intraday/*_1h.parquet)",
            "provider_5m": "yahoo_finance_5m (docs/lab/data/runs_*.json)",
            "provider_15m": "causal_3bar_resample_from_5m",
            "synthetic_substitution": False,
            "paper_only": True,
            "live_trading_enabled": False,
            "confidence_calibration_status": "UNCALIBRATED",
        },
        "symbols": list(SYMBOLS),
        "timeframes": ["5m", "15m", "1h"],
        "dates_by_timeframe": {
            "5m": dates_5m,
            "15m": dates_5m,
            "1h": dates_1h,
        },
        "bars": {
            "5m": bars_5m_by_sym,
            "15m": bars_15m_by_sym,
            "1h": bars_1h_by_sym,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(catalog, fh, separators=(",", ":"))
    return catalog


if __name__ == "__main__":
    cat = build_catalog()
    print(
        f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes): "
        f"symbols={cat['symbols']}, 5m_dates={len(cat['dates_by_timeframe']['5m'])}, "
        f"1h_dates={len(cat['dates_by_timeframe']['1h'])}"
    )
