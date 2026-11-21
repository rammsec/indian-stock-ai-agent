"""Deterministic Multi-Timeframe (15m / 30m) NSE Candle Resampler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from ..config import IST_TZ
from .schema import Candle5M


@dataclass(frozen=True)
class ResampledCandle:
    """Resampled OHLCV candle with explicit source completeness metadata."""

    symbol: str
    exchange: Literal["NSE", "BSE"]
    interval_minutes: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_bar_count: int
    expected_bar_count: int
    is_complete: bool


def resample_5m_candles(
    candles_5m: list[Candle5M],
    target_minutes: Literal[15, 30, 60] = 15,
    include_incomplete: bool = False,
) -> list[ResampledCandle]:
    """Deterministically resample 5m candles into 15m, 30m, or 60m bars aligned to 09:15 IST.

    Rules:
    - Never combines candles across different NSE trading dates.
    - Never interpolates missing OHLCV intervals.
    - Records source_bar_count and is_complete for every bucket.
    - By default drops incomplete buckets so strategies never act on partial intervals.
    """
    if target_minutes not in (15, 30, 60):
        raise ValueError("target_minutes must be 15, 30, or 60")
    if not candles_5m:
        return []

    expected_sub_bars = target_minutes // 5
    # Deduplicate and sort strictly chronologically
    seen_ts: set[datetime] = set()
    clean_bars: list[Candle5M] = []
    for c in sorted(candles_5m, key=lambda x: x.timestamp):
        ts_ist = c.timestamp.astimezone(IST_TZ)
        if ts_ist not in seen_ts:
            seen_ts.add(ts_ist)
            clean_bars.append(c)

    # Group by (session_date, bucket_start_timestamp) aligned to 09:15 IST
    buckets: dict[tuple[object, datetime], list[Candle5M]] = {}
    for c in clean_bars:
        ts_ist = c.timestamp.astimezone(IST_TZ)
        session_open = ts_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        mins_since_open = int((ts_ist - session_open).total_seconds() // 60)
        if mins_since_open < 0:
            continue
        bucket_idx = mins_since_open // target_minutes
        bucket_start = session_open + timedelta(minutes=bucket_idx * target_minutes)
        key = (ts_ist.date(), bucket_start)
        buckets.setdefault(key, []).append(c)

    resampled: list[ResampledCandle] = []
    for (_, bucket_start), bars in sorted(buckets.items(), key=lambda x: x[0][1]):
        bars_sorted = sorted(bars, key=lambda x: x.timestamp)
        is_complete = len(bars_sorted) == expected_sub_bars
        if not include_incomplete and not is_complete:
            continue

        resampled.append(
            ResampledCandle(
                symbol=bars_sorted[0].symbol,
                exchange=bars_sorted[0].exchange,
                interval_minutes=target_minutes,
                timestamp=bucket_start,
                open=bars_sorted[0].open,
                high=max(b.high for b in bars_sorted),
                low=min(b.low for b in bars_sorted),
                close=bars_sorted[-1].close,
                volume=sum(b.volume for b in bars_sorted),
                source_bar_count=len(bars_sorted),
                expected_bar_count=expected_sub_bars,
                is_complete=is_complete,
            )
        )

    return resampled


def to_candle5m_compatible(resampled: list[ResampledCandle]) -> list[Candle5M]:
    """Convert ResampledCandle list into Candle5M objects for indicator & replay reuse."""
    return [
        Candle5M(
            symbol=r.symbol,
            exchange=r.exchange,
            timestamp=r.timestamp,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
        )
        for r in resampled
    ]
