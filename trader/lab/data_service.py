"""Historical 5-minute data indexer, warm-up slicer, and quality validator for the 5-stock Lab universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from ..config import IST_TZ, LAB_DEFAULT_UNIVERSE, is_nse_trading_day
from ..data.loader import DataUnavailableError, _compute_candle_series_hash, load_historical_5m_candles
from ..data.schema import Candle5M, DataProvenance, DataSourceMode
from ..data.validator import detect_missing_candles


@dataclass(frozen=True)
class StockSessionQuality:
    """Data completeness & provenance report for a single stock on a selected historical date."""

    symbol: str
    date: str
    data_availability: Literal["Historical", "Cached Historical", "Synthetic Test", "Unavailable"]
    source_mode: str
    provider: str
    retrieved_at_ist: str
    candles_count: int
    warmup_candles_count: int
    missing_intervals_count: int
    status: Literal["Ready", "Incomplete", "Insufficient", "Unavailable"]
    sha256_hash: str


@dataclass(frozen=True)
class HistoricalDateAvailability:
    """Summary of a verified historical trading date across the 5-stock universe."""

    date: str
    all_stocks_ready: bool
    ready_count: int
    total_stocks: int
    stocks: list[StockSessionQuality]


def refresh_universe_cache(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    days: int = 60,
) -> dict[str, DataProvenance]:
    """Fetch and cache up to 60 days of genuine 5m historical candles from Yahoo Finance."""
    provenances: dict[str, DataProvenance] = {}
    for sym in symbols:
        # Force fresh fetch when refreshing cache
        candles, prov = load_historical_5m_candles(
            symbol=sym,
            exchange="NSE",
            days=days,
            mode=DataSourceMode.HISTORICAL_REAL,
            use_cache=False,
        )
        # load_historical_5m_candles writes to Parquet cache when use_cache=False?
        # Check: in loader.py, it only writes to cache if use_cache=True! Let's ensure cache is saved.
        from ..data.loader import CACHE_DIR
        import pandas as pd

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{sym}_NSE_5m.parquet"
        records = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        pd.DataFrame(records).to_parquet(cache_file, index=False)
        provenances[sym] = prov
    return provenances


def load_session_with_warmup(
    symbol: str,
    target_date: date,
    warmup_days: int = 5,
    mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
) -> tuple[list[Candle5M], list[Candle5M], StockSessionQuality]:
    """Load prior-session warm-up candles and target_date session candles with ZERO future leakage.

    Strictly slices `c.timestamp.astimezone(IST_TZ).date() <= target_date`.
    """
    if not is_nse_trading_day(target_date):
        raise ValueError(f"{target_date} is not an active NSE trading day")

    try:
        all_candles, prov = load_historical_5m_candles(
            symbol=symbol,
            exchange="NSE",
            days=60,
            mode=mode,
            use_cache=True,
        )
    except DataUnavailableError:
        quality = StockSessionQuality(
            symbol=symbol,
            date=target_date.isoformat(),
            data_availability="Unavailable",
            source_mode=mode.value,
            provider="None",
            retrieved_at_ist=datetime.now(tz=IST_TZ).isoformat(),
            candles_count=0,
            warmup_candles_count=0,
            missing_intervals_count=0,
            status="Unavailable",
            sha256_hash="UNAVAILABLE",
        )
        return [], [], quality

    # Group strictly <= target_date (NO future candles!)
    session_candles = [
        c for c in all_candles if c.timestamp.astimezone(IST_TZ).date() == target_date
    ]
    prior_candles = [
        c for c in all_candles if c.timestamp.astimezone(IST_TZ).date() < target_date
    ]
    # Keep at most warmup_days * 75 prior bars for EMA/ATR/Supertrend warm-up
    warmup_candles = prior_candles[-(warmup_days * 75) :] if prior_candles else []

    missing = detect_missing_candles(session_candles, expected_interval_minutes=5)
    n_bars = len(session_candles)
    if n_bars == 0:
        status: Literal["Ready", "Incomplete", "Insufficient", "Unavailable"] = "Unavailable"
    elif n_bars < 50:
        status = "Insufficient"
    elif len(missing) > 0 or n_bars < 70:
        status = "Incomplete"
    else:
        status = "Ready"

    avail_label: Literal["Historical", "Cached Historical", "Synthetic Test", "Unavailable"]
    if prov.source_mode == DataSourceMode.SYNTHETIC_TEST:
        avail_label = "Synthetic Test"
    elif "ParquetCache" in prov.source_name:
        avail_label = "Cached Historical"
    else:
        avail_label = "Historical"

    quality = StockSessionQuality(
        symbol=symbol,
        date=target_date.isoformat(),
        data_availability=avail_label if n_bars > 0 else "Unavailable",
        source_mode=prov.source_mode.value,
        provider=prov.source_name,
        retrieved_at_ist=prov.ingested_at.isoformat(),
        candles_count=n_bars,
        warmup_candles_count=len(warmup_candles),
        missing_intervals_count=len(missing),
        status=status,
        sha256_hash=_compute_candle_series_hash(session_candles),
    )
    return warmup_candles, session_candles, quality


def get_available_historical_dates(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    max_dates: int = 30,
) -> list[HistoricalDateAvailability]:
    """Discover all verified historical dates present in real 5m data for the 5-stock universe."""
    date_sets: dict[date, int] = {}
    symbol_candles: dict[str, dict[date, list[Candle5M]]] = {}

    for sym in symbols:
        try:
            candles, _ = load_historical_5m_candles(
                symbol=sym,
                exchange="NSE",
                days=60,
                mode=DataSourceMode.HISTORICAL_REAL,
                use_cache=True,
            )
        except DataUnavailableError:
            candles = []

        by_date: dict[date, list[Candle5M]] = {}
        for c in candles:
            d = c.timestamp.astimezone(IST_TZ).date()
            if is_nse_trading_day(d):
                by_date.setdefault(d, []).append(c)
                date_sets[d] = date_sets.get(d, 0) + 1
        symbol_candles[sym] = by_date

    # Sort descending (most recent verified dates first)
    sorted_dates = sorted(date_sets.keys(), reverse=True)[:max_dates]
    results: list[HistoricalDateAvailability] = []

    for d in sorted_dates:
        stock_qualities: list[StockSessionQuality] = []
        ready_cnt = 0
        for sym in symbols:
            bars = symbol_candles.get(sym, {}).get(d, [])
            missing = detect_missing_candles(bars, expected_interval_minutes=5) if bars else []
            n_bars = len(bars)
            if n_bars >= 70 and len(missing) == 0:
                st: Literal["Ready", "Incomplete", "Insufficient", "Unavailable"] = "Ready"
                ready_cnt += 1
            elif n_bars >= 50:
                st = "Incomplete"
                ready_cnt += 1
            elif n_bars > 0:
                st = "Insufficient"
            else:
                st = "Unavailable"

            stock_qualities.append(
                StockSessionQuality(
                    symbol=sym,
                    date=d.isoformat(),
                    data_availability="Cached Historical" if n_bars > 0 else "Unavailable",
                    source_mode=DataSourceMode.HISTORICAL_REAL.value,
                    provider=f"YahooFinance/Parquet:{sym}_NSE_5m",
                    retrieved_at_ist=datetime.now(tz=IST_TZ).isoformat(),
                    candles_count=n_bars,
                    warmup_candles_count=0,
                    missing_intervals_count=len(missing),
                    status=st,
                    sha256_hash=_compute_candle_series_hash(bars),
                )
            )

        results.append(
            HistoricalDateAvailability(
                date=d.isoformat(),
                all_stocks_ready=(ready_cnt == len(symbols)),
                ready_count=ready_cnt,
                total_stocks=len(symbols),
                stocks=stock_qualities,
            )
        )

    return results
