"""Historical 5-minute candle data loader with explicit mode separation and provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from ..config import IST_TZ, is_nse_trading_day
from .schema import Candle5M, DataProvenance, DataSourceMode
from .validator import validate_candle_series

CACHE_DIR = Path("data/intraday")


class DataUnavailableError(Exception):
    """Raised when real historical data cannot be obtained and silent synthetic substitution is blocked."""


def _compute_candle_series_hash(candles: list[Candle5M]) -> str:
    """Compute deterministic SHA-256 checksum across candle timestamps, OHLCV."""
    if not candles:
        return "EMPTY_DATASET_HASH_0000000000000000"
    raw_str = ";".join(
        f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
        for c in candles
    )
    return hashlib.sha256(raw_str.encode()).hexdigest()


def generate_synthetic_5m_session(
    symbol: str = "RELIANCE",
    exchange: str = "NSE",
    session_date: datetime.date | None = None,
    base_price: float = 2500.0,
    daily_trend_pct: float = 0.5,
    volatility_pct: float = 0.2,
    num_bars: int = 75,
    return_provenance: bool = False,
) -> list[Candle5M] | tuple[list[Candle5M], DataProvenance]:
    """Explicitly generate a deterministic synthetic 5-minute session for testing purposes.

    Strictly tagged with `DataSourceMode.SYNTHETIC_TEST`.
    """
    if session_date is None:
        session_date = datetime.now(tz=IST_TZ).date()
        while not is_nse_trading_day(session_date):
            session_date -= timedelta(days=1)

    candles: list[Candle5M] = []
    current_time = datetime(
        session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST_TZ
    )

    np.random.seed(42 + int(session_date.strftime("%Y%m%d")) % 1000)
    current_price = base_price
    drift = (daily_trend_pct / 100.0) / num_bars

    for bar_idx in range(num_bars):
        pct_change = np.random.normal(loc=drift, scale=volatility_pct / 100.0)
        open_price = round(current_price, 2)
        close_price = round(open_price * (1.0 + pct_change), 2)
        noise_high = abs(np.random.normal(0, volatility_pct / 200.0))
        noise_low = abs(np.random.normal(0, volatility_pct / 200.0))

        high_price = round(max(open_price, close_price) * (1.0 + noise_high), 2)
        low_price = round(min(open_price, close_price) * (1.0 - noise_low), 2)
        volume = float(int(np.random.lognormal(mean=9.5, sigma=0.6)))

        candle = Candle5M(
            symbol=symbol,
            exchange=exchange,  # type: ignore[arg-type]
            timestamp=current_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        candles.append(candle)

        current_price = close_price
        current_time += timedelta(minutes=5)

    sha = _compute_candle_series_hash(candles)
    provenance = DataProvenance(
        source_mode=DataSourceMode.SYNTHETIC_TEST,
        source_name="SyntheticGeneratorDeterministic",
        ingested_at=datetime.now(tz=IST_TZ),
        symbol=symbol,
        exchange=exchange,  # type: ignore[arg-type]
        bars_count=len(candles),
        start_timestamp=candles[0].timestamp,
        end_timestamp=candles[-1].timestamp,
        sha256_hash=sha,
        notes=["Synthetic fixture generated for offline deterministic unit testing"],
    )

    if return_provenance:
        return candles, provenance
    return candles


def load_historical_5m_candles(
    symbol: str,
    exchange: str = "NSE",
    days: int = 5,
    mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
    use_cache: bool = True,
) -> tuple[list[Candle5M], DataProvenance]:
    """Load 5-minute candles with explicit mode discipline and provenance.

    In `HISTORICAL_REAL` mode:
    - Queries Parquet cache and yfinance.
    - If real data cannot be fetched, RAISES `DataUnavailableError`.
    - NEVER silently substitutes synthetic data.

    In `SYNTHETIC_TEST` mode:
    - Generates synthetic test fixtures explicitly marked as such.
    """
    if mode == DataSourceMode.SYNTHETIC_TEST:
        res = generate_synthetic_5m_session(symbol=symbol, exchange=exchange, return_provenance=True)
        assert isinstance(res, tuple)
        return res

    # --- Mode is HISTORICAL_REAL ---
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{exchange}_5m.parquet"

    # Check Parquet cache
    if use_cache and cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            candles: list[Candle5M] = []
            for _, row in df.iterrows():
                ts = pd.to_datetime(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize(IST_TZ)
                else:
                    ts = ts.tz_convert(IST_TZ)

                candles.append(
                    Candle5M(
                        symbol=symbol,
                        exchange=exchange,  # type: ignore[arg-type]
                        timestamp=ts.to_pydatetime(),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
            sanitized, rep = validate_candle_series(candles)
            if sanitized:
                sha = _compute_candle_series_hash(sanitized)
                prov = DataProvenance(
                    source_mode=DataSourceMode.HISTORICAL_REAL,
                    source_name=f"ParquetCache:{cache_file.name}",
                    ingested_at=datetime.now(tz=IST_TZ),
                    symbol=symbol,
                    exchange=exchange,  # type: ignore[arg-type]
                    bars_count=len(sanitized),
                    start_timestamp=sanitized[0].timestamp,
                    end_timestamp=sanitized[-1].timestamp,
                    sha256_hash=sha,
                    notes=["Loaded from verified historical local Parquet store"],
                )
                return sanitized, prov
        except Exception:
            pass

    # Attempt fetch from yfinance for real historical data
    ticker_suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    ticker_sym = f"{symbol.upper()}{ticker_suffix}"

    try:
        yf_ticker = yf.Ticker(ticker_sym)
        df = yf_ticker.history(period=f"{days}d", interval="5m")
        if df is not None and not df.empty:
            candles = []
            for ts_index, row in df.iterrows():
                ts = pd.to_datetime(ts_index)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(IST_TZ)
                else:
                    ts = ts.tz_convert(IST_TZ)

                c = Candle5M(
                    symbol=symbol,
                    exchange=exchange,  # type: ignore[arg-type]
                    timestamp=ts.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
                candles.append(c)

            sanitized, rep = validate_candle_series(candles)
            if sanitized:
                # Save sanitized to cache
                if use_cache:
                    records = [
                        {
                            "timestamp": c.timestamp,
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                        }
                        for c in sanitized
                    ]
                    pd.DataFrame(records).to_parquet(cache_file, index=False)

                sha = _compute_candle_series_hash(sanitized)
                prov = DataProvenance(
                    source_mode=DataSourceMode.HISTORICAL_REAL,
                    source_name=f"YahooFinanceAPI:{ticker_sym}",
                    ingested_at=datetime.now(tz=IST_TZ),
                    symbol=symbol,
                    exchange=exchange,  # type: ignore[arg-type]
                    bars_count=len(sanitized),
                    start_timestamp=sanitized[0].timestamp,
                    end_timestamp=sanitized[-1].timestamp,
                    sha256_hash=sha,
                    notes=[f"Fetched {len(sanitized)} live/historical 5m bars from Yahoo Finance"],
                )
                return sanitized, prov
    except Exception as exc:
        raise DataUnavailableError(
            f"Failed to fetch verified historical market data for {symbol} ({exchange}) from Yahoo Finance: {exc}. "
            f"Silent fallback to synthetic data is blocked under strict HISTORICAL_REAL mode."
        ) from exc

    raise DataUnavailableError(
        f"No market data returned for {symbol} ({exchange}). "
        f"Silent fallback to synthetic data is blocked under strict HISTORICAL_REAL mode."
    )
