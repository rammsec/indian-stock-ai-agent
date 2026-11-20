"""Historical 5-minute candle data loader with Parquet caching and offline fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from ..config import IST_TZ
from .schema import Candle5M
from .validator import validate_candle_series

CACHE_DIR = Path("data/intraday")


def generate_synthetic_5m_session(
    symbol: str = "RELIANCE",
    exchange: str = "NSE",
    session_date: datetime.date | None = None,
    base_price: float = 2500.0,
    daily_trend_pct: float = 0.5,
    volatility_pct: float = 0.2,
    num_bars: int = 75,
) -> list[Candle5M]:
    """Generate realistic, deterministic 75 5-minute bars (09:15 to 15:30 IST) for testing.

    A full standard NSE/BSE session consists of exactly 75 5-minute intervals.
    """
    if session_date is None:
        session_date = datetime.now(tz=IST_TZ).date()

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

    return candles


def load_historical_5m_candles(
    symbol: str,
    exchange: str = "NSE",
    days: int = 5,
    use_cache: bool = True,
    fallback_synthetic: bool = True,
) -> list[Candle5M]:
    """Load 5-minute candles from Parquet cache or yfinance, with fallback to synthetic data.

    Yahoo Finance suffix: `.NS` for NSE, `.BO` for BSE.
    """
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
            sanitized, _ = validate_candle_series(candles)
            if sanitized:
                return sanitized
        except Exception:
            pass

    # Attempt fetch from yfinance
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

            sanitized, _ = validate_candle_series(candles)
            if sanitized and use_cache:
                # Save sanitized to cache
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
            return sanitized
    except Exception:
        pass

    # Fallback to deterministic synthetic fixture if allowed
    if fallback_synthetic:
        return generate_synthetic_5m_session(symbol=symbol, exchange=exchange)

    return []
