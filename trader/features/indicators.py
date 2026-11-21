"""Deterministic 5-minute technical indicator engine with zero lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import Candle5M


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative Volume Weighted Average Price (VWAP), resetting daily at 09:15 IST.

    Strictly zero-lookahead: Uses only closed bars t <= current. No backward filling.
    Typical Price = (High + Low + Close) / 3
    VWAP = cumsum(Typical Price * Volume) / cumsum(Volume)
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]

    # Extract date in IST to group by session
    dates = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date

    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()

    # If cumulative volume is 0 at start of session, fall back to typical price for that bar
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    # Forward-fill ONLY (never look forward via bfill)
    return vwap.fillna(typical_price).ffill()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average with the specified span."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's Relative Strength Index (RSI-14)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD).

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: (macd_line, signal_line, histogram)
    """
    fast_ema = compute_ema(close, span=fast_period)
    slow_ema = compute_ema(close, span=slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, span=signal_period)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR-14) using Wilder's smoothing with zero backward filling."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # For bar 0 where prev_close is NaN, true range is strictly high - low
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]

    # Wilder's exponential smoothing strictly forward without bfill
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def compute_cpr_pivots(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> dict[str, float]:
    """Compute Central Pivot Range (CPR) and classic Floor Pivots from prior session OHLC.

    - Pivot (P) = (High + Low + Close) / 3
    - Bottom CPR (BC) = (High + Low) / 2
    - Top CPR (TC) = (2 * P) - BC
    - R1 = (2 * P) - Low, S1 = (2 * P) - High
    - R2 = P + (High - Low), S2 = P - (High - Low)
    """
    p = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = (2.0 * p) - bc

    r1 = (2.0 * p) - prev_low
    s1 = (2.0 * p) - prev_high
    r2 = p + (prev_high - prev_low)
    s2 = p - (prev_high - prev_low)

    return {
        "pivot": round(p, 2),
        "cpr_bc": round(bc, 2),
        "cpr_tc": round(tc, 2),
        "cpr_width_pct": round(abs(tc - bc) / p * 100.0, 3),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
        "r2": round(r2, 2),
        "s2": round(s2, 2),
    }


def compute_all_features(candles: list[Candle5M]) -> pd.DataFrame:
    """Convert candle list into feature DataFrame with strict zero lookahead."""
    if not candles:
        return pd.DataFrame()

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
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Vectorized indicators with zero lookahead
    df["vwap"] = compute_intraday_vwap(df)
    df["ema_9"] = compute_ema(df["close"], span=9)
    df["ema_21"] = compute_ema(df["close"], span=21)
    df["ema_50"] = compute_ema(df["close"], span=50)
    df["rsi_14"] = compute_rsi(df["close"], period=14)

    macd, macd_sig, macd_hist = compute_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], period=14)
    df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # Price vs VWAP and EMA trend flags
    df["above_vwap"] = df["close"] > df["vwap"]
    df["ema_trend_bullish"] = (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])

    return df
