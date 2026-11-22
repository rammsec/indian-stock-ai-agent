"""Deterministic 5-minute technical indicator engine with zero lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import Candle5M
from ..intelligence.registry import (
    compute_atr_series,
    compute_cpr_pivots_dict,
    compute_ema_series,
    compute_macd_series,
    compute_rsi_series,
    compute_rvol_series,
    compute_session_vwap_series,
)


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative Volume Weighted Average Price (VWAP), resetting daily at 09:15 IST."""
    return compute_session_vwap_series(df, zero_vol_policy="typical_price")


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average with the specified span."""
    return compute_ema_series(series, span=span, min_periods=1)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's Relative Strength Index (RSI-14)."""
    return compute_rsi_series(
        close, period=period, warmup_fill=50.0, zero_loss_mode="legacy_50"
    )


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    return compute_macd_series(close, fast=fast_period, slow=slow_period, signal=signal_period)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR-14) using Wilder's smoothing with zero backward filling."""
    return compute_atr_series(high, low, close, period=period, method="wilder", min_periods=1)


def compute_cpr_pivots(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> dict[str, float]:
    """Compute Central Pivot Range (CPR) and classic Floor Pivots from prior session OHLC."""
    return compute_cpr_pivots_dict(prev_high, prev_low, prev_close)


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
    df["volume_ratio"] = compute_rvol_series(df["volume"], window=20, mode="inclusive")

    # Price vs VWAP and EMA trend flags
    df["above_vwap"] = df["close"] > df["vwap"]
    df["ema_trend_bullish"] = (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])

    return df
