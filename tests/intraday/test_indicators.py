"""Unit and invariant tests for 5-minute technical indicators and feature pipeline."""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from trader.config import IST_TZ
from trader.data.schema import Candle5M
from trader.data.loader import generate_synthetic_5m_session
from trader.features.indicators import (
    compute_intraday_vwap,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_cpr_pivots,
    compute_all_features,
)


def test_vwap_daily_reset():
    """Verify that cumulative VWAP resets cleanly at 09:15 IST each new session."""
    day1 = datetime(2026, 11, 19, tzinfo=IST_TZ).date()
    day2 = datetime(2026, 11, 20, tzinfo=IST_TZ).date()

    candles_day1 = generate_synthetic_5m_session(symbol="RELIANCE", session_date=day1, base_price=2000.0)
    candles_day2 = generate_synthetic_5m_session(symbol="RELIANCE", session_date=day2, base_price=3000.0)

    all_candles = candles_day1 + candles_day2
    df = compute_all_features(all_candles)

    assert len(df) == 150
    # The first bar of day 2 (index 75) should have VWAP close to its typical price (~3000), NOT an average with day 1 (~2500)
    bar_day2_open = df.iloc[75]
    expected_tp = (bar_day2_open["high"] + bar_day2_open["low"] + bar_day2_open["close"]) / 3.0
    assert abs(bar_day2_open["vwap"] - expected_tp) < 1.0


def test_ema_and_momentum_indicators():
    """Assert EMA, RSI, MACD, and ATR compute reasonable numerical values."""
    candles = generate_synthetic_5m_session(symbol="TCS", base_price=3500.0)
    df = compute_all_features(candles)

    # All columns present
    for col in ["vwap", "ema_9", "ema_21", "ema_50", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14"]:
        assert col in df.columns
        assert not df[col].isna().all()

    # RSI bounded between 0 and 100
    assert (df["rsi_14"] >= 0.0).all() and (df["rsi_14"] <= 100.0).all()

    # ATR strictly positive
    assert (df["atr_14"] > 0.0).all()


def test_cpr_pivots_math():
    """Verify classic CPR and Floor Pivot calculations."""
    pivots = compute_cpr_pivots(prev_high=100.0, prev_low=90.0, prev_close=95.0)

    # P = (100 + 90 + 95) / 3 = 95.0
    assert pivots["pivot"] == 95.0
    # BC = (100 + 90) / 2 = 95.0
    assert pivots["cpr_bc"] == 95.0
    # TC = 2 * 95 - 95 = 95.0 (virgin narrow CPR)
    assert pivots["cpr_tc"] == 95.0
    # R1 = 2 * 95 - 90 = 100.0
    assert pivots["r1"] == 100.0
    # S1 = 2 * 95 - 100 = 90.0
    assert pivots["s1"] == 90.0


def test_no_lookahead_bias():
    """Verify that feature calculation at bar t does not change when subsequent bars are added."""
    candles = generate_synthetic_5m_session(symbol="INFY", base_price=1800.0)

    # Compute features on first 40 bars
    df_partial = compute_all_features(candles[:40])
    val_at_bar_39 = df_partial.iloc[39]["ema_21"]

    # Compute features on all 75 bars
    df_full = compute_all_features(candles)
    val_at_bar_39_full = df_full.iloc[39]["ema_21"]

    # Bar 39 feature must be identical regardless of future bars
    assert abs(val_at_bar_39 - val_at_bar_39_full) < 1e-6
