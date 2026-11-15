"""Technical indicator & volatility engine for Indian equities (NSE/BSE).

Adapted from open-source GitHub repo `aswathas/pistol` (`pistol-core/compute/indicators.py`
and `volatility.py`) with Next-Day ATR/Pivot Entry, Target, and Stop-Loss calculators.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .models import PricePoint, TechnicalSnapshot


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return round(float(val), 2)
    except Exception:
        return default


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI(14)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper, middle, and lower Bollinger Bands."""
    mid = close.rolling(period, min_periods=5).mean()
    std = close.rolling(period, min_periods=5).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR-14)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=5, adjust=False).mean()


def compute_annualized_volatility(close: pd.Series, period: int = 30) -> float:
    """Annualized log-return historical volatility (%)."""
    if len(close) < 10:
        return 22.0
    lr = np.log(close / close.shift(1)).dropna()
    if lr.empty:
        return 22.0
    window = min(period, len(lr))
    std = float(lr.iloc[-window:].std())
    if math.isnan(std) or std <= 0:
        return 22.0
    return round(std * math.sqrt(252) * 100.0, 2)


def build_technical_snapshot(
    price_rows: list[dict[str, Any]],
    fallback_price: float = 0.0,
    fallback_high_52w: float = 0.0,
    fallback_low_52w: float = 0.0,
) -> TechnicalSnapshot:
    """Build full technical indicator & next-day trigger snapshot from daily OHLCV/DMA rows."""
    if not price_rows:
        p = fallback_price or 100.0
        return TechnicalSnapshot(
            current_price=p,
            prev_close=p,
            high_52w=fallback_high_52w or round(p * 1.15, 2),
            low_52w=fallback_low_52w or round(p * 0.85, 2),
            ema_20=p,
            dma_50=p,
            dma_200=p,
            atr_14=round(p * 0.018, 2),
            atr_pct=1.8,
            pivot_point=p,
            support_1=round(p * 0.985, 2),
            support_2=round(p * 0.97, 2),
            resistance_1=round(p * 1.015, 2),
            resistance_2=round(p * 1.03, 2),
        )

    df = pd.DataFrame(price_rows)
    df["date"] = df["date"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    if "open" not in df.columns or df["open"].isna().all():
        df["open"] = df["close"].shift(1).fillna(df["close"])
    else:
        df["open"] = pd.to_numeric(df["open"], errors="coerce").fillna(df["close"])

    if "high" not in df.columns or df["high"].isna().all():
        # Estimate intraday high/low from daily volatility when only close series is present
        daily_move = df["close"].diff().abs().rolling(10, min_periods=1).mean().fillna(df["close"] * 0.01)
        df["high"] = df[["open", "close"]].max(axis=1) + daily_move * 0.65
        df["low"] = df[["open", "close"]].min(axis=1) - daily_move * 0.65
    else:
        df["high"] = pd.to_numeric(df["high"], errors="coerce").fillna(df["close"])
        df["low"] = pd.to_numeric(df["low"], errors="coerce").fillna(df["close"])

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    close = df["close"]
    n = len(df)
    cur_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if n >= 2 else cur_price
    day_change_pct = ((cur_price - prev_close) / prev_close * 100.0) if prev_close else 0.0

    def _ret_pct(days_back: int) -> float:
        if n > days_back and close.iloc[-1 - days_back] > 0:
            base = float(close.iloc[-1 - days_back])
            return round((cur_price - base) / base * 100.0, 2)
        if n >= 2 and close.iloc[0] > 0:
            base = float(close.iloc[0])
            return round((cur_price - base) / base * 100.0, 2)
        return 0.0

    ret_5d = _ret_pct(5)
    ret_1m = _ret_pct(21)
    ret_6m = _ret_pct(min(126, n - 1))
    ret_1y = _ret_pct(min(250, n - 1))

    ema_20_s = close.ewm(span=20, adjust=False).mean()
    dma_50_s = (
        pd.to_numeric(df["dma50"], errors="coerce").fillna(close.rolling(50, min_periods=5).mean())
        if "dma50" in df.columns
        else close.rolling(50, min_periods=5).mean()
    )
    dma_200_s = (
        pd.to_numeric(df["dma200"], errors="coerce").fillna(close.rolling(200, min_periods=10).mean())
        if "dma200" in df.columns
        else close.rolling(200, min_periods=10).mean()
    )

    rsi_s = compute_rsi(close, 14)
    macd_l, macd_sig, macd_h = compute_macd(close)
    bb_u, bb_m, bb_l = compute_bollinger_bands(close, 20, 2.0)
    atr_s = compute_atr(df, 14)

    rsi_val = _safe_float(rsi_s.iloc[-1], 50.0)
    macd_val = _safe_float(macd_l.iloc[-1], 0.0)
    macd_sig_val = _safe_float(macd_sig.iloc[-1], 0.0)
    macd_hist_val = _safe_float(macd_h.iloc[-1], 0.0)
    ema_20_val = _safe_float(ema_20_s.iloc[-1], cur_price)
    dma_50_val = _safe_float(dma_50_s.iloc[-1], cur_price)
    dma_200_val = _safe_float(dma_200_s.iloc[-1], cur_price)
    bb_upper_val = _safe_float(bb_u.iloc[-1], cur_price * 1.04)
    bb_mid_val = _safe_float(bb_m.iloc[-1], cur_price)
    bb_lower_val = _safe_float(bb_l.iloc[-1], cur_price * 0.96)
    atr_val = _safe_float(atr_s.iloc[-1], cur_price * 0.018)
    if atr_val <= 0:
        atr_val = round(cur_price * 0.018, 2)
    atr_pct = round(atr_val / cur_price * 100.0, 2) if cur_price else 1.8

    vol_latest = _safe_float(df["volume"].iloc[-1], 0.0)
    vol_20d = _safe_float(df["volume"].iloc[-20:].mean(), vol_latest)
    vol_ratio = round(vol_latest / vol_20d, 2) if vol_20d > 0 else 1.0

    day_high = _safe_float(df["high"].iloc[-1], cur_price)
    day_low = _safe_float(df["low"].iloc[-1], cur_price)
    high_52w = fallback_high_52w or _safe_float(df["high"].max(), cur_price)
    low_52w = fallback_low_52w or _safe_float(df["low"].min(), cur_price)

    # Classic Floor Trader Pivot Points for Next-Day NSE session
    pivot = round((day_high + day_low + cur_price) / 3.0, 2)
    r1 = round(2.0 * pivot - day_low, 2)
    s1 = round(2.0 * pivot - day_high, 2)
    r2 = round(pivot + (day_high - day_low), 2)
    s2 = round(pivot - (day_high - day_low), 2)

    # Ensure sensible minimum separation using ATR if daily bar was tight
    if r1 <= cur_price:
        r1 = round(cur_price + 0.85 * atr_val, 2)
    if r2 <= r1:
        r2 = round(cur_price + 1.75 * atr_val, 2)
    if s1 >= cur_price:
        s1 = round(cur_price - 0.85 * atr_val, 2)
    if s2 >= s1:
        s2 = round(cur_price - 1.65 * atr_val, 2)

    signals: list[str] = []
    if cur_price > dma_50_val and dma_50_val > dma_200_val:
        trend_regime = "BULLISH UPTREND (Price > 50-DMA > 200-DMA)"
        signals.append("Golden Trend Alignment: Trading above both 50-DMA and 200-DMA")
    elif cur_price > dma_200_val and cur_price < dma_50_val:
        trend_regime = "PULLBACK IN STRUCTURAL UPTREND (Above 200-DMA, testing 50-DMA)"
        signals.append("Structural support intact above 200-DMA; short-term consolidation below 50-DMA")
    elif cur_price > dma_50_val and cur_price < dma_200_val:
        trend_regime = "RECOVERY / REVERSAL ATTEMPT (Above 50-DMA, below 200-DMA)"
        signals.append("Short-term momentum recovering above 50-DMA toward 200-DMA overhead resistance")
    else:
        trend_regime = "BEARISH / CORRECTIVE PHASE (Below 50-DMA & 200-DMA)"
        signals.append("Price below both 50-DMA and 200-DMA; wait for support stabilization")

    if rsi_val < 33:
        signals.append(f"RSI(14) at {rsi_val} — Oversold zone, high probability of mean-reversion bounce")
    elif rsi_val > 70:
        signals.append(f"RSI(14) at {rsi_val} — Overbought zone, watch for profit-booking near R1")
    elif 52 <= rsi_val <= 68:
        signals.append(f"RSI(14) at {rsi_val} — Healthy bullish momentum zone without overextension")
    else:
        signals.append(f"RSI(14) at {rsi_val} — Neutral momentum band")

    if macd_hist_val > 0 and macd_val > macd_sig_val:
        signals.append(f"MACD Bullish Crossover active (Histogram +{macd_hist_val:.2f})")
    else:
        signals.append(f"MACD below signal line (Histogram {macd_hist_val:.2f})")

    if vol_ratio >= 1.35:
        signals.append(f"Volume Surge: Latest session volume is {vol_ratio}x the 20-day average")

    chart_df = df.iloc[-180:].copy()
    chart_df["dma50_calc"] = dma_50_s.iloc[-180:]
    chart_df["dma200_calc"] = dma_200_s.iloc[-180:]

    series_out: list[PricePoint] = []
    for _, row in chart_df.iterrows():
        series_out.append(
            PricePoint(
                date=str(row["date"])[:10],
                close=_safe_float(row["close"]),
                open=_safe_float(row.get("open"), _safe_float(row["close"])),
                high=_safe_float(row.get("high"), _safe_float(row["close"])),
                low=_safe_float(row.get("low"), _safe_float(row["close"])),
                dma50=_safe_float(row.get("dma50_calc"), _safe_float(row["close"])),
                dma200=_safe_float(row.get("dma200_calc"), _safe_float(row["close"])),
                volume=_safe_float(row.get("volume"), 0.0),
            )
        )

    return TechnicalSnapshot(
        current_price=_safe_float(cur_price),
        prev_close=_safe_float(prev_close),
        day_change_pct=_safe_float(day_change_pct),
        day_high=day_high,
        day_low=day_low,
        high_52w=high_52w,
        low_52w=low_52w,
        return_5d_pct=ret_5d,
        return_1m_pct=ret_1m,
        return_6m_pct=ret_6m,
        return_1y_pct=ret_1y,
        rsi_14=rsi_val,
        macd_line=macd_val,
        macd_signal=macd_sig_val,
        macd_hist=macd_hist_val,
        ema_20=ema_20_val,
        dma_50=dma_50_val,
        dma_200=dma_200_val,
        bb_upper=bb_upper_val,
        bb_middle=bb_mid_val,
        bb_lower=bb_lower_val,
        atr_14=atr_val,
        atr_pct=atr_pct,
        volume_latest=vol_latest,
        volume_20d_avg=vol_20d,
        volume_Surge_ratio=vol_ratio,
        annualized_volatility_pct=compute_annualized_volatility(close),
        pivot_point=pivot,
        support_1=s1,
        support_2=s2,
        resistance_1=r1,
        resistance_2=r2,
        trend_regime=trend_regime,
        technical_signals=signals,
        price_series=series_out,
    )
