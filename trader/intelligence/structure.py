"""CYCLE-010 Phase 2 — Strictly Causal Market Structure & 20-Scenario Detection Engine.

Pure, deterministic, zero-lookahead structural analysis built on top of the
canonical Shared Intelligence Layer (`trader.intelligence.registry`).

Covers all 20 required intraday market scenarios:
1-4. Strong / Weak Bullish & Bearish Trends
5.   Sideways Consolidation
6-7. Bullish Breakout & Bearish Breakdown
8-9. False Breakout (Bull Trap) & Failed Breakdown (Bear Trap / Spring)
10.  Pullback Continuation (EMA9/EMA21/VWAP)
11.  Trend Reversal (Break of Structure / CHOCH)
12.  VWAP Reclaim & Rejection
13.  EMA(9/21) Crossover & Failed Crossover
14.  Support / Resistance & CPR Pivot Rejection
15.  Head & Shoulders & Inverse Head & Shoulders
16.  Flags, Pennants & Triangles
17.  Opening-Range Breakout (ORB) & Failed ORB
18.  High- / Low-Volume Confirmation (+ Provider Zero-Volume Gap Detection)
19.  Volatility Expansion & Contraction (ATR Ratio / Squeeze)
20.  (Combined with CentralDirectionalAgent in CandleTimingAgent for Conflict Detection)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trader.config import IST_TZ
from trader.intelligence.registry import (
    compute_atr_series,
    compute_cpr_pivots_dict,
    compute_ema_series,
    compute_macd_series,
    compute_rsi_series,
    compute_rvol_series,
    compute_session_vwap_series,
)


@dataclass(frozen=True)
class StructureSnapshot:
    """Frozen, strictly causal snapshot of market structure at the latest completed bar."""

    trend_regime: str  # STRONG_BULLISH, WEAK_BULLISH, STRONG_BEARISH, WEAK_BEARISH, SIDEWAYS_CONSOLIDATION
    breakout_event: str  # BULLISH_BREAKOUT, BEARISH_BREAKDOWN, FALSE_BREAKOUT, FAILED_BREAKDOWN, NONE
    pullback_event: str  # BULLISH_PULLBACK_CONTINUATION, BEARISH_PULLBACK_CONTINUATION, NONE
    reversal_event: str  # BULLISH_REVERSAL, BEARISH_REVERSAL, NONE
    vwap_event: str  # VWAP_RECLAIM, VWAP_REJECTION, ABOVE_VWAP, BELOW_VWAP, UNAVAILABLE
    ema_cross_event: str  # BULLISH_CROSSOVER, BEARISH_CROSSOVER, FAILED_BULLISH_CROSSOVER, FAILED_BEARISH_CROSSOVER, NONE
    sr_event: str  # RESISTANCE_REJECTION, SUPPORT_REJECTION, NONE
    chart_pattern: str  # HEAD_AND_SHOULDERS_BREAKDOWN, INVERSE_HEAD_AND_SHOULDERS_BREAKOUT, NONE
    consolidation_pattern: str  # BULL_FLAG, BEAR_FLAG, BULL_FLAG_BREAKOUT, BEAR_FLAG_BREAKDOWN, TRIANGLE_COMPRESSION, TRIANGLE_BREAKOUT, TRIANGLE_BREAKDOWN, NONE
    orb_event: str  # ORB_BULLISH_BREAKOUT, ORB_BEARISH_BREAKDOWN, FAILED_ORB_HIGH, FAILED_ORB_LOW, FORMING, INSIDE_OR, UNAVAILABLE
    volume_regime: str  # HIGH_VOLUME_CONFIRMATION, NORMAL_VOLUME, LOW_VOLUME_WEAK, VOLUME_UNREPORTED_BY_PROVIDER
    volatility_regime: str  # EXTREME_EXPANSION, EXPANSION, CONTRACTION, NORMAL
    rsi_14: float | None
    macd_hist: float | None
    ema_50: float | None
    atr_expansion_ratio: float | None
    relative_volume_ratio: float | None
    cpr_levels: dict[str, float] | None
    opening_range_high: float | None
    opening_range_low: float | None
    active_scenarios: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_regime": self.trend_regime,
            "breakout_event": self.breakout_event,
            "pullback_event": self.pullback_event,
            "reversal_event": self.reversal_event,
            "vwap_event": self.vwap_event,
            "ema_cross_event": self.ema_cross_event,
            "sr_event": self.sr_event,
            "chart_pattern": self.chart_pattern,
            "consolidation_pattern": self.consolidation_pattern,
            "orb_event": self.orb_event,
            "volume_regime": self.volume_regime,
            "volatility_regime": self.volatility_regime,
            "rsi_14": round(self.rsi_14, 2) if self.rsi_14 is not None else None,
            "macd_hist": round(self.macd_hist, 4) if self.macd_hist is not None else None,
            "ema_50": round(self.ema_50, 2) if self.ema_50 is not None else None,
            "atr_expansion_ratio": round(self.atr_expansion_ratio, 3) if self.atr_expansion_ratio is not None else None,
            "relative_volume_ratio": round(self.relative_volume_ratio, 3) if self.relative_volume_ratio is not None else None,
            "cpr_levels": dict(self.cpr_levels) if self.cpr_levels is not None else None,
            "opening_range_high": round(self.opening_range_high, 2) if self.opening_range_high is not None else None,
            "opening_range_low": round(self.opening_range_low, 2) if self.opening_range_low is not None else None,
            "active_scenarios": list(self.active_scenarios),
        }


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _find_causal_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Find confirmed fractal swing highs and lows strictly within available bars (zero lookahead).

    A pivot at index `i` is only confirmed once `i + right < len(highs)`.
    """
    n = len(highs)
    pivot_highs: list[tuple[int, float]] = []
    pivot_lows: list[tuple[int, float]] = []
    if n < left + right + 1:
        return pivot_highs, pivot_lows

    for i in range(left, n - right):
        win_h = highs[i - left : i + right + 1]
        win_l = lows[i - left : i + right + 1]
        if highs[i] == np.max(win_h) and np.sum(win_h == highs[i]) == 1:
            pivot_highs.append((i, float(highs[i])))
        if lows[i] == np.min(win_l) and np.sum(win_l == lows[i]) == 1:
            pivot_lows.append((i, float(lows[i])))
    return pivot_highs, pivot_lows


def _detect_head_and_shoulders(
    pivot_highs: list[tuple[int, float]],
    pivot_lows: list[tuple[int, float]],
    curr_close: float,
) -> str:
    """Detect Head & Shoulders breakdown or Inverse Head & Shoulders breakout from confirmed causal pivots."""
    if len(pivot_highs) >= 3 and len(pivot_lows) >= 2:
        (i1, h1), (i2, h2), (i3, h3) = pivot_highs[-3], pivot_highs[-2], pivot_highs[-1]
        inter_lows = [val for idx, val in pivot_lows if i1 < idx < i3]
        if len(inter_lows) >= 2 and h2 > h1 and h2 > h3 and h2 > 0:
            shoulder_diff_pct = abs(h1 - h3) / h2
            neckline = min(inter_lows[-2], inter_lows[-1])
            if shoulder_diff_pct <= 0.04 and curr_close < neckline:
                return "HEAD_AND_SHOULDERS_BREAKDOWN"

    if len(pivot_lows) >= 3 and len(pivot_highs) >= 2:
        (i1, l1), (i2, l2), (i3, l3) = pivot_lows[-3], pivot_lows[-2], pivot_lows[-1]
        inter_highs = [val for idx, val in pivot_highs if i1 < idx < i3]
        if len(inter_highs) >= 2 and l2 < l1 and l2 < l3 and l1 > 0:
            shoulder_diff_pct = abs(l1 - l3) / max(l1, l3)
            neckline = max(inter_highs[-2], inter_highs[-1])
            if shoulder_diff_pct <= 0.04 and curr_close > neckline:
                return "INVERSE_HEAD_AND_SHOULDERS_BREAKOUT"

    return "NONE"


def _detect_flag_pennant_triangle(
    df: pd.DataFrame,
    atr_val: float | None,
) -> str:
    """Detect causal flags, pennants, and triangles (impulse + converging consolidation or breakout)."""
    n = len(df)
    if n < 10 or atr_val is None or atr_val <= 0:
        return "NONE"

    closes = df["close"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()

    # Inspect consolidation window (bars -6..-2) and impulse pole before it (bars -10..-6)
    consol_h = highs[-6:-1]
    consol_l = lows[-6:-1]
    consol_ranges = consol_h - consol_l
    avg_consol_range = float(np.mean(consol_ranges))
    box_high = float(np.max(consol_h))
    box_low = float(np.min(consol_l))

    pole_move = closes[-6] - closes[-10]
    curr_close = float(closes[-1])

    # Triangle / Pennant converging highs and rising lows inside consolidation window
    first_half_span = float(np.max(consol_h[:2]) - np.min(consol_l[:2]))
    second_half_span = float(np.max(consol_h[3:]) - np.min(consol_l[3:]))
    is_converging = second_half_span < first_half_span * 0.80 and avg_consol_range < atr_val * 1.35

    if pole_move >= 1.5 * atr_val and avg_consol_range <= atr_val * 1.15:
        if curr_close > box_high:
            return "BULL_FLAG_BREAKOUT"
        if curr_close >= box_low:
            return "BULL_FLAG"

    if pole_move <= -1.5 * atr_val and avg_consol_range <= atr_val * 1.15:
        if curr_close < box_low:
            return "BEAR_FLAG_BREAKDOWN"
        if curr_close <= box_high:
            return "BEAR_FLAG"

    if is_converging:
        if curr_close > box_high:
            return "TRIANGLE_BREAKOUT"
        if curr_close < box_low:
            return "TRIANGLE_BREAKDOWN"
        return "TRIANGLE_COMPRESSION"

    return "NONE"


def _compute_prior_day_cpr(df: pd.DataFrame) -> dict[str, float] | None:
    """Compute CPR & Floor Pivots strictly from the previous completed trading date (zero lookahead)."""
    if "timestamp" not in df.columns or df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    dates = ts.dt.date
    latest_date = dates.iloc[-1]
    prior_df = df.loc[dates < latest_date]
    if prior_df.empty:
        return None
    prev_date = dates.loc[dates < latest_date].iloc[-1]
    prev_session = df.loc[dates == prev_date]
    if prev_session.empty:
        return None
    return compute_cpr_pivots_dict(
        prev_high=float(prev_session["high"].max()),
        prev_low=float(prev_session["low"].min()),
        prev_close=float(prev_session["close"].iloc[-1]),
    )


def _detect_orb_event(df: pd.DataFrame) -> tuple[str, float | None, float | None]:
    """Detect Opening Range Breakout (ORB) & Failed ORB strictly on current IST session bars."""
    if "timestamp" not in df.columns or df.empty:
        return "UNAVAILABLE", None, None

    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    dates = ts.dt.date
    latest_date = dates.iloc[-1]
    sess = df.loc[dates == latest_date].reset_index(drop=True)
    if len(sess) < 2:
        or_h = float(sess["high"].iloc[0]) if len(sess) == 1 else None
        or_l = float(sess["low"].iloc[0]) if len(sess) == 1 else None
        return "FORMING", or_h, or_l

    # Infer bar spacing on current session to pick opening range bars (first 15m for 5m/15m, first bar for 1h)
    sess_ts = pd.to_datetime(sess["timestamp"])
    bar_mins = int(round((sess_ts.iloc[1] - sess_ts.iloc[0]).total_seconds() / 60.0))
    or_bar_count = 3 if bar_mins <= 5 else 1
    if len(sess) <= or_bar_count:
        return "FORMING", float(sess["high"].max()), float(sess["low"].min())

    or_slice = sess.iloc[:or_bar_count]
    or_high = float(or_slice["high"].max())
    or_low = float(or_slice["low"].min())

    curr = sess.iloc[-1]
    prev = sess.iloc[-2]
    curr_h, curr_l, curr_c = float(curr["high"]), float(curr["low"]), float(curr["close"])
    rng = max(1e-9, curr_h - curr_l)
    clv = (curr_c - curr_l) / rng

    prior_post_or = sess.iloc[or_bar_count:-1]
    broke_high_earlier = (
        not prior_post_or.empty and float(prior_post_or["close"].max()) > or_high
    )
    broke_low_earlier = (
        not prior_post_or.empty and float(prior_post_or["close"].min()) < or_low
    )

    if curr_c > or_high:
        return "ORB_BULLISH_BREAKOUT", or_high, or_low
    if curr_c < or_low:
        return "ORB_BEARISH_BREAKDOWN", or_high, or_low
    if (broke_high_earlier or curr_h > or_high) and curr_c <= or_high and clv <= 0.45:
        return "FAILED_ORB_HIGH", or_high, or_low
    if (broke_low_earlier or curr_l < or_low) and curr_c >= or_low and clv >= 0.55:
        return "FAILED_ORB_LOW", or_high, or_low
    return "INSIDE_OR", or_high, or_low


def analyze_market_structure(
    df: pd.DataFrame,
    *,
    breakout_lookback: int = 10,
    max_atr_pct: float = 6.0,
) -> StructureSnapshot:
    """Compute a strictly causal StructureSnapshot over completed bars `df`."""
    if df is None or len(df) < 3:
        return StructureSnapshot(
            trend_regime="SIDEWAYS_CONSOLIDATION",
            breakout_event="NONE",
            pullback_event="NONE",
            reversal_event="NONE",
            vwap_event="UNAVAILABLE",
            ema_cross_event="NONE",
            sr_event="NONE",
            chart_pattern="NONE",
            consolidation_pattern="NONE",
            orb_event="UNAVAILABLE",
            volume_regime="NORMAL_VOLUME",
            volatility_regime="NORMAL",
            rsi_14=None,
            macd_hist=None,
            ema_50=None,
            atr_expansion_ratio=None,
            relative_volume_ratio=None,
            cpr_levels=None,
            opening_range_high=None,
            opening_range_low=None,
            active_scenarios=(),
        )

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    opens = df["open"].astype(float)
    vols = df["volume"].astype(float)

    ema9 = compute_ema_series(closes, 9)
    ema21 = compute_ema_series(closes, 21)
    ema50 = compute_ema_series(closes, 50) if len(df) >= 50 else None
    atr_s = compute_atr_series(highs, lows, closes, period=14, method="sma")
    rsi_s = compute_rsi_series(closes, period=14)
    _, _, macd_hist_s = compute_macd_series(closes)
    rvol_s = compute_rvol_series(vols, window=14, mode="prior_nonzero")
    vwap_s = compute_session_vwap_series(df)

    curr_o = float(opens.iloc[-1])
    curr_h = float(highs.iloc[-1])
    curr_l = float(lows.iloc[-1])
    curr_c = float(closes.iloc[-1])
    curr_v = float(vols.iloc[-1])
    prev_o = float(opens.iloc[-2])
    prev_h = float(highs.iloc[-2])
    prev_l = float(lows.iloc[-2])
    prev_c = float(closes.iloc[-2])

    curr_range = max(1e-9, curr_h - curr_l)
    body_ratio = abs(curr_c - curr_o) / curr_range
    upper_wick_ratio = (curr_h - max(curr_o, curr_c)) / curr_range
    lower_wick_ratio = (min(curr_o, curr_c) - curr_l) / curr_range
    clv = (curr_c - curr_l) / curr_range

    e9_curr = float(ema9.iloc[-1])
    e21_curr = float(ema21.iloc[-1])
    e9_prev = float(ema9.iloc[-2])
    e21_prev = float(ema21.iloc[-2])
    e21_slope_base = float(ema21.iloc[max(0, len(ema21) - 5)])
    ema21_slope_pct = ((e21_curr - e21_slope_base) / e21_slope_base * 100.0) if e21_slope_base > 0 else 0.0

    ema50_val = _safe_float(ema50.iloc[-1]) if ema50 is not None else None
    atr_val = _safe_float(atr_s.iloc[-1])
    rsi_val = _safe_float(rsi_s.iloc[-1])
    macd_hist_val = _safe_float(macd_hist_s.iloc[-1])

    # 18. Volume Confirmation & Yahoo Zero-Volume Unreported Detection
    if curr_v == 0.0 and curr_h > curr_l:
        volume_regime = "VOLUME_UNREPORTED_BY_PROVIDER"
        rel_vol: float | None = None
    else:
        rel_vol = _safe_float(rvol_s.iloc[-1])
        if rel_vol is not None and rel_vol >= 1.25:
            volume_regime = "HIGH_VOLUME_CONFIRMATION"
        elif rel_vol is not None and rel_vol < 0.50:
            volume_regime = "LOW_VOLUME_WEAK"
        else:
            volume_regime = "NORMAL_VOLUME"

    # 19. Volatility Expansion & Contraction
    atr_pct = (atr_val / curr_c * 100.0) if (atr_val is not None and curr_c > 0) else None
    valid_atr = atr_s.dropna()
    atr_ratio: float | None = None
    if atr_val is not None and len(valid_atr) >= 5:
        baseline_atr = float(valid_atr.tail(min(50, len(valid_atr))).mean())
        if baseline_atr > 0:
            atr_ratio = atr_val / baseline_atr

    is_inside_bar = (curr_h <= prev_h and curr_l >= prev_l) and (curr_h < prev_h or curr_l > prev_l)
    if (atr_pct is not None and atr_pct > max_atr_pct) or (atr_val is not None and curr_range > 2.5 * atr_val):
        volatility_regime = "EXTREME_EXPANSION"
    elif atr_ratio is not None and atr_ratio >= 1.25:
        volatility_regime = "EXPANSION"
    elif is_inside_bar or (atr_ratio is not None and atr_ratio <= 0.78):
        volatility_regime = "CONTRACTION"
    else:
        volatility_regime = "NORMAL"

    # 12. VWAP Reclaim & Rejection
    vwap_curr = _safe_float(vwap_s.iloc[-1])
    vwap_prev = _safe_float(vwap_s.iloc[-2]) if len(vwap_s) >= 2 else None
    if vwap_curr is None or vwap_curr <= 0:
        vwap_event = "UNAVAILABLE"
    elif (vwap_prev is not None and prev_c < vwap_prev and curr_c > vwap_curr) or (
        curr_l <= vwap_curr and curr_c > vwap_curr and clv >= 0.65
    ):
        vwap_event = "VWAP_RECLAIM"
    elif (vwap_prev is not None and prev_c > vwap_prev and curr_c < vwap_curr) or (
        curr_h >= vwap_curr and curr_c < vwap_curr and upper_wick_ratio >= 0.35 and clv <= 0.40
    ):
        vwap_event = "VWAP_REJECTION"
    elif curr_c >= vwap_curr:
        vwap_event = "ABOVE_VWAP"
    else:
        vwap_event = "BELOW_VWAP"

    # 13. EMA(9/21) Crossover & Failed Crossover
    ema_cross_event = "NONE"
    if e9_prev <= e21_prev and e9_curr > e21_curr:
        ema_cross_event = "BULLISH_CROSSOVER"
    elif e9_prev >= e21_prev and e9_curr < e21_curr:
        ema_cross_event = "BEARISH_CROSSOVER"
    elif len(df) >= 5:
        # Check for recent crossover within last 3 bars that has now failed (whipsawed back)
        for back in range(2, min(5, len(df))):
            b_prev9, b_prev21 = float(ema9.iloc[-back - 1]), float(ema21.iloc[-back - 1])
            b_curr9, b_curr21 = float(ema9.iloc[-back]), float(ema21.iloc[-back])
            if b_prev9 <= b_prev21 and b_curr9 > b_curr21 and (curr_c < e21_curr or e9_curr < e21_curr):
                ema_cross_event = "FAILED_BULLISH_CROSSOVER"
                break
            if b_prev9 >= b_prev21 and b_curr9 < b_curr21 and (curr_c > e21_curr or e9_curr > e21_curr):
                ema_cross_event = "FAILED_BEARISH_CROSSOVER"
                break

    # 1-5. Trend Regime Classification (Strong/Weak Bullish, Strong/Weak Bearish, Sideways)
    above_vwap_ok = vwap_event in ("ABOVE_VWAP", "VWAP_RECLAIM", "UNAVAILABLE")
    below_vwap_ok = vwap_event in ("BELOW_VWAP", "VWAP_REJECTION", "UNAVAILABLE")
    if curr_c > e9_curr > e21_curr and ema21_slope_pct >= 0.08 and above_vwap_ok and (
        body_ratio >= 0.40 or (rsi_val is not None and rsi_val >= 55.0)
    ):
        trend_regime = "STRONG_BULLISH"
    elif curr_c < e9_curr < e21_curr and ema21_slope_pct <= -0.08 and below_vwap_ok and (
        body_ratio >= 0.40 or (rsi_val is not None and rsi_val <= 45.0)
    ):
        trend_regime = "STRONG_BEARISH"
    elif abs(ema21_slope_pct) < 0.08 and (atr_ratio is None or atr_ratio < 1.10):
        trend_regime = "SIDEWAYS_CONSOLIDATION"
    elif curr_c > e21_curr and ema21_slope_pct > 0:
        trend_regime = "WEAK_BULLISH"
    elif curr_c < e21_curr and ema21_slope_pct < 0:
        trend_regime = "WEAK_BEARISH"
    else:
        trend_regime = "SIDEWAYS_CONSOLIDATION"

    # 6-9. Breakout / Breakdown / 1-Bar & 2-Bar False Breakout / Failed Breakdown
    prior_win = df.iloc[:-1].tail(min(breakout_lookback, len(df) - 1))
    prior_high = float(prior_win["high"].max())
    prior_low = float(prior_win["low"].min())
    prior_before_prev = df.iloc[:-2].tail(min(breakout_lookback, max(1, len(df) - 2))) if len(df) >= 4 else None
    prior_h_prev = float(prior_before_prev["high"].max()) if prior_before_prev is not None else prior_high
    prior_l_prev = float(prior_before_prev["low"].min()) if prior_before_prev is not None else prior_low

    breakout_event = "NONE"
    if (curr_h > prior_high and curr_c < prior_high and clv <= 0.45) or (
        prev_c > prior_h_prev and curr_c <= prior_h_prev
    ):
        breakout_event = "FALSE_BREAKOUT"
    elif (curr_l < prior_low and curr_c > prior_low and clv >= 0.55) or (
        prev_c < prior_l_prev and curr_c >= prior_l_prev
    ):
        breakout_event = "FAILED_BREAKDOWN"
    elif curr_c > prior_high:
        breakout_event = "BULLISH_BREAKOUT"
    elif curr_c < prior_low:
        breakout_event = "BEARISH_BREAKDOWN"

    # 10. Pullback Continuation (to EMA9 / EMA21 / VWAP in trend)
    pullback_event = "NONE"
    touch_buf = (atr_val * 0.35) if atr_val is not None else (curr_c * 0.003)
    if e9_curr > e21_curr and ema21_slope_pct > 0:
        touched_support = (
            min(curr_l, prev_l) <= e9_curr + touch_buf
            or min(curr_l, prev_l) <= e21_curr + touch_buf
            or (vwap_curr is not None and min(curr_l, prev_l) <= vwap_curr + touch_buf)
        )
        if touched_support and curr_c > curr_o and curr_c > e21_curr and (curr_c > prev_h or clv >= 0.65):
            pullback_event = "BULLISH_PULLBACK_CONTINUATION"
    elif e9_curr < e21_curr and ema21_slope_pct < 0:
        touched_resist = (
            max(curr_h, prev_h) >= e9_curr - touch_buf
            or max(curr_h, prev_h) >= e21_curr - touch_buf
            or (vwap_curr is not None and max(curr_h, prev_h) >= vwap_curr - touch_buf)
        )
        if touched_resist and curr_c < curr_o and curr_c < e21_curr and (curr_c < prev_l or clv <= 0.35):
            pullback_event = "BEARISH_PULLBACK_CONTINUATION"

    # 11 & 15. Causal Fractal Pivots -> Trend Reversal (CHOCH) & Head & Shoulders
    pivot_highs, pivot_lows = _find_causal_pivots(highs.to_numpy(), lows.to_numpy(), left=2, right=2)
    reversal_event = "NONE"
    if len(pivot_lows) >= 2 and pivot_lows[-1][1] > pivot_lows[-2][1] and curr_c < pivot_lows[-1][1] and clv <= 0.40:
        reversal_event = "BEARISH_REVERSAL"
    elif len(pivot_highs) >= 2 and pivot_highs[-1][1] < pivot_highs[-2][1] and curr_c > pivot_highs[-1][1] and clv >= 0.60:
        reversal_event = "BULLISH_REVERSAL"

    chart_pattern = _detect_head_and_shoulders(pivot_highs, pivot_lows, curr_c)

    # 14. Support / Resistance & CPR Pivot Rejection
    cpr_levels = _compute_prior_day_cpr(df)
    sr_event = "NONE"
    resist_levels = [prior_high]
    support_levels = [prior_low]
    if cpr_levels:
        resist_levels.extend([cpr_levels["r1"], cpr_levels["r2"], cpr_levels["cpr_tc"]])
        support_levels.extend([cpr_levels["s1"], cpr_levels["s2"], cpr_levels["cpr_bc"]])
    sr_tol = (atr_val * 0.15) if atr_val is not None else (curr_c * 0.0015)
    for lvl in resist_levels:
        if curr_h >= lvl - sr_tol and curr_c < lvl and clv <= 0.40 and upper_wick_ratio >= 0.30:
            sr_event = "RESISTANCE_REJECTION"
            break
    if sr_event == "NONE":
        for lvl in support_levels:
            if curr_l <= lvl + sr_tol and curr_c > lvl and clv >= 0.60 and lower_wick_ratio >= 0.30:
                sr_event = "SUPPORT_REJECTION"
                break

    # 16. Flags, Pennants & Triangles
    consolidation_pattern = _detect_flag_pennant_triangle(df, atr_val)

    # 17. Opening-Range Breakout (ORB) & Failed ORB
    orb_event, or_high, or_low = _detect_orb_event(df)

    # Collect active scenario tags for audit & UI telemetry
    active: list[str] = [trend_regime]
    for tag in (
        breakout_event,
        pullback_event,
        reversal_event,
        vwap_event,
        ema_cross_event,
        sr_event,
        chart_pattern,
        consolidation_pattern,
        orb_event,
        volume_regime,
        volatility_regime,
    ):
        if tag not in ("NONE", "UNAVAILABLE", "NORMAL", "NORMAL_VOLUME", "INSIDE_OR", "FORMING", "ABOVE_VWAP", "BELOW_VWAP"):
            active.append(tag)

    return StructureSnapshot(
        trend_regime=trend_regime,
        breakout_event=breakout_event,
        pullback_event=pullback_event,
        reversal_event=reversal_event,
        vwap_event=vwap_event,
        ema_cross_event=ema_cross_event,
        sr_event=sr_event,
        chart_pattern=chart_pattern,
        consolidation_pattern=consolidation_pattern,
        orb_event=orb_event,
        volume_regime=volume_regime,
        volatility_regime=volatility_regime,
        rsi_14=rsi_val,
        macd_hist=macd_hist_val,
        ema_50=ema50_val,
        atr_expansion_ratio=atr_ratio,
        relative_volume_ratio=rel_vol,
        cpr_levels=cpr_levels,
        opening_range_high=or_high,
        opening_range_low=or_low,
        active_scenarios=tuple(active),
    )
