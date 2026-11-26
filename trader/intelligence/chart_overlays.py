"""CYCLE-015/016/017/018: Causal Institutional Chart-Reading Overlays & Regime Signals.

Implements strictly causal (zero-lookahead) technical overlays derived from 14 authoritative
trading & price-action sources (Zerodha Varsity, Angel One, CFI, StockCharts, Schwab,
Fidelity, TrendSpider, Capital.com, Flattrade, Groww, Warrior Trading, FOREX.com, RoboForex):

1. ADX(14) / +DI / -DI Market Regime Filter (TRENDING vs CHOPPY_SIDEWAYS)
2. Supertrend(10, 3.0) Dynamic ATR Trend Overlay
3. Opening Range Breakout (ORB 15m: 09:15-09:30 IST) & Previous Day High/Low (PDH/PDL)
4. Session VWAP ±1σ / ±2σ Standard Deviation Bands (anti-chase stretch guard)
5. Liquidity Sweeps (stop-hunt wick rejections) & 3-Candle Fair Value Gaps (FVG)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Any, Literal

import numpy as np
import pandas as pd

from trader.config import IST_TZ

MarketRegimeLiteral = Literal["CHOPPY_SIDEWAYS", "TRENDING", "STRONG_EXHAUSTION_WATCH", "WARMING_UP"]
TrendDirLiteral = Literal["BULLISH", "BEARISH", "NEUTRAL"]
OrbStateLiteral = Literal["ORB_FORMING", "INSIDE_ORB", "ORB_BREAKOUT_UP", "ORB_BREAKOUT_DOWN"]
VwapStretchLiteral = Literal["NORMAL", "OVEREXTENDED_ABOVE_2SD", "OVEREXTENDED_BELOW_2SD", "UNAVAILABLE"]
StrategyProfileLiteral = Literal["BALANCED_ALL", "ORB_BREAKOUT", "SUPERTREND_ADX", "LIQUIDITY_REVERSAL"]


@dataclass(frozen=True)
class FairValueGapZone:
    """3-candle Fair Value Gap imbalance pocket."""

    direction: Literal["BULLISH", "BEARISH"]
    low: float
    high: float
    bar_index: int


@dataclass(frozen=True)
class ChartOverlaySnapshot:
    """Strictly causal snapshot of all institutional chart overlays at current bar close."""

    adx_14: float | None
    plus_di_14: float | None
    minus_di_14: float | None
    market_regime: MarketRegimeLiteral
    supertrend_value: float | None
    supertrend_direction: TrendDirLiteral
    pdh: float | None
    pdl: float | None
    orb_high: float | None
    orb_low: float | None
    orb_ready: bool
    orb_state: OrbStateLiteral
    vwap: float | None
    vwap_upper_1sd: float | None
    vwap_lower_1sd: float | None
    vwap_upper_2sd: float | None
    vwap_lower_2sd: float | None
    vwap_stretch: VwapStretchLiteral
    near_key_level: bool
    nearest_key_level_name: str | None
    liquidity_sweep: Literal["BULLISH_SWEEP", "BEARISH_SWEEP", "NONE"]
    recent_fvg: FairValueGapZone | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "adx_14": round(self.adx_14, 2) if self.adx_14 is not None else None,
            "plus_di_14": round(self.plus_di_14, 2) if self.plus_di_14 is not None else None,
            "minus_di_14": round(self.minus_di_14, 2) if self.minus_di_14 is not None else None,
            "market_regime": self.market_regime,
            "supertrend_value": round(self.supertrend_value, 2) if self.supertrend_value is not None else None,
            "supertrend_direction": self.supertrend_direction,
            "pdh": round(self.pdh, 2) if self.pdh is not None else None,
            "pdl": round(self.pdl, 2) if self.pdl is not None else None,
            "orb_high": round(self.orb_high, 2) if self.orb_high is not None else None,
            "orb_low": round(self.orb_low, 2) if self.orb_low is not None else None,
            "orb_ready": self.orb_ready,
            "orb_state": self.orb_state,
            "vwap": round(self.vwap, 2) if self.vwap is not None else None,
            "vwap_upper_1sd": round(self.vwap_upper_1sd, 2) if self.vwap_upper_1sd is not None else None,
            "vwap_lower_1sd": round(self.vwap_lower_1sd, 2) if self.vwap_lower_1sd is not None else None,
            "vwap_upper_2sd": round(self.vwap_upper_2sd, 2) if self.vwap_upper_2sd is not None else None,
            "vwap_lower_2sd": round(self.vwap_lower_2sd, 2) if self.vwap_lower_2sd is not None else None,
            "vwap_stretch": self.vwap_stretch,
            "near_key_level": self.near_key_level,
            "nearest_key_level_name": self.nearest_key_level_name,
            "liquidity_sweep": self.liquidity_sweep,
            "recent_fvg": (
                {
                    "direction": self.recent_fvg.direction,
                    "low": round(self.recent_fvg.low, 2),
                    "high": round(self.recent_fvg.high, 2),
                    "bar_index": self.recent_fvg.bar_index,
                }
                if self.recent_fvg is not None
                else None
            ),
        }


def compute_adx_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[float | None, float | None, float | None, MarketRegimeLiteral]:
    """Compute causal Wilder ADX(14), +DI(14), -DI(14) and classify market regime."""
    n = len(closes)
    if n < period + 2:
        return None, None, None, "WARMING_UP"

    tr = np.zeros(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Wilder smoothing
    sm_tr = float(np.sum(tr[1 : period + 1]))
    sm_pdm = float(np.sum(plus_dm[1 : period + 1]))
    sm_mdm = float(np.sum(minus_dm[1 : period + 1]))

    dx_vals: list[float] = []
    last_pdi = 0.0
    last_mdi = 0.0
    for i in range(period, n):
        if i > period:
            sm_tr = sm_tr - (sm_tr / period) + tr[i]
            sm_pdm = sm_pdm - (sm_pdm / period) + plus_dm[i]
            sm_mdm = sm_mdm - (sm_mdm / period) + minus_dm[i]
        pdi = (100.0 * sm_pdm / sm_tr) if sm_tr > 0.0 else 0.0
        mdi = (100.0 * sm_mdm / sm_tr) if sm_tr > 0.0 else 0.0
        last_pdi, last_mdi = pdi, mdi
        denom = pdi + mdi
        dx = (100.0 * abs(pdi - mdi) / denom) if denom > 0.0 else 0.0
        dx_vals.append(dx)

    if not dx_vals:
        return None, None, None, "WARMING_UP"

    if len(dx_vals) < period:
        adx = float(np.mean(dx_vals))
    else:
        adx = float(np.mean(dx_vals[:period]))
        for dx in dx_vals[period:]:
            adx = ((adx * (period - 1)) + dx) / period

    if adx < 20.0:
        regime: MarketRegimeLiteral = "CHOPPY_SIDEWAYS"
    elif adx <= 40.0:
        regime = "TRENDING"
    else:
        regime = "STRONG_EXHAUSTION_WATCH"
    return adx, last_pdi, last_mdi, regime


def compute_supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[float | None, TrendDirLiteral]:
    """Compute causal Supertrend(period, multiplier) value and BULLISH/BEARISH direction."""
    n = len(closes)
    if n < period + 2:
        return None, "NEUTRAL"

    tr = np.zeros(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr = np.zeros(n, dtype=float)
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period

    hl2 = (highs + lows) * 0.5
    final_upper = np.zeros(n, dtype=float)
    final_lower = np.zeros(n, dtype=float)
    direction = np.ones(n, dtype=int)  # 1 = BULLISH, -1 = BEARISH

    for i in range(period - 1, n):
        basic_upper = hl2[i] + multiplier * atr[i]
        basic_lower = hl2[i] - multiplier * atr[i]
        if i == period - 1:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            direction[i] = 1 if closes[i] >= hl2[i] else -1
            continue

        final_upper[i] = (
            basic_upper
            if (basic_upper < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower
            if (basic_lower > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if direction[i - 1] == 1:
            direction[i] = -1 if closes[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if closes[i] > final_upper[i] else -1

    last_dir: TrendDirLiteral = "BULLISH" if direction[-1] == 1 else "BEARISH"
    last_val = float(final_lower[-1] if direction[-1] == 1 else final_upper[-1])
    return last_val, last_dir


def compute_vwap_bands(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None, float | None, VwapStretchLiteral]:
    """Compute causal Session VWAP and ±1σ / ±2σ volume-weighted standard-deviation bands."""
    if len(closes) == 0:
        return None, None, None, None, None, "UNAVAILABLE"
    total_vol = float(np.sum(volumes))
    if total_vol <= 0.0:
        return None, None, None, None, None, "UNAVAILABLE"

    tp = (highs + lows + closes) / 3.0
    vwap = float(np.sum(tp * volumes) / total_vol)
    var = float(np.sum(volumes * ((tp - vwap) ** 2)) / total_vol)
    sd = float(np.sqrt(max(0.0, var)))
    u1, l1 = vwap + sd, vwap - sd
    u2, l2 = vwap + 2.0 * sd, vwap - 2.0 * sd

    last_c = float(closes[-1])
    if sd > 0.0 and last_c > u2:
        stretch: VwapStretchLiteral = "OVEREXTENDED_ABOVE_2SD"
    elif sd > 0.0 and last_c < l2:
        stretch = "OVEREXTENDED_BELOW_2SD"
    else:
        stretch = "NORMAL"
    return vwap, u1, l1, u2, l2, stretch


def detect_liquidity_sweeps_and_fvg(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_lookback: int = 10,
) -> tuple[Literal["BULLISH_SWEEP", "BEARISH_SWEEP", "NONE"], FairValueGapZone | None]:
    """Detect causal stop-hunt liquidity sweeps and recent 3-candle Fair Value Gaps."""
    n = len(closes)
    sweep: Literal["BULLISH_SWEEP", "BEARISH_SWEEP", "NONE"] = "NONE"
    if n >= swing_lookback + 2:
        prior_low = float(np.min(lows[n - swing_lookback - 1 : n - 1]))
        prior_high = float(np.max(highs[n - swing_lookback - 1 : n - 1]))
        if lows[-1] < prior_low and closes[-1] > prior_low:
            sweep = "BULLISH_SWEEP"
        elif highs[-1] > prior_high and closes[-1] < prior_high:
            sweep = "BEARISH_SWEEP"

    recent_fvg: FairValueGapZone | None = None
    scan_start = max(2, n - 12)
    for i in range(n - 1, scan_start - 1, -1):
        if lows[i] > highs[i - 2]:
            recent_fvg = FairValueGapZone(
                direction="BULLISH",
                low=float(highs[i - 2]),
                high=float(lows[i]),
                bar_index=i,
            )
            break
        if highs[i] < lows[i - 2]:
            recent_fvg = FairValueGapZone(
                direction="BEARISH",
                low=float(highs[i]),
                high=float(lows[i - 2]),
                bar_index=i,
            )
            break
    return sweep, recent_fvg


def compute_chart_overlays(
    causal_df: pd.DataFrame,
    session_date: date,
) -> ChartOverlaySnapshot:
    """Compute all causal institutional overlays strictly from `causal_df` (zero lookahead)."""
    if causal_df is None or causal_df.empty:
        return ChartOverlaySnapshot(
            adx_14=None, plus_di_14=None, minus_di_14=None, market_regime="WARMING_UP",
            supertrend_value=None, supertrend_direction="NEUTRAL",
            pdh=None, pdl=None, orb_high=None, orb_low=None, orb_ready=False, orb_state="ORB_FORMING",
            vwap=None, vwap_upper_1sd=None, vwap_lower_1sd=None, vwap_upper_2sd=None, vwap_lower_2sd=None,
            vwap_stretch="UNAVAILABLE", near_key_level=False, nearest_key_level_name=None,
            liquidity_sweep="NONE", recent_fvg=None,
        )

    highs = causal_df["high"].astype(float).to_numpy()
    lows = causal_df["low"].astype(float).to_numpy()
    closes = causal_df["close"].astype(float).to_numpy()
    volumes = causal_df["volume"].astype(float).to_numpy() if "volume" in causal_df.columns else np.zeros(len(closes))

    adx, pdi, mdi, regime = compute_adx_regime(highs, lows, closes)
    st_val, st_dir = compute_supertrend(highs, lows, closes)
    sweep, fvg = detect_liquidity_sweeps_and_fvg(highs, lows, closes)

    # Extract timestamps for PDH/PDL, ORB 15m, and Session-only VWAP bands
    if isinstance(causal_df.index, pd.DatetimeIndex):
        ts_series = pd.Series(causal_df.index)
    elif "ts_open_ist" in causal_df.columns:
        ts_series = pd.to_datetime(causal_df["ts_open_ist"])
    elif "timestamp" in causal_df.columns:
        ts_series = pd.to_datetime(causal_df["timestamp"])
    else:
        ts_series = pd.Series(dtype="datetime64[ns, Asia/Kolkata]")

    pdh: float | None = None
    pdl: float | None = None
    orb_high: float | None = None
    orb_low: float | None = None
    orb_ready = False
    orb_state: OrbStateLiteral = "ORB_FORMING"
    vwap = u1 = l1 = u2 = l2 = None
    vwap_stretch: VwapStretchLiteral = "UNAVAILABLE"

    if len(ts_series) == len(causal_df):
        ts_ist = ts_series.dt.tz_convert(IST_TZ) if ts_series.dt.tz is not None else ts_series.dt.tz_localize(IST_TZ)
        dates_arr = ts_ist.dt.date.to_numpy()
        times_arr = ts_ist.dt.time.to_numpy()

        prior_mask = dates_arr < session_date
        if np.any(prior_mask):
            prev_day = max(dates_arr[prior_mask])
            prev_mask = dates_arr == prev_day
            pdh = float(np.max(highs[prev_mask]))
            pdl = float(np.min(lows[prev_mask]))

        sess_mask = dates_arr == session_date
        if np.any(sess_mask):
            vwap, u1, l1, u2, l2, vwap_stretch = compute_vwap_bands(
                highs[sess_mask], lows[sess_mask], closes[sess_mask], volumes[sess_mask]
            )
            orb_mask = sess_mask & (times_arr >= time(9, 15, 0)) & (times_arr < time(9, 30, 0))
            if np.any(orb_mask):
                orb_high = float(np.max(highs[orb_mask]))
                orb_low = float(np.min(lows[orb_mask]))
                last_t = times_arr[sess_mask][-1]
                if last_t >= time(9, 29, 0):
                    orb_ready = True
                    last_c = float(closes[-1])
                    if last_c > orb_high:
                        orb_state = "ORB_BREAKOUT_UP"
                    elif last_c < orb_low:
                        orb_state = "ORB_BREAKOUT_DOWN"
                    else:
                        orb_state = "INSIDE_ORB"

    # Location checklist (RoboForex / Capital.com): within 0.35% of PDH, PDL, ORB H/L, or VWAP
    last_c = float(closes[-1])
    key_levels: list[tuple[str, float]] = []
    if pdh is not None:
        key_levels.append(("PDH", pdh))
    if pdl is not None:
        key_levels.append(("PDL", pdl))
    if orb_ready and orb_high is not None:
        key_levels.append(("ORB_HIGH", orb_high))
    if orb_ready and orb_low is not None:
        key_levels.append(("ORB_LOW", orb_low))
    if vwap is not None:
        key_levels.append(("VWAP", vwap))

    near_key = False
    nearest_name: str | None = None
    best_dist = float("inf")
    for name, lvl in key_levels:
        dist_pct = abs(last_c - lvl) / max(0.05, last_c)
        if dist_pct < best_dist:
            best_dist = dist_pct
            nearest_name = name
    if best_dist <= 0.0035:
        near_key = True

    return ChartOverlaySnapshot(
        adx_14=adx,
        plus_di_14=pdi,
        minus_di_14=mdi,
        market_regime=regime,
        supertrend_value=st_val,
        supertrend_direction=st_dir,
        pdh=pdh,
        pdl=pdl,
        orb_high=orb_high,
        orb_low=orb_low,
        orb_ready=orb_ready,
        orb_state=orb_state,
        vwap=vwap,
        vwap_upper_1sd=u1,
        vwap_lower_1sd=l1,
        vwap_upper_2sd=u2,
        vwap_lower_2sd=l2,
        vwap_stretch=vwap_stretch,
        near_key_level=near_key,
        nearest_key_level_name=nearest_name if near_key else None,
        liquidity_sweep=sweep,
        recent_fvg=fvg,
    )


def infer_auto_market_direction(
    mtf_5m_trend: TrendDirLiteral,
    overlays: ChartOverlaySnapshot,
    current_close: float,
) -> TrendDirLiteral:
    """Causally infer live intraday directional bias from 4 real-time chart votes.

    Votes (+1 BULLISH / -1 BEARISH):
    1. 5m MTF EMA trend
    2. Supertrend(10, 3.0) direction
    3. Session VWAP side (close > vwap vs close < vwap)
    4. ORB 15m breakout direction (post-09:30 IST)

    Returns BULLISH if score >= +2, BEARISH if score <= -2, else NEUTRAL.
    """
    score = 0
    if mtf_5m_trend == "BULLISH":
        score += 1
    elif mtf_5m_trend == "BEARISH":
        score -= 1

    if overlays.supertrend_direction == "BULLISH":
        score += 1
    elif overlays.supertrend_direction == "BEARISH":
        score -= 1

    if overlays.vwap is not None:
        if current_close > overlays.vwap:
            score += 1
        elif current_close < overlays.vwap:
            score -= 1

    if overlays.orb_ready:
        if overlays.orb_state == "ORB_BREAKOUT_UP":
            score += 1
        elif overlays.orb_state == "ORB_BREAKOUT_DOWN":
            score -= 1

    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"

