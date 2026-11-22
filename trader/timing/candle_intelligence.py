"""CYCLE-007 — Numerical Candlestick Intelligence & Price-Action Engine.

Deterministic, fail-closed, lookahead-free numerical analysis of OHLCV bars:
- Strict OHLCV & timestamp/timezone validation (IST-normalized, rejects future bars > as_of_ist)
- Per-candle anatomy (body/wick ratios, range_pct, CLV, directional flags)
- Single- and multi-candle deterministic pattern detection
- Price-action & market structure metrics (swing HH/HL vs LH/LL, EMA alignment,
  session VWAP position, breakout/rejection, ATR, relative volume)
- Multi-timeframe support (1h primary + optional 15m/5m confirmation; honestly
  records UNAVAILABLE when lower timeframes are absent — never fabricates bars)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

from trader.config import IST_TZ
from trader.intelligence.registry import (
    compute_atr_series,
    compute_ema_series,
    compute_latest_session_vwap,
    compute_rvol_series,
)

REQUIRED_OHLCV_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)


class CandleBias(str, Enum):
    """Independent directional lean derived strictly from candlestick & price-action data."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TrendStructure(str, Enum):
    """Recent swing structure classification."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"


class CandlePattern(str, Enum):
    """Deterministic single- and multi-candle numerical patterns."""

    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    BULLISH_PIN_BAR = "BULLISH_PIN_BAR"
    BEARISH_PIN_BAR = "BEARISH_PIN_BAR"
    DOJI = "DOJI"
    BULLISH_MARUBOZU = "BULLISH_MARUBOZU"
    BEARISH_MARUBOZU = "BEARISH_MARUBOZU"
    INSIDE_BAR = "INSIDE_BAR"


BULLISH_PATTERNS: frozenset[CandlePattern] = frozenset(
    {
        CandlePattern.BULLISH_ENGULFING,
        CandlePattern.HAMMER,
        CandlePattern.BULLISH_PIN_BAR,
        CandlePattern.BULLISH_MARUBOZU,
    }
)

BEARISH_PATTERNS: frozenset[CandlePattern] = frozenset(
    {
        CandlePattern.BEARISH_ENGULFING,
        CandlePattern.SHOOTING_STAR,
        CandlePattern.BEARISH_PIN_BAR,
        CandlePattern.BEARISH_MARUBOZU,
    }
)

INDECISION_PATTERNS: frozenset[CandlePattern] = frozenset(
    {
        CandlePattern.DOJI,
        CandlePattern.INSIDE_BAR,
    }
)


@dataclass(frozen=True)
class CandleAnatomy:
    """Numerical anatomical decomposition of a single OHLCV candle."""

    timestamp_ist: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    body_size: float
    candle_range: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    range_pct: float
    close_location_value: float
    is_bullish: bool
    is_bearish: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ist": self.timestamp_ist.isoformat(),
            "open": round(self.open, 4),
            "high": round(self.high, 4),
            "low": round(self.low, 4),
            "close": round(self.close, 4),
            "volume": round(self.volume, 2),
            "body_size": round(self.body_size, 4),
            "candle_range": round(self.candle_range, 4),
            "body_ratio": round(self.body_ratio, 4),
            "upper_wick_ratio": round(self.upper_wick_ratio, 4),
            "lower_wick_ratio": round(self.lower_wick_ratio, 4),
            "range_pct": round(self.range_pct, 4),
            "close_location_value": round(self.close_location_value, 4),
            "is_bullish": self.is_bullish,
            "is_bearish": self.is_bearish,
        }


@dataclass(frozen=True)
class TimeframeAnalysis:
    """Numerical candlestick & price-action analysis for a single timeframe."""

    timeframe: str
    timeframe_status: Literal["AVAILABLE", "UNAVAILABLE"]
    bar_count: int
    latest_timestamp_ist: datetime | None
    latest_candle: CandleAnatomy | None
    detected_patterns: tuple[CandlePattern, ...]
    swing_structure: TrendStructure
    ema_short: float | None
    ema_medium: float | None
    ema_alignment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    session_vwap: float | None
    vwap_position: Literal["ABOVE", "BELOW", "AT", "UNAVAILABLE"]
    vwap_distance_pct: float | None
    breakout_state: Literal[
        "BREAKOUT_HIGH",
        "BREAKDOWN_LOW",
        "REJECTION_HIGH",
        "REJECTION_LOW",
        "INSIDE_RANGE",
        "UNAVAILABLE",
    ]
    recent_swing_high: float | None
    recent_swing_low: float | None
    atr: float | None
    atr_pct: float | None
    relative_volume_ratio: float | None
    price_action_score: float
    directional_lean: CandleBias
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "timeframe_status": self.timeframe_status,
            "bar_count": self.bar_count,
            "latest_timestamp_ist": (
                self.latest_timestamp_ist.isoformat()
                if self.latest_timestamp_ist is not None
                else None
            ),
            "latest_candle": (
                self.latest_candle.to_dict() if self.latest_candle is not None else None
            ),
            "detected_patterns": [p.value for p in self.detected_patterns],
            "swing_structure": self.swing_structure.value,
            "ema_short": round(self.ema_short, 4) if self.ema_short is not None else None,
            "ema_medium": round(self.ema_medium, 4) if self.ema_medium is not None else None,
            "ema_alignment": self.ema_alignment,
            "session_vwap": round(self.session_vwap, 4) if self.session_vwap is not None else None,
            "vwap_position": self.vwap_position,
            "vwap_distance_pct": (
                round(self.vwap_distance_pct, 4) if self.vwap_distance_pct is not None else None
            ),
            "breakout_state": self.breakout_state,
            "recent_swing_high": (
                round(self.recent_swing_high, 4) if self.recent_swing_high is not None else None
            ),
            "recent_swing_low": (
                round(self.recent_swing_low, 4) if self.recent_swing_low is not None else None
            ),
            "atr": round(self.atr, 4) if self.atr is not None else None,
            "atr_pct": round(self.atr_pct, 4) if self.atr_pct is not None else None,
            "relative_volume_ratio": (
                round(self.relative_volume_ratio, 4)
                if self.relative_volume_ratio is not None
                else None
            ),
            "price_action_score": round(self.price_action_score, 4),
            "directional_lean": self.directional_lean.value,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CandleIntelligenceReport:
    """Multi-timeframe numerical candlestick intelligence report for one symbol."""

    symbol: str
    as_of_ist: datetime
    primary_timeframe: str
    primary: TimeframeAnalysis
    confirming_timeframes: dict[str, TimeframeAnalysis]
    composite_price_action_score: float
    directional_lean: CandleBias
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of_ist": self.as_of_ist.isoformat(),
            "primary_timeframe": self.primary_timeframe,
            "primary": self.primary.to_dict(),
            "confirming_timeframes": {
                tf: analysis.to_dict() for tf, analysis in self.confirming_timeframes.items()
            },
            "composite_price_action_score": round(self.composite_price_action_score, 4),
            "directional_lean": self.directional_lean.value,
            "evidence": list(self.evidence),
        }


def _ensure_ist(dt: datetime, label: str) -> datetime:
    if not isinstance(dt, datetime):
        raise ValueError(f"Expected datetime for {label}, got {type(dt).__name__}")
    if dt.tzinfo is None:
        raise ValueError(f"Timezone-naive datetime rejected for {label}: {dt!r}")
    return dt.astimezone(IST_TZ)


def validate_ohlcv_frame(
    df: pd.DataFrame,
    *,
    as_of_ist: datetime,
    symbol: str = "UNKNOWN",
    timeframe: str = "1h",
    min_bars: int = 5,
) -> pd.DataFrame:
    """Strictly validate an OHLCV DataFrame and normalize timestamps to IST.

    Fail-closed checks:
    - Rejects None, non-DataFrame, empty, or fewer than ``min_bars`` rows.
    - Rejects missing OHLCV/timestamp columns.
    - Rejects NaT/null timestamps, timezone-naive timestamps, duplicates, non-monotonic order.
    - Rejects ANY bar with timestamp > ``as_of_ist`` (strict zero-lookahead enforcement).
    - Rejects NaN/Inf/non-numeric OHLCV values, non-positive prices, negative volume,
      and OHLC geometric violations (``high < max(open, close, low)`` or
      ``low > min(open, close, high)``).
    """
    cutoff_ist = _ensure_ist(as_of_ist, "as_of_ist")
    if min_bars < 1:
        raise ValueError("min_bars must be >= 1")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"OHLCV input for {symbol} ({timeframe}) must be a pandas DataFrame")
    if df.empty:
        raise ValueError(f"Empty OHLCV DataFrame rejected for {symbol} ({timeframe})")

    missing = REQUIRED_OHLCV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required OHLCV columns for {symbol} ({timeframe}): {sorted(missing)}"
        )
    if len(df) < min_bars:
        raise ValueError(
            f"Insufficient bars for {symbol} ({timeframe}): got {len(df)}, required >= {min_bars}"
        )

    raw_ts = df["timestamp"]
    if raw_ts.isna().any():
        raise ValueError(f"NaT/null timestamps detected for {symbol} ({timeframe})")

    # Check every element if object dtype to prevent mixed naive/aware timestamps
    if raw_ts.dtype == object:
        parsed_list: list[pd.Timestamp] = []
        for idx, val in enumerate(raw_ts):
            if val is None or pd.isna(val):
                raise ValueError(f"NaT/null timestamp at index {idx} for {symbol} ({timeframe})")
            ts_item = pd.Timestamp(val)
            if ts_item.tzinfo is None:
                raise ValueError(
                    f"Timezone-naive timestamp rejected at index {idx} for {symbol} ({timeframe}): {val!r}"
                )
            parsed_list.append(ts_item.tz_convert(IST_TZ))
        ts_ist = pd.Series(parsed_list, index=df.index)
    else:
        ts_series = pd.to_datetime(raw_ts)
        if ts_series.isna().any():
            raise ValueError(f"NaT/null timestamps detected for {symbol} ({timeframe})")
        if ts_series.dt.tz is None:
            raise ValueError(
                f"Timezone-naive timestamps rejected for {symbol} ({timeframe})"
            )
        ts_ist = ts_series.dt.tz_convert(IST_TZ)

    if ts_ist.duplicated().any():
        raise ValueError(f"Duplicate timestamps detected for {symbol} ({timeframe})")
    if not ts_ist.is_monotonic_increasing:
        raise ValueError(f"Non-monotonic timestamps detected for {symbol} ({timeframe})")

    # Strict lookahead guard: reject any bar with timestamp > as_of_ist
    future_mask = ts_ist > pd.Timestamp(cutoff_ist)
    if future_mask.any():
        first_future = ts_ist[future_mask].iloc[0].isoformat()
        raise ValueError(
            f"Lookahead violation: future-dated bar detected for {symbol} ({timeframe}) "
            f"at {first_future} > as_of_ist {cutoff_ist.isoformat()}"
        )

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    if df[ohlcv_cols].isna().any().any():
        raise ValueError(f"NaN OHLCV values detected for {symbol} ({timeframe})")

    try:
        arr = df[ohlcv_cols].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric OHLCV values detected for {symbol} ({timeframe})"
        ) from exc

    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite (Inf/-Inf) OHLCV values detected for {symbol} ({timeframe})")

    opens = arr[:, 0]
    highs = arr[:, 1]
    lows = arr[:, 2]
    closes = arr[:, 3]
    volumes = arr[:, 4]

    if (opens <= 0).any() or (highs <= 0).any() or (lows <= 0).any() or (closes <= 0).any():
        raise ValueError(f"Non-positive OHLC prices detected for {symbol} ({timeframe})")
    if (volumes < 0).any():
        raise ValueError(f"Negative volume detected for {symbol} ({timeframe})")

    max_oc = np.maximum(opens, closes)
    min_oc = np.minimum(opens, closes)
    if (highs < np.maximum(max_oc, lows)).any():
        raise ValueError(
            f"OHLC invariant violation (high < max(open, close, low)) for {symbol} ({timeframe})"
        )
    if (lows > np.minimum(min_oc, highs)).any():
        raise ValueError(
            f"OHLC invariant violation (low > min(open, close, high)) for {symbol} ({timeframe})"
        )

    clean = pd.DataFrame(
        {
            "timestamp": ts_ist.reset_index(drop=True),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    return clean


def compute_candle_anatomy(
    timestamp_ist: datetime,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume: float,
) -> CandleAnatomy:
    """Compute deterministic per-candle anatomical ratios and directional flags."""
    ts = _ensure_ist(timestamp_ist, "candle.timestamp_ist")
    candle_range = max(0.0, high_px - low_px)
    body_size = abs(close_px - open_px)
    upper_wick = max(0.0, high_px - max(open_px, close_px))
    lower_wick = max(0.0, min(open_px, close_px) - low_px)

    if candle_range > 0.0:
        body_ratio = min(1.0, max(0.0, body_size / candle_range))
        upper_wick_ratio = min(1.0, max(0.0, upper_wick / candle_range))
        lower_wick_ratio = min(1.0, max(0.0, lower_wick / candle_range))
        clv = min(1.0, max(0.0, (close_px - low_px) / candle_range))
    else:
        body_ratio = 0.0
        upper_wick_ratio = 0.0
        lower_wick_ratio = 0.0
        clv = 0.5

    range_pct = (candle_range / open_px) * 100.0 if open_px > 0 else 0.0

    return CandleAnatomy(
        timestamp_ist=ts,
        open=float(open_px),
        high=float(high_px),
        low=float(low_px),
        close=float(close_px),
        volume=float(volume),
        body_size=float(body_size),
        candle_range=float(candle_range),
        body_ratio=float(body_ratio),
        upper_wick_ratio=float(upper_wick_ratio),
        lower_wick_ratio=float(lower_wick_ratio),
        range_pct=float(range_pct),
        close_location_value=float(clv),
        is_bullish=bool(close_px > open_px),
        is_bearish=bool(close_px < open_px),
    )


def detect_candle_patterns(
    curr: CandleAnatomy,
    prev: CandleAnatomy | None = None,
) -> tuple[CandlePattern, ...]:
    """Detect deterministic single- and two-candle numerical patterns."""
    patterns: list[CandlePattern] = []

    # 1. Doji (tiny body relative to range, or zero range)
    if curr.candle_range == 0.0 or curr.body_ratio <= 0.12:
        patterns.append(CandlePattern.DOJI)

    # 2. Marubozu (dominant full body, negligible wicks)
    if curr.candle_range > 0.0 and curr.body_ratio >= 0.80:
        if curr.upper_wick_ratio <= 0.10 and curr.lower_wick_ratio <= 0.10:
            if curr.is_bullish:
                patterns.append(CandlePattern.BULLISH_MARUBOZU)
            elif curr.is_bearish:
                patterns.append(CandlePattern.BEARISH_MARUBOZU)

    # 3. Hammer & Bullish Pin Bar (long lower wick rejection, close near high)
    if curr.candle_range > 0.0:
        if (
            curr.lower_wick_ratio >= 0.55
            and curr.upper_wick_ratio <= 0.18
            and curr.body_ratio <= 0.35
            and curr.close_location_value >= 0.60
        ):
            patterns.append(CandlePattern.HAMMER)
        if (
            curr.lower_wick_ratio >= 0.60
            and curr.body_ratio <= 0.30
            and curr.close_location_value >= 0.65
        ):
            patterns.append(CandlePattern.BULLISH_PIN_BAR)

        # 4. Shooting Star & Bearish Pin Bar (long upper wick rejection, close near low)
        if (
            curr.upper_wick_ratio >= 0.55
            and curr.lower_wick_ratio <= 0.18
            and curr.body_ratio <= 0.35
            and curr.close_location_value <= 0.40
        ):
            patterns.append(CandlePattern.SHOOTING_STAR)
        if (
            curr.upper_wick_ratio >= 0.60
            and curr.body_ratio <= 0.30
            and curr.close_location_value <= 0.35
        ):
            patterns.append(CandlePattern.BEARISH_PIN_BAR)

    # 5. Two-candle patterns (Engulfing & Inside Bar)
    if prev is not None:
        if (
            prev.is_bearish
            and curr.is_bullish
            and prev.body_size > 0.0
            and curr.open <= prev.close
            and curr.close >= prev.open
            and curr.body_size > prev.body_size
        ):
            patterns.append(CandlePattern.BULLISH_ENGULFING)

        if (
            prev.is_bullish
            and curr.is_bearish
            and prev.body_size > 0.0
            and curr.open >= prev.close
            and curr.close <= prev.open
            and curr.body_size > prev.body_size
        ):
            patterns.append(CandlePattern.BEARISH_ENGULFING)

        if (
            curr.high <= prev.high
            and curr.low >= prev.low
            and (curr.high < prev.high or curr.low > prev.low)
        ):
            patterns.append(CandlePattern.INSIDE_BAR)

    return tuple(patterns)


def _compute_swing_structure(df: pd.DataFrame, lookback: int = 12) -> tuple[TrendStructure, float, float]:
    """Classify recent swing structure (UPTREND HH/HL, DOWNTREND LH/LL, or RANGE)."""
    window = df.tail(min(len(df), lookback))
    recent_high = float(window["high"].max())
    recent_low = float(window["low"].min())
    if len(window) < 4:
        return TrendStructure.RANGE, recent_high, recent_low

    mid = len(window) // 2
    first_half = window.iloc[:mid]
    second_half = window.iloc[mid:]

    h1, l1 = float(first_half["high"].max()), float(first_half["low"].min())
    h2, l2 = float(second_half["high"].max()), float(second_half["low"].min())

    if h2 > h1 and l2 > l1:
        return TrendStructure.UPTREND, recent_high, recent_low
    if h2 < h1 and l2 < l1:
        return TrendStructure.DOWNTREND, recent_high, recent_low
    return TrendStructure.RANGE, recent_high, recent_low


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """Compute Average True Range over ``min(period, len(df))`` bars."""
    if df.empty:
        return None
    atr_series = compute_atr_series(
        df["high"], df["low"], df["close"], period=period, method="sma", min_periods=1
    )
    atr_val = float(atr_series.iloc[-1])
    return atr_val if math.isfinite(atr_val) else None


def _compute_session_vwap(df: pd.DataFrame) -> tuple[float | None, Literal["ABOVE", "BELOW", "AT", "UNAVAILABLE"], float | None]:
    """Compute session VWAP strictly on bars belonging to the latest bar's IST date."""
    return compute_latest_session_vwap(df, zero_vol_policy="unavailable")


def _unavailable_timeframe_analysis(timeframe: str, reason: str) -> TimeframeAnalysis:
    return TimeframeAnalysis(
        timeframe=timeframe,
        timeframe_status="UNAVAILABLE",
        bar_count=0,
        latest_timestamp_ist=None,
        latest_candle=None,
        detected_patterns=(),
        swing_structure=TrendStructure.RANGE,
        ema_short=None,
        ema_medium=None,
        ema_alignment="NEUTRAL",
        session_vwap=None,
        vwap_position="UNAVAILABLE",
        vwap_distance_pct=None,
        breakout_state="UNAVAILABLE",
        recent_swing_high=None,
        recent_swing_low=None,
        atr=None,
        atr_pct=None,
        relative_volume_ratio=None,
        price_action_score=0.0,
        directional_lean=CandleBias.NEUTRAL,
        evidence=(),
        warnings=(reason,),
    )


def _score_to_lean(score: float, threshold: float = 0.20) -> CandleBias:
    if score >= threshold:
        return CandleBias.BULLISH
    if score <= -threshold:
        return CandleBias.BEARISH
    return CandleBias.NEUTRAL


class CandleIntelligenceEngine:
    """Deterministic numerical candlestick intelligence & price-action engine."""

    def __init__(
        self,
        *,
        min_bars: int = 5,
        ema_short_span: int = 9,
        ema_medium_span: int = 21,
        atr_period: int = 14,
        swing_lookback: int = 12,
        breakout_lookback: int = 10,
        lean_threshold: float = 0.20,
    ) -> None:
        if min_bars < 2:
            raise ValueError("min_bars must be >= 2")
        if ema_short_span < 2 or ema_medium_span <= ema_short_span:
            raise ValueError("ema_short_span must be >= 2 and < ema_medium_span")
        if atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if swing_lookback < 1:
            raise ValueError("swing_lookback must be >= 1")
        if breakout_lookback < 1:
            raise ValueError("breakout_lookback must be >= 1")
        if not (0.0 < lean_threshold < 1.0):
            raise ValueError("lean_threshold must be in (0, 1)")

        self.min_bars = min_bars
        self.ema_short_span = ema_short_span
        self.ema_medium_span = ema_medium_span
        self.atr_period = atr_period
        self.swing_lookback = swing_lookback
        self.breakout_lookback = breakout_lookback
        self.lean_threshold = lean_threshold

    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        as_of_ist: datetime,
    ) -> TimeframeAnalysis:
        """Validate and analyze a single timeframe's OHLCV bars."""
        clean = validate_ohlcv_frame(
            df,
            as_of_ist=as_of_ist,
            symbol=symbol,
            timeframe=timeframe,
            min_bars=self.min_bars,
        )

        last_row = clean.iloc[-1]
        prev_row = clean.iloc[-2] if len(clean) >= 2 else None

        curr_candle = compute_candle_anatomy(
            timestamp_ist=last_row["timestamp"].to_pydatetime(),
            open_px=float(last_row["open"]),
            high_px=float(last_row["high"]),
            low_px=float(last_row["low"]),
            close_px=float(last_row["close"]),
            volume=float(last_row["volume"]),
        )
        prev_candle = (
            compute_candle_anatomy(
                timestamp_ist=prev_row["timestamp"].to_pydatetime(),
                open_px=float(prev_row["open"]),
                high_px=float(prev_row["high"]),
                low_px=float(prev_row["low"]),
                close_px=float(prev_row["close"]),
                volume=float(prev_row["volume"]),
            )
            if prev_row is not None
            else None
        )

        patterns = detect_candle_patterns(curr_candle, prev_candle)
        swing_structure, recent_high, recent_low = _compute_swing_structure(
            clean, lookback=self.swing_lookback
        )

        # Short & medium EMA alignment
        closes = clean["close"]
        ema_short = float(compute_ema_series(closes, span=self.ema_short_span).iloc[-1])
        ema_medium = float(compute_ema_series(closes, span=self.ema_medium_span).iloc[-1])
        if curr_candle.close > ema_short > ema_medium:
            ema_alignment: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "BULLISH"
        elif curr_candle.close < ema_short < ema_medium:
            ema_alignment = "BEARISH"
        else:
            ema_alignment = "NEUTRAL"

        # Session VWAP
        session_vwap, vwap_position, vwap_distance_pct = _compute_session_vwap(clean)

        # Breakout / rejection vs prior lookback window (excluding latest bar)
        prior_window = clean.iloc[:-1].tail(self.breakout_lookback)
        prior_high = float(prior_window["high"].max())
        prior_low = float(prior_window["low"].min())
        if curr_candle.close > prior_high:
            breakout_state: Literal[
                "BREAKOUT_HIGH",
                "BREAKDOWN_LOW",
                "REJECTION_HIGH",
                "REJECTION_LOW",
                "INSIDE_RANGE",
                "UNAVAILABLE",
            ] = "BREAKOUT_HIGH"
        elif curr_candle.close < prior_low:
            breakout_state = "BREAKDOWN_LOW"
        elif curr_candle.high >= prior_high and curr_candle.close < prior_high and curr_candle.close_location_value <= 0.45:
            breakout_state = "REJECTION_HIGH"
        elif curr_candle.low <= prior_low and curr_candle.close > prior_low and curr_candle.close_location_value >= 0.55:
            breakout_state = "REJECTION_LOW"
        else:
            breakout_state = "INSIDE_RANGE"

        # ATR
        atr = _compute_atr(clean, period=self.atr_period)
        atr_pct = (atr / curr_candle.close) * 100.0 if atr is not None and curr_candle.close > 0 else None

        # Relative volume ratio (vs prior non-zero volume bars)
        rvol_series = compute_rvol_series(
            clean["volume"], window=self.atr_period, mode="prior_nonzero"
        )
        raw_rvol = float(rvol_series.iloc[-1])
        warnings: list[str] = []
        if math.isfinite(raw_rvol):
            rel_vol: float | None = raw_rvol
            if curr_candle.volume == 0:
                warnings.append("Latest bar has zero reported volume")
        else:
            rel_vol = None
            warnings.append("Prior volume history is zero; relative volume unavailable")

        # Bounded price_action_score in [-1.0, +1.0]
        score = 0.0
        evidence: list[str] = []

        # 1. Pattern contribution (capped to [-0.35, +0.35])
        pat_score = 0.0
        for p in patterns:
            if p == CandlePattern.BULLISH_ENGULFING:
                pat_score += 0.30
            elif p == CandlePattern.BEARISH_ENGULFING:
                pat_score -= 0.30
            elif p in (CandlePattern.HAMMER, CandlePattern.BULLISH_PIN_BAR, CandlePattern.BULLISH_MARUBOZU):
                pat_score += 0.22
            elif p in (CandlePattern.SHOOTING_STAR, CandlePattern.BEARISH_PIN_BAR, CandlePattern.BEARISH_MARUBOZU):
                pat_score -= 0.22
        pat_score = max(-0.35, min(0.35, pat_score))
        score += pat_score
        if patterns:
            evidence.append(f"Patterns({timeframe}): {', '.join(p.value for p in patterns)}")

        # 2. Swing structure contribution (±0.20)
        if swing_structure == TrendStructure.UPTREND:
            score += 0.20
            evidence.append(f"Swing structure({timeframe}): UPTREND (HH/HL)")
        elif swing_structure == TrendStructure.DOWNTREND:
            score -= 0.20
            evidence.append(f"Swing structure({timeframe}): DOWNTREND (LH/LL)")
        else:
            evidence.append(f"Swing structure({timeframe}): RANGE")

        # 3. EMA alignment contribution (±0.20)
        if ema_alignment == "BULLISH":
            score += 0.20
            evidence.append(f"EMA alignment({timeframe}): BULLISH (close > EMA{self.ema_short_span} > EMA{self.ema_medium_span})")
        elif ema_alignment == "BEARISH":
            score -= 0.20
            evidence.append(f"EMA alignment({timeframe}): BEARISH (close < EMA{self.ema_short_span} < EMA{self.ema_medium_span})")

        # 4. Session VWAP contribution (±0.10)
        if vwap_position == "ABOVE":
            score += 0.10
            evidence.append(f"VWAP({timeframe}): ABOVE ({vwap_distance_pct:+.2f}%)")
        elif vwap_position == "BELOW":
            score -= 0.10
            evidence.append(f"VWAP({timeframe}): BELOW ({vwap_distance_pct:+.2f}%)")

        # 5. Breakout / rejection contribution (±0.15)
        if breakout_state in ("BREAKOUT_HIGH", "REJECTION_LOW"):
            score += 0.15
            evidence.append(f"Range state({timeframe}): {breakout_state}")
        elif breakout_state in ("BREAKDOWN_LOW", "REJECTION_HIGH"):
            score -= 0.15
            evidence.append(f"Range state({timeframe}): {breakout_state}")

        # 6. CLV tilt ([-0.10, +0.10])
        score += (curr_candle.close_location_value - 0.5) * 0.20

        # Clamp strictly to [-1.0, +1.0]
        score = max(-1.0, min(1.0, score))
        lean = _score_to_lean(score, threshold=self.lean_threshold)

        return TimeframeAnalysis(
            timeframe=timeframe,
            timeframe_status="AVAILABLE",
            bar_count=int(len(clean)),
            latest_timestamp_ist=curr_candle.timestamp_ist,
            latest_candle=curr_candle,
            detected_patterns=patterns,
            swing_structure=swing_structure,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_alignment=ema_alignment,
            session_vwap=session_vwap,
            vwap_position=vwap_position,
            vwap_distance_pct=vwap_distance_pct,
            breakout_state=breakout_state,
            recent_swing_high=recent_high,
            recent_swing_low=recent_low,
            atr=atr,
            atr_pct=atr_pct,
            relative_volume_ratio=rel_vol,
            price_action_score=score,
            directional_lean=lean,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
        )

    def analyze(
        self,
        *,
        symbol: str,
        as_of_ist: datetime,
        bars_primary: pd.DataFrame,
        primary_timeframe: str = "1h",
        bars_15m: pd.DataFrame | None = None,
        bars_5m: pd.DataFrame | None = None,
    ) -> CandleIntelligenceReport:
        """Run multi-timeframe numerical candlestick analysis without fabricating missing timeframes."""
        cutoff_ist = _ensure_ist(as_of_ist, "as_of_ist")
        norm_sym = symbol.strip().upper()
        if not norm_sym:
            raise ValueError("symbol must be a non-empty string")

        primary_analysis = self.analyze_timeframe(
            bars_primary,
            symbol=norm_sym,
            timeframe=primary_timeframe,
            as_of_ist=cutoff_ist,
        )

        confirming: dict[str, TimeframeAnalysis] = {}
        available_confirming_scores: list[float] = []
        all_evidence: list[str] = list(primary_analysis.evidence)

        for tf_label, tf_df in (("15m", bars_15m), ("5m", bars_5m)):
            if tf_df is None:
                confirming[tf_label] = _unavailable_timeframe_analysis(
                    tf_label,
                    f"{tf_label} OHLCV bars not provided (UNAVAILABLE; never fabricated)",
                )
            else:
                tf_analysis = self.analyze_timeframe(
                    tf_df,
                    symbol=norm_sym,
                    timeframe=tf_label,
                    as_of_ist=cutoff_ist,
                )
                confirming[tf_label] = tf_analysis
                available_confirming_scores.append(tf_analysis.price_action_score)
                all_evidence.extend(tf_analysis.evidence)

        if available_confirming_scores:
            avg_confirm = sum(available_confirming_scores) / len(available_confirming_scores)
            composite = max(-1.0, min(1.0, 0.70 * primary_analysis.price_action_score + 0.30 * avg_confirm))
        else:
            composite = primary_analysis.price_action_score

        lean = _score_to_lean(composite, threshold=self.lean_threshold)
        return CandleIntelligenceReport(
            symbol=norm_sym,
            as_of_ist=cutoff_ist,
            primary_timeframe=primary_timeframe,
            primary=primary_analysis,
            confirming_timeframes=confirming,
            composite_price_action_score=composite,
            directional_lean=lean,
            evidence=tuple(all_evidence),
        )
