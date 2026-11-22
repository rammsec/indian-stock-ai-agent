"""Canonical OHLCV data contract, provenance tiering, and causal visibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from trader.config import IST_TZ


class ProvenanceTier(str, Enum):
    """Unified market data provenance classification."""

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


NON_AUTHORITATIVE_PROVIDERS: frozenset[str] = frozenset(
    {"yahoo_finance_1h", "yahoo_finance_5m"}
)


class LookaheadViolationError(ValueError):
    """Raised when market data violates causal visibility (ts_event_ist <= ts_init_ist <= as_of_ist)."""


_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "60m": timedelta(hours=1),
    "1d": timedelta(days=1),
    "1D": timedelta(days=1),
}

_CANONICAL_COLUMNS = [
    "ts_open_ist",
    "ts_close_ist",
    "ts_init_ist",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def cap_provenance_tier(
    tier: ProvenanceTier | str,
    provider: str,
) -> ProvenanceTier | str:
    """Cap provenance tier so non-authoritative providers can NEVER emit VERIFIED."""
    is_non_auth = provider in NON_AUTHORITATIVE_PROVIDERS or provider.startswith(
        "yahoo_finance"
    )
    tier_val = tier.value if isinstance(tier, ProvenanceTier) else str(tier)
    if is_non_auth and tier_val == ProvenanceTier.VERIFIED.value:
        return ProvenanceTier.PARTIALLY_VERIFIED
    if isinstance(tier, ProvenanceTier):
        return tier
    if tier_val in ProvenanceTier._value2member_map_:
        return ProvenanceTier(tier_val)
    return tier


def _ensure_tz_aware_ist(dt: datetime, label: str) -> datetime:
    if not isinstance(dt, datetime):
        raise ValueError(f"{label} must be a datetime, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timezone-naive datetime rejected for {label}: {dt!r}")
    return dt.astimezone(IST_TZ)


def _coerce_ts_series_ist(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"NaT/null timestamp rejected in column {label}")
    # Check element-wise if object dtype to prevent silent localization of naive datetimes
    if pd.api.types.is_object_dtype(series):
        converted: list[datetime] = []
        for item in series.tolist():
            if isinstance(item, pd.Timestamp):
                if item.tzinfo is None:
                    raise ValueError(f"Timezone-naive timestamp rejected in {label}: {item!r}")
                converted.append(item.tz_convert(IST_TZ).to_pydatetime())
            elif isinstance(item, datetime):
                converted.append(_ensure_tz_aware_ist(item, label))
            else:
                dt = pd.to_datetime(item)
                if dt.tzinfo is None:
                    raise ValueError(f"Timezone-naive timestamp rejected in {label}: {item!r}")
                converted.append(dt.tz_convert(IST_TZ).to_pydatetime())
        return pd.Series(pd.to_datetime(converted, utc=True).tz_convert(IST_TZ), index=series.index)

    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError(f"Column {label} must be datetime-like, got {series.dtype}")
    if getattr(series.dt, "tz", None) is None:
        raise ValueError(f"Timezone-naive timestamp series rejected for {label}")
    return series.dt.tz_convert(IST_TZ)


@dataclass(frozen=True)
class OHLCVEvidenceFrame:
    """Immutable, causally validated OHLCV evidence container."""

    symbol: str
    exchange: str
    timeframe: str
    as_of_ist: datetime
    provenance_status: ProvenanceTier
    source_name: str
    source_sha256: str | None
    frame: pd.DataFrame
    bar_count: int
    latest_ts_event_ist: datetime | None
    latest_ts_init_ist: datetime | None
    staleness_hours: float | None
    missing_reason: str | None = None


def build_ohlcv_evidence_frame(
    df: pd.DataFrame | None,
    *,
    symbol: str,
    timeframe: str,
    as_of_ist: datetime,
    provenance_status: ProvenanceTier | str,
    source_name: str,
    exchange: str = "NSE",
    source_sha256: str | None = None,
    bar_duration: timedelta | None = None,
    ts_init_ist: datetime | None = None,
    missing_reason: str | None = None,
) -> OHLCVEvidenceFrame:
    """Validate OHLCV bars and enforce ts_event_ist <= ts_init_ist <= as_of_ist."""
    as_of_norm = _ensure_tz_aware_ist(as_of_ist, "as_of_ist")
    capped_status = cap_provenance_tier(provenance_status, source_name)
    if not isinstance(capped_status, ProvenanceTier):
        raise ValueError(f"Invalid provenance_status: {provenance_status!r}")

    if df is None or df.empty:
        if capped_status != ProvenanceTier.UNAVAILABLE:
            raise ValueError(
                f"Empty OHLCV frame rejected for {symbol} ({timeframe}) unless provenance_status is UNAVAILABLE"
            )
        empty_df = pd.DataFrame(columns=_CANONICAL_COLUMNS)
        return OHLCVEvidenceFrame(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            as_of_ist=as_of_norm,
            provenance_status=ProvenanceTier.UNAVAILABLE,
            source_name=source_name,
            source_sha256=source_sha256,
            frame=empty_df,
            bar_count=0,
            latest_ts_event_ist=None,
            latest_ts_init_ist=None,
            staleness_hours=None,
            missing_reason=missing_reason or "TIMEFRAME_UNAVAILABLE",
        )

    required_ohlcv = {"open", "high", "low", "close", "volume"}
    missing_cols = required_ohlcv - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns for {symbol}: {sorted(missing_cols)}")

    work = df.copy().reset_index(drop=True)

    # Resolve ts_open_ist
    if "ts_open_ist" in work.columns:
        ts_open = _coerce_ts_series_ist(work["ts_open_ist"], "ts_open_ist")
    elif "timestamp" in work.columns:
        ts_open = _coerce_ts_series_ist(work["timestamp"], "timestamp")
    else:
        raise ValueError(f"Missing 'ts_open_ist' or 'timestamp' column for {symbol}")

    # Resolve ts_close_ist (ts_event_ist)
    if "ts_close_ist" in work.columns:
        ts_close = _coerce_ts_series_ist(work["ts_close_ist"], "ts_close_ist")
    else:
        delta = bar_duration or _TIMEFRAME_DELTAS.get(timeframe)
        if delta is None:
            raise ValueError(
                f"Cannot infer bar_duration for timeframe {timeframe!r}; provide 'ts_close_ist' or bar_duration"
            )
        ts_close = ts_open + delta

    # Resolve ts_init_ist
    if "ts_init_ist" in work.columns:
        ts_init = _coerce_ts_series_ist(work["ts_init_ist"], "ts_init_ist")
    elif ts_init_ist is not None:
        init_norm = _ensure_tz_aware_ist(ts_init_ist, "ts_init_ist")
        ts_init = pd.Series([init_norm] * len(work), index=work.index)
    else:
        ts_init = ts_close.copy()

    # Monotonicity and duplicate checks on ts_close_ist
    if not ts_close.is_monotonic_increasing:
        raise ValueError(f"Non-monotonic ts_close_ist detected for {symbol} ({timeframe})")
    if ts_close.duplicated().any():
        raise ValueError(f"Duplicate ts_close_ist detected for {symbol} ({timeframe})")

    # Numeric & OHLC integrity checks
    for col in ("open", "high", "low", "close", "volume"):
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite (NaN/Inf) values in '{col}' for {symbol} ({timeframe})")
        if col == "volume":
            if (vals < 0.0).any():
                raise ValueError(f"Negative volume rejected for {symbol} ({timeframe})")
        else:
            if (vals <= 0.0).any():
                raise ValueError(f"Non-positive price in '{col}' rejected for {symbol} ({timeframe})")
        work[col] = vals

    if (work["high"] < work[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"OHLC violation (high < max(open, close)) for {symbol} ({timeframe})")
    if (work["low"] > work[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"OHLC violation (low > min(open, close)) for {symbol} ({timeframe})")
    if (work["high"] < work["low"]).any():
        raise ValueError(f"OHLC violation (high < low) for {symbol} ({timeframe})")

    # Causal visibility invariant: ts_event_ist <= ts_init_ist <= as_of_ist
    if (ts_close > ts_init).any():
        raise LookaheadViolationError(
            f"Lookahead violation for {symbol} ({timeframe}): ts_event_ist > ts_init_ist"
        )
    if (ts_init > as_of_norm).any() or (ts_close > as_of_norm).any():
        raise LookaheadViolationError(
            f"Lookahead violation for {symbol} ({timeframe}): bar ts_close_ist/ts_init_ist > as_of_ist ({as_of_norm.isoformat()})"
        )

    canonical_df = pd.DataFrame(
        {
            "ts_open_ist": ts_open,
            "ts_close_ist": ts_close,
            "ts_init_ist": ts_init,
            "open": work["open"].astype(float),
            "high": work["high"].astype(float),
            "low": work["low"].astype(float),
            "close": work["close"].astype(float),
            "volume": work["volume"].astype(float),
        }
    )

    latest_event = canonical_df["ts_close_ist"].iloc[-1].to_pydatetime()
    latest_init = canonical_df["ts_init_ist"].iloc[-1].to_pydatetime()
    staleness_hours = (as_of_norm - latest_event).total_seconds() / 3600.0

    return OHLCVEvidenceFrame(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        as_of_ist=as_of_norm,
        provenance_status=capped_status,
        source_name=source_name,
        source_sha256=source_sha256,
        frame=canonical_df,
        bar_count=len(canonical_df),
        latest_ts_event_ist=latest_event,
        latest_ts_init_ist=latest_init,
        staleness_hours=staleness_hours,
        missing_reason=missing_reason,
    )
