"""Canonical feature registry and pure, lookahead-free indicator functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal

import numpy as np
import pandas as pd


class DuplicateFeatureRegistrationError(ValueError):
    """Raised when registering a feature_id that already exists in a FeatureRegistry."""


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable specification of a canonical feature formula variant."""

    feature_id: str
    formula_variant: str
    min_bars_required: int
    output_dtype: str
    is_recursive: bool
    description: str


class FeatureRegistry:
    """Immutable-view registry enforcing one canonical formula_variant per feature_id."""

    def __init__(self, specs: Iterable[FeatureSpec] = ()) -> None:
        self._specs: dict[str, FeatureSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: FeatureSpec) -> None:
        """Register a FeatureSpec, raising DuplicateFeatureRegistrationError on collision."""
        if spec.feature_id in self._specs:
            raise DuplicateFeatureRegistrationError(
                f"Duplicate feature_id registration rejected: {spec.feature_id!r}"
            )
        self._specs[spec.feature_id] = spec

    @property
    def specs(self) -> MappingProxyType[str, FeatureSpec]:
        """Read-only mapping view of registered features."""
        return MappingProxyType(self._specs)

    def get(self, feature_id: str) -> FeatureSpec | None:
        return self._specs.get(feature_id)

    def __contains__(self, feature_id: object) -> bool:
        return feature_id in self._specs

    def __getitem__(self, feature_id: str) -> FeatureSpec:
        return self._specs[feature_id]

    def __len__(self) -> int:
        return len(self._specs)


def compute_ema_series(
    series: pd.Series,
    span: int,
    min_periods: int = 1,
) -> pd.Series:
    """Calculate Exponential Moving Average with adjust=False."""
    return series.astype(float).ewm(span=span, min_periods=min_periods, adjust=False).mean()


def compute_atr_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    method: Literal["wilder", "sma"] = "wilder",
    min_periods: int = 1,
) -> pd.Series:
    """Calculate Average True Range using either canonical Wilder EWM or legacy SMA."""
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if len(true_range) > 0:
        true_range.iloc[0] = h.iloc[0] - l.iloc[0]

    if method == "wilder":
        return true_range.ewm(
            alpha=1.0 / period, min_periods=min_periods, adjust=False
        ).mean()
    if method == "sma":
        return true_range.rolling(window=period, min_periods=min_periods).mean()
    raise ValueError(f"Unsupported ATR method: {method!r}. Expected 'wilder' or 'sma'.")


def _extract_ist_dates(df: pd.DataFrame) -> pd.Series:
    ts_col = (
        "ts_open_ist"
        if "ts_open_ist" in df.columns
        else ("timestamp" if "timestamp" in df.columns else "ts_close_ist")
    )
    if ts_col not in df.columns:
        raise ValueError("DataFrame must contain 'ts_open_ist', 'timestamp', or 'ts_close_ist'")
    ts = df[ts_col]
    if getattr(ts.dt, "tz", None) is None:
        raise ValueError(f"Timezone-naive timestamp series rejected in {ts_col}")
    return ts.dt.tz_convert("Asia/Kolkata").dt.date


def compute_session_vwap_series(
    df: pd.DataFrame,
    zero_vol_policy: Literal["typical_price", "unavailable"] = "typical_price",
) -> pd.Series:
    """Calculate cumulative session VWAP resetting daily at 09:15 IST."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    typical_price = (
        df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)
    ) / 3.0
    vol = df["volume"].astype(float)
    tp_vol = typical_price * vol

    dates = _extract_ist_dates(df)
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    if zero_vol_policy == "typical_price":
        return vwap.fillna(typical_price).ffill()
    if zero_vol_policy == "unavailable":
        return vwap
    raise ValueError(
        f"Unsupported zero_vol_policy: {zero_vol_policy!r}. Expected 'typical_price' or 'unavailable'."
    )


def compute_latest_session_vwap(
    df: pd.DataFrame,
    zero_vol_policy: Literal["typical_price", "unavailable"] = "unavailable",
) -> tuple[float | None, Literal["ABOVE", "BELOW", "AT", "UNAVAILABLE"], float | None]:
    """Compute latest session VWAP, relative position, and percentage distance."""
    if df.empty:
        return None, "UNAVAILABLE", None

    vwap_series = compute_session_vwap_series(df, zero_vol_policy=zero_vol_policy)
    vwap = float(vwap_series.iloc[-1])
    if not math.isfinite(vwap) or vwap <= 0.0:
        return None, "UNAVAILABLE", None

    latest_close = float(df["close"].iloc[-1])
    dist_pct = ((latest_close - vwap) / vwap) * 100.0
    if abs(dist_pct) <= 0.05:
        pos: Literal["ABOVE", "BELOW", "AT", "UNAVAILABLE"] = "AT"
    elif latest_close > vwap:
        pos = "ABOVE"
    else:
        pos = "BELOW"
    return vwap, pos, dist_pct


def compute_rsi_series(
    close: pd.Series,
    period: int = 14,
    warmup_fill: float | None = 50.0,
    zero_loss_mode: Literal["legacy_50", "strict_100"] = "legacy_50",
) -> pd.Series:
    """Calculate Wilder's RSI with explicit zero-loss handling."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    if zero_loss_mode == "legacy_50":
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if warmup_fill is not None:
            rsi = rsi.fillna(warmup_fill)
        return rsi

    if zero_loss_mode == "strict_100":
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        zero_loss_mask = (avg_loss == 0.0) & avg_gain.notna()
        rsi = rsi.mask(zero_loss_mask & (avg_gain > 0.0), 100.0)
        rsi = rsi.mask(zero_loss_mask & (avg_gain == 0.0), 50.0)
        if warmup_fill is not None:
            rsi = rsi.fillna(warmup_fill)
        return rsi

    raise ValueError(
        f"Unsupported zero_loss_mode: {zero_loss_mode!r}. Expected 'legacy_50' or 'strict_100'."
    )


def compute_macd_series(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    fast_ema = compute_ema_series(close, span=fast)
    slow_ema = compute_ema_series(close, span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema_series(macd_line, span=signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_cpr_pivots_dict(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> dict[str, float]:
    """Compute Central Pivot Range (CPR) and classic Floor Pivots from prior session OHLC."""
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


def compute_rvol_series(
    volume: pd.Series,
    window: int = 20,
    mode: Literal["inclusive", "prior_nonzero", "prior_floored"] = "inclusive",
) -> pd.Series:
    """Calculate Relative Volume (RVOL) with explicit current-bar and zero-volume semantics."""
    vol = volume.astype(float)

    if mode == "inclusive":
        vol_sma = vol.rolling(window=window, min_periods=1).mean()
        return vol / vol_sma.replace(0, np.nan)

    if mode == "prior_floored":
        prior_mean = vol.shift(1).rolling(window=window, min_periods=1).mean()
        return vol / prior_mean.replace(0, np.nan)

    if mode == "prior_nonzero":
        vals = vol.to_numpy(dtype=float)
        out = np.full(len(vals), np.nan, dtype=float)
        for i in range(len(vals)):
            curr = vals[i]
            if curr == 0.0:
                out[i] = 0.0
                continue
            start = max(0, i - window)
            prior = vals[start:i]
            nonzero = prior[prior > 0.0]
            if len(nonzero) > 0 and curr > 0.0:
                out[i] = curr / float(np.mean(nonzero))
        return pd.Series(out, index=volume.index, dtype=float)

    raise ValueError(
        f"Unsupported RVOL mode: {mode!r}. Expected 'inclusive', 'prior_nonzero', or 'prior_floored'."
    )


def compute_supertrend_series(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend (band value, in_uptrend boolean) using canonical Wilder ATR."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index), pd.Series(dtype=bool, index=df.index)

    hl2 = (df["high"].astype(float) + df["low"].astype(float)) / 2.0
    atr = compute_atr_series(
        df["high"], df["low"], df["close"], period=period, method="wilder"
    )

    upperband = (hl2 + (multiplier * atr)).copy()
    lowerband = (hl2 - (multiplier * atr)).copy()
    closes = df["close"].astype(float)

    in_uptrend = pd.Series(True, index=df.index, dtype=bool)
    supertrend = pd.Series(0.0, index=df.index, dtype=float)
    if len(df) > 0:
        supertrend.iloc[0] = float(lowerband.iloc[0])

    for i in range(1, len(df)):
        curr_close = float(closes.iloc[i])
        prev_close = float(closes.iloc[i - 1])

        if not (
            lowerband.iloc[i] > lowerband.iloc[i - 1]
            or prev_close < lowerband.iloc[i - 1]
        ):
            lowerband.iloc[i] = lowerband.iloc[i - 1]

        if not (
            upperband.iloc[i] < upperband.iloc[i - 1]
            or prev_close > upperband.iloc[i - 1]
        ):
            upperband.iloc[i] = upperband.iloc[i - 1]

        if bool(in_uptrend.iloc[i - 1]):
            in_uptrend.iloc[i] = not (curr_close < float(lowerband.iloc[i]))
        else:
            in_uptrend.iloc[i] = curr_close > float(upperband.iloc[i])

        supertrend.iloc[i] = (
            lowerband.iloc[i] if bool(in_uptrend.iloc[i]) else upperband.iloc[i]
        )

    return supertrend, in_uptrend


DEFAULT_FEATURE_REGISTRY = FeatureRegistry(
    specs=[
        FeatureSpec(
            feature_id="atr_wilder_14",
            formula_variant="wilder_ewm_alpha_1_over_14",
            min_bars_required=14,
            output_dtype="float64",
            is_recursive=True,
            description="Canonical 14-period Wilder EWM Average True Range (alpha=1/14, adjust=False).",
        ),
        FeatureSpec(
            feature_id="atr_sma_14",
            formula_variant="sma_14_legacy",
            min_bars_required=14,
            output_dtype="float64",
            is_recursive=False,
            description="Explicit legacy 14-period SMA True Range for CYCLE-007/008 parity.",
        ),
        FeatureSpec(
            feature_id="ema_9",
            formula_variant="ewm_span_9_adjust_false",
            min_bars_required=9,
            output_dtype="float64",
            is_recursive=True,
            description="9-period Exponential Moving Average (adjust=False).",
        ),
        FeatureSpec(
            feature_id="ema_21",
            formula_variant="ewm_span_21_adjust_false",
            min_bars_required=21,
            output_dtype="float64",
            is_recursive=True,
            description="21-period Exponential Moving Average (adjust=False).",
        ),
        FeatureSpec(
            feature_id="ema_50",
            formula_variant="ewm_span_50_adjust_false",
            min_bars_required=50,
            output_dtype="float64",
            is_recursive=True,
            description="50-period Exponential Moving Average (adjust=False).",
        ),
        FeatureSpec(
            feature_id="session_vwap_ist",
            formula_variant="ist_session_cumsum_typical_price",
            min_bars_required=1,
            output_dtype="float64",
            is_recursive=False,
            description="Intraday cumulative VWAP resetting daily at 09:15 IST.",
        ),
        FeatureSpec(
            feature_id="rsi_wilder_14",
            formula_variant="wilder_rsi_14_explicit_zero_loss",
            min_bars_required=15,
            output_dtype="float64",
            is_recursive=True,
            description="14-period Wilder RSI with explicit zero-loss mode (legacy_50 | strict_100).",
        ),
        FeatureSpec(
            feature_id="rvol_inclusive_20",
            formula_variant="inclusive_sma_20",
            min_bars_required=20,
            output_dtype="float64",
            is_recursive=False,
            description="20-bar Relative Volume including current bar in denominator.",
        ),
        FeatureSpec(
            feature_id="rvol_prior_nonzero_14",
            formula_variant="exclusive_prior_nonzero_14",
            min_bars_required=15,
            output_dtype="float64",
            is_recursive=False,
            description="14-bar Relative Volume against prior non-zero volume bars.",
        ),
        FeatureSpec(
            feature_id="rvol_prior_10",
            formula_variant="exclusive_prior_10",
            min_bars_required=11,
            output_dtype="float64",
            is_recursive=False,
            description="10-bar Relative Volume against prior 10 bars.",
        ),
        FeatureSpec(
            feature_id="macd_12_26_9",
            formula_variant="ema_12_26_signal_9",
            min_bars_required=26,
            output_dtype="float64",
            is_recursive=True,
            description="MACD (12, 26, 9) line, signal, and histogram.",
        ),
        FeatureSpec(
            feature_id="supertrend_10_3",
            formula_variant="supertrend_wilder_10_mult_3",
            min_bars_required=10,
            output_dtype="float64",
            is_recursive=True,
            description="Supertrend (period=10, multiplier=3.0) using Wilder ATR.",
        ),
        FeatureSpec(
            feature_id="cpr_pivots",
            formula_variant="classic_floor_cpr",
            min_bars_required=1,
            output_dtype="dict[str, float]",
            is_recursive=False,
            description="Central Pivot Range (P, BC, TC, width%) and S1/S2/R1/R2 levels.",
        ),
    ]
)
