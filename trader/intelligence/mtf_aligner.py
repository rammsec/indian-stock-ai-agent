"""Multi-timeframe IST alignment and NSE session bucket utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from trader.config import IST_TZ
from trader.intelligence.data_contract import _coerce_ts_series_ist, _ensure_tz_aware_ist


def expected_5m_sub_bars_for_bucket(
    bucket_open_ist: datetime,
    target_minutes: int,
) -> int:
    """Return expected 5m sub-bars for a resampled NSE bucket.

    NSE regular session runs 09:15–15:30 IST (75 x 5m bars).
    For 60m resampling, bars 1..6 (09:15..14:15) have 12 x 5m bars,
    while the final closing bar (15:15–15:30) has 3 x 5m bars.
    """
    if target_minutes <= 0 or target_minutes % 5 != 0:
        raise ValueError(f"target_minutes must be a positive multiple of 5, got {target_minutes}")
    bucket_ist = _ensure_tz_aware_ist(bucket_open_ist, "bucket_open_ist")
    if target_minutes == 60 and bucket_ist.hour == 15 and bucket_ist.minute == 15:
        return 3
    return target_minutes // 5


def align_htf_to_ltf_ist(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_cols: Sequence[str],
    *,
    ts_col: str = "ts_close_ist",
    suffix: str = "_htf",
    prefix: str = "",
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    """Align higher-timeframe features onto lower-timeframe bars using strict backward merge_asof on IST ts_close_ist."""
    if ts_col not in ltf_df.columns:
        raise ValueError(f"ltf_df missing required timestamp column {ts_col!r}")
    if ts_col not in htf_df.columns:
        raise ValueError(f"htf_df missing required timestamp column {ts_col!r}")

    missing_htf = [c for c in htf_cols if c not in htf_df.columns]
    if missing_htf:
        raise ValueError(f"htf_df missing requested columns: {missing_htf}")

    ltf_work = ltf_df.copy()
    htf_work = htf_df.copy()

    ltf_work[ts_col] = _coerce_ts_series_ist(ltf_work[ts_col], f"ltf.{ts_col}")
    htf_work[ts_col] = _coerce_ts_series_ist(htf_work[ts_col], f"htf.{ts_col}")

    ltf_work = ltf_work.sort_values(ts_col).reset_index(drop=True)
    htf_work = htf_work.sort_values(ts_col).reset_index(drop=True)

    rename_map: dict[str, str] = {}
    selected_cols: list[str] = [ts_col]
    for col in htf_cols:
        if col == ts_col:
            continue
        selected_cols.append(col)
        if prefix:
            rename_map[col] = f"{prefix}{col}"
        elif col in ltf_work.columns:
            rename_map[col] = f"{col}{suffix}"

    htf_sub = htf_work[selected_cols].rename(columns=rename_map)

    return pd.merge_asof(
        ltf_work,
        htf_sub,
        on=ts_col,
        direction="backward",
        allow_exact_matches=allow_exact_matches,
    )
