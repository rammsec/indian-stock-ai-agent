"""Phase 11.2: Comprehensive 1-Hour Historical Data Provenance & Integrity Validator."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..config import IST_TZ, LAB_DEFAULT_UNIVERSE, is_nse_trading_day
from ..intelligence.data_contract import cap_provenance_tier
from .loader import CACHE_DIR
from .resampler import ResampledCandle
from .schema import Candle5M

ProvenanceStatus = Literal["VERIFIED", "PARTIALLY_VERIFIED", "FAILED", "INSUFFICIENT_DATA"]
VALIDATOR_VERSION = "11.2.0"
EXPECTED_1H_BARS_PER_SESSION = 7  # 09:15, 10:15, 11:15, 12:15, 13:15, 14:15, 15:15 IST


def validate_1h_provenance(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    cache_dir: Path = CACHE_DIR,
    min_bars_per_session: int = 6,
    min_sessions_required: int = 60,
    provider_name: str = "yahoo_finance_1h",
    resampling_methodology: str = "DIRECT_PROVIDER_1H_OHLCV_NO_INTERPOLATION",
    *,
    enforce_provider_authority_cap: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Run all 15 mandatory provenance & integrity checks across the 1-hour NSE Parquet archive."""
    per_symbol: dict[str, Any] = {}
    any_failed = False
    any_partial = False
    insufficient = False
    total_missing_intervals = 0
    total_duplicates = 0
    total_invalid_ohlc = 0

    for sym in symbols:
        source_file = cache_dir / f"{sym}_1h.parquet"
        if not source_file.exists():
            any_failed = True
            per_symbol[sym] = {
                "symbol": sym,
                "timeframe": "1h",
                "provider": provider_name,
                "source_file": str(source_file),
                "source_sha256": None,
                "dataset_fingerprint": None,
                "first_timestamp": None,
                "last_timestamp": None,
                "session_count": 0,
                "valid_session_count": 0,
                "incomplete_session_count": 0,
                "missing_interval_count": 0,
                "duplicate_count": 0,
                "invalid_ohlc_count": 0,
                "negative_volume_count": 0,
                "out_of_session_count": 0,
                "timezone_validation": "FAIL_MISSING_FILE",
                "latest_session_complete": False,
                "resampling_methodology": resampling_methodology,
                "final_provenance_status": "FAILED",
                "failure_reasons": ["Source file does not exist"],
            }
            continue

        try:
            raw_bytes = source_file.read_bytes()
            file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            df = pd.read_parquet(source_file)
        except Exception as exc:
            any_failed = True
            per_symbol[sym] = {
                "symbol": sym,
                "timeframe": "1h",
                "provider": provider_name,
                "source_file": str(source_file),
                "final_provenance_status": "FAILED",
                "failure_reasons": [f"Unreadable parquet: {exc}"],
            }
            continue

        if df.empty or "timestamp" not in df.columns:
            any_failed = True
            per_symbol[sym] = {
                "symbol": sym,
                "timeframe": "1h",
                "provider": provider_name,
                "source_file": str(source_file),
                "source_sha256": file_sha256,
                "final_provenance_status": "FAILED",
                "failure_reasons": ["Empty dataset or missing timestamp column"],
            }
            continue

        # Timezone validation
        ts_series = df["timestamp"]
        tz_valid = True
        tz_msg = "PASS_ASIA_KOLKATA"
        try:
            if getattr(ts_series.dt, "tz", None) is None:
                tz_valid = False
                tz_msg = "FAIL_NAIVE_TIMESTAMP"
                df["timestamp"] = pd.to_datetime(ts_series).dt.tz_localize(IST_TZ)
            else:
                tz_name = str(ts_series.dt.tz)
                if "Kolkata" not in tz_name and "+05:30" not in tz_name and "UTC" not in tz_name:
                    tz_valid = False
                    tz_msg = f"FAIL_UNEXPECTED_TZ:{tz_name}"
                df["timestamp"] = pd.to_datetime(ts_series, utc=True).dt.tz_convert(IST_TZ)
        except Exception as exc:
            tz_valid = False
            tz_msg = f"FAIL_INVALID_TS:{exc}"

        if not tz_valid:
            any_failed = True

        # Duplicate check
        dup_count = int(df["timestamp"].duplicated().sum())
        total_duplicates += dup_count
        if dup_count > 0:
            any_failed = True

        # OHLC consistency & non-negative volume checks
        invalid_ohlc_mask = (
            df[["open", "high", "low", "close"]].isna().any(axis=1)
            | (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
            | (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        )
        invalid_ohlc_count = int(invalid_ohlc_mask.sum())
        total_invalid_ohlc += invalid_ohlc_count
        if invalid_ohlc_count > 0:
            any_failed = True

        neg_vol_count = int((df["volume"].isna() | (df["volume"] < 0)).sum())
        if neg_vol_count > 0:
            any_failed = True

        # Market session boundary check (09:15..15:15 IST)
        hhmm = df["timestamp"].dt.strftime("%H:%M")
        out_of_session_count = int(((hhmm < "09:15") | (hhmm > "15:15")).sum())
        if out_of_session_count > 0:
            any_failed = True

        # Session counts & missing intervals against expected NSE trading calendar
        df_sorted = df.sort_values("timestamp")
        df_sorted["d"] = df_sorted["timestamp"].dt.date
        counts_by_day = df_sorted.groupby("d").size().to_dict()

        first_d: date = df_sorted["d"].min()
        last_d: date = df_sorted["d"].max()
        expected_trading_days = [
            d for d in pd.date_range(first_d, last_d, freq="D").date if is_nse_trading_day(d)
        ]

        valid_sessions = 0
        incomplete_sessions = 0
        missing_intervals = 0
        for d in expected_trading_days:
            cnt = int(counts_by_day.get(d, 0))
            if cnt >= min_bars_per_session:
                valid_sessions += 1
            else:
                incomplete_sessions += 1
            missing_intervals += max(0, EXPECTED_1H_BARS_PER_SESSION - cnt)

        total_missing_intervals += missing_intervals
        latest_bars = int(counts_by_day.get(last_d, 0))
        latest_complete = latest_bars >= min_bars_per_session

        # Canonical content fingerprint
        fp_hasher = hashlib.sha256()
        for row in df_sorted.itertuples(index=False):
            fp_hasher.update(
                f"{row.timestamp.isoformat()}|{row.open:.4f}|{row.high:.4f}|{row.low:.4f}|{row.close:.4f}|{int(row.volume)}\n".encode()
            )
        dataset_fp = fp_hasher.hexdigest()

        failure_reasons: list[str] = []
        if not tz_valid:
            failure_reasons.append(tz_msg)
        if dup_count > 0:
            failure_reasons.append(f"{dup_count} duplicate timestamps")
        if invalid_ohlc_count > 0:
            failure_reasons.append(f"{invalid_ohlc_count} invalid OHLC rows")
        if neg_vol_count > 0:
            failure_reasons.append(f"{neg_vol_count} negative volume rows")
        if out_of_session_count > 0:
            failure_reasons.append(f"{out_of_session_count} out-of-session bars")

        if failure_reasons:
            sym_status: ProvenanceStatus = "FAILED"
        elif valid_sessions < min_sessions_required:
            sym_status = "INSUFFICIENT_DATA"
            insufficient = True
        elif incomplete_sessions > 0 or not latest_complete:
            sym_status = "PARTIALLY_VERIFIED"
            any_partial = True
        else:
            sym_status = "VERIFIED"

        if enforce_provider_authority_cap:
            capped = cap_provenance_tier(sym_status, provider_name)
            sym_status = capped.value if hasattr(capped, "value") else capped  # type: ignore[assignment]
            if sym_status == "PARTIALLY_VERIFIED":
                any_partial = True

        per_symbol[sym] = {
            "symbol": sym,
            "timeframe": "1h",
            "provider": provider_name,
            "source_file": str(source_file),
            "source_sha256": file_sha256,
            "dataset_fingerprint": dataset_fp,
            "first_timestamp": df_sorted["timestamp"].iloc[0].isoformat(),
            "last_timestamp": df_sorted["timestamp"].iloc[-1].isoformat(),
            "session_count": len(expected_trading_days),
            "valid_session_count": valid_sessions,
            "incomplete_session_count": incomplete_sessions,
            "missing_interval_count": missing_intervals,
            "duplicate_count": dup_count,
            "invalid_ohlc_count": invalid_ohlc_count,
            "negative_volume_count": neg_vol_count,
            "out_of_session_count": out_of_session_count,
            "timezone_validation": tz_msg,
            "latest_session_bars": latest_bars,
            "latest_session_complete": latest_complete,
            "resampling_methodology": resampling_methodology,
            "final_provenance_status": sym_status,
            "failure_reasons": failure_reasons,
        }

    if any_failed:
        overall_status: ProvenanceStatus = "FAILED"
        verified = False
    elif insufficient:
        overall_status = "INSUFFICIENT_DATA"
        verified = False
    elif any_partial:
        overall_status = "PARTIALLY_VERIFIED"
        verified = False
    else:
        overall_status = "VERIFIED"
        verified = True

    combined_fp_hasher = hashlib.sha256()
    for sym in sorted(per_symbol.keys()):
        combined_fp_hasher.update(f"{sym}:{per_symbol[sym].get('dataset_fingerprint', '')}\n".encode())

    report = {
        "validator_version": VALIDATOR_VERSION,
        "verification_timestamp": datetime.now(tz=IST_TZ).isoformat(),
        "timeframe": "1h",
        "provider": provider_name,
        "resampling_methodology": resampling_methodology,
        "verified": verified,
        "qualification_allowed": verified,
        "exploratory_research_allowed": overall_status in ("VERIFIED", "PARTIALLY_VERIFIED"),
        "final_provenance_status": overall_status,
        "dataset_fingerprint": combined_fp_hasher.hexdigest(),
        "total_missing_intervals": total_missing_intervals,
        "total_duplicates": total_duplicates,
        "total_invalid_ohlc": total_invalid_ohlc,
        "symbols": per_symbol,
    }
    return verified, report


def validate_resampled_1h_consistency(
    candles_5m: list[Candle5M],
    resampled_1h: list[ResampledCandle],
) -> tuple[bool, list[str]]:
    """Verify 5m->1h resampled candles match underlying 5m OHLCV without interpolation or fabricated volume."""
    errors: list[str] = []
    if not resampled_1h:
        return len(candles_5m) == 0, ["Empty resampled output"] if candles_5m else []

    by_day_5m: dict[date, list[Candle5M]] = {}
    for c in candles_5m:
        by_day_5m.setdefault(c.timestamp.astimezone(IST_TZ).date(), []).append(c)

    for r in resampled_1h:
        r_ist = r.timestamp.astimezone(IST_TZ)
        d = r_ist.date()
        day_5m = sorted(by_day_5m.get(d, []), key=lambda x: x.timestamp)
        bucket_end = r_ist.timestamp() + 3600
        src = [c for c in day_5m if r_ist.timestamp() <= c.timestamp.astimezone(IST_TZ).timestamp() < bucket_end]
        if len(src) != r.source_bar_count:
            errors.append(f"{r_ist.isoformat()}: source_bar_count {r.source_bar_count} != {len(src)}")
            continue
        if not src:
            errors.append(f"{r_ist.isoformat()}: fabricated bar with 0 source candles")
            continue
        if abs(r.open - src[0].open) > 1e-6:
            errors.append(f"{r_ist.isoformat()}: open mismatch")
        if abs(r.high - max(c.high for c in src)) > 1e-6:
            errors.append(f"{r_ist.isoformat()}: high mismatch")
        if abs(r.low - min(c.low for c in src)) > 1e-6:
            errors.append(f"{r_ist.isoformat()}: low mismatch")
        if abs(r.close - src[-1].close) > 1e-6:
            errors.append(f"{r_ist.isoformat()}: close mismatch")
        if r.volume != sum(c.volume for c in src):
            errors.append(f"{r_ist.isoformat()}: volume mismatch")

    return len(errors) == 0, errors
