"""Strict 5-minute candle validation, anomaly detection, and session filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, time
import pandas as pd

from ..config import IST_TZ, MARKET_OPEN_TIME, MARKET_CLOSE_TIME
from .schema import Candle5M, CandleValidationResult


def is_within_intraday_session(ts: datetime) -> bool:
    """Check if timestamp falls within the normal Indian market session (09:15 to 15:30 IST)."""
    t = ts.astimezone(IST_TZ).time()
    return MARKET_OPEN_TIME <= t <= MARKET_CLOSE_TIME


def detect_missing_candles(
    candles: list[Candle5M],
    expected_interval_minutes: int = 5,
) -> list[str]:
    """Identify missing intervals within the same trading session."""
    if len(candles) < 2:
        return []

    missing: list[str] = []
    for i in range(1, len(candles)):
        c_prev = candles[i - 1]
        c_curr = candles[i]

        # Only check intervals on the same calendar day
        t_prev = c_prev.timestamp.astimezone(IST_TZ)
        t_curr = c_curr.timestamp.astimezone(IST_TZ)

        if t_prev.date() == t_curr.date():
            delta = t_curr - t_prev
            expected_delta = timedelta(minutes=expected_interval_minutes)
            if delta > expected_delta:
                missing.append(
                    f"Gap detected on {t_prev.date()}: missing bars between "
                    f"{t_prev.strftime('%H:%M')} and {t_curr.strftime('%H:%M')} ({int(delta.total_seconds() / 60)}m gap)"
                )

    return missing


def validate_candle_series(
    candles: list[Candle5M],
    filter_outside_market_hours: bool = True,
    max_single_bar_return_pct: float = 12.0,
) -> tuple[list[Candle5M], CandleValidationResult]:
    """Perform rigorous validation, deduplication, and anomaly checks on a series of 5m candles.

    Returns:
        tuple[list[Candle5M], CandleValidationResult]: Sanitized candles and a comprehensive report.
    """
    if not candles:
        return [], CandleValidationResult(
            symbol="UNKNOWN",
            exchange="UNKNOWN",
            total_input_bars=0,
            valid_bars_count=0,
            dropped_bars_count=0,
            is_valid=False,
            issues=["Empty candle list provided"],
        )

    symbol = candles[0].symbol
    exchange = candles[0].exchange
    issues: list[str] = []
    valid_candles: list[Candle5M] = []
    seen_timestamps: set[datetime] = set()

    for idx, c in enumerate(candles):
        ts_ist = c.timestamp.astimezone(IST_TZ)

        # 1. Filter outside regular trading hours
        if filter_outside_market_hours and not is_within_intraday_session(ts_ist):
            issues.append(f"Bar at {ts_ist.isoformat()} outside market hours (09:15-15:30 IST); dropped")
            continue

        # 2. Check for duplicate timestamps
        if ts_ist in seen_timestamps:
            issues.append(f"Duplicate timestamp {ts_ist.isoformat()} detected; dropped")
            continue

        # 3. Check for suspicious price spike relative to prior valid bar
        if valid_candles:
            prior_close = valid_candles[-1].close
            bar_return_pct = abs((c.close - prior_close) / prior_close) * 100.0
            if bar_return_pct > max_single_bar_return_pct:
                issues.append(
                    f"Suspicious price jump of {bar_return_pct:.2f}% at {ts_ist.strftime('%Y-%m-%d %H:%M')} "
                    f"(Close {c.close} vs prior {prior_close}); flagged"
                )

        # 4. Check timestamp monotonicity
        if valid_candles and ts_ist <= valid_candles[-1].timestamp.astimezone(IST_TZ):
            issues.append(f"Non-monotonic timestamp {ts_ist.isoformat()} <= prior bar; dropped")
            continue

        seen_timestamps.add(ts_ist)
        valid_candles.append(c)

    # 5. Detect missing intervals
    missing_intervals = detect_missing_candles(valid_candles, expected_interval_minutes=5)

    is_valid = len(valid_candles) > 0 and len(valid_candles) == len(candles) and len(missing_intervals) == 0

    report = CandleValidationResult(
        symbol=symbol,
        exchange=exchange,
        total_input_bars=len(candles),
        valid_bars_count=len(valid_candles),
        dropped_bars_count=len(candles) - len(valid_candles),
        is_valid=is_valid,
        issues=issues,
        missing_intervals=missing_intervals,
    )

    return valid_candles, report
