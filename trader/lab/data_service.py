"""Historical 5-minute data indexer, warm-up slicer, and strict quality validator for the 5-stock Lab universe."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd

from ..config import IST_TZ, LAB_DEFAULT_UNIVERSE, is_nse_trading_day
from ..data.loader import CACHE_DIR, DataUnavailableError, load_historical_5m_candles
from ..data.schema import Candle5M, DataProvenance, DataSourceMode
from ..intelligence.parsing import compute_candle_series_hash
from ..data.validator import detect_missing_candles

EXPECTED_SESSION_BARS = 75  # 09:15 to 15:25 IST inclusive (75 x 5m intervals)


@dataclass(frozen=True)
class StockSessionQuality:
    """Data completeness & provenance report for a single stock on a selected historical date."""

    symbol: str
    date: str
    data_availability: Literal["Historical", "Cached Historical", "Synthetic Test", "Unavailable"]
    source_mode: str
    provider: str
    retrieved_at_ist: str
    candles_count: int
    expected_candles: int
    warmup_candles_count: int
    missing_intervals_count: int
    missing_intervals_detail: list[str]
    status: Literal["Ready", "Incomplete", "Unavailable"]
    quality_reason: str
    sha256_hash: str


@dataclass(frozen=True)
class HistoricalDateAvailability:
    """Summary of a verified historical trading date across the 5-stock universe."""

    date: str
    date_status: Literal["FULLY_READY", "PARTIAL", "UNAVAILABLE", "EXCHANGE_HOLIDAY"]
    all_stocks_ready: bool
    ready_count: int
    incomplete_count: int
    unavailable_count: int
    total_stocks: int
    stocks: list[StockSessionQuality]


def _classify_stock_quality(
    symbol: str,
    target_date: date,
    session_candles: list[Candle5M],
    warmup_count: int,
    avail_label: Literal["Historical", "Cached Historical", "Synthetic Test", "Unavailable"],
    source_mode: str,
    provider: str,
    retrieved_at_ist: str,
) -> StockSessionQuality:
    """Strictly classify stock completeness: Ready ONLY if 75/75 bars and 0 missing intervals."""
    if not is_nse_trading_day(target_date):
        return StockSessionQuality(
            symbol=symbol,
            date=target_date.isoformat(),
            data_availability="Unavailable",
            source_mode=source_mode,
            provider=provider,
            retrieved_at_ist=retrieved_at_ist,
            candles_count=0,
            expected_candles=0,
            warmup_candles_count=warmup_count,
            missing_intervals_count=0,
            missing_intervals_detail=[],
            status="Unavailable",
            quality_reason=f"{target_date.isoformat()} is an official exchange holiday or weekend",
            sha256_hash="EXCHANGE_HOLIDAY",
        )

    n_bars = len(session_candles)
    if n_bars == 0:
        return StockSessionQuality(
            symbol=symbol,
            date=target_date.isoformat(),
            data_availability="Unavailable",
            source_mode=source_mode,
            provider=provider,
            retrieved_at_ist=retrieved_at_ist,
            candles_count=0,
            expected_candles=EXPECTED_SESSION_BARS,
            warmup_candles_count=warmup_count,
            missing_intervals_count=EXPECTED_SESSION_BARS,
            missing_intervals_detail=["Entire session missing (0/75 bars)"],
            status="Unavailable",
            quality_reason="Provider returned 0 records for this trading session",
            sha256_hash="UNAVAILABLE",
        )

    missing = detect_missing_candles(session_candles, expected_interval_minutes=5)
    if n_bars >= EXPECTED_SESSION_BARS and len(missing) == 0:
        status: Literal["Ready", "Incomplete", "Unavailable"] = "Ready"
        reason = f"Complete {n_bars}/{EXPECTED_SESSION_BARS} 5-minute session bars"
    else:
        status = "Incomplete"
        gap_note = f"; gaps: {', '.join(missing[:3])}" if missing else ""
        reason = f"Incomplete session ({n_bars}/{EXPECTED_SESSION_BARS} bars, {len(missing)} gap(s){gap_note})"

    return StockSessionQuality(
        symbol=symbol,
        date=target_date.isoformat(),
        data_availability=avail_label,
        source_mode=source_mode,
        provider=provider,
        retrieved_at_ist=retrieved_at_ist,
        candles_count=n_bars,
        expected_candles=EXPECTED_SESSION_BARS,
        warmup_candles_count=warmup_count,
        missing_intervals_count=len(missing),
        missing_intervals_detail=missing,
        status=status,
        quality_reason=reason,
        sha256_hash=compute_candle_series_hash(session_candles),
    )


def refresh_universe_cache(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    days: int = 60,
) -> dict[str, DataProvenance]:
    """Fetch and cache up to 60 days of genuine 5m historical candles from Yahoo Finance."""
    provenances: dict[str, DataProvenance] = {}
    for sym in symbols:
        candles, prov = load_historical_5m_candles(
            symbol=sym,
            exchange="NSE",
            days=days,
            mode=DataSourceMode.HISTORICAL_REAL,
            use_cache=False,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{sym}_NSE_5m.parquet"
        records = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        pd.DataFrame(records).to_parquet(cache_file, index=False)
        provenances[sym] = prov
    return provenances


def load_session_with_warmup(
    symbol: str,
    target_date: date,
    warmup_days: int = 5,
    mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
) -> tuple[list[Candle5M], list[Candle5M], StockSessionQuality]:
    """Load prior-session warm-up candles and target_date session candles with ZERO future leakage."""
    now_ist = datetime.now(tz=IST_TZ).isoformat()
    if not is_nse_trading_day(target_date):
        q = _classify_stock_quality(
            symbol=symbol,
            target_date=target_date,
            session_candles=[],
            warmup_count=0,
            avail_label="Unavailable",
            source_mode=mode.value,
            provider="None",
            retrieved_at_ist=now_ist,
        )
        return [], [], q

    try:
        all_candles, prov = load_historical_5m_candles(
            symbol=symbol,
            exchange="NSE",
            days=60,
            mode=mode,
            use_cache=True,
        )
    except DataUnavailableError:
        q = _classify_stock_quality(
            symbol=symbol,
            target_date=target_date,
            session_candles=[],
            warmup_count=0,
            avail_label="Unavailable",
            source_mode=mode.value,
            provider="ProviderOutage",
            retrieved_at_ist=now_ist,
        )
        return [], [], q

    session_candles = [
        c for c in all_candles if c.timestamp.astimezone(IST_TZ).date() == target_date
    ]
    prior_candles = [
        c for c in all_candles if c.timestamp.astimezone(IST_TZ).date() < target_date
    ]
    warmup_candles = prior_candles[-(warmup_days * EXPECTED_SESSION_BARS) :] if prior_candles else []

    if prov.source_mode == DataSourceMode.SYNTHETIC_TEST:
        avail_label: Literal["Historical", "Cached Historical", "Synthetic Test", "Unavailable"] = "Synthetic Test"
    elif "ParquetCache" in prov.source_name:
        avail_label = "Cached Historical"
    else:
        avail_label = "Historical"

    quality = _classify_stock_quality(
        symbol=symbol,
        target_date=target_date,
        session_candles=session_candles,
        warmup_count=len(warmup_candles),
        avail_label=avail_label if session_candles else "Unavailable",
        source_mode=prov.source_mode.value,
        provider=prov.source_name,
        retrieved_at_ist=prov.ingested_at.isoformat(),
    )
    return warmup_candles, session_candles, quality


def generate_pre_backtest_quality_report(
    target_date: date,
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    mode: DataSourceMode = DataSourceMode.HISTORICAL_REAL,
) -> HistoricalDateAvailability:
    """Produce an explicit pre-backtest data-quality report for a target date."""
    if not is_nse_trading_day(target_date):
        stocks = [
            _classify_stock_quality(s, target_date, [], 0, "Unavailable", mode.value, "NSECalendar", datetime.now(tz=IST_TZ).isoformat())
            for s in symbols
        ]
        return HistoricalDateAvailability(
            date=target_date.isoformat(),
            date_status="EXCHANGE_HOLIDAY",
            all_stocks_ready=False,
            ready_count=0,
            incomplete_count=0,
            unavailable_count=len(symbols),
            total_stocks=len(symbols),
            stocks=stocks,
        )

    stocks = []
    ready_cnt = 0
    inc_cnt = 0
    unavail_cnt = 0
    for sym in symbols:
        _, _, q = load_session_with_warmup(sym, target_date, warmup_days=5, mode=mode)
        stocks.append(q)
        if q.status == "Ready":
            ready_cnt += 1
        elif q.status == "Incomplete":
            inc_cnt += 1
        else:
            unavail_cnt += 1

    if ready_cnt == len(symbols):
        d_status: Literal["FULLY_READY", "PARTIAL", "UNAVAILABLE", "EXCHANGE_HOLIDAY"] = "FULLY_READY"
    elif ready_cnt > 0 or inc_cnt > 0:
        d_status = "PARTIAL"
    else:
        d_status = "UNAVAILABLE"

    return HistoricalDateAvailability(
        date=target_date.isoformat(),
        date_status=d_status,
        all_stocks_ready=(ready_cnt == len(symbols)),
        ready_count=ready_cnt,
        incomplete_count=inc_cnt,
        unavailable_count=unavail_cnt,
        total_stocks=len(symbols),
        stocks=stocks,
    )


def get_available_historical_dates(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    max_dates: int = 60,
) -> list[HistoricalDateAvailability]:
    """Discover all historical dates in real 5m data with strict READY vs INCOMPLETE vs PARTIAL classification."""
    date_sets: dict[date, int] = {}
    symbol_candles: dict[str, dict[date, list[Candle5M]]] = {}
    symbol_provs: dict[str, DataProvenance | None] = {}

    for sym in symbols:
        try:
            candles, prov = load_historical_5m_candles(
                symbol=sym,
                exchange="NSE",
                days=60,
                mode=DataSourceMode.HISTORICAL_REAL,
                use_cache=True,
            )
            symbol_provs[sym] = prov
        except DataUnavailableError:
            candles = []
            symbol_provs[sym] = None

        by_date: dict[date, list[Candle5M]] = {}
        for c in candles:
            d = c.timestamp.astimezone(IST_TZ).date()
            if is_nse_trading_day(d):
                by_date.setdefault(d, []).append(c)
                date_sets[d] = date_sets.get(d, 0) + 1
        symbol_candles[sym] = by_date

    sorted_dates = sorted(date_sets.keys(), reverse=True)[:max_dates]
    results: list[HistoricalDateAvailability] = []
    now_ist = datetime.now(tz=IST_TZ).isoformat()

    for d in sorted_dates:
        stock_qualities: list[StockSessionQuality] = []
        ready_cnt = 0
        inc_cnt = 0
        unavail_cnt = 0

        for sym in symbols:
            bars = symbol_candles.get(sym, {}).get(d, [])
            prov = symbol_provs.get(sym)
            prov_name = prov.source_name if prov else "None"
            ret_at = prov.ingested_at.isoformat() if prov else now_ist
            avail: Literal["Historical", "Cached Historical", "Synthetic Test", "Unavailable"] = (
                "Cached Historical" if bars else "Unavailable"
            )
            q = _classify_stock_quality(
                symbol=sym,
                target_date=d,
                session_candles=bars,
                warmup_count=0,
                avail_label=avail,
                source_mode=DataSourceMode.HISTORICAL_REAL.value,
                provider=prov_name,
                retrieved_at_ist=ret_at,
            )
            stock_qualities.append(q)
            if q.status == "Ready":
                ready_cnt += 1
            elif q.status == "Incomplete":
                inc_cnt += 1
            else:
                unavail_cnt += 1

        if ready_cnt == len(symbols):
            d_status: Literal["FULLY_READY", "PARTIAL", "UNAVAILABLE", "EXCHANGE_HOLIDAY"] = "FULLY_READY"
        elif ready_cnt > 0 or inc_cnt > 0:
            d_status = "PARTIAL"
        else:
            d_status = "UNAVAILABLE"

        results.append(
            HistoricalDateAvailability(
                date=d.isoformat(),
                date_status=d_status,
                all_stocks_ready=(ready_cnt == len(symbols)),
                ready_count=ready_cnt,
                incomplete_count=inc_cnt,
                unavailable_count=unavail_cnt,
                total_stocks=len(symbols),
                stocks=stock_qualities,
            )
        )

    return results


def filter_research_eligible_dates(
    avail_dates: list[HistoricalDateAvailability],
    min_bars_per_stock: int = 70,
) -> list[HistoricalDateAvailability]:
    """Filter dates meeting configurable research threshold while preserving PARTIAL disclosure."""
    eligible = []
    for d in avail_dates:
        if all(s.candles_count >= min_bars_per_stock for s in d.stocks):
            eligible.append(d)
    return eligible


def compute_data_quality_analytics(
    symbols: tuple[str, ...] | list[str] = LAB_DEFAULT_UNIVERSE,
    min_bars_for_research: int = 70,
) -> dict[str, Any]:
    """Compute comprehensive Phase 8 data-quality analytics across the 5-stock universe."""
    avail_dates = get_available_historical_dates(symbols=symbols, max_dates=90)
    eligible = filter_research_eligible_dates(avail_dates, min_bars_per_stock=min_bars_for_research)

    coverage_by_stock: dict[str, dict[str, Any]] = {}
    missing_by_time: dict[str, int] = {}
    coverage_by_provider: dict[str, int] = {}

    for sym in symbols:
        try:
            candles, prov = load_historical_5m_candles(sym, "NSE", days=60, mode=DataSourceMode.HISTORICAL_REAL, use_cache=True)
            prov_name = prov.source_name
        except DataUnavailableError:
            candles = []
            prov_name = "Unavailable"

        coverage_by_provider[prov_name] = coverage_by_provider.get(prov_name, 0) + len(candles)
        zero_vol = sum(1 for c in candles if c.volume <= 0)
        outliers = 0
        for i in range(1, len(candles)):
            if candles[i - 1].close > 0:
                ret = abs(candles[i].close - candles[i - 1].close) / candles[i - 1].close
                if ret > 0.10:
                    outliers += 1

        sym_days = [s for d in avail_dates for s in d.stocks if s.symbol == sym]
        ready_days = sum(1 for s in sym_days if s.status == "Ready")
        inc_days = sum(1 for s in sym_days if s.status == "Incomplete")
        total_missing = sum(s.missing_intervals_count for s in sym_days)

        for s in sym_days:
            for detail in s.missing_intervals_detail:
                # Extract HH:MM window from gap string
                if "between " in detail:
                    part = detail.split("between ")[1].split(" (")[0]
                    missing_by_time[part] = missing_by_time.get(part, 0) + 1

        coverage_by_stock[sym] = {
            "symbol": sym,
            "total_bars": len(candles),
            "sessions_present": len(sym_days),
            "ready_sessions": ready_days,
            "incomplete_sessions": inc_days,
            "total_missing_intervals": total_missing,
            "zero_volume_bars": zero_vol,
            "outlier_bars_gt_10pct": outliers,
            "avg_bars_per_session": round(sum(s.candles_count for s in sym_days) / len(sym_days), 1) if sym_days else 0.0,
            "provider": prov_name,
        }

    return {
        "total_available_days": len(avail_dates),
        "fully_ready_sessions": sum(1 for d in avail_dates if d.date_status == "FULLY_READY"),
        "partial_sessions": sum(1 for d in avail_dates if d.date_status == "PARTIAL"),
        "unavailable_sessions": sum(1 for d in avail_dates if d.date_status == "UNAVAILABLE"),
        "research_eligible_sessions": len(eligible),
        "research_min_bars_threshold": min_bars_for_research,
        "last_refresh_ist": datetime.now(tz=IST_TZ).isoformat(),
        "coverage_by_stock": coverage_by_stock,
        "coverage_by_provider": coverage_by_provider,
        "missing_windows_histogram": dict(sorted(missing_by_time.items(), key=lambda x: -x[1])[:10]),
    }

