"""CYCLE-010 Phase 5: Read-Only Local NSE Tick/Seconds Recorder & Causal Multi-Timeframe Archiver.

HARD SAFETY INVARIANTS:
- Strictly READ-ONLY market-data ingestion (`LIVE_TRADING_ENABLED = False`, `READ_ONLY_MODE = True`).
- Zero broker order placement, modification, cancellation, or execution routing.
- Uses optional local `.env` / environment credentials for Zerodha KiteTicker or
  Angel One SmartAPI WebSocket v2 feeds; never hardcodes or logs secrets.
- Causal tick-to-bar resampling aligned strictly to 09:15:00 IST with 15:30:00 IST clamping.
- Never merges ticks across different NSE trading dates and never fabricates missing ticks.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import IST_TZ

LIVE_TRADING_ENABLED: Final[bool] = False
READ_ONLY_MODE: Final[bool] = True

NSE_SESSION_OPEN_TIME: Final[time] = time(9, 15, 0)
NSE_SESSION_CLOSE_TIME: Final[time] = time(15, 30, 0)

SUPPORTED_TIMEFRAMES: Final[tuple[str, ...]] = ("1s", "5s", "1m", "5m", "15m")
TIMEFRAME_SECONDS: Final[dict[str, int]] = {
    "1s": 1,
    "5s": 5,
    "1m": 60,
    "5m": 300,
    "15m": 900,
}

ProviderName = Literal["zerodha_kite", "angel_one_smartapi", "offline_replay"]
TimeframeName = Literal["1s", "5s", "1m", "5m", "15m"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TickEvent(_StrictModel):
    """Single normalized NSE/BSE market tick with timezone-aware IST timestamp."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    timestamp_ist: datetime
    ltp: float = Field(gt=0.0)
    volume_delta: float = Field(default=0.0, ge=0.0)
    cumulative_volume: float | None = Field(default=None, ge=0.0)
    bid: float | None = Field(default=None, gt=0.0)
    ask: float | None = Field(default=None, gt=0.0)
    provider: ProviderName = "offline_replay"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("symbol must be non-empty")
        return cleaned

    @field_validator("timestamp_ist")
    @classmethod
    def require_tz_aware_ist(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("timestamp_ist must be timezone-aware (Asia/Kolkata / IST)")
        return v.astimezone(IST_TZ)


class RecordedBar(_StrictModel):
    """Causally aggregated OHLCV bar from genuine recorded ticks."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    timeframe: TimeframeName
    ts_open_ist: datetime
    ts_close_ist: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    tick_count: int = Field(ge=1)
    source_provider: ProviderName

    @field_validator("ts_open_ist", "ts_close_ist")
    @classmethod
    def require_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Bar timestamps must be timezone-aware IST")
        return v.astimezone(IST_TZ)


class RecorderProvenance(_StrictModel):
    """Cryptographic provenance manifest for a recorded symbol/date tick archive."""

    symbol: str
    trading_date: str
    provider: ProviderName
    source_mode: Literal["LIVE_RECORDED_READONLY", "OFFLINE_REPLAY_TEST"]
    live_trading_enabled: Literal[False] = False
    read_only: Literal[True] = True
    recorded_at_ist: datetime
    raw_tick_count: int = Field(ge=0)
    out_of_order_ticks: int = Field(ge=0)
    rejected_outside_hours_ticks: int = Field(ge=0)
    bar_counts: dict[str, int] = Field(default_factory=dict)
    file_sha256: dict[str, str] = Field(default_factory=dict)


@dataclass
class TickRecorderSummary:
    """In-memory summary returned after a recording or offline replay run."""

    provider: ProviderName
    symbols: list[str]
    total_ticks_accepted: int = 0
    out_of_order_ticks: int = 0
    rejected_outside_hours_ticks: int = 0
    bars_written_by_timeframe: dict[str, int] = field(default_factory=dict)
    written_files: list[str] = field(default_factory=list)


def assert_read_only_recorder_config(
    *,
    live_trading_enabled: bool = False,
    read_only: bool = True,
) -> None:
    """Fail-closed guard ensuring the recorder can never be configured for order routing."""
    if live_trading_enabled or LIVE_TRADING_ENABLED:
        raise RuntimeError(
            "SAFETY VIOLATION: Tick recorder is strictly read-only; live_trading_enabled must be False."
        )
    if not read_only or not READ_ONLY_MODE:
        raise RuntimeError(
            "SAFETY VIOLATION: Tick recorder requires read_only=True at all times."
        )


def is_within_nse_session_ist(ts: datetime) -> bool:
    """Return True iff `ts` falls on a weekday strictly inside [09:15:00, 15:30:00) IST."""
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError("Timestamp must be timezone-aware")
    ts_ist = ts.astimezone(IST_TZ)
    if ts_ist.weekday() >= 5:
        return False
    t = ts_ist.time()
    return NSE_SESSION_OPEN_TIME <= t < NSE_SESSION_CLOSE_TIME


def align_tick_bucket_ist(ts_ist: datetime, timeframe: TimeframeName) -> tuple[datetime, datetime]:
    """Compute (ts_open_ist, ts_close_ist) aligned to 09:15:00 IST, clamped at 15:30:00 IST."""
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    step_sec = TIMEFRAME_SECONDS[timeframe]
    ts_ist = ts_ist.astimezone(IST_TZ)
    session_open = ts_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    session_close = ts_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    elapsed_sec = int((ts_ist - session_open).total_seconds())
    if elapsed_sec < 0:
        raise ValueError(f"Timestamp {ts_ist.isoformat()} is before NSE session open 09:15:00 IST")
    bucket_idx = elapsed_sec // step_sec
    ts_open = session_open + timedelta(seconds=bucket_idx * step_sec)
    ts_close = min(ts_open + timedelta(seconds=step_sec), session_close)
    return ts_open, ts_close


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class CausalTickAggregator:
    """Deterministic causal multi-timeframe tick-to-OHLCV bar aggregator.

    Properties:
    - Rejects ticks outside NSE regular session hours [09:15:00, 15:30:00) IST.
    - Tracks out-of-order ticks and sorts ticks chronologically inside each bucket
      so `open` is always the earliest tick and `close` is always the latest tick.
    - Computes incremental volume from `cumulative_volume` monotonically per
      (symbol, trading_date) when provided, otherwise uses `volume_delta`.
    - Never merges ticks across different trading dates.
    - `flush_closed_bars(watermark_ist)` emits ONLY bars whose `ts_close_ist <= watermark_ist`
      (strict zero-lookahead guarantee).
    """

    def __init__(self, timeframes: Iterable[TimeframeName] = SUPPORTED_TIMEFRAMES) -> None:
        tf_list = tuple(timeframes)
        if not tf_list:
            raise ValueError("At least one timeframe is required")
        for tf in tf_list:
            if tf not in TIMEFRAME_SECONDS:
                raise ValueError(f"Unsupported timeframe: {tf!r}")
        self.timeframes: tuple[TimeframeName, ...] = tf_list
        self.accepted_ticks: int = 0
        self.out_of_order_ticks: int = 0
        self.rejected_outside_hours_ticks: int = 0
        self._last_ts_by_symbol_day: dict[tuple[str, date], datetime] = {}
        self._last_cum_vol_by_symbol_day: dict[tuple[str, date], float] = {}
        # Key: (symbol, trading_date, timeframe, ts_open_ist) -> list[(ts_ist, seq, ltp, vol, exchange, provider)]
        self._open_buckets: dict[
            tuple[str, date, TimeframeName, datetime],
            list[tuple[datetime, int, float, float, Literal["NSE", "BSE"], ProviderName]],
        ] = defaultdict(list)
        self._bucket_close: dict[tuple[str, date, TimeframeName, datetime], datetime] = {}
        self._seq: int = 0

    def ingest_tick(self, tick: TickEvent) -> bool:
        """Ingest one TickEvent. Returns True if accepted, False if rejected outside NSE hours."""
        if not is_within_nse_session_ist(tick.timestamp_ist):
            self.rejected_outside_hours_ticks += 1
            return False

        ts_ist = tick.timestamp_ist.astimezone(IST_TZ)
        day_key = (tick.symbol, ts_ist.date())

        prev_ts = self._last_ts_by_symbol_day.get(day_key)
        if prev_ts is not None and ts_ist < prev_ts:
            self.out_of_order_ticks += 1
        elif prev_ts is None or ts_ist > prev_ts:
            self._last_ts_by_symbol_day[day_key] = ts_ist

        # Resolve per-tick volume delta without double-counting cumulative volume
        if tick.cumulative_volume is not None:
            prev_cum = self._last_cum_vol_by_symbol_day.get(day_key)
            if prev_cum is None:
                vol_delta = tick.volume_delta if tick.volume_delta > 0 else 0.0
                self._last_cum_vol_by_symbol_day[day_key] = tick.cumulative_volume
            else:
                vol_delta = max(0.0, tick.cumulative_volume - prev_cum)
                if tick.cumulative_volume > prev_cum:
                    self._last_cum_vol_by_symbol_day[day_key] = tick.cumulative_volume
        else:
            vol_delta = tick.volume_delta

        self._seq += 1
        entry = (ts_ist, self._seq, tick.ltp, vol_delta, tick.exchange, tick.provider)
        for tf in self.timeframes:
            ts_open, ts_close = align_tick_bucket_ist(ts_ist, tf)
            bkey = (tick.symbol, ts_ist.date(), tf, ts_open)
            self._open_buckets[bkey].append(entry)
            self._bucket_close[bkey] = ts_close

        self.accepted_ticks += 1
        return True

    def _build_bar(
        self,
        bkey: tuple[str, date, TimeframeName, datetime],
        entries: list[tuple[datetime, int, float, float, Literal["NSE", "BSE"], ProviderName]],
    ) -> RecordedBar:
        symbol, _, tf, ts_open = bkey
        ts_close = self._bucket_close[bkey]
        ordered = sorted(entries, key=lambda x: (x[0], x[1]))
        prices = [e[2] for e in ordered]
        total_vol = sum(e[3] for e in ordered)
        return RecordedBar(
            symbol=symbol,
            exchange=ordered[0][4],
            timeframe=tf,
            ts_open_ist=ts_open,
            ts_close_ist=ts_close,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_vol,
            tick_count=len(ordered),
            source_provider=ordered[-1][5],
        )

    def flush_closed_bars(self, watermark_ist: datetime) -> list[RecordedBar]:
        """Emit and remove all buckets whose `ts_close_ist <= watermark_ist`."""
        if watermark_ist.tzinfo is None or watermark_ist.tzinfo.utcoffset(watermark_ist) is None:
            raise ValueError("watermark_ist must be timezone-aware")
        wm = watermark_ist.astimezone(IST_TZ)
        ready_keys = [k for k, close_ts in self._bucket_close.items() if close_ts <= wm]
        ready_keys.sort(key=lambda k: (k[0], k[2], k[3]))
        bars: list[RecordedBar] = []
        for k in ready_keys:
            entries = self._open_buckets.pop(k)
            bars.append(self._build_bar(k, entries))
            self._bucket_close.pop(k, None)
        return bars

    def finalize_all(self) -> list[RecordedBar]:
        """Flush all remaining open buckets (e.g., at 15:30 IST or graceful shutdown)."""
        keys = sorted(self._open_buckets.keys(), key=lambda k: (k[0], k[2], k[3]))
        bars: list[RecordedBar] = []
        for k in keys:
            entries = self._open_buckets.pop(k)
            bars.append(self._build_bar(k, entries))
            self._bucket_close.pop(k, None)
        return bars


class TickParquetArchive:
    """Writes partitioned raw ticks, multi-timeframe OHLCV bars, and SHA-256 provenance."""

    def __init__(self, base_dir: str | Path = "data/ticks") -> None:
        self.base_dir = Path(base_dir)

    def write_day_archive(
        self,
        *,
        symbol: str,
        trading_date: date,
        provider: ProviderName,
        raw_ticks: list[TickEvent],
        bars_by_timeframe: dict[TimeframeName, list[RecordedBar]],
        out_of_order_ticks: int = 0,
        rejected_outside_hours_ticks: int = 0,
    ) -> tuple[Path, RecorderProvenance]:
        """Persist raw ticks + resampled Parquet files + provenance.json for one (symbol, date)."""
        assert_read_only_recorder_config()
        day_dir = self.base_dir / symbol.upper() / trading_date.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)

        file_sha256: dict[str, str] = {}
        bar_counts: dict[str, int] = {}

        # 1. Raw ticks Parquet
        raw_path = day_dir / "raw_ticks.parquet"
        raw_rows = [
            {
                "symbol": t.symbol,
                "exchange": t.exchange,
                "timestamp_ist": t.timestamp_ist.astimezone(IST_TZ),
                "ltp": t.ltp,
                "volume_delta": t.volume_delta,
                "cumulative_volume": t.cumulative_volume,
                "bid": t.bid,
                "ask": t.ask,
                "provider": t.provider,
            }
            for t in sorted(raw_ticks, key=lambda x: x.timestamp_ist)
        ]
        raw_df = pd.DataFrame(raw_rows)
        raw_df.to_parquet(raw_path, index=False)
        file_sha256[raw_path.name] = sha256_file(raw_path)

        # 2. Multi-timeframe OHLCV Parquet files
        for tf, bars in sorted(bars_by_timeframe.items()):
            bars_sorted = sorted(bars, key=lambda b: b.ts_open_ist)
            bar_path = day_dir / f"bars_{tf}.parquet"
            bar_rows = [
                {
                    "symbol": b.symbol,
                    "exchange": b.exchange,
                    "timeframe": b.timeframe,
                    "ts_open_ist": b.ts_open_ist.astimezone(IST_TZ),
                    "ts_close_ist": b.ts_close_ist.astimezone(IST_TZ),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "tick_count": b.tick_count,
                    "source_provider": b.source_provider,
                }
                for b in bars_sorted
            ]
            bar_df = pd.DataFrame(bar_rows)
            bar_df.to_parquet(bar_path, index=False)
            file_sha256[bar_path.name] = sha256_file(bar_path)
            bar_counts[tf] = len(bars_sorted)

        source_mode: Literal["LIVE_RECORDED_READONLY", "OFFLINE_REPLAY_TEST"] = (
            "OFFLINE_REPLAY_TEST" if provider == "offline_replay" else "LIVE_RECORDED_READONLY"
        )
        provenance = RecorderProvenance(
            symbol=symbol.upper(),
            trading_date=trading_date.isoformat(),
            provider=provider,
            source_mode=source_mode,
            live_trading_enabled=False,
            read_only=True,
            recorded_at_ist=datetime.now(IST_TZ),
            raw_tick_count=len(raw_rows),
            out_of_order_ticks=out_of_order_ticks,
            rejected_outside_hours_ticks=rejected_outside_hours_ticks,
            bar_counts=bar_counts,
            file_sha256=file_sha256,
        )
        prov_path = day_dir / "provenance.json"
        prov_path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
        return day_dir, provenance


def load_recorded_bars_parquet(path: str | Path) -> list[RecordedBar]:
    """Load a recorded multi-timeframe Parquet file back into typed `RecordedBar` objects."""
    df = pd.read_parquet(Path(path))
    if df.empty:
        return []
    bars: list[RecordedBar] = []
    for row in df.to_dict(orient="records"):
        ts_open = pd.Timestamp(row["ts_open_ist"]).to_pydatetime()
        ts_close = pd.Timestamp(row["ts_close_ist"]).to_pydatetime()
        bars.append(
            RecordedBar(
                symbol=str(row["symbol"]),
                exchange=row["exchange"],
                timeframe=row["timeframe"],
                ts_open_ist=ts_open,
                ts_close_ist=ts_close,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                tick_count=int(row["tick_count"]),
                source_provider=row["source_provider"],
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Read-Only Market-Data WebSocket Adapters (ZERO order-routing methods)
# ---------------------------------------------------------------------------


class ReadOnlyTickAdapter(ABC):
    """Abstract read-only tick feed adapter. Intentionally exposes NO order methods."""

    provider_name: ProviderName

    @abstractmethod
    def validate_credentials(self) -> None:
        """Verify required local credentials/inputs exist without logging secret values."""

    @abstractmethod
    def stream(
        self,
        symbols: list[str],
        on_tick: Callable[[TickEvent], None],
    ) -> None:
        """Connect and invoke `on_tick` for each normalized TickEvent until stopped."""

    @abstractmethod
    def stop(self) -> None:
        """Gracefully close the read-only feed."""


def parse_zerodha_kite_tick(
    raw_tick: dict[str, Any],
    token_to_symbol: dict[int, str],
) -> TickEvent | None:
    """Normalize a Zerodha KiteTicker dict payload into a `TickEvent`."""
    token = raw_tick.get("instrument_token")
    if token not in token_to_symbol:
        return None
    ltp = float(raw_tick["last_price"])
    raw_ts = raw_tick.get("exchange_timestamp") or raw_tick.get("timestamp")
    if isinstance(raw_ts, datetime):
        ts_ist = raw_ts.astimezone(IST_TZ) if raw_ts.tzinfo else raw_ts.replace(tzinfo=IST_TZ)
    elif isinstance(raw_ts, str):
        parsed = datetime.fromisoformat(raw_ts)
        ts_ist = parsed.astimezone(IST_TZ) if parsed.tzinfo else parsed.replace(tzinfo=IST_TZ)
    else:
        ts_ist = datetime.now(IST_TZ)

    cum_vol = raw_tick.get("volume_traded")
    if cum_vol is None:
        cum_vol = raw_tick.get("volume")
    last_qty = float(raw_tick.get("last_traded_quantity") or raw_tick.get("last_quantity") or 0.0)

    depth = raw_tick.get("depth") or {}
    buy_depth = depth.get("buy") or []
    sell_depth = depth.get("sell") or []
    bid = float(buy_depth[0]["price"]) if buy_depth and buy_depth[0].get("price", 0) > 0 else None
    ask = float(sell_depth[0]["price"]) if sell_depth and sell_depth[0].get("price", 0) > 0 else None

    return TickEvent(
        symbol=token_to_symbol[token],
        exchange="NSE",
        timestamp_ist=ts_ist,
        ltp=ltp,
        volume_delta=max(0.0, last_qty),
        cumulative_volume=float(cum_vol) if cum_vol is not None else None,
        bid=bid,
        ask=ask,
        provider="zerodha_kite",
    )


def parse_angel_one_tick(
    raw_tick: dict[str, Any],
    token_to_symbol: dict[str, str],
) -> TickEvent | None:
    """Normalize an Angel One SmartAPI WebSocket v2 dict payload into a `TickEvent`."""
    token = str(raw_tick.get("token", ""))
    if token not in token_to_symbol:
        return None
    # Angel One v2 reports prices in paise (x100) for integer fields; support float rupees too.
    raw_ltp = raw_tick.get("last_traded_price")
    if raw_ltp is None:
        return None
    ltp_val = float(raw_ltp)
    if isinstance(raw_ltp, int) and ltp_val > 10000:
        ltp_val = ltp_val / 100.0

    exch_ts_ms = raw_tick.get("exchange_timestamp")
    if isinstance(exch_ts_ms, (int, float)):
        ts_ist = datetime.fromtimestamp(float(exch_ts_ms) / 1000.0, tz=IST_TZ)
    elif isinstance(exch_ts_ms, datetime):
        ts_ist = exch_ts_ms.astimezone(IST_TZ) if exch_ts_ms.tzinfo else exch_ts_ms.replace(tzinfo=IST_TZ)
    else:
        ts_ist = datetime.now(IST_TZ)

    cum_vol = raw_tick.get("volume_trade_for_the_day")
    last_qty = float(raw_tick.get("last_traded_quantity") or 0.0)

    return TickEvent(
        symbol=token_to_symbol[token],
        exchange="NSE",
        timestamp_ist=ts_ist,
        ltp=ltp_val,
        volume_delta=max(0.0, last_qty),
        cumulative_volume=float(cum_vol) if cum_vol is not None else None,
        provider="angel_one_smartapi",
    )


class ZerodhaKiteReadOnlyAdapter(ReadOnlyTickAdapter):
    """Read-only Zerodha KiteTicker WebSocket adapter using local env vars."""

    provider_name: ProviderName = "zerodha_kite"

    def __init__(self, symbol_to_token: dict[str, int] | None = None) -> None:
        assert_read_only_recorder_config()
        self.symbol_to_token = {k.upper(): int(v) for k, v in (symbol_to_token or {}).items()}
        self._kws: Any = None

    def validate_credentials(self) -> None:
        api_key = os.environ.get("KITE_API_KEY", "").strip()
        access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
        if not api_key or not access_token:
            raise RuntimeError(
                "Missing Zerodha Kite credentials. Set KITE_API_KEY and KITE_ACCESS_TOKEN "
                "in your local environment (.env) before starting live tick recording."
            )

    def stream(self, symbols: list[str], on_tick: Callable[[TickEvent], None]) -> None:
        self.validate_credentials()
        try:
            from kiteconnect import KiteTicker  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Optional dependency `kiteconnect` is not installed. "
                "Install locally with `pip install kiteconnect` to record Zerodha ticks."
            ) from exc

        token_to_symbol = {
            self.symbol_to_token[s.upper()]: s.upper()
            for s in symbols
            if s.upper() in self.symbol_to_token
        }
        if not token_to_symbol:
            raise ValueError("No instrument tokens configured for requested symbols.")

        kws = KiteTicker(os.environ["KITE_API_KEY"], os.environ["KITE_ACCESS_TOKEN"])
        self._kws = kws

        def _on_ticks(ws: Any, ticks: list[dict[str, Any]]) -> None:
            for raw in ticks:
                evt = parse_zerodha_kite_tick(raw, token_to_symbol)
                if evt is not None:
                    on_tick(evt)

        def _on_connect(ws: Any, response: Any) -> None:
            tokens = list(token_to_symbol.keys())
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)

        kws.on_ticks = _on_ticks
        kws.on_connect = _on_connect
        kws.connect(threaded=False)

    def stop(self) -> None:
        if self._kws is not None:
            self._kws.close()


class AngelOneSmartApiReadOnlyAdapter(ReadOnlyTickAdapter):
    """Read-only Angel One SmartAPI WebSocket v2 adapter using local env vars."""

    provider_name: ProviderName = "angel_one_smartapi"

    def __init__(self, symbol_to_token: dict[str, str] | None = None) -> None:
        assert_read_only_recorder_config()
        self.symbol_to_token = {k.upper(): str(v) for k, v in (symbol_to_token or {}).items()}
        self._sws: Any = None

    def validate_credentials(self) -> None:
        missing = [
            key
            for key in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_AUTH_TOKEN", "ANGEL_FEED_TOKEN")
            if not os.environ.get(key, "").strip()
        ]
        if missing:
            raise RuntimeError(
                f"Missing Angel One SmartAPI credentials: {', '.join(missing)}. "
                "Set them in your local environment (.env) before starting live tick recording."
            )

    def stream(self, symbols: list[str], on_tick: Callable[[TickEvent], None]) -> None:
        self.validate_credentials()
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Optional dependency `smartapi-python` is not installed. "
                "Install locally with `pip install smartapi-python` to record Angel One ticks."
            ) from exc

        token_to_symbol = {
            self.symbol_to_token[s.upper()]: s.upper()
            for s in symbols
            if s.upper() in self.symbol_to_token
        }
        if not token_to_symbol:
            raise ValueError("No Angel One symbol tokens configured for requested symbols.")

        sws = SmartWebSocketV2(
            os.environ["ANGEL_AUTH_TOKEN"],
            os.environ["ANGEL_API_KEY"],
            os.environ["ANGEL_CLIENT_CODE"],
            os.environ["ANGEL_FEED_TOKEN"],
        )
        self._sws = sws

        def _on_open(wsapp: Any) -> None:
            sws.subscribe(
                "bharatalpha_readonly_recorder",
                2,  # Quote mode (read-only)
                [{"exchangeType": 1, "tokens": list(token_to_symbol.keys())}],
            )

        def _on_data(wsapp: Any, message: dict[str, Any]) -> None:
            evt = parse_angel_one_tick(message, token_to_symbol)
            if evt is not None:
                on_tick(evt)

        sws.on_open = _on_open
        sws.on_data = _on_data
        sws.connect()

    def stop(self) -> None:
        if self._sws is not None:
            self._sws.close_connection()


class OfflineJsonlTickAdapter(ReadOnlyTickAdapter):
    """Deterministic offline JSONL tick stream adapter for testing and replay."""

    provider_name: ProviderName = "offline_replay"

    def __init__(self, jsonl_path: str | Path) -> None:
        assert_read_only_recorder_config()
        self.jsonl_path = Path(jsonl_path)
        self._stopped = False

    def validate_credentials(self) -> None:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Offline tick JSONL file not found: {self.jsonl_path}")

    def stream(self, symbols: list[str], on_tick: Callable[[TickEvent], None]) -> None:
        self.validate_credentials()
        allowed = {s.upper() for s in symbols}
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if self._stopped:
                    break
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                tick = TickEvent(**payload)
                if not allowed or tick.symbol in allowed:
                    on_tick(tick)

    def stop(self) -> None:
        self._stopped = True


class ReadOnlyNseTickRecorder:
    """End-to-end read-only NSE tick recorder and multi-timeframe Parquet archiver."""

    def __init__(
        self,
        *,
        adapter: ReadOnlyTickAdapter,
        output_dir: str | Path = "data/ticks",
        timeframes: Iterable[TimeframeName] = SUPPORTED_TIMEFRAMES,
    ) -> None:
        assert_read_only_recorder_config()
        self.adapter = adapter
        self.archive = TickParquetArchive(base_dir=output_dir)
        self.timeframes: tuple[TimeframeName, ...] = tuple(timeframes)
        self.aggregator = CausalTickAggregator(timeframes=self.timeframes)
        self._raw_ticks_by_symbol_day: dict[tuple[str, date], list[TickEvent]] = defaultdict(list)

    def _handle_tick(self, tick: TickEvent) -> None:
        if self.aggregator.ingest_tick(tick):
            day = tick.timestamp_ist.astimezone(IST_TZ).date()
            self._raw_ticks_by_symbol_day[(tick.symbol, day)].append(tick)

    def run(self, symbols: list[str]) -> TickRecorderSummary:
        """Stream ticks from the read-only adapter and persist partitioned Parquet archives."""
        assert_read_only_recorder_config()
        clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not clean_symbols:
            raise ValueError("At least one NSE symbol is required")

        self.adapter.stream(clean_symbols, self._handle_tick)
        all_bars = self.aggregator.finalize_all()

        bars_grouped: dict[tuple[str, date], dict[TimeframeName, list[RecordedBar]]] = defaultdict(
            lambda: {tf: [] for tf in self.timeframes}
        )
        for bar in all_bars:
            day = bar.ts_open_ist.astimezone(IST_TZ).date()
            bars_grouped[(bar.symbol, day)][bar.timeframe].append(bar)

        summary = TickRecorderSummary(
            provider=self.adapter.provider_name,
            symbols=clean_symbols,
            total_ticks_accepted=self.aggregator.accepted_ticks,
            out_of_order_ticks=self.aggregator.out_of_order_ticks,
            rejected_outside_hours_ticks=self.aggregator.rejected_outside_hours_ticks,
            bars_written_by_timeframe={tf: 0 for tf in self.timeframes},
        )

        for (sym, day), raw_ticks in sorted(self._raw_ticks_by_symbol_day.items()):
            tf_map = bars_grouped.get((sym, day), {tf: [] for tf in self.timeframes})
            day_dir, prov = self.archive.write_day_archive(
                symbol=sym,
                trading_date=day,
                provider=self.adapter.provider_name,
                raw_ticks=raw_ticks,
                bars_by_timeframe=tf_map,
                out_of_order_ticks=self.aggregator.out_of_order_ticks,
                rejected_outside_hours_ticks=self.aggregator.rejected_outside_hours_ticks,
            )
            summary.written_files.append(str(day_dir / "provenance.json"))
            for tf, count in prov.bar_counts.items():
                summary.bars_written_by_timeframe[tf] = summary.bars_written_by_timeframe.get(tf, 0) + count

        return summary
