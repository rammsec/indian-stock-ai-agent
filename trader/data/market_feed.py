"""Pluggable BaseMarketFeed Interface & Offline/Local/Read-Only Broker Adapters.

All feeds return canonical, lookahead-free ``OHLCVEvidenceFrame`` instances
enforcing ``ts_close_ist <= ts_init_ist <= as_of_ist`` and provenance capping.
Broker adapters here are strictly READ-ONLY market-data feeds (zero order-routing).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import pandas as pd

from ..config import IST_TZ
from ..intelligence.data_contract import (
    OHLCVEvidenceFrame,
    ProvenanceTier,
    build_ohlcv_evidence_frame,
)
from ..intelligence.parsing import sha256_file

TimeframeLiteral = Literal["1m", "5m", "15m", "1h"]

_TF_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


class BrokerFeedNotConfiguredError(RuntimeError):
    """Raised when a read-only broker feed lacks required local API credentials."""


def _normalize_raw_ohlcv(
    df: pd.DataFrame,
    timeframe: TimeframeLiteral,
    as_of_ist: datetime,
    *,
    tz_default: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Normalize raw OHLCV columns into IST-aware ts_open_ist / ts_close_ist / ts_init_ist and filter <= as_of_ist."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ts_open_ist", "ts_close_ist", "ts_init_ist", "open", "high", "low", "close", "volume"]
        )

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]

    if "ts_open_ist" not in out.columns:
        if "timestamp_ist" in out.columns:
            ts = pd.to_datetime(out["timestamp_ist"])
        elif "timestamp" in out.columns:
            ts = pd.to_datetime(out["timestamp"])
        elif "date" in out.columns:
            ts = pd.to_datetime(out["date"])
        elif "datetime" in out.columns:
            ts = pd.to_datetime(out["datetime"])
        elif isinstance(out.index, pd.DatetimeIndex):
            ts = pd.to_datetime(out.index)
        else:
            raise ValueError("OHLCV DataFrame must contain 'ts_open_ist', 'timestamp', 'date', or DatetimeIndex")

        if getattr(ts.dt if hasattr(ts, "dt") else ts, "tz", None) is None:
            ts = ts.dt.tz_localize(tz_default) if hasattr(ts, "dt") else ts.tz_localize(tz_default)
        else:
            ts = ts.dt.tz_convert("Asia/Kolkata") if hasattr(ts, "dt") else ts.tz_convert("Asia/Kolkata")
        out["ts_open_ist"] = ts.values if not hasattr(ts, "dt") else ts

    else:
        ts_open = pd.to_datetime(out["ts_open_ist"])
        if getattr(ts_open.dt, "tz", None) is None:
            ts_open = ts_open.dt.tz_localize(tz_default)
        else:
            ts_open = ts_open.dt.tz_convert("Asia/Kolkata")
        out["ts_open_ist"] = ts_open

    delta = pd.Timedelta(minutes=_TF_MINUTES[timeframe])
    if "ts_close_ist" not in out.columns:
        ts_open_dt = pd.to_datetime(out["ts_open_ist"])
        ts_close_dt = ts_open_dt + delta
        if timeframe == "1h":
            # NSE final shortened 1h session bar runs 15:15-15:30 IST (15 minutes), not 15:15-16:15 IST
            is_final_bar = (ts_open_dt.dt.hour == 15) & (ts_open_dt.dt.minute == 15)
            ts_close_dt = ts_close_dt.where(~is_final_bar, ts_open_dt + pd.Timedelta(minutes=15))
        out["ts_close_ist"] = ts_close_dt
    else:
        ts_close = pd.to_datetime(out["ts_close_ist"])
        if getattr(ts_close.dt, "tz", None) is None:
            ts_close = ts_close.dt.tz_localize(tz_default)
        else:
            ts_close = ts_close.dt.tz_convert("Asia/Kolkata")
        out["ts_close_ist"] = ts_close

    if "ts_init_ist" not in out.columns:
        out["ts_init_ist"] = out["ts_close_ist"]
    else:
        ts_init = pd.to_datetime(out["ts_init_ist"])
        if getattr(ts_init.dt, "tz", None) is None:
            ts_init = ts_init.dt.tz_localize(tz_default)
        else:
            ts_init = ts_init.dt.tz_convert("Asia/Kolkata")
        out["ts_init_ist"] = ts_init

    as_of_ts = pd.Timestamp(as_of_ist.astimezone(IST_TZ))
    out = out.loc[
        (pd.to_datetime(out["ts_close_ist"]) <= as_of_ts)
        & (pd.to_datetime(out["ts_init_ist"]) <= as_of_ts)
    ].copy()
    out = out.sort_values("ts_close_ist").drop_duplicates(subset=["ts_close_ist"], keep="last")
    return out.reset_index(drop=True)


class BaseMarketFeed(ABC):
    """Abstract pluggable market data feed returning canonical OHLCVEvidenceFrames."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical provider identifier (e.g. 'local_parquet', 'yahoo_finance', 'zerodha_kite')."""

    @property
    @abstractmethod
    def is_authoritative(self) -> bool:
        """True only for exchange/broker-authoritative feeds."""

    @property
    @abstractmethod
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        """Supported bar timeframes."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        """Return causal OHLCVEvidenceFrame with ts_close_ist <= ts_init_ist <= as_of_ist."""

    def get_latest_bar(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
    ) -> dict[str, Any] | None:
        """Return the latest closed bar at or before as_of_ist, or None if unavailable."""
        ev = self.get_bars(symbol=symbol, timeframe=timeframe, as_of_ist=as_of_ist, lookback_bars=5)
        if ev.bar_count == 0 or ev.frame.empty:
            return None
        row = ev.frame.iloc[-1]
        return {
            "symbol": ev.symbol,
            "timeframe": ev.timeframe,
            "ts_open_ist": row["ts_open_ist"],
            "ts_close_ist": row["ts_close_ist"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "provenance_status": ev.provenance_status.value,
        }


class ParquetCSVMarketFeed(BaseMarketFeed):
    """Offline/local Parquet & CSV market feed (supports committed 1h archive + local 5m/15m)."""

    def __init__(
        self,
        data_dir: str | Path = Path("data/intraday"),
        *,
        exchange: str = "NSE",
        provenance_tier: ProvenanceTier = ProvenanceTier.PARTIALLY_VERIFIED,
        source_prefix: str = "yahoo_finance",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.exchange = exchange
        self._provenance_tier = provenance_tier
        self._source_prefix = source_prefix

    @property
    def provider_name(self) -> str:
        return "local_parquet_csv"

    @property
    def is_authoritative(self) -> bool:
        return False

    @property
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        return ("1m", "5m", "15m", "1h")

    def _resolve_file(self, symbol: str, timeframe: TimeframeLiteral) -> Path | None:
        sym = symbol.strip().upper()
        candidates = [
            self.data_dir / f"{sym}_{timeframe}.parquet",
            self.data_dir / f"{sym}_{self.exchange}_{timeframe}.parquet",
            self.data_dir / f"{sym}_{timeframe}.csv",
            self.data_dir / f"{sym}_{self.exchange}_{timeframe}.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        if as_of_ist.tzinfo is None:
            raise ValueError("as_of_ist must be timezone-aware")
        as_of_ist = as_of_ist.astimezone(IST_TZ)
        sym = symbol.strip().upper()
        src_name = f"{self._source_prefix}_{timeframe}"
        path = self._resolve_file(sym, timeframe)

        if path is None:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=src_name,
                exchange=self.exchange,
                missing_reason="LOCAL_FILE_NOT_FOUND",
            )

        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        norm = _normalize_raw_ohlcv(raw, timeframe, as_of_ist)
        if lookback_bars > 0 and len(norm) > lookback_bars:
            norm = norm.iloc[-lookback_bars:].reset_index(drop=True)

        if norm.empty:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=src_name,
                source_sha256=sha256_file(path),
                exchange=self.exchange,
                missing_reason="NO_CAUSAL_BARS_BEFORE_AS_OF",
            )

        return build_ohlcv_evidence_frame(
            symbol=sym,
            timeframe=timeframe,
            df=norm,
            as_of_ist=as_of_ist,
            provenance_status=self._provenance_tier,
            source_name=src_name,
            source_sha256=sha256_file(path),
            exchange=self.exchange,
        )


class YFinanceMarketFeed(BaseMarketFeed):
    """Local Yahoo Finance feed for 5m / 15m / 1h bars (capped at PARTIALLY_VERIFIED)."""

    def __init__(
        self,
        *,
        exchange_suffix: str = ".NS",
        fetcher_fn: Callable[[str, str, str], pd.DataFrame] | None = None,
    ) -> None:
        self.exchange_suffix = exchange_suffix
        self._fetcher_fn = fetcher_fn

    @property
    def provider_name(self) -> str:
        return "yahoo_finance"

    @property
    def is_authoritative(self) -> bool:
        return False

    @property
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        return ("5m", "15m", "1h")

    def _default_fetch(self, yf_symbol: str, interval: str, period: str) -> pd.DataFrame:
        import yfinance as yf  # lazy import for OfflineExecutionGuard safety

        return yf.download(yf_symbol, interval=interval, period=period, progress=False, auto_adjust=False)

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        if as_of_ist.tzinfo is None:
            raise ValueError("as_of_ist must be timezone-aware")
        as_of_ist = as_of_ist.astimezone(IST_TZ)
        sym = symbol.strip().upper()
        yf_sym = sym if "." in sym else f"{sym}{self.exchange_suffix}"
        src_name = f"yahoo_finance_{timeframe}"
        period = "5d" if timeframe == "5m" else ("30d" if timeframe == "15m" else "60d")

        try:
            fetcher = self._fetcher_fn or self._default_fetch
            raw = fetcher(yf_sym, timeframe, period)
            norm = _normalize_raw_ohlcv(raw, timeframe, as_of_ist)
            if lookback_bars > 0 and len(norm) > lookback_bars:
                norm = norm.iloc[-lookback_bars:].reset_index(drop=True)
        except Exception as exc:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=src_name,
                missing_reason=f"YFINANCE_FETCH_ERROR:{type(exc).__name__}",
            )

        if norm.empty:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=src_name,
                missing_reason="YFINANCE_EMPTY",
            )

        return build_ohlcv_evidence_frame(
            symbol=sym,
            timeframe=timeframe,
            df=norm,
            as_of_ist=as_of_ist,
            provenance_status=ProvenanceTier.PARTIALLY_VERIFIED,
            source_name=src_name,
        )


class ZerodhaKiteReadOnlyFeed(BaseMarketFeed):
    """Read-only Zerodha Kite Connect market-data adapter (zero order-routing capability).

    Reads ``KITE_API_KEY`` and ``KITE_ACCESS_TOKEN`` from environment variables when
    not passed explicitly. Plug in your read-only token locally to stream real 5m/15m/1h
    bars into the Research Agent and Candle Bar Agent in paper-only mode.
    """

    _INTERVAL_MAP: dict[str, str] = {"5m": "5minute", "15m": "15minute", "1h": "60minute"}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        access_token: str | None = None,
        instrument_tokens: dict[str, int] | None = None,
        http_get_fn: Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] | None = None,
        fail_to_unavailable: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("KITE_API_KEY", "")
        self.access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN", "")
        self.instrument_tokens = {k.upper(): int(v) for k, v in (instrument_tokens or {}).items()}
        self._http_get_fn = http_get_fn
        self.fail_to_unavailable = fail_to_unavailable

    @property
    def provider_name(self) -> str:
        return "zerodha_kite_readonly"

    @property
    def is_authoritative(self) -> bool:
        return True

    @property
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        return ("5m", "15m", "1h")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.access_token)

    def _default_http_get(self, url: str, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
        import httpx

        resp = httpx.get(url, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        if as_of_ist.tzinfo is None:
            raise ValueError("as_of_ist must be timezone-aware")
        as_of_ist = as_of_ist.astimezone(IST_TZ)
        sym = symbol.strip().upper()

        if not self.is_configured:
            if not self.fail_to_unavailable:
                raise BrokerFeedNotConfiguredError("KITE_API_KEY / KITE_ACCESS_TOKEN not configured")
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=self.provider_name,
                missing_reason="KITE_CREDENTIALS_NOT_CONFIGURED",
            )

        token = self.instrument_tokens.get(sym)
        if token is None:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=self.provider_name,
                missing_reason=f"MISSING_INSTRUMENT_TOKEN:{sym}",
            )

        interval = self._INTERVAL_MAP[timeframe]
        from_dt = as_of_ist - timedelta(days=15 if timeframe == "5m" else 45)
        url = f"https://api.kite.trade/instruments/historical/{token}/{interval}"
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.api_key}:{self.access_token}",
        }
        params = {
            "from": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "to": as_of_ist.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            getter = self._http_get_fn or self._default_http_get
            payload = getter(url, headers, params)
            candles = payload.get("data", {}).get("candles", [])
            raw = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            norm = _normalize_raw_ohlcv(raw, timeframe, as_of_ist)
            if lookback_bars > 0 and len(norm) > lookback_bars:
                norm = norm.iloc[-lookback_bars:].reset_index(drop=True)
        except Exception as exc:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=self.provider_name,
                missing_reason=f"KITE_FETCH_ERROR:{type(exc).__name__}",
            )

        if norm.empty:
            return build_ohlcv_evidence_frame(
                symbol=sym,
                timeframe=timeframe,
                df=None,
                as_of_ist=as_of_ist,
                provenance_status=ProvenanceTier.UNAVAILABLE,
                source_name=self.provider_name,
                missing_reason="KITE_EMPTY",
            )

        return build_ohlcv_evidence_frame(
            symbol=sym,
            timeframe=timeframe,
            df=norm,
            as_of_ist=as_of_ist,
            provenance_status=ProvenanceTier.VERIFIED,
            source_name=self.provider_name,
        )


class CompositeMarketFeed(BaseMarketFeed):
    """Prioritized fallback chain across multiple BaseMarketFeed providers."""

    def __init__(self, feeds: Sequence[BaseMarketFeed]) -> None:
        if not feeds:
            raise ValueError("CompositeMarketFeed requires at least one feed")
        self.feeds = tuple(feeds)

    @property
    def provider_name(self) -> str:
        return "+".join(f.provider_name for f in self.feeds)

    @property
    def is_authoritative(self) -> bool:
        return any(f.is_authoritative for f in self.feeds)

    @property
    def supported_timeframes(self) -> tuple[TimeframeLiteral, ...]:
        return ("5m", "15m", "1h")

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeframeLiteral,
        as_of_ist: datetime,
        lookback_bars: int = 200,
    ) -> OHLCVEvidenceFrame:
        last_ev: OHLCVEvidenceFrame | None = None
        for feed in self.feeds:
            if timeframe not in feed.supported_timeframes:
                continue
            ev = feed.get_bars(symbol=symbol, timeframe=timeframe, as_of_ist=as_of_ist, lookback_bars=lookback_bars)
            if ev.provenance_status != ProvenanceTier.UNAVAILABLE and ev.bar_count > 0:
                return ev
            last_ev = ev
        assert last_ev is not None
        return last_ev
