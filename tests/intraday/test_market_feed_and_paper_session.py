"""Tests for BaseMarketFeed adapters and PaperFeedSignalSession (Tracks 3 & 4)."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trader.config import IST_TZ
from trader.data.market_feed import (
    CompositeMarketFeed,
    ParquetCSVMarketFeed,
    YFinanceMarketFeed,
    ZerodhaKiteReadOnlyFeed,
)
from trader.intelligence.data_contract import ProvenanceTier
from trader.timing.paper_feed_session import PaperFeedSignalSession

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_INTRADAY_DIR = REPO_ROOT / "data" / "intraday"


def _make_synthetic_ohlcv(symbol: str = "RELIANCE", n_sessions: int = 2, freq_minutes: int = 5) -> pd.DataFrame:
    rows = []
    base_date = datetime(2026, 11, 19, 9, 15, tzinfo=IST_TZ)
    bars_per_session = int(375 // freq_minutes)
    for s in range(n_sessions):
        sess_start = base_date + timedelta(days=s)
        for i in range(bars_per_session):
            o_time = sess_start + timedelta(minutes=i * freq_minutes)
            c_time = o_time + timedelta(minutes=freq_minutes)
            px = 2500.0 + s * 10.0 + i * 0.5
            rows.append({
                "ts_open_ist": o_time,
                "ts_close_ist": c_time,
                "ts_init_ist": c_time,
                "open": px,
                "high": px + 2.0,
                "low": px - 1.5,
                "close": px + 0.8,
                "volume": 1000 + i * 20,
            })
    return pd.DataFrame(rows)


def test_parquet_csv_market_feed_on_committed_1h_archive() -> None:
    feed = ParquetCSVMarketFeed(data_dir=DATA_INTRADAY_DIR)
    as_of = datetime(2026, 11, 20, 12, 30, tzinfo=IST_TZ)
    ev = feed.get_bars("RELIANCE", "1h", as_of, lookback_bars=50)

    assert ev.symbol == "RELIANCE"
    assert ev.timeframe == "1h"
    assert ev.bar_count == 50
    assert ev.provenance_status == ProvenanceTier.PARTIALLY_VERIFIED
    assert pd.Timestamp(ev.latest_ts_event_ist) <= pd.Timestamp(as_of)


def test_parquet_csv_market_feed_supports_local_5m_and_15m_csv(tmp_path: Path) -> None:
    df_5m = _make_synthetic_ohlcv("RELIANCE", n_sessions=2, freq_minutes=5)
    df_15m = _make_synthetic_ohlcv("RELIANCE", n_sessions=2, freq_minutes=15)
    df_5m.to_csv(tmp_path / "RELIANCE_5m.csv", index=False)
    df_15m.to_csv(tmp_path / "RELIANCE_15m.csv", index=False)

    last_date = pd.to_datetime(df_5m["ts_open_ist"]).dt.tz_convert("Asia/Kolkata").dt.date.iloc[-1]
    as_of = datetime.combine(last_date, time(11, 30), tzinfo=IST_TZ)

    feed = ParquetCSVMarketFeed(data_dir=tmp_path)
    ev5 = feed.get_bars("RELIANCE", "5m", as_of, lookback_bars=20)
    ev15 = feed.get_bars("RELIANCE", "15m", as_of, lookback_bars=20)
    assert ev5.bar_count == 20
    assert ev15.bar_count == 20
    assert pd.Timestamp(ev5.latest_ts_event_ist) <= pd.Timestamp(as_of)


def test_yfinance_market_feed_caps_provenance_and_fails_closed() -> None:
    df = _make_synthetic_ohlcv("TCS", n_sessions=2, freq_minutes=5)
    raw = df.rename(columns={"ts_open_ist": "timestamp"})[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    as_of = pd.Timestamp(df["ts_close_ist"].iloc[-5]).to_pydatetime().astimezone(IST_TZ)

    feed = YFinanceMarketFeed(fetcher_fn=lambda _sym, _iv, _per: raw)
    ev = feed.get_bars("TCS", "5m", as_of, lookback_bars=15)
    assert ev.provenance_status == ProvenanceTier.PARTIALLY_VERIFIED
    assert ev.bar_count == 15

    empty_feed = YFinanceMarketFeed(fetcher_fn=lambda _s, _i, _p: pd.DataFrame())
    ev_empty = empty_feed.get_bars("TCS", "5m", as_of)
    assert ev_empty.provenance_status == ProvenanceTier.UNAVAILABLE
    assert ev_empty.bar_count == 0


def test_zerodha_kite_readonly_feed_unconfigured_and_mocked_verified() -> None:
    unconf = ZerodhaKiteReadOnlyFeed(api_key="", access_token="")
    as_of = datetime(2026, 11, 20, 12, 30, tzinfo=IST_TZ)
    ev_unconf = unconf.get_bars("RELIANCE", "5m", as_of)
    assert ev_unconf.provenance_status == ProvenanceTier.UNAVAILABLE
    assert ev_unconf.missing_reason == "KITE_CREDENTIALS_NOT_CONFIGURED"

    # Zero order-placement methods on read-only feed
    assert not hasattr(unconf, "place_order")
    assert not hasattr(unconf, "submit_order")

    df = _make_synthetic_ohlcv("RELIANCE", n_sessions=2, freq_minutes=5)
    candles = [
        [r["ts_open_ist"].isoformat(), r["open"], r["high"], r["low"], r["close"], r["volume"]]
        for _, r in df.iterrows()
    ]
    as_of_real = pd.Timestamp(df["ts_close_ist"].iloc[-3]).to_pydatetime().astimezone(IST_TZ)

    kite = ZerodhaKiteReadOnlyFeed(
        api_key="test_key",
        access_token="test_token",
        instrument_tokens={"RELIANCE": 738561},
        http_get_fn=lambda _u, _h, _p: {"status": "success", "data": {"candles": candles}},
    )
    ev_ok = kite.get_bars("RELIANCE", "5m", as_of_real, lookback_bars=20)
    assert ev_ok.provenance_status == ProvenanceTier.VERIFIED
    assert ev_ok.bar_count == 20


def test_composite_feed_and_paper_signal_session_end_to_end(tmp_path: Path) -> None:
    df_5m = _make_synthetic_ohlcv("RELIANCE", n_sessions=5, freq_minutes=5)
    df_15m = _make_synthetic_ohlcv("RELIANCE", n_sessions=5, freq_minutes=15)
    df_1h = _make_synthetic_ohlcv("RELIANCE", n_sessions=10, freq_minutes=60)
    df_5m.to_parquet(tmp_path / "RELIANCE_5m.parquet", index=False)
    df_15m.to_parquet(tmp_path / "RELIANCE_15m.parquet", index=False)
    df_1h.to_parquet(tmp_path / "RELIANCE_1h.parquet", index=False)

    last_date = pd.to_datetime(df_1h["ts_open_ist"]).dt.tz_convert("Asia/Kolkata").dt.date.iloc[-1]
    as_of = datetime.combine(last_date, time(12, 30), tzinfo=IST_TZ)
    obs_date = (last_date - timedelta(days=1)).isoformat()

    composite = CompositeMarketFeed(
        [
            ZerodhaKiteReadOnlyFeed(api_key="", access_token=""),  # falls back cleanly
            ParquetCSVMarketFeed(data_dir=tmp_path),
        ]
    )

    predictions_payload = {
        "generated_at": (as_of - timedelta(hours=1)).isoformat(),
        "predictions": [
            {
                "symbol": "RELIANCE",
                "direction": "BULLISH",
                "prob_next_day_up": 0.68,
                "expected_next_day_return_pct": 1.25,
                "signal_tier": "HIGH",
                "observation_date": obs_date,
            }
        ],
    }
    intelligence_payload = {
        "model_version": "5.0",
        "generated_at": (as_of - timedelta(hours=1)).isoformat(),
        "stocks": {
            "RELIANCE": {
                "overall_signal": "BULLISH",
                "composite_score": 0.68,
                "news_sentiment_score": 0.35,
                "observation_date": obs_date,
            }
        },
    }
    regime_payload = {
        "generated_at": (as_of - timedelta(hours=1)).isoformat(),
        "observation_date": obs_date,
        "regime": "BULL_LOW_VOL",
        "volatility_regime": "LOW",
    }

    session = PaperFeedSignalSession(feed=composite)
    report = session.generate_daily_calls(
        symbols=["RELIANCE"],
        as_of_ist=as_of,
        primary_timeframe="1h",
        predictions_payload=predictions_payload,
        intelligence_payload=intelligence_payload,
        regime_payload=regime_payload,
    )

    assert report.paper_only is True
    assert len(report.calls) == 1
    call = report.calls[0]
    assert call.symbol == "RELIANCE"
    assert call.bars_5m_count > 0
    assert call.bars_15m_count > 0
    assert call.bars_1h_count > 0
    assert call.timing is not None
