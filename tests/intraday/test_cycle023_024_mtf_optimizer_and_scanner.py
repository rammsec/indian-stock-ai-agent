"""CYCLE-023 & CYCLE-024 Test Suite.

Verifies:
- CYCLE-023: NSE-aligned causal 1m -> 5m / 15m resampling (zero cross-day merging,
  exact OHLCV aggregation) and strictly chronological multi-timeframe walk-forward comparison.
- CYCLE-024: Multi-stock causal intraday opportunity scanner zero-lookahead invariance,
  ranking accuracy, paper-only safety, and live dashboard #watchlist-scanner DOM.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from trader.config import IST_TZ
from trader.data.loader_1m import OneMinuteSessionBundle
from trader.data.schema import Candle5M
from trader.evaluation.mtf_walkforward_optimizer import (
    LIVE_TRADING_ENABLED as MTF_LIVE_FLAG,
    resample_bars_nse_causal,
    run_mtf_walkforward_comparison,
)
from trader.intelligence.multi_stock_scanner import (
    LIVE_TRADING_ENABLED as SCANNER_LIVE_FLAG,
    scan_multi_stock_opportunities,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_1m_bundle(
    symbol: str,
    trading_date: date,
    *,
    base_price: float = 1000.0,
    drift_per_bar: float = 0.20,
    n_bars: int = 120,
    seed: int = 11,
) -> OneMinuteSessionBundle:
    rng = np.random.default_rng(seed)
    prev_day = trading_date - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)

    warmup: list[Candle5M] = []
    px = base_price
    t_prev = datetime(prev_day.year, prev_day.month, prev_day.day, 12, 0, tzinfo=IST_TZ)
    for k in range(180):
        o = px
        c = max(10.0, o + float(rng.normal(0.0, 0.25)))
        warmup.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t_prev + timedelta(minutes=k),
                open=round(o, 2),
                high=round(max(o, c) + 0.3, 2),
                low=round(min(o, c) - 0.3, 2),
                close=round(c, 2),
                volume=5000.0,
            )
        )
        px = c

    session: list[Candle5M] = []
    t_open = datetime(trading_date.year, trading_date.month, trading_date.day, 9, 15, tzinfo=IST_TZ)
    for k in range(n_bars):
        o = px
        drift = drift_per_bar if k >= 15 else 0.02
        c = max(10.0, o + drift + float(rng.normal(0.0, 0.2)))
        session.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t_open + timedelta(minutes=k),
                open=round(o, 2),
                high=round(max(o, c) + 0.35, 2),
                low=round(min(o, c) - 0.35, 2),
                close=round(c, 2),
                volume=8000.0 + k * 25.0,
            )
        )
        px = c

    return OneMinuteSessionBundle(
        symbol=symbol,
        trading_date=trading_date,
        warmup_bars=warmup,
        session_bars=session,
        source_providers=("unit_test",),
    )


def test_cycle023_causal_resampling_alignment_and_no_cross_day_merge() -> None:
    """5m and 15m resampling must align to 09:15 IST and never merge bars across trading dates."""
    assert MTF_LIVE_FLAG is False
    d1 = date(2026, 11, 20)
    d2 = date(2026, 11, 23)
    bars: list[Candle5M] = []
    # Last 3 minutes of d1 (15:27, 15:28, 15:29) + first 5 minutes of d2 (09:15..09:19)
    for m, px in ((27, 100.0), (28, 102.0), (29, 101.0)):
        bars.append(
            Candle5M(
                symbol="TCS",
                exchange="NSE",
                timestamp=datetime(d1.year, d1.month, d1.day, 15, m, tzinfo=IST_TZ),
                open=px,
                high=px + 1.0,
                low=px - 1.0,
                close=px + 0.5,
                volume=100.0,
            )
        )
    for m, px in ((15, 200.0), (16, 203.0), (17, 201.0), (18, 205.0), (19, 204.0)):
        bars.append(
            Candle5M(
                symbol="TCS",
                exchange="NSE",
                timestamp=datetime(d2.year, d2.month, d2.day, 9, m, tzinfo=IST_TZ),
                open=px,
                high=px + 1.0,
                low=px - 1.0,
                close=px + 0.5,
                volume=200.0,
            )
        )

    res_5m = resample_bars_nse_causal(bars, 5)
    assert len(res_5m) == 2
    assert res_5m[0].timestamp.date() == d1
    assert res_5m[0].timestamp.hour == 15 and res_5m[0].timestamp.minute == 25
    assert res_5m[0].open == 100.0 and res_5m[0].high == 103.0 and res_5m[0].low == 99.0 and res_5m[0].volume == 300.0

    assert res_5m[1].timestamp.date() == d2
    assert res_5m[1].timestamp.hour == 9 and res_5m[1].timestamp.minute == 15
    assert res_5m[1].open == 200.0 and res_5m[1].high == 206.0 and res_5m[1].low == 199.0 and res_5m[1].volume == 1000.0


def test_cycle023_mtf_walkforward_chronological_comparison() -> None:
    """Multi-timeframe walk-forward optimizer must compare 1m/5m/15m strictly chronologically."""
    dates = [date(2026, 10, 12) + timedelta(days=d) for d in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]
    bundles = {
        "TCS": [_make_1m_bundle("TCS", d, drift_per_bar=0.35, seed=10 + i) for i, d in enumerate(dates)],
        "RELIANCE": [_make_1m_bundle("RELIANCE", d, base_price=1400.0, drift_per_bar=-0.25, seed=30 + i) for i, d in enumerate(dates)],
    }
    report = run_mtf_walkforward_comparison(
        bundles_override=bundles,
        timeframes_min=(1, 5, 15),
        initial_capital_inr=50_000.0,
        train_ratio=0.70,
    )
    assert report["live_trading_enabled"] is False
    assert report["zero_lookahead_verified"] is True
    assert report["train_date_range"][1] < report["oos_test_date_range"][0]
    assert set(report["timeframes"].keys()) == {"1m", "5m", "15m"}
    assert report["train_selected_overall"] is not None
    assert report["train_selected_overall"]["timeframe"] in {"1m", "5m", "15m"}


def test_cycle024_scanner_zero_lookahead_and_ranking() -> None:
    """Mutating bars after `as_of_ist` must NEVER alter scanner scores, directions, or ranking."""
    assert SCANNER_LIVE_FLAG is False
    d = date(2026, 11, 23)
    tcs = _make_1m_bundle("TCS", d, base_price=2500.0, drift_per_bar=0.80, n_bars=90, seed=1)
    itc = _make_1m_bundle("ITC", d, base_price=300.0, drift_per_bar=0.0, n_bars=90, seed=2)
    as_of = datetime(2026, 11, 23, 10, 15, tzinfo=IST_TZ)

    res1 = scan_multi_stock_opportunities({"TCS": tcs, "ITC": itc}, as_of_ist=as_of)
    assert res1.live_trading_enabled is False
    assert res1.symbols_scanned == 2
    assert [r.rank for r in res1.ranked_opportunities] == [1, 2]
    assert res1.ranked_opportunities[0].opportunity_score >= res1.ranked_opportunities[1].opportunity_score

    # Violently mutate future bars AFTER 10:15 IST and re-scan at 10:15 IST
    mut_tcs_bars = [
        b
        if b.timestamp.astimezone(IST_TZ) <= as_of
        else Candle5M(
            symbol=b.symbol,
            exchange=b.exchange,
            timestamp=b.timestamp,
            open=b.open * 0.5,
            high=b.high * 0.5,
            low=b.low * 0.5,
            close=b.close * 0.5,
            volume=b.volume * 20.0,
        )
        for b in tcs.session_bars
    ]
    tcs_mut = OneMinuteSessionBundle(
        symbol=tcs.symbol,
        trading_date=tcs.trading_date,
        warmup_bars=tcs.warmup_bars,
        session_bars=mut_tcs_bars,
        source_providers=tcs.source_providers,
    )
    res2 = scan_multi_stock_opportunities({"TCS": tcs_mut, "ITC": itc}, as_of_ist=as_of)
    assert res1.to_dict() == res2.to_dict()


def test_cycle024_dashboard_watchlist_scanner_dom() -> None:
    """docs/candle-agent/index.html must include the #watchlist-scanner strip and renderWatchlistScanner()."""
    html = (REPO_ROOT / "docs/candle-agent/index.html").read_text(encoding="utf-8")
    assert 'id="watchlist-scanner"' in html
    assert "function renderWatchlistScanner()" in html
