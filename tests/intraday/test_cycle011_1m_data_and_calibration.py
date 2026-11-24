"""CYCLE-011 Tests: Genuine 1m NSE Data Loader, 1m Browser Terminal Contract, and Walk-Forward Calibration."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trader.authority import PermissionType
from trader.config import IST_TZ
from trader.data.loader_1m import (
    DEFAULT_1M_DIR,
    DEFAULT_SYMBOLS_1M,
    OneMinuteSessionBundle,
    load_1m_sessions,
    normalize_1m_dataframe,
    sha256_file_1m,
)
from trader.data.schema import Candle5M
from trader.evaluation.walkforward_1m import (
    run_walkforward_1m_calibration,
    split_bundles_chronologically,
)
from trader.timing.central_directional_agent import DirectionalBias


def _make_1m_session_bundle(symbol: str, trade_date: date, base_price: float = 1000.0, n_bars: int = 40) -> OneMinuteSessionBundle:
    """Build a small deterministic in-memory 1m bundle strictly for offline unit tests."""
    warmup: list[Candle5M] = []
    prev_day = trade_date - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    t_w = datetime(prev_day.year, prev_day.month, prev_day.day, 14, 30, tzinfo=IST_TZ)
    for i in range(30):
        p = base_price + i * 0.25
        warmup.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t_w + timedelta(minutes=i),
                open=p,
                high=p + 0.6,
                low=p - 0.4,
                close=p + 0.2,
                volume=1500.0 + i * 20.0,
            )
        )

    session: list[Candle5M] = []
    t0 = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, tzinfo=IST_TZ)
    for i in range(n_bars):
        p = base_price + 8.0 + i * 0.4
        session.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t0 + timedelta(minutes=i),
                open=p,
                high=p + 0.8,
                low=p - 0.3,
                close=p + 0.35,
                volume=2200.0 + i * 35.0,
            )
        )

    return OneMinuteSessionBundle(
        symbol=symbol,
        trading_date=trade_date,
        warmup_bars=warmup,
        session_bars=session,
        source_providers=("unit_test_fixture",),
    )


class TestOneMinuteNormalizationAndProvenance:
    """Verify 1m IST normalization, NSE session filtering, and committed Parquet provenance."""

    def test_normalize_1m_dataframe_filters_invalid_rows(self) -> None:
        df = pd.DataFrame(
            [
                # Valid 09:15 IST Monday
                {"timestamp_ist": "2026-11-16T09:15:00+05:30", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 500, "source": "hf"},
                # Duplicate timestamp -> kept last
                {"timestamp_ist": "2026-11-16T09:15:00+05:30", "open": 100.0, "high": 101.2, "low": 99.5, "close": 100.8, "volume": 600, "source": "yf"},
                # Pre-open -> dropped
                {"timestamp_ist": "2026-11-16T09:14:00+05:30", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 100, "source": "hf"},
                # Post-close -> dropped
                {"timestamp_ist": "2026-11-16T15:30:00+05:30", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 100, "source": "hf"},
                # Weekend Saturday -> dropped
                {"timestamp_ist": "2026-11-14T10:00:00+05:30", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 100, "source": "hf"},
                # Corrupted OHLC (high < close) -> dropped
                {"timestamp_ist": "2026-11-16T09:16:00+05:30", "open": 100.0, "high": 99.0, "low": 98.0, "close": 100.5, "volume": 100, "source": "hf"},
            ]
        )
        clean = normalize_1m_dataframe(df, "TCS")
        assert len(clean) == 1
        assert clean["close"].iloc[0] == pytest.approx(100.8)
        assert clean["source"].iloc[0] == "yf"

    def test_genuine_1m_archive_and_provenance_integrity(self) -> None:
        prov_path = DEFAULT_1M_DIR / "provenance_1m.json"
        if not prov_path.exists() or not (DEFAULT_1M_DIR / "TCS_1m.parquet").exists():
            pytest.skip("Local 1m Parquet archive not present in this checkout (gitignored)")
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov["live_trading_enabled"] is False
        assert prov["synthetic_substitution"] is False
        for sym in DEFAULT_SYMBOLS_1M:
            assert sym in prov["symbols"]
            info = prov["symbols"][sym]
            p = DEFAULT_1M_DIR / info["parquet_file"]
            assert p.exists()
            assert sha256_file_1m(p) == info["sha256"]
            assert info["bars_count"] >= 1000
            bundles = load_1m_sessions(sym, base_dir=DEFAULT_1M_DIR, max_days=2)
            assert len(bundles) == 2
            assert bundles[0].trading_date < bundles[1].trading_date
            assert all(b.timestamp < bundles[1].session_bars[0].timestamp for b in bundles[1].warmup_bars)


class TestWalkForward1MCalibration:
    """Verify chronological split invariants and end-to-end 1m walk-forward calibration."""

    def test_chronological_split_strictly_separates_train_and_oos(self) -> None:
        bundles = [
            _make_1m_session_bundle("TCS", date(2026, 11, 16)),
            _make_1m_session_bundle("TCS", date(2026, 11, 17)),
            _make_1m_session_bundle("TCS", date(2026, 11, 18)),
            _make_1m_session_bundle("TCS", date(2026, 11, 19)),
        ]
        train, test = split_bundles_chronologically(bundles, train_ratio=0.75)
        assert len(train) == 3
        assert len(test) == 1
        assert max(b.trading_date for b in train) < min(b.trading_date for b in test)

    def test_walkforward_1m_calibration_runs_end_to_end(self) -> None:
        bundles = [
            _make_1m_session_bundle("TCS", date(2026, 11, 16), n_bars=25),
            _make_1m_session_bundle("TCS", date(2026, 11, 17), n_bars=25),
        ]
        rep = run_walkforward_1m_calibration(
            bundles,
            direction=DirectionalBias.BULLISH,
            human_permission=PermissionType.ALLOW_BOTH,
            train_ratio=0.50,
            min_calibration_samples=3,
        )
        d = rep.to_dict()
        assert d["symbols"] == ["TCS"]
        assert d["train_metrics"]["sessions_count"] == 1
        assert d["oos_test_metrics"]["sessions_count"] == 1
        assert d["train_metrics"]["date_end"] < d["oos_test_metrics"]["date_start"]


class TestBrowser1MCatalogAndMarkers:
    """Verify docs/candle-agent/ supports 1m timeframe and BUY/SHORT/EXIT canvas markers."""

    def test_catalog_has_1m_and_html_has_1m_and_markers(self) -> None:
        cat_path = Path("docs/candle-agent/data/session_catalog.json")
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        assert "1m" in cat["timeframes"]
        assert len(cat["dates_by_timeframe"]["1m"]) >= 5
        for sym in DEFAULT_SYMBOLS_1M:
            assert len(cat["bars"]["1m"][sym]) >= 375

        html = Path("docs/candle-agent/index.html").read_text(encoding="utf-8")
        assert '<option value="1m">' in html
        assert "▲ BUY" in html
        assert "▼ SHORT" in html
        assert "◆ EXIT" in html

    def test_node_js_engine_runs_full_375_bar_1m_session(self) -> None:
        script = """
        const fs = require("fs");
        const eng = require("./docs/candle-agent/engine.js");
        const cat = JSON.parse(fs.readFileSync("./docs/candle-agent/data/session_catalog.json", "utf8"));
        const sym = "TCS";
        const d = cat.dates_by_timeframe["1m"][0];
        const all = cat.bars["1m"][sym];
        const sessionBars = all.filter(b => b.t.startsWith(d));
        const warmupBars = all.filter(b => b.t < sessionBars[0].t).slice(-50);
        const sim = new eng.IntradaySessionSimulator({
          symbol: sym, sessionDate: d, timeframe: "1m",
          warmupBars, sessionBars, centralDirection: "BULLISH", humanPermission: "ALLOW_BOTH", initialCapital: 50000
        });
        while (sim.barIdx < sim.sessionBars.length) sim.stepNextBar();
        console.log(JSON.stringify({ bars: sim.barIdx, decisions: sim.decisions.length, orders: sim.orders.length }));
        """
        proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=20)
        out = json.loads(proc.stdout.strip())
        assert out["bars"] >= 350
        assert out["decisions"] == out["bars"]
