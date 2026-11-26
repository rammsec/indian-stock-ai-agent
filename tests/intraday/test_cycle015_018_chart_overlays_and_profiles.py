"""CYCLE-015..018 Tests: Causal ADX(14), Supertrend(10,3), ORB 15m, PDH/PDL, VWAP ±1σ/±2σ, Sweeps/FVG, & Strategy Profiles."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader.config import IST_TZ
from trader.intelligence.chart_overlays import (
    compute_adx_regime,
    compute_chart_overlays,
    compute_supertrend,
    compute_vwap_bands,
    detect_liquidity_sweeps_and_fvg,
)


def test_python_adx_supertrend_vwap_and_sweep_fvg_primitives() -> None:
    # Strong monotonic uptrend -> ADX trending, Supertrend BULLISH
    n = 40
    closes = np.linspace(1000.0, 1080.0, n)
    highs = closes + 1.5
    lows = closes - 1.5
    vols = np.full(n, 5000.0)

    adx, pdi, mdi, regime = compute_adx_regime(highs, lows, closes)
    assert adx is not None and adx >= 20.0
    assert pdi is not None and mdi is not None and pdi > mdi
    assert regime in ("TRENDING", "STRONG_EXHAUSTION_WATCH")

    st_val, st_dir = compute_supertrend(highs, lows, closes)
    assert st_val is not None
    assert st_dir == "BULLISH"

    vw, u1, l1, u2, l2, stretch = compute_vwap_bands(highs, lows, closes, vols)
    assert vw is not None and u2 is not None and l2 is not None
    assert l2 <= l1 <= vw <= u1 <= u2
    assert stretch in ("NORMAL", "OVEREXTENDED_ABOVE_2SD")

    # Explicit Bullish Liquidity Sweep + Bullish FVG
    h_sw = np.array([102.0] * 12 + [101.0, 105.0, 109.0])
    l_sw = np.array([100.0] * 12 + [98.5, 101.5, 106.0])
    c_sw = np.array([101.0] * 12 + [100.5, 104.5, 108.5])
    _, fvg = detect_liquidity_sweeps_and_fvg(h_sw, l_sw, c_sw)
    assert fvg is not None and fvg.direction == "BULLISH"

    sweep, _ = detect_liquidity_sweeps_and_fvg(h_sw[:13], l_sw[:13], c_sw[:13])
    assert sweep == "BULLISH_SWEEP"


def test_python_js_chart_overlays_exact_parity_on_real_1m_data() -> None:
    cat = json.loads(Path("docs/candle-agent/data/session_catalog.json").read_text(encoding="utf-8"))
    sym = "TCS"
    d_str = cat["dates_by_timeframe"]["1m"][0]
    all_bars = cat["bars"]["1m"][sym]
    sess_bars = [b for b in all_bars if b["t"].startswith(d_str)]
    warmup_bars = [b for b in all_bars if b["t"] < sess_bars[0]["t"]][-50:]
    combined = warmup_bars + sess_bars[:60]

    # Python snapshot
    df = pd.DataFrame(
        [
            {
                "timestamp": datetime.strptime(b["t"], "%Y-%m-%d %H:%M").replace(tzinfo=IST_TZ),
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
            }
            for b in combined
        ]
    )
    py_ov = compute_chart_overlays(df, date.fromisoformat(d_str))

    # JS snapshot
    script = f"""
    const fs = require("fs");
    const eng = require("./docs/candle-agent/engine.js");
    const cat = JSON.parse(fs.readFileSync("./docs/candle-agent/data/session_catalog.json", "utf8"));
    const all = cat.bars["1m"]["{sym}"];
    const sess = all.filter(b => b.t.startsWith("{d_str}"));
    const warm = all.filter(b => b.t < sess[0].t).slice(-50);
    const ov = eng.computeChartOverlays(warm.concat(sess.slice(0, 60)), "{d_str}");
    console.log(JSON.stringify({{
      adx14: ov.adx14, marketRegime: ov.marketRegime,
      supertrendDirection: ov.supertrendDirection, supertrendValue: ov.supertrendValue,
      pdh: ov.pdh, pdl: ov.pdl, orbHigh: ov.orbHigh, orbLow: ov.orbLow,
      orbReady: ov.orbReady, orbState: ov.orbState, vwapStretch: ov.vwapStretch
    }}));
    """
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=20)
    js_ov = json.loads(proc.stdout.strip())

    assert js_ov["adx14"] == pytest.approx(py_ov.adx_14, abs=1e-4)
    assert js_ov["marketRegime"] == py_ov.market_regime
    assert js_ov["supertrendDirection"] == py_ov.supertrend_direction
    assert js_ov["supertrendValue"] == pytest.approx(py_ov.supertrend_value, abs=1e-4)
    assert js_ov["pdh"] == pytest.approx(py_ov.pdh)
    assert js_ov["pdl"] == pytest.approx(py_ov.pdl)
    assert js_ov["orbHigh"] == pytest.approx(py_ov.orb_high)
    assert js_ov["orbLow"] == pytest.approx(py_ov.orb_low)
    assert js_ov["orbReady"] is py_ov.orb_ready
    assert js_ov["orbState"] == py_ov.orb_state
    assert js_ov["vwapStretch"] == py_ov.vwap_stretch


def test_all_four_strategy_profiles_and_ui_overlays() -> None:
    script = """
    const fs = require("fs");
    const eng = require("./docs/candle-agent/engine.js");
    const cat = JSON.parse(fs.readFileSync("./docs/candle-agent/data/session_catalog.json", "utf8"));
    const d = cat.dates_by_timeframe["1m"][0];
    const all = cat.bars["1m"]["TCS"];
    const sess = all.filter(b => b.t.startsWith(d)).slice(0, 120);
    const warm = all.filter(b => b.t < sess[0].t).slice(-50);
    const profiles = ["BALANCED_ALL", "ORB_BREAKOUT", "SUPERTREND_ADX", "LIQUIDITY_REVERSAL"];
    const out = {};
    for (const p of profiles) {
      const sim = new eng.IntradaySessionSimulator({
        symbol: "TCS", sessionDate: d, timeframe: "1m",
        warmupBars: warm, sessionBars: sess, centralDirection: "BULLISH",
        humanPermission: "ALLOW_BOTH", strategyProfile: p, initialCapital: 50000
      });
      while (sim.barIdx < sim.sessionBars.length) sim.stepNextBar();
      out[p] = { bars: sim.barIdx, entries: sim.entriesOpenedThisSession, hasOverlays: Boolean(sim.lastOverlays) };
    }
    console.log(JSON.stringify(out));
    """
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=25)
    res = json.loads(proc.stdout.strip())
    for p in ("BALANCED_ALL", "ORB_BREAKOUT", "SUPERTREND_ADX", "LIQUIDITY_REVERSAL"):
        assert res[p]["bars"] == 120
        assert res[p]["hasOverlays"] is True

    html = Path("docs/candle-agent/index.html").read_text(encoding="utf-8")
    assert 'id="select-profile"' in html
    assert "ORB_BREAKOUT" in html
    assert "SUPERTREND_ADX" in html
    assert "LIQUIDITY_REVERSAL" in html
    assert "supertrendSeries" in html
    assert "vwapUpper2Series" in html
