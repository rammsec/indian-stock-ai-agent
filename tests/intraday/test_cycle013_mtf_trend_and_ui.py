"""CYCLE-013 Tests: Causal 5m MTF Trend Alignment Gate, Wider R:R Targets, & Upgraded Dashboard KPI UI."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from trader.config import IST_TZ
from trader.timing.monitoring_session import evaluate_causal_5m_mtf_trend


def _make_1m_df(direction: str, n_bars: int = 120) -> pd.DataFrame:
    t0 = datetime(2026, 11, 20, 9, 15, tzinfo=IST_TZ)
    rows = []
    px = 2400.0
    step = 0.5 if direction == "BULLISH" else -0.5
    for i in range(n_bars):
        px += step
        rows.append(
            {
                "timestamp": t0 + timedelta(minutes=i),
                "open": px - step * 0.5,
                "high": px + 0.5,
                "low": px - 0.5,
                "close": px,
                "volume": 5000.0,
            }
        )
    return pd.DataFrame(rows)


def test_python_causal_5m_mtf_trend_bullish_bearish_neutral() -> None:
    assert evaluate_causal_5m_mtf_trend(_make_1m_df("BULLISH", n_bars=120)) == "BULLISH"
    assert evaluate_causal_5m_mtf_trend(_make_1m_df("BEARISH", n_bars=120)) == "BEARISH"
    # Fewer than 5 completed 5m bars (< 25 1m bars) -> NEUTRAL
    assert evaluate_causal_5m_mtf_trend(_make_1m_df("BULLISH", n_bars=10)) == "NEUTRAL"


def test_js_engine_mtf_5m_trend_gate_and_telemetry() -> None:
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
    const trends = new Set(sim.decisions.map(x => x.mtf5mTrend));
    const mtfBlocks = sim.decisions.flatMap(x => x.rejectionReasons || []).filter(r => r.startsWith("MTF_5M_TREND_MISALIGNED:"));
    console.log(JSON.stringify({
      requireMtf: sim.requireMtfTrendAlignment,
      minRr: sim.minWinnerRrMultiple,
      trends: Array.from(trends),
      mtfBlocks: mtfBlocks.length,
      entries: sim.entriesOpenedThisSession
    }));
    """
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=20)
    out = json.loads(proc.stdout.strip())
    assert out["requireMtf"] is True
    assert out["minRr"] == 2.5
    assert any(t in ("BULLISH", "BEARISH") for t in out["trends"])
    assert out["entries"] <= 4


def test_dashboard_ui_contains_kpi_telemetry_strip_and_legend() -> None:
    html = Path("docs/candle-agent/index.html").read_text(encoding="utf-8")
    for dom_id in ("kpi-price", "kpi-position", "kpi-pnl", "kpi-trades", "kpi-mtf", "kpi-winrate"):
        assert f'id="{dom_id}"' in html, f"Missing upgraded KPI tile #{dom_id}"
    assert "kpi-strip" in html
    assert "5m MTF Trend Gate" in html
