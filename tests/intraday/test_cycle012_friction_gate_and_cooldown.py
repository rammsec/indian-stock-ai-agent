"""CYCLE-012 Tests: Friction-to-Reward Edge Gate, Post-Exit Cooldown, & Daily Entry Cap (Python + JS Parity)."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta

from trader.authority import PermissionType
from trader.config import IST_TZ
from trader.data.loader_1m import OneMinuteSessionBundle
from trader.data.schema import Candle5M
from trader.evaluation.walkforward_1m import InMemory1MReplayFeed
from trader.timing.central_directional_agent import DirectionalBias
from trader.timing.monitoring_session import IntradayCandleMonitoringSession


def _make_trending_1m_bundle(symbol: str = "TCS", trade_date: date = date(2026, 11, 20), n_bars: int = 80) -> OneMinuteSessionBundle:
    """Create a deterministic 1m fixture with alternating swings to trigger multiple entry candidates."""
    warmup: list[Candle5M] = []
    t_w = datetime(2026, 11, 19, 14, 30, tzinfo=IST_TZ)
    for i in range(40):
        p = 2400.0 + i * 0.2
        warmup.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t_w + timedelta(minutes=i),
                open=p,
                high=p + 0.6,
                low=p - 0.3,
                close=p + 0.3,
                volume=5000.0,
            )
        )

    session: list[Candle5M] = []
    t0 = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, tzinfo=IST_TZ)
    price = 2410.0
    for i in range(n_bars):
        step = 1.8 if (i % 6) in (0, 1, 2, 3) else -2.5
        o = price
        c = max(2300.0, price + step)
        h = max(o, c) + 0.8
        l = min(o, c) - 0.8
        price = c
        session.append(
            Candle5M(
                symbol=symbol,
                exchange="NSE",
                timestamp=t0 + timedelta(minutes=i),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=12000.0 if step > 0 else 4000.0,
            )
        )
    return OneMinuteSessionBundle(
        symbol=symbol,
        trading_date=trade_date,
        warmup_bars=warmup,
        session_bars=session,
        source_providers=("unit_test_cycle012",),
    )


def test_python_friction_gate_blocks_low_edge_entries() -> None:
    bundle = _make_trending_1m_bundle()
    feed = InMemory1MReplayFeed(bundle)
    sess = IntradayCandleMonitoringSession(feed=feed, initial_capital=50000.0)
    # Extremely high min_reward_to_friction_ratio (100x) must block every candidate via FRICTION_EDGE_TOO_LOW
    res = sess.run_session(
        symbol="TCS",
        session_date=bundle.trading_date,
        timeframe="1m",
        direction=DirectionalBias.BULLISH,
        human_permission=PermissionType.ALLOW_BOTH,
        warmup_lookback_bars=40,
        min_reward_to_friction_ratio=100.0,
        cooldown_bars_after_exit=0,
        max_entries_per_session=20,
    )
    assert len(res.closed_trades) == 0
    all_rejs = [r for d in res.decisions for r in d.rejection_reasons]
    assert any(r.startswith("FRICTION_EDGE_TOO_LOW:") for r in all_rejs)


def test_python_daily_entry_cap_enforced() -> None:
    bundle = _make_trending_1m_bundle()
    feed = InMemory1MReplayFeed(bundle)
    sess = IntradayCandleMonitoringSession(feed=feed, initial_capital=50000.0)
    res = sess.run_session(
        symbol="TCS",
        session_date=bundle.trading_date,
        timeframe="1m",
        direction=DirectionalBias.BULLISH,
        human_permission=PermissionType.ALLOW_BOTH,
        warmup_lookback_bars=40,
        min_reward_to_friction_ratio=0.0,
        cooldown_bars_after_exit=0,
        max_entries_per_session=1,
    )
    assert len(res.closed_trades) <= 1


def test_js_engine_cycle012_friction_cap_and_cooldown_parity() -> None:
    script = """
    const fs = require("fs");
    const eng = require("./docs/candle-agent/engine.js");
    const cat = JSON.parse(fs.readFileSync("./docs/candle-agent/data/session_catalog.json", "utf8"));
    const sym = "TCS";
    const d = cat.dates_by_timeframe["1m"][0];
    const all = cat.bars["1m"][sym];
    const sessionBars = all.filter(b => b.t.startsWith(d));
    const warmupBars = all.filter(b => b.t < sessionBars[0].t).slice(-50);

    // 1. Default 1m discipline (3x friction, 5-bar cooldown, max 4 entries)
    const simDefault = new eng.IntradaySessionSimulator({
      symbol: sym, sessionDate: d, timeframe: "1m",
      warmupBars, sessionBars, centralDirection: "BULLISH", humanPermission: "ALLOW_BOTH", initialCapital: 50000
    });
    while (simDefault.barIdx < simDefault.sessionBars.length) simDefault.stepNextBar();

    // 2. High friction ratio (100x) must block all entries with FRICTION_EDGE_TOO_LOW
    const simHighFric = new eng.IntradaySessionSimulator({
      symbol: sym, sessionDate: d, timeframe: "1m",
      warmupBars, sessionBars, centralDirection: "BULLISH", humanPermission: "ALLOW_BOTH", initialCapital: 50000,
      minRewardToFrictionRatio: 100.0, cooldownBarsAfterExit: 0, maxEntriesPerSession: 20
    });
    while (simHighFric.barIdx < simHighFric.sessionBars.length) simHighFric.stepNextBar();
    const fricReasons = simHighFric.decisions.flatMap(x => x.rejectionReasons || []).filter(r => r.startsWith("FRICTION_EDGE_TOO_LOW:"));

    console.log(JSON.stringify({
      defaultEntries: simDefault.entriesOpenedThisSession,
      defaultCap: simDefault.maxEntriesPerSession,
      highFricEntries: simHighFric.entriesOpenedThisSession,
      fricBlocks: fricReasons.length
    }));
    """
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=20)
    out = json.loads(proc.stdout.strip())
    assert out["defaultCap"] == 4
    assert out["defaultEntries"] <= 4
    assert out["highFricEntries"] == 0
    assert out["fricBlocks"] > 0
