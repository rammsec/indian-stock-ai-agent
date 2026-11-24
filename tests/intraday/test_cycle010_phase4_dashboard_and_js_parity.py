"""CYCLE-010 Phase 4 Acceptance & Python-JS Numerical Parity Tests:
- Verified multi-timeframe (5m / 15m / 1h) session_catalog.json integrity & 15:15-15:30 IST clamping
- Unified docs/candle-agent/index.html & docs/trader/index.html UI contracts
- Node.js execution parity of docs/candle-agent/engine.js vs Python (NSE tick rounding,
  Indian statutory intraday costs, 15% position cap, protective exits, full-day 5m & 1h sessions)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from trader.portfolio.cost_calculator import calculate_intraday_costs
from trader.portfolio.ledger import round_to_nse_tick

CATALOG_PATH = Path("docs/candle-agent/data/session_catalog.json")
ENGINE_JS_PATH = Path("docs/candle-agent/engine.js")
CANDLE_HTML_PATH = Path("docs/candle-agent/index.html")
TRADER_HTML_PATH = Path("docs/trader/index.html")
TRADER_JS_PATH = Path("docs/trader/app.js")


def test_session_catalog_provenance_symbols_and_1515_bar_clamping() -> None:
    assert CATALOG_PATH.exists(), "docs/candle-agent/data/session_catalog.json must exist"
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    prov = cat["provenance"]
    assert prov["status"] == "PARTIALLY_VERIFIED"
    assert prov["synthetic_substitution"] is False
    assert prov["paper_only"] is True
    assert prov["live_trading_enabled"] is False
    assert prov["confidence_calibration_status"] == "UNCALIBRATED"

    assert cat["symbols"] == ["TCS", "RELIANCE", "ITC", "TATAELXSI", "BIRLACABLE"]
    assert "INFY" not in cat["symbols"]
    assert cat["timeframes"] == ["5m", "15m", "1h"]
    assert len(cat["dates_by_timeframe"]["5m"]) == 10
    assert len(cat["dates_by_timeframe"]["15m"]) == 10
    assert len(cat["dates_by_timeframe"]["1h"]) == 60

    # Verify NSE 15:15 IST 1h closing bars are strictly clamped to 15:30 IST (never 16:15)
    tcs_1h = cat["bars"]["1h"]["TCS"]
    bars_1515 = [b for b in tcs_1h if b["t"].endswith("15:15")]
    assert len(bars_1515) >= 50
    for b in bars_1515:
        assert b["tc"].endswith("15:30"), f"Expected 15:30 close for 15:15 bar, got {b}"


def test_dashboard_html_and_trader_launcher_contracts() -> None:
    candle_html = CANDLE_HTML_PATH.read_text(encoding="utf-8")
    assert 'src="engine.js"' in candle_html
    assert 'id="select-symbol"' in candle_html
    assert 'id="select-timeframe"' in candle_html
    assert 'id="select-date"' in candle_html
    assert 'id="select-direction"' in candle_html
    assert 'id="select-human-perm"' in candle_html
    assert 'id="btn-play"' in candle_html
    assert 'id="btn-step"' in candle_html
    assert 'id="btn-stop-all"' in candle_html
    assert "UNCALIBRATED" in candle_html
    assert "PAPER-ONLY" in candle_html
    # Must never silently fall back to TCS 2026-11-09
    assert "allCandleData['TCS']" not in candle_html

    trader_html = TRADER_HTML_PATH.read_text(encoding="utf-8")
    assert 'id="candle-agent-launcher"' in trader_html
    trader_js = TRADER_JS_PATH.read_text(encoding="utf-8")
    assert "data.metadata || data.meta" in trader_js


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_engine_numerical_parity_with_python_and_full_day_sessions() -> None:
    """Execute docs/candle-agent/engine.js in Node.js and assert exact parity with Python."""
    js_script = r"""
const fs = require("fs");
const E = require("./docs/candle-agent/engine.js");
const cat = JSON.parse(fs.readFileSync("./docs/candle-agent/data/session_catalog.json", "utf8"));

const buyCost = E.calculateIndianIntradayCosts(2400.0, 5, "BUY", "NSE", 5.0);
const sellCost = E.calculateIndianIntradayCosts(2450.0, 5, "SELL", "NSE", 5.0);

// Run full-day 1h session on genuine TCS 2026-10-29
const tcs1h = cat.bars["1h"]["TCS"];
const sess1h = tcs1h.filter(b => b.t.startsWith("2026-10-29"));
const warm1h = tcs1h.filter(b => b.t < sess1h[0].t).slice(-50);
const sim1h = new E.IntradaySessionSimulator({
  symbol: "TCS", sessionDate: "2026-10-29", timeframe: "1h",
  warmupBars: warm1h, sessionBars: sess1h,
  centralDirection: "BULLISH", humanPermission: "ALLOW_LONG", initialCapital: 50000
});
while (sim1h.stepNextBar() !== null) {}

// Run full-day 5m session on genuine TCS 2026-11-20 (75 bars!)
const tcs5m = cat.bars["5m"]["TCS"];
const sess5m = tcs5m.filter(b => b.t.startsWith("2026-11-20"));
const warm5m = tcs5m.filter(b => b.t < sess5m[0].t).slice(-50);
const sim5m = new E.IntradaySessionSimulator({
  symbol: "TCS", sessionDate: "2026-11-20", timeframe: "5m",
  warmupBars: warm5m, sessionBars: sess5m,
  centralDirection: "BULLISH", humanPermission: "ALLOW_LONG", initialCapital: 50000
});
while (sim5m.stepNextBar() !== null) {}

// Verify protective exit allowed even when human permission flips ALLOW_LONG -> ALLOW_SHORT
const exitPermFlip = E.humanPermissionAllows("ALLOW_SHORT", "EXIT_LONG");
const coverPermFlip = E.humanPermissionAllows("ALLOW_LONG", "COVER_SHORT");

// Verify 15% single-stock cap on ₹50,000 budget @ ₹2,400 -> max 3 shares (₹7,200 <= ₹7,500)
const capSim = new E.IntradaySessionSimulator({
  symbol: "TCS", sessionDate: "2026-11-20", timeframe: "1h",
  warmupBars: [], sessionBars: [], centralDirection: "BULLISH", humanPermission: "ALLOW_LONG", initialCapital: 50000
});
const sizedQty = capSim._sizeOrder("BUY", 2400.0, 2364.0);

console.log(JSON.stringify({
  tickUp: E.roundToNseTick(2484.53),
  tickDown: E.roundToNseTick(2484.52),
  buyTotal: buyCost.totalFrictionInr,
  sellTotal: sellCost.totalFrictionInr,
  sizedQty,
  exitPermFlip: exitPermFlip.allowed,
  coverPermFlip: coverPermFlip.allowed,
  sim1hBars: sim1h.decisions.length,
  sim1hRiskState: sim1h.riskState,
  sim1hFlatAtClose: sim1h.position === null,
  sim1hOutcomeCount: sim1h.outcomeReport ? sim1h.outcomeReport.totalDecisions : 0,
  sim5mBars: sim5m.decisions.length,
  sim5mRiskState: sim5m.riskState,
  sim5mFlatAtClose: sim5m.position === null,
  sim5mOutcomeCount: sim5m.outcomeReport ? sim5m.outcomeReport.totalDecisions : 0,
}));
"""
    proc = subprocess.run(
        ["node", "-e", js_script],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    res = json.loads(proc.stdout.strip())

    # 1. Tick rounding exact parity
    assert res["tickUp"] == round_to_nse_tick(2484.53) == 2484.55
    assert res["tickDown"] == round_to_nse_tick(2484.52) == 2484.50

    # 2. Statutory Indian intraday costs exact parity
    py_buy = calculate_intraday_costs(price=2400.0, quantity=5, side="BUY", exchange="NSE").total_friction_inr
    py_sell = calculate_intraday_costs(price=2450.0, quantity=5, side="SELL", exchange="NSE").total_friction_inr
    assert abs(res["buyTotal"] - py_buy) < 1e-3
    assert abs(res["sellTotal"] - py_sell) < 1e-3

    # 3. 15% max position weight cap (₹50k * 15% = ₹7,500 -> max 3 shares @ ₹2,400)
    assert res["sizedQty"] == 3

    # 4. Protective exits never blocked on directional flip
    assert res["exitPermFlip"] is True
    assert res["coverPermFlip"] is True

    # 5. Full-day 1h (7 bars) and 5m (75 bars) continuous sessions complete cleanly
    assert res["sim1hBars"] == 7
    assert res["sim1hRiskState"] == "REDUCING"
    assert res["sim1hFlatAtClose"] is True
    assert res["sim1hOutcomeCount"] == 7

    assert res["sim5mBars"] >= 70
    assert res["sim5mRiskState"] == "REDUCING"
    assert res["sim5mFlatAtClose"] is True
    assert res["sim5mOutcomeCount"] == res["sim5mBars"]
