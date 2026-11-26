"""CYCLE-014 Tests: Chandelier ATR Trailing Stop, Fee-Aware Breakeven Lock, & Early-Exit Suppression (Python + JS Parity)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from trader.timing.monitoring_session import compute_chandelier_trailing_stop


def test_python_chandelier_long_monotonic_and_breakeven_lock() -> None:
    # Before +1.2R (highest=1008 < 1000 + 1.2*10 = 1012), no breakeven lock yet
    sl1, be1 = compute_chandelier_trailing_stop(
        side="LONG",
        entry_price=1000.0,
        current_close=1007.0,
        current_stop_loss=990.0,
        highest_price=1008.0,
        lowest_price=999.0,
        atr_14=5.0,
        initial_risk_distance=10.0,
        round_trip_fee_per_share=2.5,
        chandelier_atr_mult=2.2,
        breakeven_trigger_r=1.2,
    )
    assert be1 is False
    # highest - 2.2*5 = 997.0 > 990.0 -> ratchets up!
    assert sl1 == pytest.approx(997.0)

    # Once highest reaches 1014 (>= 1012 = +1.2R), breakeven + fees + 0.05 = 1002.55 locks in!
    sl2, be2 = compute_chandelier_trailing_stop(
        side="LONG",
        entry_price=1000.0,
        current_close=1013.0,
        current_stop_loss=sl1,
        highest_price=1014.0,
        lowest_price=999.0,
        atr_14=5.0,
        initial_risk_distance=10.0,
        round_trip_fee_per_share=2.5,
        chandelier_atr_mult=2.2,
        breakeven_trigger_r=1.2,
    )
    assert be2 is True
    assert sl2 is not None and sl2 >= 1002.55

    # Monotonicity: if price pulls back and ATR spikes, stop must NEVER widen downward
    sl3, _ = compute_chandelier_trailing_stop(
        side="LONG",
        entry_price=1000.0,
        current_close=1005.0,
        current_stop_loss=sl2,
        highest_price=1014.0,
        lowest_price=999.0,
        atr_14=20.0,
        initial_risk_distance=10.0,
        round_trip_fee_per_share=2.5,
        chandelier_atr_mult=2.2,
        breakeven_trigger_r=1.2,
    )
    assert sl3 == sl2


def test_python_chandelier_short_monotonic_and_breakeven_lock() -> None:
    sl, be = compute_chandelier_trailing_stop(
        side="SHORT",
        entry_price=1000.0,
        current_close=985.0,
        current_stop_loss=1010.0,
        highest_price=1001.0,
        lowest_price=984.0,
        atr_14=4.0,
        initial_risk_distance=10.0,
        round_trip_fee_per_share=2.5,
        chandelier_atr_mult=2.2,
        breakeven_trigger_r=1.2,
    )
    assert be is True
    # min(be_cap=997.45, ch=984+8.8=992.80) = 992.80
    assert sl == pytest.approx(992.80)


def test_python_js_chandelier_exact_parity_and_ui_overlay() -> None:
    script = """
    const eng = require("./docs/candle-agent/engine.js");
    const rLong = eng.computeChandelierTrailingStop({
      side: "LONG", entryPrice: 1000.0, currentClose: 1013.0, currentStopLoss: 997.0,
      highestPrice: 1014.0, lowestPrice: 999.0, atr14: 5.0, initialRiskDistance: 10.0,
      roundTripFeePerShare: 2.5, chandelierAtrMult: 2.2, breakevenTriggerR: 1.2
    });
    const rShort = eng.computeChandelierTrailingStop({
      side: "SHORT", entryPrice: 1000.0, currentClose: 985.0, currentStopLoss: 1010.0,
      highestPrice: 1001.0, lowestPrice: 984.0, atr14: 4.0, initialRiskDistance: 10.0,
      roundTripFeePerShare: 2.5, chandelierAtrMult: 2.2, breakevenTriggerR: 1.2
    });
    console.log(JSON.stringify({ rLong, rShort }));
    """
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=15)
    out = json.loads(proc.stdout.strip())
    py_long_sl, py_long_be = compute_chandelier_trailing_stop(
        side="LONG", entry_price=1000.0, current_close=1013.0, current_stop_loss=997.0,
        highest_price=1014.0, lowest_price=999.0, atr_14=5.0, initial_risk_distance=10.0,
        round_trip_fee_per_share=2.5, chandelier_atr_mult=2.2, breakeven_trigger_r=1.2,
    )
    py_short_sl, py_short_be = compute_chandelier_trailing_stop(
        side="SHORT", entry_price=1000.0, current_close=985.0, current_stop_loss=1010.0,
        highest_price=1001.0, lowest_price=984.0, atr_14=4.0, initial_risk_distance=10.0,
        round_trip_fee_per_share=2.5, chandelier_atr_mult=2.2, breakeven_trigger_r=1.2,
    )
    assert out["rLong"]["stopLoss"] == pytest.approx(py_long_sl)
    assert out["rLong"]["breakevenLocked"] is py_long_be
    assert out["rShort"]["stopLoss"] == pytest.approx(py_short_sl)
    assert out["rShort"]["breakevenLocked"] is py_short_be

    html = Path("docs/candle-agent/index.html").read_text(encoding="utf-8")
    assert "chandelierStopHistory" in html
    assert "BE LOCKED" in html
