"""CYCLE-004 Regression Tests: CNC Delivery Charges, Multi-Day Swing & Cycle Progression."""

from __future__ import annotations

from trader.config import LAB_DEFAULT_UNIVERSE
from trader.data.loader_1h import get_available_1h_dates
from trader.research.cycle004 import calculate_cnc_delivery_costs, run_cnc_swing_backtest
from trader.research.orchestrator import ChiefResearchOrchestrator


def test_cnc_delivery_cost_model_zero_brokerage_and_statutory() -> None:
    buy_cost = calculate_cnc_delivery_costs("BUY", 25_000.0)
    assert buy_cost.brokerage_inr == 0.0
    assert buy_cost.stt_inr == 25.0  # 0.1%
    assert buy_cost.stamp_duty_inr == 3.75  # 0.015%
    assert buy_cost.dp_charge_inr == 0.0

    sell_cost = calculate_cnc_delivery_costs("SELL", 25_000.0)
    assert sell_cost.brokerage_inr == 0.0
    assert sell_cost.stt_inr == 25.0
    assert sell_cost.stamp_duty_inr == 0.0
    assert sell_cost.dp_charge_inr == 15.93


def test_cnc_swing_backtest_runs_and_reconciles() -> None:
    elig, _ = get_available_1h_dates(LAB_DEFAULT_UNIVERSE, min_bars_per_stock=6)
    res = run_cnc_swing_backtest(
        elig[-15:],
        "STRAT-019-60M-CNC-ORB-SWING",
        "60m CNC Delivery ORB Swing",
        mode="ORB_SWING",
        initial_capital_inr=50_000.0,
        slippage_bps=5.0,
    )
    assert res["sessions_evaluated"] == 15
    assert res["total_brokerage_inr"] == 0.0
    assert abs((50_000.0 + res["gross_pnl_inr"] - res["total_fees_inr"]) - res["final_equity_inr"]) <= 1.0


def test_build_cycle_progression_returns_chronological_cycles() -> None:
    prog = ChiefResearchOrchestrator().build_cycle_progression()
    assert len(prog) >= 3
    assert [p["cycle_id"] for p in prog[:3]] == ["CYCLE-001", "CYCLE-002", "CYCLE-003"]
