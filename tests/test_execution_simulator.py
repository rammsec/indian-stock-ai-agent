"""Unit tests for the realistic next-day Indian equity execution simulator."""

from __future__ import annotations

import pandas as pd

from backend.backtesting.execution_simulator import ExecutionConfig, simulate_oos_execution


def _make_signal_and_ohlc(
    next_open: float,
    next_high: float,
    next_low: float,
    next_close: float,
    prev_close: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oos_df = pd.DataFrame(
        [
            {
                "symbol": "RELIANCE",
                "observation_date": "2025-06-02",
                "prob_next_day_up": 0.60,
                "expected_next_day_return_pct": 0.80,
                "expected_5d_return_pct": 1.50,
                "pm_atr_14_pct": 2.0,
                "vd_amihud_illiquidity_20d": 0.001,
                "regime_label": "Weak Bull",
                "target_next_day_ret": (next_close / prev_close - 1.0) * 100.0,
            },
            {
                "symbol": "RELIANCE",
                "observation_date": "2025-06-03",
                "prob_next_day_up": 0.45,
                "expected_next_day_return_pct": -0.20,
                "expected_5d_return_pct": -0.40,
                "pm_atr_14_pct": 2.0,
                "vd_amihud_illiquidity_20d": 0.001,
                "regime_label": "Weak Bull",
                "target_next_day_ret": 0.0,
            },
        ]
    )
    ohlc_df = pd.DataFrame(
        [
            {
                "symbol": "RELIANCE",
                "observation_date": "2025-06-02",
                "open": 99.5,
                "high": 101.0,
                "low": 99.0,
                "close": prev_close,
            },
            {
                "symbol": "RELIANCE",
                "observation_date": "2025-06-03",
                "open": next_open,
                "high": next_high,
                "low": next_low,
                "close": next_close,
            },
        ]
    )
    return oos_df, ohlc_df


def test_ambiguous_daily_ohlc_sl_and_tp_resolves_conservatively_to_sl() -> None:
    # ATR = 2%, sl = 100 * (1 - 1.5*0.02) = 97.0, tp = 100 * (1 + 2.5*0.02) = 105.0
    # Both low=96.0 <= 97.0 AND high=106.0 >= 105.0 -> must conservatively exit at SL (97.0, -3.0%)
    oos_df, ohlc_df = _make_signal_and_ohlc(next_open=100.0, next_high=106.0, next_low=96.0, next_close=104.0)
    cfg = ExecutionConfig(entry_timing="next_open", use_sl_tp=True, max_position_weight=0.20)
    sim_df, daily_port, stats = simulate_oos_execution(oos_df, ohlc_df=ohlc_df, config=cfg)

    assert stats["ambiguous_ohlc_conservative_sl"] == 1
    assert sim_df.loc[0, "exit_reason"] == "AMBIGUOUS_OHLC_CONSERVATIVE_SL"
    assert abs(sim_df.loc[0, "gross_trade_ret_pct"] - (-3.0)) < 1e-4
    # Portfolio weight capped at 20% -> portfolio gross return is 0.20 * (-3.0%) = -0.60%
    assert abs(daily_port.loc[0, "gross_ret_pct"] - (-0.60)) < 1e-4


def test_opening_gap_up_filter_skips_excessive_overnight_gap() -> None:
    # Next open = 104.0 (+4.0% gap) > max_gap_up_pct (2.5%) -> must skip entry
    oos_df, ohlc_df = _make_signal_and_ohlc(next_open=104.0, next_high=105.0, next_low=103.0, next_close=104.5)
    cfg = ExecutionConfig(entry_timing="next_open", max_gap_up_pct=2.5)
    sim_df, _, stats = simulate_oos_execution(oos_df, ohlc_df=ohlc_df, config=cfg)

    assert stats["skipped_gap_up"] == 1
    assert stats["executed_trades"] == 0
    assert sim_df.loc[0, "position"] == 0.0
    assert sim_df.loc[0, "exit_reason"] == "SKIPPED_GAP_UP"


def test_intraday_mis_has_lower_cost_than_delivery_cnc() -> None:
    oos_df, ohlc_df = _make_signal_and_ohlc(next_open=100.0, next_high=101.5, next_low=99.5, next_close=101.0)
    sim_cnc, _, _ = simulate_oos_execution(
        oos_df, ohlc_df=ohlc_df, config=ExecutionConfig(cost_mode="delivery", max_hold_sessions=1)
    )
    sim_mis, _, _ = simulate_oos_execution(
        oos_df, ohlc_df=ohlc_df, config=ExecutionConfig(cost_mode="intraday", max_hold_sessions=1)
    )
    assert sim_mis.loc[0, "cost_drag_pct"] < sim_cnc.loc[0, "cost_drag_pct"]
