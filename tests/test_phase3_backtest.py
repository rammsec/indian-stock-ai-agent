"""Automated Test Suite for Phase 3 — Prediction Engine, Indian Cost Model & Walk-Forward Backtester."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from backend.prediction import (
    classify_cost_aware_action,
    compute_dynamic_slippage_pct,
    estimate_indian_roundtrip_cost,
)
from backend.storage.duckdb_store import DuckDBStore


def test_indian_equity_transaction_cost_model_statutory_components() -> None:
    """Verify Indian CNC delivery and MIS intraday statutory cost components & dynamic slippage."""
    deliv_largecap = estimate_indian_roundtrip_cost(
        mode="delivery", atr_14_pct=1.4, amihud_illiquidity=0.001
    )
    assert deliv_largecap.stt_pct == 0.20          # 0.10% Buy + 0.10% Sell
    assert deliv_largecap.stamp_duty_pct == 0.015  # 0.015% Buy
    assert deliv_largecap.dp_charge_pct == 0.015   # ~1.5 bps amortized DP charge
    assert 0.32 <= deliv_largecap.total_roundtrip_pct <= 0.45
    assert 32.0 <= deliv_largecap.total_roundtrip_bps <= 45.0

    # Higher volatility + higher illiquidity small-cap must incur higher slippage
    deliv_smallcap = estimate_indian_roundtrip_cost(
        mode="delivery", atr_14_pct=3.5, amihud_illiquidity=0.12
    )
    assert deliv_smallcap.slippage_pct > deliv_largecap.slippage_pct
    assert deliv_smallcap.total_roundtrip_pct > deliv_largecap.total_roundtrip_pct

    intra = estimate_indian_roundtrip_cost(
        mode="intraday", atr_14_pct=1.5, amihud_illiquidity=0.002
    )
    assert intra.stt_pct == 0.025                  # 0.025% Sell side only
    assert intra.stamp_duty_pct == 0.003           # 0.003% Buy side
    assert intra.brokerage_pct == 0.06             # 0.03% per leg
    assert 0.16 <= intra.total_roundtrip_pct <= 0.30


def test_cost_and_regime_aware_abstention_gate() -> None:
    """Verify NO TRADE abstention on quality failure, low agreement, sub-cost edge, and Bear/High-Vol regimes."""
    # 1. Data quality failure -> NO TRADE
    act1, reason1 = classify_cost_aware_action(
        prob_up=0.68,
        exp_1d_pct=1.2,
        exp_5d_pct=3.5,
        downside_mae_pct=-1.0,
        upside_mfe_pct=2.2,
        agreement_score=0.90,
        confidence_score=85.0,
        delivery_cost_pct=0.35,
        intraday_cost_pct=0.20,
        market_regime="Strong Bull",
        trade_eligible=False,
    )
    assert act1 == "NO TRADE"
    assert reason1 is not None and "quality" in reason1.lower()

    # 2. Low model agreement -> NO TRADE
    act2, reason2 = classify_cost_aware_action(
        prob_up=0.62,
        exp_1d_pct=0.8,
        exp_5d_pct=2.0,
        downside_mae_pct=-1.0,
        upside_mfe_pct=1.8,
        agreement_score=0.35,
        confidence_score=65.0,
        delivery_cost_pct=0.35,
        intraday_cost_pct=0.20,
        market_regime="Strong Bull",
        trade_eligible=True,
    )
    assert act2 == "NO TRADE"
    assert reason2 is not None and "agreement" in reason2.lower()

    # 3. Expected return smaller than Indian round-trip transaction costs -> NO TRADE
    act3, reason3 = classify_cost_aware_action(
        prob_up=0.52,
        exp_1d_pct=0.10,
        exp_5d_pct=0.22,
        downside_mae_pct=-1.1,
        upside_mfe_pct=1.1,
        agreement_score=0.75,
        confidence_score=60.0,
        delivery_cost_pct=0.36,
        intraday_cost_pct=0.20,
        market_regime="Weak Bull",
        trade_eligible=True,
    )
    assert act3 == "NO TRADE"
    assert reason3 is not None and "transaction costs" in reason3.lower()

    # 4. Bear / High Volatility regime blocks STRONG BUY / moderate BUY
    act4, reason4 = classify_cost_aware_action(
        prob_up=0.58,
        exp_1d_pct=0.45,
        exp_5d_pct=0.95,
        downside_mae_pct=-1.1,
        upside_mfe_pct=1.6,
        agreement_score=0.80,
        confidence_score=74.0,
        delivery_cost_pct=0.35,
        intraday_cost_pct=0.20,
        market_regime="Bear",
        trade_eligible=True,
    )
    assert act4 == "HOLD / WATCH"
    assert reason4 is not None and "regime" in reason4.lower()


def test_walk_forward_backtest_summary_and_duckdb_views() -> None:
    """Verify walk-forward backtest artifacts, model comparisons, calibration curve, and DuckDB views."""
    store = DuckDBStore()
    assert store.backtest_summary_json.exists(), "walk_forward_summary.json must exist in /data/backtests/"
    assert store.backtest_predictions_parquet.exists(), "walk_forward_predictions.parquet must exist"
    assert store.predictions_parquet.exists(), "prediction_log.parquet must exist in /data/predictions/"
    assert store.latest_predictions_json.exists(), "latest_predictions.json must exist in /data/predictions/"

    summary = json.loads(store.backtest_summary_json.read_text())
    assert summary["total_walk_forward_folds"] >= 5
    assert summary["purge_gap_days"] >= 1
    assert len(summary["symbols_evaluated"]) >= 15

    # All 4 required models must be evaluated
    models = summary["models"]
    for req_model in ["ml_ensemble", "linear_ridge_logistic", "technical_rule", "naive_momentum"]:
        assert req_model in models, f"Missing required benchmark model: {req_model}"
        m = models[req_model]
        assert m["total_predictions"] >= 3000
        assert 0.0 <= m["abstention_rate_pct"] <= 100.0
        assert 0.0 <= m["brier_score"] <= 0.50
        assert m["transaction_cost_drag_pct"] >= 0.0

    # Calibration curve must have bins
    assert len(summary["calibration_curve"]) >= 3
    assert len(summary["by_regime_performance"]) >= 2
    assert len(summary["by_symbol_performance"]) >= 15
    assert len(summary["top_global_features"]) >= 10

    # Query via DuckDB views `v_backtest_predictions` and `v_predictions`
    bt_q = store.query_df(
        "SELECT model_name, COUNT(*) AS cnt, AVG(prob_next_day_up) AS avg_prob "
        "FROM v_backtest_predictions GROUP BY model_name ORDER BY model_name"
    )
    assert len(bt_q) == 4

    pred_q = store.query_df(
        "SELECT symbol, prob_next_day_up, expected_next_day_return_pct, net_expected_1d_return_pct, "
        "recommended_action, top_bullish_drivers, top_bearish_drivers FROM v_predictions ORDER BY symbol"
    )
    assert len(pred_q) >= 15
    assert (pred_q["prob_next_day_up"] >= 0.15).all() and (pred_q["prob_next_day_up"] <= 0.85).all()
    # Net expected return must strictly account for transaction costs (strictly less than gross expected return)
    assert (pred_q["net_expected_1d_return_pct"] < pred_q["expected_next_day_return_pct"]).all()
