"""Phase 3 Prediction & Walk-Forward Backtesting Runner (Sections 8, 9 & 10).

Executes:
1. Purged expanding-window walk-forward validation across 15 NSE stocks comparing:
   - `ml_ensemble` (`BharatAlphaQuantEnsemble` — Calibrated LightGBM + XGBoost + Linear)
   - `linear_ridge_logistic` (`LinearRidgeLogisticBaseline`)
   - `technical_rule` (`TechnicalRuleBaseline`)
   - `naive_momentum` (`NaiveMomentumBaseline`)
   - `Nifty 50 Buy-and-Hold` & `Equal-Weight Universe Buy-and-Hold`
2. Realistic Indian equity transaction cost modeling (STT, Stamp Duty, NSE Exchange
   Charges, SEBI Fees, 18% GST, DP Charges, and dynamic ATR/Amihud slippage).
3. Latest next-trading-day predictions across all 15 stocks with exact TreeSHAP
   feature attributions (`top_bullish_drivers`, `top_bearish_drivers`).
4. Persists:
   - `/data/backtests/walk_forward_predictions.parquet`
   - `/data/backtests/walk_forward_summary.json`
   - `/data/predictions/prediction_log.parquet`
   - `/data/predictions/latest_predictions.json`
   - `docs/data/backtest_summary.json` & `docs/data/latest_predictions.json`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.backtesting.walk_forward import run_walk_forward_backtest
from backend.features.selector import get_candidate_feature_columns
from backend.prediction.cost_model import estimate_indian_roundtrip_cost
from backend.prediction.ensemble_model import BharatAlphaQuantEnsemble, build_latest_prediction_records
from backend.storage.duckdb_store import DuckDBStore

DOCS_DATA_DIR = ROOT_DIR / "docs" / "data"


def run_phase3_pipeline() -> dict[str, Any]:
    store = DuckDBStore()
    if not store.features_parquet.exists():
        raise FileNotFoundError(
            "Phase 2 feature files not found. Run `scripts/run_phase2_features.py` first."
        )

    feature_df = pd.read_parquet(store.features_parquet)
    candidate_features = get_candidate_feature_columns(feature_df)
    ohlc_df = pd.read_parquet(store.equities_parquet) if store.equities_parquet.exists() else None

    print("=" * 84)
    print("BHARATALPHA AI — PHASE 3 WALK-FORWARD BACKTESTING & PREDICTION ENGINE")
    print("=" * 84)

    # 1. Print Indian Equity Transaction Cost Baseline Breakdown
    deliv_cost_ex = estimate_indian_roundtrip_cost(mode="delivery", atr_14_pct=1.6, amihud_illiquidity=0.002)
    intra_cost_ex = estimate_indian_roundtrip_cost(mode="intraday", atr_14_pct=1.6, amihud_illiquidity=0.002)
    print(
        f"\n[1/4] Indian NSE Equity Round-Trip Cost Model (Large-Cap Reference):\n"
        f"  • CNC Delivery Round-Trip : {deliv_cost_ex.total_roundtrip_pct:.3f}% ({deliv_cost_ex.total_roundtrip_bps:.1f} bps) "
        f"[STT={deliv_cost_ex.stt_pct:.3f}%, Stamp={deliv_cost_ex.stamp_duty_pct:.3f}%, "
        f"DP={deliv_cost_ex.dp_charge_pct:.3f}%, Slippage={deliv_cost_ex.slippage_pct:.3f}%]\n"
        f"  • MIS Intraday Round-Trip : {intra_cost_ex.total_roundtrip_pct:.3f}% ({intra_cost_ex.total_roundtrip_bps:.1f} bps) "
        f"[Brokerage={intra_cost_ex.brokerage_pct:.3f}%, STT={intra_cost_ex.stt_pct:.3f}%, "
        f"Slippage={intra_cost_ex.slippage_pct:.3f}%]"
    )

    # 2. Run Purged & Embargoed Walk-Forward Time-Series Backtest (strictly in-fold feature selection)
    print(f"\n[2/4] Running Purged Walk-Forward Backtest ({len(candidate_features)} candidate features, in-fold selection)...")
    summary, all_oos_df, final_ensemble = run_walk_forward_backtest(
        feature_df=feature_df,
        selected_features=candidate_features,
        initial_train_dates=240,
        step_dates=42,
        purge_gap_days=5,
        embargo_sessions=1,
        ohlc_df=ohlc_df,
        top_k_features=40,
    )

    bt_rows = store.upsert_backtest_predictions(all_oos_df)
    summary_dict = summary.model_dump()
    store.backtest_summary_json.write_text(json.dumps(summary_dict, indent=2))
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / "backtest_summary.json").write_text(json.dumps(summary_dict, indent=2))

    print(
        f"  -> Completed {summary.total_walk_forward_folds} walk-forward folds "
        f"({summary.test_start_date} to {summary.test_end_date}, purge={summary.purge_gap_days}s, embargo={summary.embargo_sessions}s, mode={summary.execution_mode})\n"
        f"  -> Feature Stability Jaccard            : {summary.feature_stability_jaccard:.4f}\n"
        f"  -> Saved {bt_rows} out-of-sample rows to {store.backtest_predictions_parquet}\n"
        f"  -> Nifty 50 Buy & Hold OOS Return        : {summary.nifty_50_buy_hold_return_pct:+.2f}%\n"
        f"  -> Equal-Weight Universe OOS Return      : {summary.equal_weight_universe_return_pct:+.2f}%"
    )

    print("\n  Model Benchmark Comparison (Out-of-Sample Walk-Forward):")
    header = (
        f"  {'Model':<23} | {'Acc%':>5} | {'HC Hit%':>7} | {'AUC':>6} | {'Brier':>7} | "
        f"{'IC_1D':>7} | {'Gross%':>7} | {'Cost%':>6} | {'Net%':>7} | {'Sharpe(N)':>9} | {'MaxDD%':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m_name, m in summary.models.items():
        print(
            f"  {m_name:<23} | {m.directional_accuracy_all_pct:>5.1f} | "
            f"{m.high_conviction_hit_rate_pct:>7.1f} | {m.roc_auc:>6.3f} | {m.brier_score:>7.4f} | "
            f"{m.rank_ic_1d:>+7.4f} | {m.gross_cumulative_return_pct:>+7.2f} | "
            f"{m.transaction_cost_drag_pct:>6.2f} | {m.net_cumulative_return_pct:>+7.2f} | "
            f"{m.annualized_sharpe_net:>+9.2f} | {m.max_drawdown_net_pct:>+7.2f}"
        )

    # 3. Generate Latest Trading Day Predictions & Permanent Log
    print("\n[3/4] Generating Calibrated Next-Day Predictions & TreeSHAP Attributions...")
    latest_rows_per_sym = (
        feature_df.sort_values(["symbol", "observation_date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    # Read current market regime & quality gate eligibility if present
    market_regime_label = "Sideways"
    if store.regime_parquet.exists():
        reg_df = pd.read_parquet(store.regime_parquet).sort_values("observation_date")
        if not reg_df.empty:
            market_regime_label = str(reg_df["regime"].iloc[-1])

    quality_map: dict[str, bool] = {}
    if store.quality_parquet.exists():
        q_df = pd.read_parquet(store.quality_parquet)
        for _, q_row in q_df.iterrows():
            quality_map[str(q_row["symbol"]).upper()] = bool(q_row["trade_eligible"])

    latest_records = build_latest_prediction_records(
        ensemble=final_ensemble,
        latest_feature_df=latest_rows_per_sym,
        market_regime_label=market_regime_label,
        quality_map=quality_map,
    )

    # Train genuine 5D and 10D ensembles so live multi-horizon predictions never reuse 1D prob or scaled 5D return
    for horizon_tag, target_col, purge_n in (("5D", "target_5d_ret", 5), ("10D", "target_10d_ret", 10)):
        if target_col in feature_df.columns and int(feature_df[target_col].notna().sum()) >= 120:
            df_h = feature_df.copy()
            df_h["target_next_day_ret"] = pd.to_numeric(df_h[target_col], errors="coerce")
            df_h["target_next_day_up"] = (df_h["target_next_day_ret"] > 0).astype(float)
            ens_h = BharatAlphaQuantEnsemble(top_k_features=40, purge_sessions=purge_n, random_state=42)
            ens_h.fit(df_h, candidate_features)
            pred_h = ens_h.predict_df(latest_rows_per_sym, candidate_features)
            h_by_sym = {
                str(latest_rows_per_sym.iloc[i]["symbol"]).upper(): pred_h.iloc[i].to_dict()
                for i in range(len(latest_rows_per_sym))
            }
            for rec in latest_records:
                row_h = h_by_sym.get(rec.symbol)
                if row_h is not None:
                    p_h = round(float(row_h["prob_next_day_up"]), 4)
                    r_h = round(float(row_h["expected_next_day_return_pct"]), 4)
                    if horizon_tag == "5D":
                        rec.prob_5d_up = p_h
                        rec.expected_5d_return_pct = r_h
                        rec.net_expected_5d_return_pct = round(r_h - rec.roundtrip_delivery_cost_pct, 4)
                    else:
                        rec.prob_10d_up = p_h
                        rec.expected_10d_return_pct = r_h

    # Attach exact last_close, atr_14_pct, and NaiveMomentumBaseline 5D benchmark
    from backend.prediction.baselines import NaiveMomentumBaseline

    mom_model = NaiveMomentumBaseline()
    mom_model.fit(feature_df[feature_df["target_next_day_ret"].notna()], [])
    mom_preds = mom_model.predict_df(latest_rows_per_sym, [])
    mom_by_sym = {
        str(latest_rows_per_sym.iloc[i]["symbol"]).upper(): mom_preds.iloc[i].to_dict()
        for i in range(len(latest_rows_per_sym))
    }
    atr_by_sym = {
        str(latest_rows_per_sym.iloc[i]["symbol"]).upper(): float(latest_rows_per_sym.iloc[i]["pm_atr_14_pct"])
        for i in range(len(latest_rows_per_sym))
        if pd.notna(latest_rows_per_sym.iloc[i].get("pm_atr_14_pct"))
    }
    close_by_sym: dict[str, float] = {}
    if ohlc_df is not None and not ohlc_df.empty:
        latest_eq = ohlc_df.sort_values(["symbol", "observation_date"]).groupby("symbol", as_index=False).tail(1)
        for _, eq_r in latest_eq.iterrows():
            if pd.notna(eq_r.get("close")):
                close_by_sym[str(eq_r["symbol"]).upper()] = float(eq_r["close"])

    for rec in latest_records:
        rec.last_close = round(close_by_sym[rec.symbol], 2) if rec.symbol in close_by_sym else None
        rec.atr_14_pct = round(atr_by_sym[rec.symbol], 3) if rec.symbol in atr_by_sym else None
        if rec.symbol in mom_by_sym:
            rec.momentum_prob_up = round(float(mom_by_sym[rec.symbol]["prob_next_day_up"]), 4)
            rec.momentum_5d_return_pct = round(float(mom_by_sym[rec.symbol]["expected_5d_return_pct"]), 4)

    pred_log_cnt = store.upsert_predictions(latest_records)
    if ohlc_df is not None:
        store.evaluate_realized_outcomes(ohlc_df)
    latest_payload = {
        "generated_at": summary.generated_at,
        "market_regime": market_regime_label,
        "predictions": [r.model_dump() for r in latest_records],
    }
    store.latest_predictions_json.write_text(json.dumps(latest_payload, indent=2))
    (DOCS_DATA_DIR / "latest_predictions.json").write_text(json.dumps(latest_payload, indent=2))

    # Also embed `quantitative_prediction` into existing `docs/data/{SYMBOL}.json` files
    for rec in latest_records:
        sym_json = DOCS_DATA_DIR / f"{rec.symbol}.json"
        if sym_json.exists():
            try:
                bundle_data = json.loads(sym_json.read_text())
                if isinstance(bundle_data, dict) and "overview" in bundle_data:
                    bundle_data["quantitative_prediction"] = rec.model_dump()
                    sym_json.write_text(json.dumps(bundle_data, indent=2))
            except Exception:
                pass

    print(f"  -> Logged {pred_log_cnt} prediction records into {store.predictions_parquet}")
    print(f"\n[4/4] Latest Calibrated Predictions (Regime: {market_regime_label}):")
    for rec in latest_records:
        top_bull = rec.top_bullish_drivers[0].feature if rec.top_bullish_drivers else "none"
        top_bear = rec.top_bearish_drivers[0].feature if rec.top_bearish_drivers else "none"
        print(
            f"  • {rec.symbol:<11} ({rec.observation_date}) | P(Up)={rec.prob_next_day_up:.3f} | "
            f"E[1D]={rec.expected_next_day_return_pct:+.2f}% (Net {rec.net_expected_1d_return_pct:+.2f}%) | "
            f"E[5D]={rec.expected_5d_return_pct:+.2f}% | MAE={rec.downside_risk_mae_pct:.2f}% | "
            f"Conf={rec.confidence_score:>4.1f} | {rec.recommended_action:<16} | +{top_bull} / -{top_bear}"
        )

    return summary_dict


if __name__ == "__main__":
    run_phase3_pipeline()
