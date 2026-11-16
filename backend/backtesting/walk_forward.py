"""Walk-Forward Time-Series Backtesting Engine (Phase 3 — Section 8 & 10).

Executes strict chronological expanding-window walk-forward validation across
historical Indian equity data with zero random K-fold leakage:
- Purges `purge_gap_days` between `train_end_date` and `test_start_date`.
- Compares 4 models side-by-side on identical out-of-sample folds:
  1. `ml_ensemble` (`BharatAlphaQuantEnsemble` — Calibrated LightGBM + XGBoost + Linear)
  2. `linear_ridge_logistic` (`LinearRidgeLogisticBaseline`)
  3. `technical_rule` (`TechnicalRuleBaseline`)
  4. `naive_momentum` (`NaiveMomentumBaseline`)
- Benchmarks against Nifty 50 Buy-and-Hold and Equal-Weight Universe Buy-and-Hold.
- Applies realistic Indian NSE transaction costs (STT, Stamp Duty, Exchange Fees,
  SEBI Fees, 18% GST, DP charges, and dynamic ATR/Amihud slippage) with turnover-aware
  position persistence (charging round-trip costs only when positions are opened/closed).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..models import BacktestComparisonSummary, ModelEvaluationMetrics
from ..prediction.baselines import (
    LinearRidgeLogisticBaseline,
    NaiveMomentumBaseline,
    TechnicalRuleBaseline,
)
from ..prediction.ensemble_model import BharatAlphaQuantEnsemble
from .execution_simulator import ExecutionConfig, simulate_oos_execution


def _infer_row_regime_label(df: pd.DataFrame) -> pd.Series:
    """Derive point-in-time market regime label for each historical bar from `ms_` features."""
    vix = pd.to_numeric(df.get("ms_india_vix_level", 15.0), errors="coerce").fillna(15.0)
    dist50 = pd.to_numeric(df.get("ms_nifty_dist_sma50_pct", 0.0), errors="coerce").fillna(0.0)
    dist200 = pd.to_numeric(df.get("ms_nifty_dist_sma200_pct", 0.0), errors="coerce").fillna(0.0)
    n_ret20 = pd.to_numeric(df.get("ms_nifty_ret_20d_pct", 0.0), errors="coerce").fillna(0.0)

    conds = [
        vix >= 21.0,
        (dist200 < -1.0) & (dist50 < 0.0),
        (dist200 > 0.0) & (dist50 > 0.5) & (n_ret20 > 1.5),
        (dist200 > 0.0) & (dist50 >= -0.5),
    ]
    choices = ["High Volatility", "Bear", "Strong Bull", "Weak Bull"]
    return pd.Series(np.select(conds, choices, default="Sideways"), index=df.index)


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    """Compute Expected Calibration Error (ECE) across quantile probability bins."""
    if len(y_true) == 0:
        return 0.0
    s_prob = pd.Series(y_prob)
    try:
        bins = pd.qcut(s_prob, q=n_bins, duplicates="drop")
    except Exception:
        return 0.0
    ece = 0.0
    n = len(y_true)
    for b_idx in bins.cat.categories:
        mask = (bins == b_idx).values
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return round(float(ece), 5)


def _evaluate_model_oos_predictions(
    model_name: str,
    oos_df: pd.DataFrame,
    ohlc_df: pd.DataFrame | None = None,
    execution_config: ExecutionConfig | None = None,
) -> tuple[ModelEvaluationMetrics, pd.DataFrame, dict[str, Any]]:
    """Evaluate out-of-sample predictions with realistic next-day execution & Indian costs."""
    df = oos_df.sort_values(["symbol", "observation_date"]).copy()
    valid_mask = df["target_next_day_ret"].notna() & df["target_next_day_up"].notna()
    df = df.loc[valid_mask].reset_index(drop=True)

    y_true_up = df["target_next_day_up"].astype(int).values
    y_true_r1 = pd.to_numeric(df["target_next_day_ret"], errors="coerce").values
    y_true_r5 = pd.to_numeric(df.get("target_5d_ret"), errors="coerce")

    p_up = pd.to_numeric(df["prob_next_day_up"], errors="coerce").fillna(0.50).clip(0.01, 0.99).values
    exp_r5 = pd.to_numeric(df["expected_5d_return_pct"], errors="coerce").fillna(0.0)

    # Classification & Calibration metrics across all OOS rows
    pred_dir = (p_up >= 0.50).astype(int)
    dir_acc_all = float((pred_dir == y_true_up).mean() * 100.0) if len(y_true_up) > 0 else 50.0

    try:
        auc_val = float(roc_auc_score(y_true_up, p_up))
    except Exception:
        auc_val = 0.50

    brier_val = float(brier_score_loss(y_true_up, p_up))
    logloss_val = float(log_loss(y_true_up, p_up, labels=[0, 1]))
    ece_val = _compute_ece(y_true_up, p_up, n_bins=5)

    # Spearman Rank IC (1D and 5D)
    s_p = pd.Series(p_up)
    s_r1 = pd.Series(y_true_r1)
    ic_1d_raw = s_p.rank().corr(s_r1.rank())
    ic_1d = float(ic_1d_raw) if pd.notna(ic_1d_raw) else 0.0

    ic_5d = 0.0
    if y_true_r5 is not None:
        m5 = y_true_r5.notna()
        if int(m5.sum()) >= 20:
            ic_5d_raw = exp_r5[m5].rank().corr(y_true_r5[m5].rank())
            if pd.notna(ic_5d_raw):
                ic_5d = float(ic_5d_raw)

    # High-conviction subset (`|p_up - 0.50| >= 0.045`)
    hc_mask = np.abs(p_up - 0.50) >= 0.045
    if int(hc_mask.sum()) >= 10:
        hc_hit_rate = float((pred_dir[hc_mask] == y_true_up[hc_mask]).mean() * 100.0)
    else:
        top_q = np.argsort(np.abs(p_up - 0.50))[-max(10, len(p_up) // 5):]
        hc_hit_rate = float((pred_dir[top_q] == y_true_up[top_q]).mean() * 100.0)

    df["regime_label"] = _infer_row_regime_label(df)
    cfg = execution_config or ExecutionConfig()
    df, daily_port, exec_stats = simulate_oos_execution(df, ohlc_df=ohlc_df, config=cfg)

    active_days_count = int((df["position"] > 0).sum())
    abstention_rate = round((1.0 - active_days_count / max(1, len(df))) * 100.0, 2)

    gross_curve = (1.0 + daily_port["gross_ret_pct"] / 100.0).cumprod()
    net_curve = (1.0 + daily_port["net_ret_pct"] / 100.0).cumprod()

    gross_cum_pct = float((gross_curve.iloc[-1] - 1.0) * 100.0) if len(gross_curve) > 0 else 0.0
    net_cum_pct = float((net_curve.iloc[-1] - 1.0) * 100.0) if len(net_curve) > 0 else 0.0
    cost_drag_total_pct = max(0.0, gross_cum_pct - net_cum_pct)

    # Sharpe, Sortino, and Max Drawdown on daily portfolio returns
    r_gross = daily_port["gross_ret_pct"].values
    r_net = daily_port["net_ret_pct"].values

    std_gross = float(np.std(r_gross, ddof=1)) if len(r_gross) > 1 else 1.0
    std_net = float(np.std(r_net, ddof=1)) if len(r_net) > 1 else 1.0
    sharpe_gross = (float(np.mean(r_gross)) / std_gross * math.sqrt(252)) if std_gross > 1e-9 else 0.0
    sharpe_net = (float(np.mean(r_net)) / std_net * math.sqrt(252)) if std_net > 1e-9 else 0.0

    downside_r = r_net[r_net < 0.0]
    down_std = float(np.sqrt(np.mean(downside_r ** 2))) if len(downside_r) > 0 else 1.0
    sortino_net = (float(np.mean(r_net)) / down_std * math.sqrt(252)) if down_std > 1e-9 else 0.0

    running_peak = net_curve.cummax()
    dd_series = (net_curve / running_peak - 1.0) * 100.0
    max_dd_net_pct = float(dd_series.min()) if len(dd_series) > 0 else 0.0

    # Trade-level win/loss & profit factor on active stock-day positions
    active_rows = df[df["position"] > 0]
    if len(active_rows) > 0:
        wins = active_rows.loc[active_rows["net_trade_ret_pct"] > 0, "net_trade_ret_pct"]
        losses = active_rows.loc[active_rows["net_trade_ret_pct"] < 0, "net_trade_ret_pct"]
        gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_losses = float(losses.abs().sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_wins / gross_losses) if gross_losses > 1e-9 else (2.5 if gross_wins > 0 else 0.0)
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.abs().mean()) if len(losses) > 0 else 1.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 1e-9 else 1.0
        expectancy_bps = float(active_rows["net_trade_ret_pct"].mean() * 100.0)
    else:
        profit_factor = 0.0
        win_loss_ratio = 0.0
        expectancy_bps = 0.0

    metrics = ModelEvaluationMetrics(
        model_name=model_name,
        total_predictions=len(df),
        active_trades_count=active_days_count,
        abstention_rate_pct=abstention_rate,
        directional_accuracy_all_pct=round(dir_acc_all, 2),
        high_conviction_hit_rate_pct=round(hc_hit_rate, 2),
        roc_auc=round(auc_val, 4),
        brier_score=round(brier_val, 5),
        log_loss_val=round(logloss_val, 5),
        calibration_ece=ece_val,
        rank_ic_1d=round(ic_1d, 5),
        rank_ic_5d=round(ic_5d, 5),
        gross_cumulative_return_pct=round(gross_cum_pct, 2),
        transaction_cost_drag_pct=round(cost_drag_total_pct, 2),
        net_cumulative_return_pct=round(net_cum_pct, 2),
        annualized_sharpe_gross=round(sharpe_gross, 3),
        annualized_sharpe_net=round(sharpe_net, 3),
        sortino_ratio_net=round(sortino_net, 3),
        max_drawdown_net_pct=round(max_dd_net_pct, 2),
        profit_factor_net=round(profit_factor, 3),
        avg_win_loss_ratio=round(win_loss_ratio, 3),
        expectancy_per_trade_bps=round(expectancy_bps, 2),
    )
    return metrics, df, exec_stats


def _compute_jaccard_stability(feature_sets: list[list[str]]) -> float:
    """Compute average pairwise Jaccard similarity of selected features across consecutive folds."""
    if len(feature_sets) < 2:
        return 1.0
    scores: list[float] = []
    for i in range(len(feature_sets) - 1):
        a = set(feature_sets[i])
        b = set(feature_sets[i + 1])
        union = len(a | b)
        scores.append(len(a & b) / union if union > 0 else 0.0)
    return round(float(np.mean(scores)), 4)


def run_walk_forward_backtest(
    feature_df: pd.DataFrame,
    selected_features: list[str],
    initial_train_dates: int = 240,
    step_dates: int = 42,
    purge_gap_days: int = 5,
    embargo_sessions: int = 1,
    ohlc_df: pd.DataFrame | None = None,
    execution_config: ExecutionConfig | None = None,
    top_k_features: int = 40,
) -> tuple[BacktestComparisonSummary, pd.DataFrame, BharatAlphaQuantEnsemble]:
    """Run purged & embargoed expanding-window walk-forward validation across all models & benchmarks."""
    df_all = feature_df.sort_values(["observation_date", "symbol"]).reset_index(drop=True)
    all_dates = sorted(df_all["observation_date"].unique().tolist())
    symbols = sorted(df_all["symbol"].unique().tolist())

    total_gap = purge_gap_days + embargo_sessions
    if len(all_dates) <= initial_train_dates + total_gap + 10:
        raise ValueError(
            f"Insufficient historical trading dates ({len(all_dates)}) for initial_train_dates={initial_train_dates}."
        )

    model_builders = {
        "ml_ensemble": lambda: BharatAlphaQuantEnsemble(
            top_k_features=top_k_features,
            purge_sessions=purge_gap_days,
            random_state=42,
        ),
        "linear_ridge_logistic": lambda: LinearRidgeLogisticBaseline(c_logreg=0.05, ridge_alpha=120.0),
        "technical_rule": lambda: TechnicalRuleBaseline(),
        "naive_momentum": lambda: NaiveMomentumBaseline(),
    }

    oos_collector: dict[str, list[pd.DataFrame]] = {m: [] for m in model_builders}
    fold_feature_sets: list[list[str]] = []
    fold_count = 0
    cursor = initial_train_dates

    while cursor + total_gap < len(all_dates):
        train_end_date = all_dates[cursor - 1]
        test_start_idx = cursor + total_gap
        test_end_idx = min(len(all_dates) - 1, test_start_idx + step_dates - 1)
        test_start_date = all_dates[test_start_idx]
        test_end_date = all_dates[test_end_idx]

        train_slice = df_all[df_all["observation_date"] <= train_end_date].copy()
        test_slice = df_all[
            (df_all["observation_date"] >= test_start_date)
            & (df_all["observation_date"] <= test_end_date)
        ].copy()

        if len(test_slice) == 0:
            break

        fold_count += 1
        for m_name, builder in model_builders.items():
            model = builder()
            model.fit(train_slice, selected_features)
            if m_name == "ml_ensemble":
                fold_feature_sets.append(list(model.active_features_))
            pred_out = model.predict_df(test_slice, selected_features)

            merged_fold = test_slice[
                [
                    "symbol",
                    "observation_date",
                    "sector",
                    "pm_atr_14_pct",
                    "vd_amihud_illiquidity_20d",
                    "ms_india_vix_level",
                    "ms_nifty_dist_sma50_pct",
                    "ms_nifty_dist_sma200_pct",
                    "ms_nifty_ret_1d_pct",
                    "ms_nifty_ret_20d_pct",
                    "target_next_day_ret",
                    "target_next_day_up",
                    "target_5d_ret",
                    "target_next_day_mae",
                    "target_next_day_mfe",
                ]
            ].copy()
            merged_fold["fold_id"] = fold_count
            merged_fold["model_name"] = m_name
            for p_col in pred_out.columns:
                merged_fold[p_col] = pred_out[p_col].values
            oos_collector[m_name].append(merged_fold)

        cursor += step_dates
        if test_end_idx >= len(all_dates) - 1:
            break

    cfg = execution_config or ExecutionConfig()
    metrics_by_model: dict[str, ModelEvaluationMetrics] = {}
    evaluated_frames: dict[str, pd.DataFrame] = {}
    exec_diagnostics: dict[str, dict[str, Any]] = {}

    for m_name, fold_list in oos_collector.items():
        oos_full = pd.concat(fold_list, ignore_index=True)
        m_metrics, eval_df, m_exec_stats = _evaluate_model_oos_predictions(
            m_name,
            oos_full,
            ohlc_df=ohlc_df,
            execution_config=cfg,
        )
        metrics_by_model[m_name] = m_metrics
        evaluated_frames[m_name] = eval_df
        exec_diagnostics[m_name] = m_exec_stats

    ml_oos_df = evaluated_frames["ml_ensemble"]
    test_start_dt = str(ml_oos_df["observation_date"].min())
    test_end_dt = str(ml_oos_df["observation_date"].max())

    # Compute Nifty 50 Buy-and-Hold and Equal-Weight Universe Buy-and-Hold over the exact OOS window
    nifty_daily = (
        ml_oos_df.groupby("observation_date", as_index=False)["ms_nifty_ret_1d_pct"]
        .first()
        .sort_values("observation_date")
    )
    nifty_cum_ret = float(((1.0 + nifty_daily["ms_nifty_ret_1d_pct"].fillna(0.0) / 100.0).prod() - 1.0) * 100.0)

    eq_univ_daily = (
        ml_oos_df.groupby("observation_date", as_index=False)["target_next_day_ret"]
        .mean()
        .sort_values("observation_date")
    )
    eq_univ_cum_ret = float(((1.0 + eq_univ_daily["target_next_day_ret"].fillna(0.0) / 100.0).prod() - 1.0) * 100.0)

    # Build Calibration Reliability Curve (5 quantile bins on `ml_ensemble`)
    ml_oos_df["prob_bin"] = pd.qcut(ml_oos_df["prob_next_day_up"], q=5, duplicates="drop")
    calibration_curve: list[dict[str, float]] = []
    for _, b_grp in ml_oos_df.groupby("prob_bin", observed=False):
        if len(b_grp) > 0:
            calibration_curve.append(
                {
                    "bin_min_prob": round(float(b_grp["prob_next_day_up"].min()), 4),
                    "bin_max_prob": round(float(b_grp["prob_next_day_up"].max()), 4),
                    "mean_predicted_prob": round(float(b_grp["prob_next_day_up"].mean()), 4),
                    "empirical_win_rate": round(float(b_grp["target_next_day_up"].mean()), 4),
                    "mean_next_day_ret_pct": round(float(b_grp["target_next_day_ret"].mean()), 4),
                    "count": float(len(b_grp)),
                }
            )

    # Breakdown by Market Regime, Sector, and Symbol for `ml_ensemble`
    by_regime: dict[str, dict[str, float]] = {}
    for reg_name, r_grp in ml_oos_df.groupby("regime_label"):
        act = r_grp[r_grp["position"] > 0]
        by_regime[str(reg_name)] = {
            "observations": float(len(r_grp)),
            "active_trades": float(len(act)),
            "directional_accuracy_pct": round(
                float(((r_grp["prob_next_day_up"] >= 0.5) == r_grp["target_next_day_up"]).mean() * 100.0), 2
            ),
            "avg_net_trade_ret_bps": round(float(act["net_trade_ret_pct"].mean() * 100.0), 2) if len(act) > 0 else 0.0,
        }

    by_sector: dict[str, dict[str, float]] = {}
    for sec_name, s_grp in ml_oos_df.groupby("sector"):
        act = s_grp[s_grp["position"] > 0]
        by_sector[str(sec_name)] = {
            "observations": float(len(s_grp)),
            "active_trades": float(len(act)),
            "directional_accuracy_pct": round(
                float(((s_grp["prob_next_day_up"] >= 0.5) == s_grp["target_next_day_up"]).mean() * 100.0), 2
            ),
            "avg_net_trade_ret_bps": round(float(act["net_trade_ret_pct"].mean() * 100.0), 2) if len(act) > 0 else 0.0,
        }

    by_symbol: dict[str, dict[str, float]] = {}
    for sym_name, sym_grp in ml_oos_df.groupby("symbol"):
        act = sym_grp[sym_grp["position"] > 0]
        sym_net_cum = float(((1.0 + sym_grp["net_trade_ret_pct"] / 100.0).prod() - 1.0) * 100.0)
        by_symbol[str(sym_name)] = {
            "observations": float(len(sym_grp)),
            "active_trades": float(len(act)),
            "directional_accuracy_pct": round(
                float(((sym_grp["prob_next_day_up"] >= 0.5) == sym_grp["target_next_day_up"]).mean() * 100.0), 2
            ),
            "net_cumulative_return_pct": round(sym_net_cum, 2),
        }

    # Fit final production `BharatAlphaQuantEnsemble` on all historical rows where `target_next_day_ret` is known
    final_ensemble = BharatAlphaQuantEnsemble(
        top_k_features=top_k_features,
        purge_sessions=purge_gap_days,
        random_state=42,
    )
    final_ensemble.fit(df_all[df_all["target_next_day_ret"].notna()], selected_features)

    top_global_features = [
        {"feature": f_name, "importance": imp}
        for f_name, imp in list(final_ensemble.feature_importances_.items())[:20]
    ]

    summary = BacktestComparisonSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        train_start_date=str(all_dates[0]),
        test_start_date=test_start_dt,
        test_end_date=test_end_dt,
        total_walk_forward_folds=fold_count,
        purge_gap_days=purge_gap_days,
        embargo_sessions=embargo_sessions,
        execution_mode=cfg.name,
        execution_diagnostics=exec_diagnostics,
        feature_stability_jaccard=_compute_jaccard_stability(fold_feature_sets),
        symbols_evaluated=symbols,
        nifty_50_buy_hold_return_pct=round(nifty_cum_ret, 2),
        equal_weight_universe_return_pct=round(eq_univ_cum_ret, 2),
        models=metrics_by_model,
        calibration_curve=calibration_curve,
        by_regime_performance=by_regime,
        by_sector_performance=by_sector,
        by_symbol_performance=by_symbol,
        top_global_features=top_global_features,
    )

    all_models_oos_df = pd.concat(list(evaluated_frames.values()), ignore_index=True)
    return summary, all_models_oos_df, final_ensemble


def compare_execution_assumptions(
    oos_df: pd.DataFrame,
    ohlc_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare identical OOS predictions under alternative entry, cost, and sizing assumptions."""
    configs = [
        ExecutionConfig(
            name="same_close_legacy",
            entry_timing="same_close_legacy",
            cost_mode="delivery",
            use_sl_tp=False,
            max_position_weight=1.0,
            max_portfolio_exposure=1.0,
        ),
        ExecutionConfig(
            name="next_open_delivery_capped",
            entry_timing="next_open",
            cost_mode="delivery",
            use_sl_tp=True,
            max_position_weight=0.20,
            max_portfolio_exposure=1.0,
        ),
        ExecutionConfig(
            name="next_open_intraday_mis",
            entry_timing="next_open",
            cost_mode="intraday",
            max_hold_sessions=1,
            use_sl_tp=True,
            max_position_weight=0.20,
            max_portfolio_exposure=1.0,
        ),
    ]
    results: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        m, _, stats = _evaluate_model_oos_predictions(
            "ml_ensemble",
            oos_df,
            ohlc_df=ohlc_df,
            execution_config=cfg,
        )
        results[cfg.name] = {
            "net_return_pct": m.net_cumulative_return_pct,
            "net_sharpe": m.annualized_sharpe_net,
            "max_drawdown_pct": m.max_drawdown_net_pct,
            "profit_factor": m.profit_factor_net,
            "expectancy_bps": m.expectancy_per_trade_bps,
            "active_trades": m.active_trades_count,
            "diagnostics": stats,
        }
    return results


def run_nested_feature_count_ablation(
    feature_df: pd.DataFrame,
    candidate_features: list[str],
    ohlc_df: pd.DataFrame | None = None,
    k_values: tuple[int, ...] = (20, 40, 80),
    initial_train_dates: int = 240,
    step_dates: int = 42,
    purge_gap_days: int = 5,
    embargo_sessions: int = 1,
) -> dict[str, Any]:
    """Run nested walk-forward validation across `top_k_features` in {20, 40, 80}."""
    df_all = feature_df.sort_values(["observation_date", "symbol"]).reset_index(drop=True)
    all_dates = sorted(df_all["observation_date"].unique().tolist())
    total_gap = purge_gap_days + embargo_sessions

    fixed_k_results: dict[int, dict[str, Any]] = {}
    for k in k_values:
        summary, _, _ = run_walk_forward_backtest(
            df_all,
            candidate_features,
            initial_train_dates=initial_train_dates,
            step_dates=step_dates,
            purge_gap_days=purge_gap_days,
            embargo_sessions=embargo_sessions,
            ohlc_df=ohlc_df,
            top_k_features=k,
        )
        ml = summary.models["ml_ensemble"]
        fixed_k_results[k] = {
            "roc_auc": ml.roc_auc,
            "brier_score": ml.brier_score,
            "log_loss": ml.log_loss_val,
            "calibration_ece": ml.calibration_ece,
            "rank_ic_1d": ml.rank_ic_1d,
            "net_return_pct": ml.net_cumulative_return_pct,
            "net_sharpe": ml.annualized_sharpe_net,
            "expectancy_bps": ml.expectancy_per_trade_bps,
            "profit_factor": ml.profit_factor_net,
            "feature_stability_jaccard": summary.feature_stability_jaccard,
        }

    # Nested outer loop: pick best k on chronological inner validation inside each outer train fold
    nested_oos: list[pd.DataFrame] = []
    chosen_ks: list[int] = []
    cursor = initial_train_dates
    while cursor + total_gap < len(all_dates):
        train_dates = all_dates[:cursor]
        test_start_idx = cursor + total_gap
        test_end_idx = min(len(all_dates) - 1, test_start_idx + step_dates - 1)
        train_slice = df_all[df_all["observation_date"] <= train_dates[-1]].copy()
        test_slice = df_all[
            (df_all["observation_date"] >= all_dates[test_start_idx])
            & (df_all["observation_date"] <= all_dates[test_end_idx])
        ].copy()
        if len(test_slice) == 0:
            break

        inner_split = int(len(train_dates) * 0.80)
        inner_train_end = train_dates[max(10, inner_split - purge_gap_days - 1)]
        inner_val_start = train_dates[min(len(train_dates) - 1, inner_split)]
        in_tr = train_slice[train_slice["observation_date"] <= inner_train_end]
        in_va = train_slice[train_slice["observation_date"] >= inner_val_start]

        best_k = k_values[0]
        best_ll = 999.0
        if len(in_tr) >= 120 and len(in_va) >= 30:
            y_va = in_va["target_next_day_up"].astype(int).values
            for k in k_values:
                m = BharatAlphaQuantEnsemble(top_k_features=k, purge_sessions=purge_gap_days, random_state=42)
                m.fit(in_tr, candidate_features)
                p_va = np.clip(m.predict_df(in_va, candidate_features)["prob_next_day_up"].values, 0.01, 0.99)
                ll = float(log_loss(y_va, p_va, labels=[0, 1]))
                if ll < best_ll:
                    best_ll = ll
                    best_k = k
        chosen_ks.append(best_k)

        outer_m = BharatAlphaQuantEnsemble(top_k_features=best_k, purge_sessions=purge_gap_days, random_state=42)
        outer_m.fit(train_slice, candidate_features)
        pred_out = outer_m.predict_df(test_slice, candidate_features)
        merged = test_slice[
            [
                "symbol", "observation_date", "sector", "pm_atr_14_pct",
                "vd_amihud_illiquidity_20d", "ms_india_vix_level",
                "ms_nifty_dist_sma50_pct", "ms_nifty_dist_sma200_pct",
                "ms_nifty_ret_1d_pct", "ms_nifty_ret_20d_pct",
                "target_next_day_ret", "target_next_day_up", "target_5d_ret",
            ]
        ].copy()
        for c in pred_out.columns:
            merged[c] = pred_out[c].values
        nested_oos.append(merged)
        cursor += step_dates
        if test_end_idx >= len(all_dates) - 1:
            break

    nested_summary: dict[str, Any] = {"chosen_k_per_fold": chosen_ks}
    if nested_oos:
        nm, _, _ = _evaluate_model_oos_predictions(
            "nested_ml_ensemble",
            pd.concat(nested_oos, ignore_index=True),
            ohlc_df=ohlc_df,
        )
        nested_summary.update(
            {
                "roc_auc": nm.roc_auc,
                "brier_score": nm.brier_score,
                "log_loss": nm.log_loss_val,
                "calibration_ece": nm.calibration_ece,
                "net_return_pct": nm.net_cumulative_return_pct,
                "net_sharpe": nm.annualized_sharpe_net,
                "expectancy_bps": nm.expectancy_per_trade_bps,
                "profit_factor": nm.profit_factor_net,
            }
        )

    return {"fixed_k": fixed_k_results, "nested_cv": nested_summary}


def evaluate_regime_filtered_variants(
    oos_df: pd.DataFrame,
    ohlc_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate out-of-sample regime/volatility filter variants without cherry-picking."""
    df = oos_df.copy()
    df["regime_label"] = _infer_row_regime_label(df)

    variants: dict[str, pd.Series] = {
        "baseline_block_high_vol": df["regime_label"] != "High Volatility",
        "block_bear_and_high_vol": ~df["regime_label"].isin(["High Volatility", "Bear"]),
        "bull_and_sideways_only": df["regime_label"].isin(["Strong Bull", "Sideways"]),
        "nifty_above_sma50_only": pd.to_numeric(df["ms_nifty_dist_sma50_pct"], errors="coerce").fillna(0.0) > 0.0,
    }
    out: dict[str, dict[str, Any]] = {}
    for name, allow_mask in variants.items():
        mod = df.copy()
        # Gate entry by setting prob below entry threshold when regime filter blocks
        mod.loc[~allow_mask, "prob_next_day_up"] = np.minimum(mod.loc[~allow_mask, "prob_next_day_up"], 0.49)
        m, _, _ = _evaluate_model_oos_predictions(name, mod, ohlc_df=ohlc_df)
        out[name] = {
            "active_trades": m.active_trades_count,
            "net_return_pct": m.net_cumulative_return_pct,
            "net_sharpe": m.annualized_sharpe_net,
            "max_drawdown_pct": m.max_drawdown_net_pct,
            "profit_factor": m.profit_factor_net,
            "expectancy_bps": m.expectancy_per_trade_bps,
        }
    return out
