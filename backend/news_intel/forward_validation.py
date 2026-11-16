"""Phase 5.3 Forward Validation, Immutable Snapshots, Matured Outcomes & Paper Research Portfolio."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import IST_TZ
from ..models import (
    ForwardHorizonEvaluation,
    ForwardOutcomeRecord,
    ForwardPredictionSnapshot,
    ForwardValidationPayload,
    PaperPortfolioReport,
    PaperStrategySummary,
    PaperTradeRecord,
    Phase5IntelligencePayload,
)
from ..storage.duckdb_store import DuckDBStore
from .relationships import PILOT_UNIVERSE_5

MIN_FORWARD_SAMPLES = 10
DEFAULT_CNC_ROUNDTRIP_COST_PCT = 0.347
STARTING_CAPITAL_INR = 1_000_000.0
MAX_POSITION_PCT = 0.10
MAX_EXPOSURE_PCT = 0.80


def record_forward_prediction_snapshots(
    store: DuckDBStore,
    payload: Phase5IntelligencePayload,
    quant_by_sym: dict[str, dict[str, Any]],
    model_version: str = "5.3.0-pilot",
    feature_set_version: str = "237+17",
) -> list[ForwardPredictionSnapshot]:
    """Persist immutable daily prediction snapshots before outcomes are known."""
    snapshots: list[ForwardPredictionSnapshot] = []
    for sym, s_item in payload.stocks.items():
        q_pred = quant_by_sym.get(sym.upper()) or {}
        obs_date = str(q_pred.get("observation_date") or s_item.observation_date)
        pred_id = f"fwd_{obs_date}_{sym.upper()}"
        h1 = s_item.horizon_outlooks.get("1D")
        h5 = s_item.horizon_outlooks.get("5D")
        h10 = s_item.horizon_outlooks.get("10D")

        snap = ForwardPredictionSnapshot(
            prediction_id=pred_id,
            symbol=sym.upper(),
            prediction_timestamp=payload.latest_prediction_at,
            observation_date=obs_date,
            model_version=model_version,
            feature_set_version=feature_set_version,
            prob_1d_up=h1.prob_positive if h1 and h1.is_model_generated else None,
            expected_1d_return_pct=h1.expected_net_return_pct if h1 and h1.is_model_generated else None,
            prob_5d_up=h5.prob_positive if h5 and h5.is_model_generated else None,
            expected_5d_return_pct=h5.expected_net_return_pct if h5 and h5.is_model_generated else None,
            prob_10d_up=h10.prob_positive if h10 and h10.is_model_generated else None,
            expected_10d_return_pct=h10.expected_net_return_pct if h10 and h10.is_model_generated else None,
            news_score=round(float(s_item.news_sentiment_score), 4),
            cross_stock_impact=round(float(s_item.news_features.get("news_cross_stock_net_impact", 0.0)), 4),
            combined_signal=h1.combined_signal if h1 else "UNAVAILABLE",
            recommendation_1d=h1.decision if h1 else "UNAVAILABLE",
            recommendation_5d=h5.decision if h5 else "UNAVAILABLE",
            recommendation_10d=h10.decision if h10 else "UNAVAILABLE",
            data_freshness=h1.data_freshness if h1 else "Unknown",
            model_availability={
                "1D": bool(h1 and h1.is_model_generated),
                "5D": bool(h5 and h5.is_model_generated),
                "10D": bool(h10 and h10.is_model_generated),
            },
            benchmark_prob_up=q_pred.get("momentum_prob_up"),
            benchmark_5d_return_pct=q_pred.get("momentum_5d_return_pct"),
            metadata={
                "market_regime": q_pred.get("market_regime", "Sideways"),
                "quant_prob_1d": h1.quant_prob_positive if h1 else None,
                "quant_prob_5d": h5.quant_prob_positive if h5 else None,
                "quant_prob_10d": h10.quant_prob_positive if h10 else None,
                "roundtrip_delivery_cost_pct": q_pred.get("roundtrip_delivery_cost_pct", DEFAULT_CNC_ROUNDTRIP_COST_PCT),
            },
        )
        snapshots.append(snap)

    if snapshots:
        store.upsert_forward_predictions(snapshots)
    return snapshots


def evaluate_forward_outcomes(
    store: DuckDBStore,
    equities_df: pd.DataFrame | None = None,
) -> list[ForwardOutcomeRecord]:
    """Attach realized 1D, 5D, and 10D outcomes strictly after each horizon matures."""
    if not store.forward_predictions_parquet.exists():
        return []
    snap_df = pd.read_parquet(store.forward_predictions_parquet)
    if snap_df.empty:
        return []

    if equities_df is None:
        if not store.equities_parquet.exists():
            return []
        equities_df = pd.read_parquet(store.equities_parquet)
    if equities_df.empty:
        return []

    cols = [c for c in ("symbol", "observation_date", "open", "close", "adj_close") if c in equities_df.columns]
    eq = equities_df[cols].dropna(subset=["close"]).copy()
    eq["symbol"] = eq["symbol"].astype(str).str.upper()
    eq["observation_date"] = eq["observation_date"].astype(str)
    eq = eq.sort_values(["symbol", "observation_date"]).reset_index(drop=True)

    sym_map: dict[str, pd.DataFrame] = {
        str(sym): grp.reset_index(drop=True) for sym, grp in eq.groupby("symbol", sort=False)
    }

    now_iso = datetime.now(IST_TZ).isoformat()
    outcomes: list[ForwardOutcomeRecord] = []

    for _, row in snap_df.iterrows():
        pred_id = str(row["prediction_id"])
        sym = str(row["symbol"]).upper()
        obs_dt = str(row["observation_date"])

        meta_raw = row.get("metadata")
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw.startswith("{") else (meta_raw or {})
        cost_pct = float(meta.get("roundtrip_delivery_cost_pct") or DEFAULT_CNC_ROUNDTRIP_COST_PCT)

        grp = sym_map.get(sym)
        base_idx = None
        if grp is not None:
            matches = grp.index[grp["observation_date"] == obs_dt].tolist()
            if matches:
                base_idx = int(matches[0])

        for horizon, hold_n in (("1D", 1), ("5D", 5), ("10D", 10)):
            if base_idx is None or grp is None or base_idx + hold_n >= len(grp):
                outcomes.append(
                    ForwardOutcomeRecord(
                        prediction_id=pred_id,
                        symbol=sym,
                        observation_date=obs_dt,
                        horizon=horizon,  # type: ignore[arg-type]
                        status="PENDING",
                    )
                )
                continue

            entry_idx = base_idx + 1
            exit_idx = base_idx + hold_n
            entry_dt = str(grp.loc[entry_idx, "observation_date"])
            exit_dt = str(grp.loc[exit_idx, "observation_date"])

            entry_open = grp.loc[entry_idx, "open"] if "open" in grp.columns else np.nan
            entry_close = float(grp.loc[entry_idx, "close"])
            entry_px = float(entry_open) if pd.notna(entry_open) and float(entry_open) > 0 else entry_close
            exit_px = float(grp.loc[exit_idx, "close"])

            if entry_px <= 0 or exit_px <= 0:
                outcomes.append(
                    ForwardOutcomeRecord(
                        prediction_id=pred_id,
                        symbol=sym,
                        observation_date=obs_dt,
                        horizon=horizon,  # type: ignore[arg-type]
                        status="PENDING",
                    )
                )
                continue

            # Corporate-action (split/bonus/dividend) adjustment via adj_close ratio
            ca_adjusted = False
            adj_factor = 1.0
            if "adj_close" in grp.columns:
                ac_in = grp.loc[entry_idx, "adj_close"]
                ac_out = grp.loc[exit_idx, "adj_close"]
                if pd.notna(ac_in) and pd.notna(ac_out) and float(ac_in) > 0 and entry_close > 0:
                    ratio_in = float(ac_in) / entry_close
                    ratio_out = float(ac_out) / exit_px
                    if ratio_in > 0 and abs((ratio_out / ratio_in) - 1.0) > 1e-4:
                        adj_factor = ratio_out / ratio_in
                        ca_adjusted = True

            gross_ret = ((exit_px * adj_factor) / entry_px - 1.0) * 100.0
            net_ret = gross_ret - cost_pct
            outcomes.append(
                ForwardOutcomeRecord(
                    prediction_id=pred_id,
                    symbol=sym,
                    observation_date=obs_dt,
                    horizon=horizon,  # type: ignore[arg-type]
                    status="MATURED",
                    entry_date=entry_dt,
                    entry_price=round(entry_px, 4),
                    exit_date=exit_dt,
                    exit_price=round(exit_px, 4),
                    gross_return_pct=round(float(gross_ret), 4),
                    transaction_cost_pct=round(float(cost_pct), 4),
                    net_return_pct=round(float(net_ret), 4),
                    benchmark_return_pct=round(float(net_ret), 4),
                    corporate_action_adjusted=ca_adjusted,
                    evaluated_at=now_iso,
                )
            )

    if outcomes:
        store.upsert_forward_outcomes(outcomes)
    return outcomes


def compute_forward_horizon_metrics(
    snapshots_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    horizon: str,
    model_version: str = "5.3.0-pilot",
    min_samples: int = MIN_FORWARD_SAMPLES,
) -> ForwardHorizonEvaluation:
    """Compute independent forward evaluation metrics for 1D, 5D, or 10D."""
    prob_col = {"1D": "prob_1d_up", "5D": "prob_5d_up", "10D": "prob_10d_up"}[horizon]
    total_snaps = len(snapshots_df)
    avail_count = int(snapshots_df[prob_col].notna().sum()) if (not snapshots_df.empty and prob_col in snapshots_df.columns) else 0
    coverage = round(avail_count / total_snaps * 100.0, 1) if total_snaps > 0 else 0.0

    h_out = outcomes_df[outcomes_df["horizon"] == horizon].copy() if not outcomes_df.empty else pd.DataFrame()
    matured = h_out[h_out["status"] == "MATURED"].copy() if not h_out.empty else pd.DataFrame()
    pending_cnt = int((h_out["status"] == "PENDING").sum()) if not h_out.empty else 0

    if snapshots_df.empty or matured.empty:
        return ForwardHorizonEvaluation(
            horizon=horizon,  # type: ignore[arg-type]
            model_version=model_version,
            min_sample_threshold=min_samples,
            total_snapshots=total_snaps,
            matured_count=0,
            pending_count=pending_cnt,
            prediction_coverage_pct=coverage,
            status="INSUFFICIENT_DATA",
        )

    merged = matured.merge(snapshots_df[["prediction_id", prob_col, "benchmark_prob_up"]], on="prediction_id", how="inner")
    valid = merged.dropna(subset=[prob_col, "gross_return_pct", "net_return_pct"]).copy()
    n_eval = len(valid)

    if n_eval < min_samples:
        return ForwardHorizonEvaluation(
            horizon=horizon,  # type: ignore[arg-type]
            model_version=model_version,
            min_sample_threshold=min_samples,
            total_snapshots=total_snaps,
            matured_count=n_eval,
            pending_count=pending_cnt,
            prediction_coverage_pct=coverage,
            status="INSUFFICIENT_DATA",
        )

    y_gross = pd.to_numeric(valid["gross_return_pct"], errors="coerce")
    y_net = pd.to_numeric(valid["net_return_pct"], errors="coerce")
    y_up = (y_gross > 0).astype(int).values
    p_up = pd.to_numeric(valid[prob_col], errors="coerce").clip(0.01, 0.99).values
    pred_up = (p_up >= 0.50).astype(int)

    acc = round(float((pred_up == y_up).mean() * 100.0), 2)
    brier = round(float(brier_score_loss(y_up, p_up)), 4)
    try:
        auc = round(float(roc_auc_score(y_up, p_up)), 4) if len(set(y_up)) > 1 else None
    except Exception:
        auc = None
    ic_raw = pd.Series(p_up).rank().corr(y_gross.rank())
    rank_ic = round(float(ic_raw), 4) if pd.notna(ic_raw) else None

    long_mask = p_up >= 0.50
    eval_net = y_net[long_mask] if int(long_mask.sum()) > 0 else y_net
    eval_gross = y_gross[long_mask] if int(long_mask.sum()) > 0 else y_gross

    gross_cum = round(float((np.prod(1.0 + eval_gross.values / 100.0) - 1.0) * 100.0), 2)
    net_cum = round(float((np.prod(1.0 + eval_net.values / 100.0) - 1.0) * 100.0), 2)
    eq_curve = np.cumprod(1.0 + eval_net.values / 100.0)
    peak = np.maximum.accumulate(eq_curve)
    max_dd = round(float(((eq_curve / np.maximum(peak, 1e-9)) - 1.0).min() * 100.0), 2)
    std_r = float(np.std(eval_net.values, ddof=1)) if len(eval_net) > 1 else 0.0
    sharpe = round(float((np.mean(eval_net.values) / std_r) * np.sqrt(252.0)), 3) if std_r > 1e-9 else 0.0
    wins = eval_net[eval_net > 0].sum()
    losses = abs(eval_net[eval_net < 0].sum())
    pf = round(float(wins / losses), 3) if losses > 1e-9 else (2.0 if wins > 0 else 0.0)
    win_rate = round(float((eval_net > 0).mean() * 100.0), 2)

    vs_mom: float | None = None
    b_sub = valid.dropna(subset=["benchmark_prob_up"])
    if len(b_sub) >= min_samples:
        b_long = pd.to_numeric(b_sub["benchmark_prob_up"], errors="coerce") >= 0.50
        b_rets = pd.to_numeric(b_sub.loc[b_long, "net_return_pct"], errors="coerce")
        b_cum = float((np.prod(1.0 + b_rets.values / 100.0) - 1.0) * 100.0) if len(b_rets) > 0 else 0.0
        vs_mom = round(net_cum - b_cum, 2)

    status = "WARNING" if (acc < 48.0 or brier > 0.27) else "OK"
    return ForwardHorizonEvaluation(
        horizon=horizon,  # type: ignore[arg-type]
        model_version=model_version,
        min_sample_threshold=min_samples,
        total_snapshots=total_snaps,
        matured_count=n_eval,
        pending_count=pending_cnt,
        prediction_coverage_pct=coverage,
        status=status,  # type: ignore[arg-type]
        directional_accuracy_pct=acc,
        roc_auc=auc,
        brier_score=brier,
        rank_ic=rank_ic,
        gross_return_pct=gross_cum,
        net_return_pct=net_cum,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        profit_factor=pf,
        win_rate_pct=win_rate,
        vs_naive_momentum_net_diff_pct=vs_mom,
    )


def run_paper_research_portfolio(
    snapshots_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    indices_df: pd.DataFrame | None = None,
) -> PaperPortfolioReport:
    """Simulate separate 1D, 5D, and 10D ₹10,00,000 paper portfolios with strict exposure/overlap rules."""
    strategies: dict[str, PaperStrategySummary] = {}
    all_trades: list[PaperTradeRecord] = []

    if snapshots_df.empty or outcomes_df.empty:
        for h in ("1D", "5D", "10D"):
            strategies[h] = PaperStrategySummary(horizon=h, status="INSUFFICIENT_DATA")  # type: ignore[arg-type]
        return PaperPortfolioReport(
            strategies=strategies,
            benchmarks={
                "naive_momentum": {"net_return_pct": 0.0, "status": "INSUFFICIENT_DATA"},
                "nifty_50_buy_hold": {"net_return_pct": 0.0, "status": "INSUFFICIENT_DATA"},
                "cash_no_trade": {"net_return_pct": 0.0, "status": "ACTIVE"},
            },
            recent_trades=[],
        )

    snaps = snapshots_df.sort_values(["observation_date", "symbol"]).copy()
    all_dates = sorted(snaps["observation_date"].astype(str).unique().tolist())

    for horizon, hold_n in (("1D", 1), ("5D", 5), ("10D", 10)):
        rec_col = {"1D": "recommendation_1d", "5D": "recommendation_5d", "10D": "recommendation_10d"}[horizon]
        h_out = outcomes_df[outcomes_df["horizon"] == horizon].copy()
        merged = snaps.merge(h_out, on=["prediction_id", "symbol", "observation_date"], how="left")

        equity = STARTING_CAPITAL_INR
        gross_pnl_total = 0.0
        cost_total = 0.0
        peak_equity = equity
        max_dd = 0.0
        peak_exposure_pct = 0.0
        active_positions: list[dict[str, Any]] = []  # {symbol, exit_date, capital}
        executed_cnt = 0
        rejected_cnt = 0
        win_cnt = 0

        for _, row in merged.iterrows():
            sym = str(row["symbol"]).upper()
            sig_dt = str(row["observation_date"])
            pred_id = str(row["prediction_id"])
            rec = str(row.get(rec_col) or "UNAVAILABLE")

            # Settle any positions whose exit_date <= sig_dt
            still_open: list[dict[str, Any]] = []
            for pos in active_positions:
                if pos["exit_date"] <= sig_dt:
                    equity += pos["net_pnl"]
                    peak_equity = max(peak_equity, equity)
                    dd = ((equity / max(peak_equity, 1.0)) - 1.0) * 100.0
                    max_dd = min(max_dd, dd)
                else:
                    still_open.append(pos)
            active_positions = still_open

            current_exposure_inr = sum(p["capital"] for p in active_positions)
            current_exposure_pct = (current_exposure_inr / max(equity, 1.0)) * 100.0
            peak_exposure_pct = max(peak_exposure_pct, current_exposure_pct)

            if rec == "UNAVAILABLE":
                rejected_cnt += 1
                all_trades.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=sig_dt,
                        action="REJECTED",
                        reason="REJECTED_UNAVAILABLE_MODEL",
                    )
                )
                continue

            if rec != "BUY":
                rejected_cnt += 1
                all_trades.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=sig_dt,
                        action="REJECTED",
                        reason=f"REJECTED_NON_BUY_SIGNAL_{rec}",
                    )
                )
                continue

            if any(p["symbol"] == sym for p in active_positions):
                rejected_cnt += 1
                all_trades.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=sig_dt,
                        action="REJECTED",
                        reason="REJECTED_OVERLAPPING_POSITION",
                    )
                )
                continue

            alloc_inr = min(equity * MAX_POSITION_PCT, max(0.0, equity * MAX_EXPOSURE_PCT - current_exposure_inr))
            if alloc_inr < 1000.0 or (current_exposure_inr + alloc_inr) / max(equity, 1.0) > MAX_EXPOSURE_PCT + 1e-6:
                rejected_cnt += 1
                all_trades.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=sig_dt,
                        action="REJECTED",
                        reason="REJECTED_MAX_EXPOSURE_80PCT",
                    )
                )
                continue

            if str(row.get("status") or "PENDING") != "MATURED":
                rejected_cnt += 1
                all_trades.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=sig_dt,
                        action="REJECTED",
                        reason="PENDING_MATURITY",
                    )
                )
                continue

            g_ret = float(row["gross_return_pct"])
            c_pct = float(row["transaction_cost_pct"])
            n_ret = float(row["net_return_pct"])
            gross_pnl = alloc_inr * (g_ret / 100.0)
            cost_inr = alloc_inr * (c_pct / 100.0)
            net_pnl = alloc_inr * (n_ret / 100.0)

            gross_pnl_total += gross_pnl
            cost_total += cost_inr
            executed_cnt += 1
            if net_pnl > 0:
                win_cnt += 1

            exit_dt = str(row["exit_date"])
            active_positions.append({"symbol": sym, "exit_date": exit_dt, "capital": alloc_inr, "net_pnl": net_pnl})
            new_exp_pct = ((current_exposure_inr + alloc_inr) / max(equity, 1.0)) * 100.0
            peak_exposure_pct = max(peak_exposure_pct, new_exp_pct)

            all_trades.append(
                PaperTradeRecord(
                    strategy_horizon=horizon,  # type: ignore[arg-type]
                    prediction_id=pred_id,
                    symbol=sym,
                    signal_date=sig_dt,
                    action="ENTRY_EXIT",
                    reason="BUY_SIGNAL_EXECUTED",
                    entry_date=str(row["entry_date"]),
                    exit_date=exit_dt,
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    allocated_capital_inr=round(alloc_inr, 2),
                    position_weight_pct=round(alloc_inr / max(equity, 1.0) * 100.0, 2),
                    gross_return_pct=round(g_ret, 4),
                    transaction_cost_inr=round(cost_inr, 2),
                    net_return_pct=round(n_ret, 4),
                    net_pnl_inr=round(net_pnl, 2),
                )
            )

        # Settle remaining matured positions at end of window
        for pos in active_positions:
            equity += pos["net_pnl"]
            peak_equity = max(peak_equity, equity)
            dd = ((equity / max(peak_equity, 1.0)) - 1.0) * 100.0
            max_dd = min(max_dd, dd)

        strategies[horizon] = PaperStrategySummary(
            horizon=horizon,  # type: ignore[arg-type]
            starting_capital_inr=STARTING_CAPITAL_INR,
            ending_capital_inr=round(equity, 2),
            net_return_pct=round((equity / STARTING_CAPITAL_INR - 1.0) * 100.0, 2),
            gross_return_pct=round((gross_pnl_total / STARTING_CAPITAL_INR) * 100.0, 2),
            total_cost_inr=round(cost_total, 2),
            max_drawdown_pct=round(max_dd, 2),
            peak_exposure_pct=round(peak_exposure_pct, 2),
            executed_trades=executed_cnt,
            rejected_signals=rejected_cnt,
            win_rate_pct=round(win_cnt / executed_cnt * 100.0, 2) if executed_cnt > 0 else None,
            status="ACTIVE" if executed_cnt > 0 else "INSUFFICIENT_DATA",
        )

    # Comparable benchmarks over the same snapshot dates
    mom_net_pct = 0.0
    matured_1d = outcomes_df[(outcomes_df["horizon"] == "1D") & (outcomes_df["status"] == "MATURED")]
    if not matured_1d.empty:
        m_join = snaps.merge(matured_1d, on=["prediction_id", "symbol", "observation_date"], how="inner")
        m_buy = m_join[pd.to_numeric(m_join["benchmark_prob_up"], errors="coerce") >= 0.535]
        if not m_buy.empty:
            mom_net_pct = round(float(m_buy["net_return_pct"].mean() * len(m_buy) * MAX_POSITION_PCT), 2)

    nifty_ret_pct = 0.0
    if indices_df is not None and not indices_df.empty and all_dates:
        n50 = indices_df[indices_df["index_code"].astype(str).str.upper() == "NIFTY 50"].copy()
        n50["observation_date"] = n50["observation_date"].astype(str)
        n50 = n50[n50["observation_date"].isin(all_dates)].sort_values("observation_date")
        if len(n50) >= 2 and float(n50["close"].iloc[0]) > 0:
            nifty_ret_pct = round((float(n50["close"].iloc[-1]) / float(n50["close"].iloc[0]) - 1.0) * 100.0, 2)

    return PaperPortfolioReport(
        starting_capital_inr=STARTING_CAPITAL_INR,
        max_position_pct=MAX_POSITION_PCT * 100.0,
        max_total_exposure_pct=MAX_EXPOSURE_PCT * 100.0,
        strategies=strategies,
        benchmarks={
            "naive_momentum": {"net_return_pct": mom_net_pct, "status": "ACTIVE" if not matured_1d.empty else "INSUFFICIENT_DATA"},
            "nifty_50_buy_hold": {"net_return_pct": nifty_ret_pct, "status": "ACTIVE" if len(all_dates) >= 2 else "INSUFFICIENT_DATA"},
            "cash_no_trade": {"net_return_pct": 0.0, "status": "ACTIVE"},
        },
        recent_trades=all_trades[-30:],
    )


def build_forward_validation_payload(
    store: DuckDBStore | None = None,
    equities_df: pd.DataFrame | None = None,
    indices_df: pd.DataFrame | None = None,
    persist: bool = True,
) -> ForwardValidationPayload:
    """Assemble the complete Phase 5.3 Forward Validation & Paper Portfolio payload."""
    st = store or DuckDBStore()
    now_ist = datetime.now(IST_TZ)
    dq_warnings: list[str] = []
    alerts: list[str] = []

    valid_events = 0
    quarantined_events = 0
    last_news_at: str | None = None

    if st.events_parquet.exists():
        ev_df = pd.read_parquet(st.events_parquet)
        valid_events = len(ev_df)
        if not ev_df.empty and "collection_timestamp" in ev_df.columns:
            last_news_at = str(ev_df["collection_timestamp"].dropna().max() or "") or None
        if not last_news_at and not ev_df.empty and "publication_timestamp" in ev_df.columns:
            last_news_at = str(ev_df["publication_timestamp"].dropna().max() or "") or None
    else:
        dq_warnings.append("structured_events.parquet not yet populated; forward news collection initialized.")

    if st.quarantined_events_parquet.exists():
        q_df = pd.read_parquet(st.quarantined_events_parquet)
        quarantined_events = len(q_df)

    if last_news_at:
        try:
            dt = datetime.fromisoformat(last_news_at.replace("Z", "+00:00"))
            age_h = (now_ist - dt.astimezone(IST_TZ)).total_seconds() / 3600.0
            if age_h > 48.0:
                alerts.append(f"Stale news collection: last event collected {age_h:.1f}h ago.")
        except Exception:
            pass
    else:
        alerts.append("No forward news events collected yet.")

    snaps_df = pd.read_parquet(st.forward_predictions_parquet) if st.forward_predictions_parquet.exists() else pd.DataFrame()
    outs_df = pd.read_parquet(st.forward_outcomes_parquet) if st.forward_outcomes_parquet.exists() else pd.DataFrame()

    if snaps_df.empty:
        dq_warnings.append("forward_predictions.parquet has 0 immutable snapshots; run daily pipeline to record snapshots.")

    horizon_evals: dict[str, ForwardHorizonEvaluation] = {}
    for h in ("1D", "5D", "10D"):
        he = compute_forward_horizon_metrics(snaps_df, outs_df, h)
        horizon_evals[h] = he
        if he.status == "WARNING":
            alerts.append(f"Forward validation drift warning on {h}: Acc={he.directional_accuracy_pct}%, Brier={he.brier_score}.")

    if indices_df is None and st.indices_parquet.exists():
        indices_df = pd.read_parquet(st.indices_parquet)

    paper_report = run_paper_research_portfolio(snaps_df, outs_df, indices_df=indices_df)
    if persist and paper_report.recent_trades:
        st.upsert_paper_trades(paper_report.recent_trades)

    matured_total = int((outs_df["status"] == "MATURED").sum()) if not outs_df.empty else 0
    pending_total = int((outs_df["status"] == "PENDING").sum()) if not outs_df.empty else 0

    recent_snaps: list[ForwardPredictionSnapshot] = []
    if not snaps_df.empty:
        for _, r in snaps_df.tail(15).iterrows():
            d = r.to_dict()
            for jf in ("model_availability", "metadata"):
                if isinstance(d.get(jf), str):
                    try:
                        d[jf] = json.loads(d[jf])
                    except Exception:
                        d[jf] = {}
            recent_snaps.append(ForwardPredictionSnapshot.model_validate(d))

    recent_outs: list[ForwardOutcomeRecord] = []
    if not outs_df.empty:
        for _, r in outs_df.tail(15).iterrows():
            d = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
            recent_outs.append(ForwardOutcomeRecord.model_validate(d))

    payload = ForwardValidationPayload(
        generated_at=now_ist.isoformat(),
        pilot_universe=list(PILOT_UNIVERSE_5),
        total_real_news_events=valid_events + quarantined_events,
        valid_news_events=valid_events,
        quarantined_news_events=quarantined_events,
        last_news_collection_at=last_news_at,
        total_immutable_snapshots=len(snaps_df),
        matured_outcomes_total=matured_total,
        pending_outcomes_total=pending_total,
        horizon_evaluations=horizon_evals,
        paper_portfolio=paper_report,
        recent_snapshots=recent_snaps,
        recent_outcomes=recent_outs,
        data_quality_warnings=dq_warnings,
        monitoring_alerts=alerts,
    )

    if persist:
        out_path = st.predictions_dir / "forward_validation.json"
        out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    return payload
