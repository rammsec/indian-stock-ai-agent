"""Phase 5.3.1 Forward Validation, Immutable Snapshots, Matured Outcomes & Look-Ahead-Free Paper Portfolio."""

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
    NewsProvenanceSummary,
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

        b_p1 = q_pred.get("momentum_prob_up")
        b_r1 = q_pred.get("momentum_1d_return_pct")
        b_p5 = q_pred.get("momentum_prob_5d_up")
        b_r5 = q_pred.get("momentum_5d_return_pct")
        b_p10 = q_pred.get("momentum_prob_10d_up")
        b_r10 = q_pred.get("momentum_10d_return_pct")

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
            benchmark_prob_up=b_p1,
            benchmark_5d_return_pct=b_r5,
            benchmark_prob_1d_up=b_p1,
            benchmark_expected_1d_return_pct=b_r1,
            benchmark_prob_5d_up=b_p5,
            benchmark_expected_5d_return_pct=b_r5,
            benchmark_prob_10d_up=b_p10,
            benchmark_expected_10d_return_pct=b_r10,
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


def _compute_ca_adj_factor(grp: pd.DataFrame, entry_idx: int, exit_idx: int) -> tuple[float, bool]:
    """Compute corporate-action adjustment factor from adj_close ratios between entry and exit."""
    if "adj_close" not in grp.columns:
        return 1.0, False
    entry_close = grp.loc[entry_idx, "close"]
    exit_close = grp.loc[exit_idx, "close"]
    ac_in = grp.loc[entry_idx, "adj_close"]
    ac_out = grp.loc[exit_idx, "adj_close"]
    if (
        pd.notna(ac_in)
        and pd.notna(ac_out)
        and pd.notna(entry_close)
        and pd.notna(exit_close)
        and float(ac_in) > 0
        and float(entry_close) > 0
        and float(exit_close) > 0
    ):
        ratio_in = float(ac_in) / float(entry_close)
        ratio_out = float(ac_out) / float(exit_close)
        if ratio_in > 0 and abs((ratio_out / ratio_in) - 1.0) > 1e-4:
            return float(ratio_out / ratio_in), True
    return 1.0, False


def evaluate_forward_outcomes(
    store: DuckDBStore,
    equities_df: pd.DataFrame | None = None,
) -> list[ForwardOutcomeRecord]:
    """Attach realized 1D, 5D, and 10D outcomes strictly after each NSE trading-session horizon matures.

    Execution convention:
    - 1D: Entry at next eligible NSE session's open (t+1), exit at that same session's close (t+1).
    - 5D: Entry at next eligible NSE session's open (t+1), exit at close of 5th NSE session (t+5).
    - 10D: Entry at next eligible NSE session's open (t+1), exit at close of 10th NSE session (t+10).
    - Missing or non-positive execution/exit prices are marked INVALID (never fabricated).
    """
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
    eq = equities_df[cols].copy()
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
                        status_reason="Awaiting NSE trading session maturity.",
                    )
                )
                continue

            entry_idx = base_idx + 1
            exit_idx = base_idx + hold_n
            entry_dt = str(grp.loc[entry_idx, "observation_date"])
            exit_dt = str(grp.loc[exit_idx, "observation_date"])

            entry_open = grp.loc[entry_idx, "open"] if "open" in grp.columns else np.nan
            exit_close = grp.loc[exit_idx, "close"] if "close" in grp.columns else np.nan
            if pd.isna(entry_open) or float(entry_open) <= 0 or pd.isna(exit_close) or float(exit_close) <= 0:
                outcomes.append(
                    ForwardOutcomeRecord(
                        prediction_id=pred_id,
                        symbol=sym,
                        observation_date=obs_dt,
                        horizon=horizon,  # type: ignore[arg-type]
                        status="INVALID",
                        status_reason="Missing or non-positive open[t+1] or close[t+H] price; refusing to fabricate return.",
                        entry_date=entry_dt,
                        exit_date=exit_dt,
                        evaluated_at=now_iso,
                    )
                )
                continue

            entry_px = float(entry_open)
            exit_px = float(exit_close)
            adj_factor, ca_adjusted = _compute_ca_adj_factor(grp, entry_idx, exit_idx)

            gross_ret = ((exit_px * adj_factor) / entry_px - 1.0) * 100.0
            net_ret = gross_ret - cost_pct
            outcomes.append(
                ForwardOutcomeRecord(
                    prediction_id=pred_id,
                    symbol=sym,
                    observation_date=obs_dt,
                    horizon=horizon,  # type: ignore[arg-type]
                    status="MATURED",
                    status_reason="Matured on NSE trading calendar.",
                    entry_date=entry_dt,
                    entry_price=round(entry_px, 4),
                    exit_date=exit_dt,
                    exit_price=round(exit_px, 4),
                    gross_return_pct=round(float(gross_ret), 4),
                    transaction_cost_pct=round(float(cost_pct), 4),
                    net_return_pct=round(float(net_ret), 4),
                    benchmark_return_pct=None,
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
    """Compute independent forward evaluation metrics and horizon-matched benchmark metrics for 1D, 5D, or 10D."""
    prob_col = {"1D": "prob_1d_up", "5D": "prob_5d_up", "10D": "prob_10d_up"}[horizon]
    bench_col = {"1D": "benchmark_prob_1d_up", "5D": "benchmark_prob_5d_up", "10D": "benchmark_prob_10d_up"}[horizon]

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

    keep_cols = ["prediction_id", prob_col]
    if bench_col in snapshots_df.columns:
        keep_cols.append(bench_col)
    merged = matured.merge(snapshots_df[keep_cols], on="prediction_id", how="inner")
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

    # Horizon-matched benchmark metrics (never compare 5D/10D against 1D benchmark)
    b_acc: float | None = None
    b_auc: float | None = None
    b_brier: float | None = None
    b_net: float | None = None
    vs_mom: float | None = None
    if bench_col in valid.columns:
        b_sub = valid.dropna(subset=[bench_col])
        if len(b_sub) >= min_samples:
            bp = pd.to_numeric(b_sub[bench_col], errors="coerce").clip(0.01, 0.99).values
            by_up = (pd.to_numeric(b_sub["gross_return_pct"], errors="coerce") > 0).astype(int).values
            b_acc = round(float(((bp >= 0.50).astype(int) == by_up).mean() * 100.0), 2)
            b_brier = round(float(brier_score_loss(by_up, bp)), 4)
            try:
                b_auc = round(float(roc_auc_score(by_up, bp)), 4) if len(set(by_up)) > 1 else None
            except Exception:
                b_auc = None
            b_long = bp >= 0.50
            b_rets = pd.to_numeric(b_sub.loc[b_long, "net_return_pct"], errors="coerce")
            b_net = round(float((np.prod(1.0 + b_rets.values / 100.0) - 1.0) * 100.0), 2) if len(b_rets) > 0 else 0.0
            vs_mom = round(net_cum - b_net, 2)

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
        benchmark_accuracy_pct=b_acc,
        benchmark_roc_auc=b_auc,
        benchmark_brier_score=b_brier,
        benchmark_net_return_pct=b_net,
    )


def _simulate_strategy_timeline(
    horizon: str,
    hold_n: int,
    snaps_df: pd.DataFrame,
    sym_map: dict[str, pd.DataFrame],
    signal_getter: Any,
    is_benchmark: bool = False,
) -> tuple[PaperStrategySummary, list[PaperTradeRecord]]:
    """Chronological, zero-lookahead paper portfolio simulation across NSE trading sessions.

    - A signal on observation_date t is evaluated using ONLY data available at t.
    - Qualifying BUY enters at the next eligible NSE trading session's open (t+1).
    - Entry occurs immediately once t+1 open is observed, even if t+H has NOT matured yet.
    - Open positions are marked to market on each subsequent session and settled at close[t+H].
    """
    events: list[PaperTradeRecord] = []
    cash = STARTING_CAPITAL_INR
    realized_pnl = 0.0
    gross_pnl_total = 0.0
    cost_total = 0.0
    peak_equity = STARTING_CAPITAL_INR
    max_dd = 0.0
    peak_exposure_pct = 0.0
    signals_received = 0
    entries_executed = 0
    matured_cnt = 0
    rejected_cnt = 0
    win_cnt = 0

    # Active open positions keyed by symbol
    open_positions: dict[str, dict[str, Any]] = {}

    # Build chronological session calendar from equities + snapshot dates
    eq_dates = sorted({d for grp in sym_map.values() for d in grp["observation_date"].tolist()})
    snap_dates = sorted(snaps_df["observation_date"].astype(str).unique().tolist())
    all_sessions = sorted(set(eq_dates) | set(snap_dates))

    # Pre-index snapshots by (entry_session_date -> list of signal rows) using NSE calendar
    # A signal on obs_dt enters at the next trading session in sym_map[sym]
    signals_by_obs: dict[str, list[dict[str, Any]]] = {}
    for _, r in snaps_df.sort_values(["observation_date", "symbol"]).iterrows():
        d_str = str(r["observation_date"])
        signals_by_obs.setdefault(d_str, []).append(r.to_dict())

    pending_entries_by_date: dict[str, list[dict[str, Any]]] = {}

    for sess_dt in all_sessions:
        # 1. Settle any open positions whose scheduled exit_date <= sess_dt
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            if pos["exit_date"] is not None and sess_dt >= pos["exit_date"]:
                grp = sym_map.get(sym)
                exit_idx = pos["exit_idx"]
                if grp is not None and exit_idx < len(grp):
                    exit_px_raw = grp.loc[exit_idx, "close"]
                    if pd.notna(exit_px_raw) and float(exit_px_raw) > 0:
                        adj_factor, _ = _compute_ca_adj_factor(grp, pos["entry_idx"], exit_idx)
                        g_ret = ((float(exit_px_raw) * adj_factor) / pos["entry_price"] - 1.0) * 100.0
                        c_pct = pos["cost_pct"]
                        n_ret = g_ret - c_pct
                        g_pnl = pos["capital"] * (g_ret / 100.0)
                        c_inr = pos["capital"] * (c_pct / 100.0)
                        n_pnl = pos["capital"] * (n_ret / 100.0)

                        cash += pos["capital"] + n_pnl
                        realized_pnl += n_pnl
                        gross_pnl_total += g_pnl
                        cost_total += c_inr
                        matured_cnt += 1
                        if n_pnl > 0:
                            win_cnt += 1
                        del open_positions[sym]

                        inv_now = sum(p["capital"] for p in open_positions.values())
                        eq_now = cash + inv_now + sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
                        peak_equity = max(peak_equity, eq_now)
                        max_dd = min(max_dd, ((eq_now / max(peak_equity, 1.0)) - 1.0) * 100.0)

                        if not is_benchmark:
                            events.append(
                                PaperTradeRecord(
                                    strategy_horizon=horizon,  # type: ignore[arg-type]
                                    prediction_id=pos["prediction_id"],
                                    symbol=sym,
                                    signal_date=pos["signal_date"],
                                    event_date=pos["exit_date"],
                                    action="POSITION_EXITED",
                                    reason="SCHEDULED_HORIZON_EXIT",
                                    entry_date=pos["entry_date"],
                                    exit_date=pos["exit_date"],
                                    entry_price=round(pos["entry_price"], 4),
                                    exit_price=round(float(exit_px_raw), 4),
                                    allocated_capital_inr=round(pos["capital"], 2),
                                    position_weight_pct=round(pos["weight_pct"], 2),
                                    gross_return_pct=round(g_ret, 4),
                                    transaction_cost_inr=round(c_inr, 2),
                                    net_return_pct=round(n_ret, 4),
                                    net_pnl_inr=round(n_pnl, 2),
                                    realized_pnl_inr=round(n_pnl, 2),
                                    cash_after_inr=round(cash, 2),
                                    invested_after_inr=round(inv_now, 2),
                                    equity_after_inr=round(eq_now, 2),
                                    holding_sessions=hold_n,
                                )
                            )

        # 2. Execute scheduled entries at today's open (t+1 open for yesterday's signals)
        for sig in pending_entries_by_date.get(sess_dt, []):
            sym = str(sig["symbol"]).upper()
            pred_id = str(sig["prediction_id"])
            obs_dt = str(sig["observation_date"])
            rec = signal_getter(sig)

            inv_now = sum(p["capital"] for p in open_positions.values())
            unrl_now = sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
            eq_now = cash + inv_now + unrl_now

            if rec == "UNAVAILABLE":
                rejected_cnt += 1
                if not is_benchmark:
                    events.append(
                        PaperTradeRecord(
                            strategy_horizon=horizon,  # type: ignore[arg-type]
                            prediction_id=pred_id,
                            symbol=sym,
                            signal_date=obs_dt,
                            event_date=sess_dt,
                            action="SIGNAL_REJECTED",
                            reason="REJECTED_UNAVAILABLE_MODEL",
                            cash_after_inr=round(cash, 2),
                            invested_after_inr=round(inv_now, 2),
                            equity_after_inr=round(eq_now, 2),
                        )
                    )
                continue

            if rec != "BUY":
                rejected_cnt += 1
                if not is_benchmark:
                    events.append(
                        PaperTradeRecord(
                            strategy_horizon=horizon,  # type: ignore[arg-type]
                            prediction_id=pred_id,
                            symbol=sym,
                            signal_date=obs_dt,
                            event_date=sess_dt,
                            action="SIGNAL_REJECTED",
                            reason=f"REJECTED_NON_BUY_SIGNAL_{rec}",
                            cash_after_inr=round(cash, 2),
                            invested_after_inr=round(inv_now, 2),
                            equity_after_inr=round(eq_now, 2),
                        )
                    )
                continue

            if sym in open_positions:
                rejected_cnt += 1
                if not is_benchmark:
                    events.append(
                        PaperTradeRecord(
                            strategy_horizon=horizon,  # type: ignore[arg-type]
                            prediction_id=pred_id,
                            symbol=sym,
                            signal_date=obs_dt,
                            event_date=sess_dt,
                            action="SIGNAL_REJECTED",
                            reason="REJECTED_OVERLAPPING_POSITION",
                            cash_after_inr=round(cash, 2),
                            invested_after_inr=round(inv_now, 2),
                            equity_after_inr=round(eq_now, 2),
                        )
                    )
                continue

            alloc_inr = min(eq_now * MAX_POSITION_PCT, max(0.0, eq_now * MAX_EXPOSURE_PCT - inv_now), cash)
            if alloc_inr < 1000.0 or (inv_now + alloc_inr) / max(eq_now, 1.0) > MAX_EXPOSURE_PCT + 1e-6:
                rejected_cnt += 1
                if not is_benchmark:
                    events.append(
                        PaperTradeRecord(
                            strategy_horizon=horizon,  # type: ignore[arg-type]
                            prediction_id=pred_id,
                            symbol=sym,
                            signal_date=obs_dt,
                            event_date=sess_dt,
                            action="SIGNAL_REJECTED",
                            reason="REJECTED_MAX_EXPOSURE_80PCT",
                            cash_after_inr=round(cash, 2),
                            invested_after_inr=round(inv_now, 2),
                            equity_after_inr=round(eq_now, 2),
                        )
                    )
                continue

            grp = sym_map.get(sym)
            entry_idx = sig["_entry_idx"]
            entry_open = grp.loc[entry_idx, "open"] if grp is not None and "open" in grp.columns else np.nan
            if pd.isna(entry_open) or float(entry_open) <= 0:
                rejected_cnt += 1
                if not is_benchmark:
                    events.append(
                        PaperTradeRecord(
                            strategy_horizon=horizon,  # type: ignore[arg-type]
                            prediction_id=pred_id,
                            symbol=sym,
                            signal_date=obs_dt,
                            event_date=sess_dt,
                            action="SIGNAL_REJECTED",
                            reason="REJECTED_INVALID_ENTRY_PRICE",
                            cash_after_inr=round(cash, 2),
                            invested_after_inr=round(inv_now, 2),
                            equity_after_inr=round(eq_now, 2),
                        )
                    )
                continue

            entry_px = float(entry_open)
            meta_raw = sig.get("metadata")
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw.startswith("{") else (meta_raw or {})
            cost_pct = float(meta.get("roundtrip_delivery_cost_pct") or DEFAULT_CNC_ROUNDTRIP_COST_PCT)
            exit_idx = sig["_base_idx"] + hold_n
            exit_dt = str(grp.loc[exit_idx, "observation_date"]) if grp is not None and exit_idx < len(grp) else None

            cash -= alloc_inr
            entries_executed += 1
            weight_pct = alloc_inr / max(eq_now, 1.0) * 100.0
            open_positions[sym] = {
                "prediction_id": pred_id,
                "symbol": sym,
                "signal_date": obs_dt,
                "entry_date": sess_dt,
                "entry_idx": entry_idx,
                "entry_price": entry_px,
                "exit_idx": exit_idx,
                "exit_date": exit_dt,
                "capital": alloc_inr,
                "weight_pct": weight_pct,
                "cost_pct": cost_pct,
                "unrealized_pnl": 0.0,
            }

            new_inv = sum(p["capital"] for p in open_positions.values())
            peak_exposure_pct = max(peak_exposure_pct, (new_inv / max(eq_now, 1.0)) * 100.0)

            if not is_benchmark:
                events.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=obs_dt,
                        event_date=sess_dt,
                        action="ENTRY_EXECUTED",
                        reason="BUY_SIGNAL_EXECUTED_AT_NEXT_OPEN",
                        entry_date=sess_dt,
                        exit_date=exit_dt,
                        entry_price=round(entry_px, 4),
                        allocated_capital_inr=round(alloc_inr, 2),
                        position_weight_pct=round(weight_pct, 2),
                        cash_after_inr=round(cash, 2),
                        invested_after_inr=round(new_inv, 2),
                        equity_after_inr=round(eq_now, 2),
                    )
                )

            # Same-session exit for 1D if close[t+1] is already available
            if exit_dt is not None and exit_dt == sess_dt and grp is not None and exit_idx < len(grp):
                exit_px_raw = grp.loc[exit_idx, "close"]
                if pd.notna(exit_px_raw) and float(exit_px_raw) > 0:
                    adj_factor, _ = _compute_ca_adj_factor(grp, entry_idx, exit_idx)
                    g_ret = ((float(exit_px_raw) * adj_factor) / entry_px - 1.0) * 100.0
                    n_ret = g_ret - cost_pct
                    g_pnl = alloc_inr * (g_ret / 100.0)
                    c_inr = alloc_inr * (cost_pct / 100.0)
                    n_pnl = alloc_inr * (n_ret / 100.0)

                    cash += alloc_inr + n_pnl
                    realized_pnl += n_pnl
                    gross_pnl_total += g_pnl
                    cost_total += c_inr
                    matured_cnt += 1
                    if n_pnl > 0:
                        win_cnt += 1
                    del open_positions[sym]

                    inv_after = sum(p["capital"] for p in open_positions.values())
                    eq_after = cash + inv_after + sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
                    peak_equity = max(peak_equity, eq_after)
                    max_dd = min(max_dd, ((eq_after / max(peak_equity, 1.0)) - 1.0) * 100.0)

                    if not is_benchmark:
                        events.append(
                            PaperTradeRecord(
                                strategy_horizon=horizon,  # type: ignore[arg-type]
                                prediction_id=pred_id,
                                symbol=sym,
                                signal_date=obs_dt,
                                event_date=sess_dt,
                                action="POSITION_EXITED",
                                reason="SCHEDULED_1D_CLOSE_EXIT",
                                entry_date=sess_dt,
                                exit_date=sess_dt,
                                entry_price=round(entry_px, 4),
                                exit_price=round(float(exit_px_raw), 4),
                                allocated_capital_inr=round(alloc_inr, 2),
                                position_weight_pct=round(weight_pct, 2),
                                gross_return_pct=round(g_ret, 4),
                                transaction_cost_inr=round(c_inr, 2),
                                net_return_pct=round(n_ret, 4),
                                net_pnl_inr=round(n_pnl, 2),
                                realized_pnl_inr=round(n_pnl, 2),
                                cash_after_inr=round(cash, 2),
                                invested_after_inr=round(inv_after, 2),
                                equity_after_inr=round(eq_after, 2),
                                holding_sessions=1,
                            )
                        )

        # 3. Mark remaining open positions to market at sess_dt close
        for sym, pos in open_positions.items():
            grp = sym_map.get(sym)
            if grp is not None:
                m_idx_list = grp.index[grp["observation_date"] == sess_dt].tolist()
                if m_idx_list:
                    m_idx = int(m_idx_list[0])
                    c_px = grp.loc[m_idx, "close"]
                    if pd.notna(c_px) and float(c_px) > 0:
                        adj_f, _ = _compute_ca_adj_factor(grp, pos["entry_idx"], m_idx)
                        unrl_ret = ((float(c_px) * adj_f) / pos["entry_price"] - 1.0) * 100.0
                        pos["unrealized_pnl"] = pos["capital"] * (unrl_ret / 100.0)

        inv_eod = sum(p["capital"] for p in open_positions.values())
        unrl_eod = sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
        eq_eod = cash + inv_eod + unrl_eod
        peak_equity = max(peak_equity, eq_eod)
        max_dd = min(max_dd, ((eq_eod / max(peak_equity, 1.0)) - 1.0) * 100.0)
        peak_exposure_pct = max(peak_exposure_pct, (inv_eod / max(eq_eod, 1.0)) * 100.0)

        # 4. Record SIGNAL_RECEIVED on sess_dt and schedule next eligible NSE open (t+1)
        for sig in signals_by_obs.get(sess_dt, []):
            signals_received += 1
            sym = str(sig["symbol"]).upper()
            pred_id = str(sig["prediction_id"])
            rec = signal_getter(sig)
            if not is_benchmark:
                events.append(
                    PaperTradeRecord(
                        strategy_horizon=horizon,  # type: ignore[arg-type]
                        prediction_id=pred_id,
                        symbol=sym,
                        signal_date=sess_dt,
                        event_date=sess_dt,
                        action="SIGNAL_RECEIVED",
                        reason=f"SIGNAL_{rec}",
                        cash_after_inr=round(cash, 2),
                        invested_after_inr=round(inv_eod, 2),
                        equity_after_inr=round(eq_eod, 2),
                    )
                )
            grp = sym_map.get(sym)
            if grp is not None:
                matches = grp.index[grp["observation_date"] == sess_dt].tolist()
                if matches:
                    b_idx = int(matches[0])
                    if b_idx + 1 < len(grp):
                        entry_dt = str(grp.loc[b_idx + 1, "observation_date"])
                        sig_copy = dict(sig)
                        sig_copy["_base_idx"] = b_idx
                        sig_copy["_entry_idx"] = b_idx + 1
                        pending_entries_by_date.setdefault(entry_dt, []).append(sig_copy)

    # Emit POSITION_OPEN summary events for positions still open at end of available data
    inv_final = sum(p["capital"] for p in open_positions.values())
    unrl_final = sum(p.get("unrealized_pnl", 0.0) for p in open_positions.values())
    eq_final = cash + inv_final + unrl_final
    if not is_benchmark:
        for sym, pos in open_positions.items():
            events.append(
                PaperTradeRecord(
                    strategy_horizon=horizon,  # type: ignore[arg-type]
                    prediction_id=pos["prediction_id"],
                    symbol=sym,
                    signal_date=pos["signal_date"],
                    event_date=all_sessions[-1] if all_sessions else pos["entry_date"],
                    action="POSITION_OPEN",
                    reason="AWAITING_SCHEDULED_EXIT",
                    entry_date=pos["entry_date"],
                    exit_date=pos["exit_date"],
                    entry_price=round(pos["entry_price"], 4),
                    allocated_capital_inr=round(pos["capital"], 2),
                    position_weight_pct=round(pos["weight_pct"], 2),
                    unrealized_pnl_inr=round(pos.get("unrealized_pnl", 0.0), 2),
                    cash_after_inr=round(cash, 2),
                    invested_after_inr=round(inv_final, 2),
                    equity_after_inr=round(eq_final, 2),
                )
            )

    summary = PaperStrategySummary(
        horizon=horizon,  # type: ignore[arg-type]
        starting_capital_inr=STARTING_CAPITAL_INR,
        ending_capital_inr=round(eq_final, 2),
        cash_inr=round(cash, 2),
        invested_capital_inr=round(inv_final, 2),
        realized_pnl_inr=round(realized_pnl, 2),
        unrealized_pnl_inr=round(unrl_final, 2),
        net_return_pct=round((eq_final / STARTING_CAPITAL_INR - 1.0) * 100.0, 2),
        gross_return_pct=round(((gross_pnl_total + unrl_final) / STARTING_CAPITAL_INR) * 100.0, 2),
        total_cost_inr=round(cost_total, 2),
        max_drawdown_pct=round(max_dd, 2),
        peak_exposure_pct=round(peak_exposure_pct, 2),
        signals_received=signals_received,
        entries_executed=entries_executed,
        open_positions_count=len(open_positions),
        matured_positions_count=matured_cnt,
        executed_trades=matured_cnt,
        rejected_signals=rejected_cnt,
        win_rate_pct=round(win_cnt / matured_cnt * 100.0, 2) if matured_cnt > 0 else None,
        status="ACTIVE" if entries_executed > 0 else "INSUFFICIENT_DATA",
    )
    return summary, events


def run_paper_research_portfolio(
    snapshots_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    indices_df: pd.DataFrame | None = None,
    equities_df: pd.DataFrame | None = None,
) -> PaperPortfolioReport:
    """Simulate separate 1D, 5D, and 10D ₹10,00,000 paper portfolios with zero look-ahead bias."""
    strategies: dict[str, PaperStrategySummary] = {}
    all_trades: list[PaperTradeRecord] = []

    if snapshots_df.empty:
        for h in ("1D", "5D", "10D"):
            strategies[h] = PaperStrategySummary(horizon=h, status="INSUFFICIENT_DATA")  # type: ignore[arg-type]
        return PaperPortfolioReport(
            strategies=strategies,
            benchmarks={
                "naive_momentum": {"net_return_pct": None, "max_drawdown_pct": None, "status": "UNAVAILABLE"},
                "nifty_50_buy_hold": {"net_return_pct": None, "max_drawdown_pct": None, "status": "UNAVAILABLE"},
                "cash_no_trade": {"net_return_pct": 0.0, "max_drawdown_pct": 0.0, "status": "ACTIVE"},
            },
            recent_trades=[],
        )

    sym_map: dict[str, pd.DataFrame] = {}
    if equities_df is not None and not equities_df.empty:
        cols = [c for c in ("symbol", "observation_date", "open", "close", "adj_close") if c in equities_df.columns]
        eq = equities_df[cols].copy()
        eq["symbol"] = eq["symbol"].astype(str).str.upper()
        eq["observation_date"] = eq["observation_date"].astype(str)
        eq = eq.sort_values(["symbol", "observation_date"]).reset_index(drop=True)
        sym_map = {str(sym): grp.reset_index(drop=True) for sym, grp in eq.groupby("symbol", sort=False)}

    for horizon, hold_n in (("1D", 1), ("5D", 5), ("10D", 10)):
        rec_col = {"1D": "recommendation_1d", "5D": "recommendation_5d", "10D": "recommendation_10d"}[horizon]
        summary, evts = _simulate_strategy_timeline(
            horizon=horizon,
            hold_n=hold_n,
            snaps_df=snapshots_df,
            sym_map=sym_map,
            signal_getter=lambda s, rc=rec_col: str(s.get(rc) or "UNAVAILABLE"),
            is_benchmark=False,
        )
        strategies[horizon] = summary
        all_trades.extend(evts)

    # Independent Benchmark 1: Naive Momentum (portfolio simulation with same capital/exposure/cost rules)
    def _mom_signal(s: dict[str, Any]) -> str:
        p = s.get("benchmark_prob_1d_up")
        if p is None or (isinstance(p, float) and np.isnan(p)):
            p = s.get("benchmark_prob_up")
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "UNAVAILABLE"
        return "BUY" if float(p) >= 0.535 else "HOLD"

    mom_summary, _ = _simulate_strategy_timeline(
        horizon="1D",
        hold_n=1,
        snaps_df=snapshots_df,
        sym_map=sym_map,
        signal_getter=_mom_signal,
        is_benchmark=True,
    )
    mom_bench = {
        "net_return_pct": mom_summary.net_return_pct if mom_summary.entries_executed > 0 else None,
        "max_drawdown_pct": mom_summary.max_drawdown_pct if mom_summary.entries_executed > 0 else None,
        "ending_capital_inr": mom_summary.ending_capital_inr if mom_summary.entries_executed > 0 else None,
        "status": "ACTIVE" if mom_summary.entries_executed > 0 else "INSUFFICIENT_DATA",
    }

    # Independent Benchmark 2: Nifty 50 Buy-and-Hold (80% max exposure comparable, daily equity curve)
    all_dates = sorted(snapshots_df["observation_date"].astype(str).unique().tolist())
    nifty_bench: dict[str, float | str | None] = {
        "net_return_pct": None,
        "max_drawdown_pct": None,
        "ending_capital_inr": None,
        "status": "UNAVAILABLE",
    }
    if indices_df is not None and not indices_df.empty and all_dates:
        n50 = indices_df[indices_df["index_code"].astype(str).str.upper() == "NIFTY 50"].copy()
        n50["observation_date"] = n50["observation_date"].astype(str)
        n50 = n50[n50["observation_date"] >= all_dates[0]].sort_values("observation_date")
        if len(n50) >= 2 and float(n50["close"].iloc[0]) > 0:
            closes = pd.to_numeric(n50["close"], errors="coerce").dropna().values
            if len(closes) >= 2 and closes[0] > 0:
                invested = STARTING_CAPITAL_INR * MAX_EXPOSURE_PCT
                cash_res = STARTING_CAPITAL_INR - invested
                cost_drag = invested * (0.05 / 100.0)
                eq_curve = cash_res + (invested - cost_drag) * (closes / closes[0])
                peak = np.maximum.accumulate(eq_curve)
                dd = float(((eq_curve / np.maximum(peak, 1e-9)) - 1.0).min() * 100.0)
                nifty_bench = {
                    "net_return_pct": round(float((eq_curve[-1] / STARTING_CAPITAL_INR - 1.0) * 100.0), 2),
                    "max_drawdown_pct": round(dd, 2),
                    "ending_capital_inr": round(float(eq_curve[-1]), 2),
                    "status": "ACTIVE",
                }

    return PaperPortfolioReport(
        starting_capital_inr=STARTING_CAPITAL_INR,
        max_position_pct=MAX_POSITION_PCT * 100.0,
        max_total_exposure_pct=MAX_EXPOSURE_PCT * 100.0,
        strategies=strategies,
        benchmarks={
            "naive_momentum": mom_bench,
            "nifty_50_buy_hold": nifty_bench,
            "cash_no_trade": {
                "net_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "ending_capital_inr": STARTING_CAPITAL_INR,
                "status": "ACTIVE",
            },
        },
        recent_trades=all_trades[-60:],
    )


def compute_news_provenance(
    store: DuckDBStore,
    snaps_df: pd.DataFrame,
) -> NewsProvenanceSummary:
    """Audit stored news events for timestamps, duplicates, quarantine counts, and forward provenance."""
    ev_df = pd.read_parquet(store.events_parquet) if store.events_parquet.exists() else pd.DataFrame()
    q_df = pd.read_parquet(store.quarantined_events_parquet) if store.quarantined_events_parquet.exists() else pd.DataFrame()

    valid_cnt = len(ev_df)
    quar_cnt = len(q_df)
    total_cnt = valid_cnt + quar_cnt
    if ev_df.empty:
        return NewsProvenanceSummary(
            total_events=total_cnt,
            valid_events=0,
            quarantined_events=quar_cnt,
            is_genuinely_forward_collected=False,
            provenance_note="No structured news events stored yet.",
        )

    pub_min = str(ev_df["publication_timestamp"].dropna().min()) if "publication_timestamp" in ev_df.columns else None
    pub_max = str(ev_df["publication_timestamp"].dropna().max()) if "publication_timestamp" in ev_df.columns else None
    coll_min = (
        str(ev_df["collection_timestamp"].dropna().min())
        if "collection_timestamp" in ev_df.columns and ev_df["collection_timestamp"].notna().any()
        else None
    )
    coll_max = (
        str(ev_df["collection_timestamp"].dropna().max())
        if "collection_timestamp" in ev_df.columns and ev_df["collection_timestamp"].notna().any()
        else None
    )
    uniq_urls = int(ev_df["source_url"].nunique()) if "source_url" in ev_df.columns else 0
    dup_cnt = int(ev_df["is_duplicate_report"].sum()) if "is_duplicate_report" in ev_df.columns else 0

    after_cutoff = 0
    if not snaps_df.empty and "prediction_timestamp" in snaps_df.columns and "publication_timestamp" in ev_df.columns:
        max_pred_ts = pd.to_datetime(snaps_df["prediction_timestamp"], errors="coerce").max()
        pub_ts = pd.to_datetime(ev_df["publication_timestamp"], errors="coerce")
        if pd.notna(max_pred_ts):
            after_cutoff = int((pub_ts > max_pred_ts).sum())

    # Honest provenance check: if collection_timestamp is missing on rows or pub_min predates first snapshot
    has_coll = "collection_timestamp" in ev_df.columns and ev_df["collection_timestamp"].notna().all()
    is_forward = bool(has_coll and not snaps_df.empty)
    if not has_coll:
        note = (
            f"Pre-baseline RSS archive ({pub_min[:10] if pub_min else 'N/A'} to {pub_max[:10] if pub_max else 'N/A'}) "
            "collected retroactively before Phase 5.3 snapshot logging began; only newly collected events with "
            "collection_timestamp >= prediction_timestamp are prospective."
        )
    else:
        note = "All stored events carry verified collection_timestamp and publication_timestamp provenance."

    return NewsProvenanceSummary(
        total_events=total_cnt,
        valid_events=valid_cnt,
        quarantined_events=quar_cnt,
        unique_source_urls=uniq_urls,
        duplicate_reports_count=dup_cnt,
        publication_date_min=pub_min,
        publication_date_max=pub_max,
        collection_date_min=coll_min,
        collection_date_max=coll_max,
        events_after_prediction_cutoff=after_cutoff,
        is_genuinely_forward_collected=is_forward,
        provenance_note=note,
    )


def build_forward_validation_payload(
    store: DuckDBStore | None = None,
    equities_df: pd.DataFrame | None = None,
    indices_df: pd.DataFrame | None = None,
    persist: bool = True,
) -> ForwardValidationPayload:
    """Assemble the complete Phase 5.3.1 Forward Validation & Paper Portfolio payload."""
    st = store or DuckDBStore()
    now_ist = datetime.now(IST_TZ)
    dq_warnings: list[str] = []
    alerts: list[str] = []

    snaps_df = pd.read_parquet(st.forward_predictions_parquet) if st.forward_predictions_parquet.exists() else pd.DataFrame()
    outs_df = pd.read_parquet(st.forward_outcomes_parquet) if st.forward_outcomes_parquet.exists() else pd.DataFrame()
    provenance = compute_news_provenance(st, snaps_df)

    if not provenance.is_genuinely_forward_collected and provenance.valid_events > 0:
        dq_warnings.append(provenance.provenance_note)

    last_news_at = provenance.collection_date_max or provenance.publication_date_max
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

    if snaps_df.empty:
        dq_warnings.append("forward_predictions.parquet has 0 immutable snapshots; run daily pipeline to record snapshots.")

    horizon_evals: dict[str, ForwardHorizonEvaluation] = {}
    for h in ("1D", "5D", "10D"):
        he = compute_forward_horizon_metrics(snaps_df, outs_df, h)
        horizon_evals[h] = he
        if he.status == "WARNING":
            alerts.append(f"Forward {h} model accuracy/calibration warning (Acc={he.directional_accuracy_pct}%, Brier={he.brier_score}).")

    if equities_df is None and st.equities_parquet.exists():
        equities_df = pd.read_parquet(st.equities_parquet)
    if indices_df is None and st.indices_parquet.exists():
        indices_df = pd.read_parquet(st.indices_parquet)

    paper_report = run_paper_research_portfolio(snaps_df, outs_df, indices_df=indices_df, equities_df=equities_df)
    if persist and paper_report.recent_trades:
        st.upsert_paper_trades(paper_report.recent_trades)

    recent_snaps = [ForwardPredictionSnapshot(**r) for r in snaps_df.tail(25).to_dict(orient="records")] if not snaps_df.empty else []
    recent_outs = [ForwardOutcomeRecord(**r) for r in outs_df.tail(50).to_dict(orient="records")] if not outs_df.empty else []

    matured_total = int((outs_df["status"] == "MATURED").sum()) if not outs_df.empty else 0
    pending_total = int((outs_df["status"] == "PENDING").sum()) if not outs_df.empty else 0

    payload = ForwardValidationPayload(
        generated_at=now_ist.isoformat(),
        pilot_universe=list(PILOT_UNIVERSE_5),
        total_real_news_events=provenance.total_events,
        valid_news_events=provenance.valid_events,
        quarantined_news_events=provenance.quarantined_events,
        last_news_collection_at=last_news_at,
        total_immutable_snapshots=len(snaps_df),
        matured_outcomes_total=matured_total,
        pending_outcomes_total=pending_total,
        horizon_evaluations=horizon_evals,
        paper_portfolio=paper_report,
        news_provenance=provenance,
        recent_snapshots=recent_snaps,
        recent_outcomes=recent_outs,
        data_quality_warnings=dq_warnings,
        monitoring_alerts=alerts,
    )

    if persist:
        out_path = Path(st.predictions_dir) / "forward_validation.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload.model_dump(), indent=2), encoding="utf-8")

    return payload
