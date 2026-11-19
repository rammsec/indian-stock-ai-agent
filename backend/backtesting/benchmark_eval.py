"""Reproducible Benchmark & Statistical Evaluation Engine (Read-Only Consumer).

Completely decoupled from frozen prediction logic (`backend/prediction/`,
`backend/features/`, `backend/news_intel/`). Evaluates canonical forward
predictions against horizon-aligned, cost-matched benchmarks:
  - `bharat_alpha`: Canonical BharatAlpha multi-horizon predictions
  - `naive_momentum`: Horizon-matched naive momentum baseline (`benchmark_prob_{h}_up`)
  - `equal_weight_universe`: Equal-weight pilot universe over identical `[entry_date, exit_date]`
  - `nifty_50_buy_hold`: Nifty 50 index (`NIFTY_50` / `NIFTY 50`) over identical `[entry_date, exit_date]`
  - `cash_no_trade`: Risk-free 0% cash baseline

Key integrity guarantees:
  1. Strict look-ahead exclusion: `entry_date` must be strictly after `observation_date`
     and `prediction_timestamp` date (`next_open` at `t+1`, exit at `close[t+H]`).
  2. Daily `(symbol, observation_date)` deduplication: prevents same-day session
     (`PRE_MARKET` / `INTRADAY` / `POST_MARKET`) double-counting.
  3. `NIFTY_50` / `NIFTY 50` normalization so live `indices_daily.parquet` matches.
  4. Daily-portfolio Sharpe & compounding (never sequential stock-level compounding).
  5. Wilson score & bootstrap 95% confidence intervals + strict `INSUFFICIENT_DATA`
     gating when `matured_count < min_samples` or `unique_dates < min_unique_dates`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

DEFAULT_CNC_ROUNDTRIP_COST_PCT = 0.347
MIN_EVAL_OBSERVATIONS = 30
MIN_EVAL_UNIQUE_DATES = 10

_SESSION_PRIORITY = {"POST_MARKET": 3, "INTRADAY": 2, "PRE_MARKET": 1}


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float | None
    high: float | None
    method: str
    confidence_level: float = 0.95


@dataclass
class StrategyBenchmarkMetrics:
    strategy_name: str
    horizon: str
    status: str  # "OK" | "WARNING" | "INSUFFICIENT_DATA"
    insufficiency_reason: str | None = None
    matured_observations: int = 0
    pending_observations: int = 0
    unique_observation_dates: int = 0
    unique_symbols: int = 0
    long_signals_count: int = 0
    buy_recommendations_count: int = 0
    directional_accuracy_pct: float | None = None
    directional_accuracy_ci95: ConfidenceInterval | None = None
    roc_auc: float | None = None
    roc_auc_ci95: ConfidenceInterval | None = None
    brier_score: float | None = None
    brier_score_ci95: ConfidenceInterval | None = None
    rank_ic: float | None = None
    rank_ic_ci95: ConfidenceInterval | None = None
    gross_return_pct: float | None = None
    net_return_pct: float | None = None
    daily_portfolio_sharpe: float | None = None
    max_drawdown_pct: float | None = None
    win_rate_pct: float | None = None
    win_rate_ci95: ConfidenceInterval | None = None
    profit_factor: float | None = None


@dataclass
class HorizonBenchmarkReport:
    horizon: str
    min_observations_required: int
    min_unique_dates_required: int
    matured_observations: int
    pending_observations: int
    unique_observation_dates: int
    unique_symbols: int
    entry_window_start: str | None
    entry_window_end: str | None
    exit_window_end: str | None
    cost_assumption_pct: float
    status: str
    strategies: dict[str, StrategyBenchmarkMetrics] = field(default_factory=dict)


def _normalize_index_code(val: Any) -> str:
    return str(val or "").strip().upper().replace(" ", "_")


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> ConfidenceInterval:
    """Compute Wilson score 95% confidence interval for a binomial proportion (in %)."""
    if total <= 0:
        return ConfidenceInterval(low=None, high=None, method="wilson_95")
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return ConfidenceInterval(
        low=round(max(0.0, center - half) * 100.0, 2),
        high=round(min(1.0, center + half) * 100.0, 2),
        method="wilson_95",
    )


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Any,
    n_boot: int = 500,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute non-parametric percentile bootstrap 95% CI."""
    n = len(values)
    if n < 5:
        return ConfidenceInterval(low=None, high=None, method="bootstrap_95")
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            val = stat_fn(idx)
            if val is not None and not np.isnan(val):
                stats.append(float(val))
        except Exception:
            continue
    if len(stats) < 20:
        return ConfidenceInterval(low=None, high=None, method="bootstrap_95")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return ConfidenceInterval(low=round(float(lo), 4), high=round(float(hi), 4), method="bootstrap_95")


def deduplicate_daily_canonical_snapshots(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly one canonical snapshot per (symbol, observation_date).

    Excludes `is_duplicate_run == True` first, then resolves multiple NSE sessions
    on the same date deterministically (`POST_MARKET` > `INTRADAY` > `PRE_MARKET`,
    tie-broken by latest `prediction_timestamp`) to prevent same-day double-counting.
    """
    if snapshots_df.empty:
        return snapshots_df.copy()

    df = snapshots_df.copy()
    if "is_duplicate_run" in df.columns:
        df = df[~df["is_duplicate_run"].fillna(False).astype(bool)].copy()
    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["observation_date"] = df["observation_date"].astype(str).str.strip()
    if "nse_session" in df.columns:
        df["_sess_prio"] = df["nse_session"].astype(str).str.upper().map(_SESSION_PRIORITY).fillna(0)
    else:
        df["_sess_prio"] = 0
    ts_col = df["prediction_timestamp"].astype(str) if "prediction_timestamp" in df.columns else ""
    df["_ts_sort"] = ts_col

    df = (
        df.sort_values(["observation_date", "symbol", "_sess_prio", "_ts_sort", "prediction_id"])
        .drop_duplicates(subset=["symbol", "observation_date"], keep="last")
        .drop(columns=["_sess_prio", "_ts_sort"])
        .reset_index(drop=True)
    )
    return df


def _compute_ca_adj_factor(grp: pd.DataFrame, entry_idx: int, exit_idx: int) -> float:
    if "adj_close" not in grp.columns:
        return 1.0
    ec = grp.loc[entry_idx, "close"]
    xc = grp.loc[exit_idx, "close"]
    ac_in = grp.loc[entry_idx, "adj_close"]
    ac_out = grp.loc[exit_idx, "adj_close"]
    if pd.notna(ac_in) and pd.notna(ac_out) and pd.notna(ec) and pd.notna(xc):
        if float(ac_in) > 0 and float(ec) > 0 and float(xc) > 0:
            r_in = float(ac_in) / float(ec)
            r_out = float(ac_out) / float(xc)
            if r_in > 0 and abs((r_out / r_in) - 1.0) > 1e-4:
                return float(r_out / r_in)
    return 1.0


def build_aligned_outcomes(
    canonical_snaps_df: pd.DataFrame,
    equities_df: pd.DataFrame,
    indices_df: pd.DataFrame | None = None,
    cost_pct: float = DEFAULT_CNC_ROUNDTRIP_COST_PCT,
) -> pd.DataFrame:
    """Attach strictly look-ahead-free outcomes and window-aligned NIFTY_50 returns.

    Entry occurs at `open[t+1]` on the first NSE session strictly after
    `max(observation_date, prediction_timestamp[:10])`. Exit occurs at `close[t+H]`.
    """
    if canonical_snaps_df.empty or equities_df.empty:
        return pd.DataFrame()

    eq = equities_df.copy()
    eq["symbol"] = eq["symbol"].astype(str).str.upper().str.strip()
    eq["observation_date"] = eq["observation_date"].astype(str).str.strip()
    eq = eq.sort_values(["symbol", "observation_date"]).drop_duplicates(subset=["symbol", "observation_date"], keep="last")
    sym_map = {str(s): grp.reset_index(drop=True) for s, grp in eq.groupby("symbol", sort=False)}

    n50_map: dict[str, tuple[float | None, float | None]] = {}
    if indices_df is not None and not indices_df.empty and "index_code" in indices_df.columns:
        idx_norm = indices_df["index_code"].apply(_normalize_index_code)
        n50 = indices_df[idx_norm == "NIFTY_50"].copy()
        n50["observation_date"] = n50["observation_date"].astype(str).str.strip()
        n50 = n50.sort_values("observation_date").drop_duplicates(subset=["observation_date"], keep="last")
        for _, r in n50.iterrows():
            op = float(r["open"]) if "open" in n50.columns and pd.notna(r.get("open")) else None
            cl = float(r["close"]) if "close" in n50.columns and pd.notna(r.get("close")) else None
            n50_map[str(r["observation_date"])] = (op, cl)

    rows: list[dict[str, Any]] = []
    for _, snap in canonical_snaps_df.iterrows():
        pred_id = str(snap["prediction_id"])
        sym = str(snap["symbol"]).upper().strip()
        obs_dt = str(snap["observation_date"]).strip()
        pred_ts = str(snap.get("prediction_timestamp") or "")
        pred_dt = pred_ts[:10] if len(pred_ts) >= 10 else obs_dt
        cutoff_dt = max(obs_dt, pred_dt)

        grp = sym_map.get(sym)
        entry_idx: int | None = None
        if grp is not None and not grp.empty:
            after_mask = grp["observation_date"] > cutoff_dt
            if after_mask.any():
                entry_idx = int(grp.index[after_mask][0])

        for horizon, hold_n in (("1D", 1), ("5D", 5), ("10D", 10)):
            prob_col = {"1D": "prob_1d_up", "5D": "prob_5d_up", "10D": "prob_10d_up"}[horizon]
            rec_col = {"1D": "recommendation_1d", "5D": "recommendation_5d", "10D": "recommendation_10d"}[horizon]
            bench_col = {"1D": "benchmark_prob_1d_up", "5D": "benchmark_prob_5d_up", "10D": "benchmark_prob_10d_up"}[horizon]

            p_model = snap.get(prob_col)
            rec_model = str(snap.get(rec_col) or "UNAVAILABLE")
            p_bench = snap.get(bench_col)
            if (p_bench is None or pd.isna(p_bench)) and horizon == "1D":
                p_bench = snap.get("benchmark_prob_up")

            base_rec: dict[str, Any] = {
                "prediction_id": pred_id,
                "symbol": sym,
                "observation_date": obs_dt,
                "horizon": horizon,
                "model_prob_up": float(p_model) if pd.notna(p_model) else np.nan,
                "model_recommendation": rec_model,
                "bench_prob_up": float(p_bench) if pd.notna(p_bench) else np.nan,
                "status": "PENDING",
                "entry_date": None,
                "exit_date": None,
                "gross_return_pct": np.nan,
                "net_return_pct": np.nan,
                "nifty_gross_return_pct": np.nan,
                "nifty_net_return_pct": np.nan,
            }

            if grp is None or grp.empty or entry_idx is None:
                rows.append(base_rec)
                continue

            exit_idx = entry_idx + (hold_n - 1)
            if exit_idx >= len(grp):
                rows.append(base_rec)
                continue

            entry_dt = str(grp.loc[entry_idx, "observation_date"])
            exit_dt = str(grp.loc[exit_idx, "observation_date"])
            entry_open = grp.loc[entry_idx, "open"] if "open" in grp.columns else np.nan
            exit_close = grp.loc[exit_idx, "close"] if "close" in grp.columns else np.nan

            if pd.isna(entry_open) or float(entry_open) <= 0 or pd.isna(exit_close) or float(exit_close) <= 0:
                base_rec["status"] = "INVALID"
                rows.append(base_rec)
                continue

            adj = _compute_ca_adj_factor(grp, entry_idx, exit_idx)
            gross_ret = ((float(exit_close) * adj) / float(entry_open) - 1.0) * 100.0
            net_ret = gross_ret - cost_pct

            n_gross = np.nan
            n_net = np.nan
            if entry_dt in n50_map and exit_dt in n50_map:
                n_op, n_cl_in = n50_map[entry_dt]
                _, n_cl_out = n50_map[exit_dt]
                n_in = n_op if (n_op is not None and n_op > 0) else n_cl_in
                if n_in is not None and n_in > 0 and n_cl_out is not None and n_cl_out > 0:
                    n_gross = ((n_cl_out / n_in) - 1.0) * 100.0
                    n_net = n_gross - cost_pct

            base_rec.update(
                {
                    "status": "MATURED",
                    "entry_date": entry_dt,
                    "exit_date": exit_dt,
                    "gross_return_pct": float(gross_ret),
                    "net_return_pct": float(net_ret),
                    "nifty_gross_return_pct": float(n_gross) if pd.notna(n_gross) else np.nan,
                    "nifty_net_return_pct": float(n_net) if pd.notna(n_net) else np.nan,
                }
            )
            rows.append(base_rec)

    return pd.DataFrame(rows)


def _evaluate_prob_strategy(
    name: str,
    horizon: str,
    matured_df: pd.DataFrame,
    pending_cnt: int,
    prob_col: str,
    rec_col: str | None,
    min_samples: int,
    min_unique_dates: int,
) -> StrategyBenchmarkMetrics:
    sub = matured_df.dropna(subset=[prob_col, "gross_return_pct", "net_return_pct"]).copy()
    n_obs = len(sub)
    n_dates = int(sub["observation_date"].nunique()) if n_obs > 0 else 0
    n_syms = int(sub["symbol"].nunique()) if n_obs > 0 else 0

    if n_obs < min_samples or n_dates < min_unique_dates:
        reason = (
            f"Matured observations={n_obs} (min {min_samples}), "
            f"unique_dates={n_dates} (min {min_unique_dates})."
        )
        return StrategyBenchmarkMetrics(
            strategy_name=name,
            horizon=horizon,
            status="INSUFFICIENT_DATA",
            insufficiency_reason=reason,
            matured_observations=n_obs,
            pending_observations=pending_cnt,
            unique_observation_dates=n_dates,
            unique_symbols=n_syms,
        )

    y_gross = pd.to_numeric(sub["gross_return_pct"], errors="coerce").values
    y_net = pd.to_numeric(sub["net_return_pct"], errors="coerce").values
    y_up = (y_gross > 0).astype(int)
    p_up = pd.to_numeric(sub[prob_col], errors="coerce").clip(0.01, 0.99).values
    pred_up = (p_up >= 0.50).astype(int)

    correct = int((pred_up == y_up).sum())
    acc = round(correct / n_obs * 100.0, 2)
    acc_ci = wilson_ci(correct, n_obs)

    brier = round(float(brier_score_loss(y_up, p_up)), 4)
    brier_ci = bootstrap_ci(y_up, lambda idx: brier_score_loss(y_up[idx], p_up[idx]))

    auc: float | None = None
    auc_ci: ConfidenceInterval | None = None
    if len(set(y_up.tolist())) > 1:
        auc = round(float(roc_auc_score(y_up, p_up)), 4)
        auc_ci = bootstrap_ci(
            y_up,
            lambda idx: roc_auc_score(y_up[idx], p_up[idx]) if len(set(y_up[idx].tolist())) > 1 else np.nan,
        )

    ic_raw = pd.Series(p_up).rank().corr(pd.Series(y_gross).rank())
    rank_ic = round(float(ic_raw), 4) if pd.notna(ic_raw) else None
    ic_ci = bootstrap_ci(
        y_gross,
        lambda idx: pd.Series(p_up[idx]).rank().corr(pd.Series(y_gross[idx]).rank()),
    )

    long_mask = p_up >= 0.50
    long_cnt = int(long_mask.sum())
    buy_cnt = int((sub[rec_col] == "BUY").sum()) if (rec_col and rec_col in sub.columns) else long_cnt

    # Daily-portfolio aggregation (equal-weight across active long signals per entry_date)
    sub_work = sub.copy()
    sub_work["_is_long"] = long_mask
    daily_gross: list[float] = []
    daily_net: list[float] = []
    for _, grp in sub_work.groupby("entry_date", sort=True):
        act = grp[grp["_is_long"]]
        if len(act) == 0:
            daily_gross.append(0.0)
            daily_net.append(0.0)
        else:
            daily_gross.append(float(act["gross_return_pct"].mean()))
            daily_net.append(float(act["net_return_pct"].mean()))

    dg_arr = np.array(daily_gross, dtype=float)
    dn_arr = np.array(daily_net, dtype=float)
    gross_cum = round(float((np.prod(1.0 + dg_arr / 100.0) - 1.0) * 100.0), 2)
    net_cum = round(float((np.prod(1.0 + dn_arr / 100.0) - 1.0) * 100.0), 2)

    eq_curve = np.cumprod(1.0 + dn_arr / 100.0)
    peak = np.maximum.accumulate(eq_curve)
    max_dd = round(float(((eq_curve / np.maximum(peak, 1e-9)) - 1.0).min() * 100.0), 2)

    std_d = float(np.std(dn_arr, ddof=1)) if len(dn_arr) > 1 else 0.0
    sharpe = round(float((np.mean(dn_arr) / std_d) * math.sqrt(252.0)), 3) if std_d > 1e-9 else 0.0

    trade_rets = y_net[long_mask] if long_cnt > 0 else np.array([], dtype=float)
    if len(trade_rets) > 0:
        wins_cnt = int((trade_rets > 0).sum())
        win_rate = round(wins_cnt / len(trade_rets) * 100.0, 2)
        win_ci = wilson_ci(wins_cnt, len(trade_rets))
        gross_w = float(trade_rets[trade_rets > 0].sum())
        gross_l = float(abs(trade_rets[trade_rets < 0].sum()))
        pf = round(gross_w / gross_l, 3) if gross_l > 1e-9 else (2.5 if gross_w > 0 else 0.0)
    else:
        win_rate = None
        win_ci = None
        pf = None

    status = "WARNING" if (acc < 48.0 or brier > 0.27) else "OK"
    return StrategyBenchmarkMetrics(
        strategy_name=name,
        horizon=horizon,
        status=status,
        matured_observations=n_obs,
        pending_observations=pending_cnt,
        unique_observation_dates=n_dates,
        unique_symbols=n_syms,
        long_signals_count=long_cnt,
        buy_recommendations_count=buy_cnt,
        directional_accuracy_pct=acc,
        directional_accuracy_ci95=acc_ci,
        roc_auc=auc,
        roc_auc_ci95=auc_ci,
        brier_score=brier,
        brier_score_ci95=brier_ci,
        rank_ic=rank_ic,
        rank_ic_ci95=ic_ci,
        gross_return_pct=gross_cum,
        net_return_pct=net_cum,
        daily_portfolio_sharpe=sharpe,
        max_drawdown_pct=max_dd,
        win_rate_pct=win_rate,
        win_rate_ci95=win_ci,
        profit_factor=pf,
    )


def _evaluate_passive_benchmark(
    name: str,
    horizon: str,
    matured_df: pd.DataFrame,
    pending_cnt: int,
    gross_col: str,
    net_col: str,
    min_samples: int,
    min_unique_dates: int,
) -> StrategyBenchmarkMetrics:
    sub = matured_df.dropna(subset=[gross_col, net_col]).copy()
    n_obs = len(sub)
    n_dates = int(sub["observation_date"].nunique()) if n_obs > 0 else 0
    n_syms = int(sub["symbol"].nunique()) if n_obs > 0 else 0

    if n_obs < min_samples or n_dates < min_unique_dates:
        return StrategyBenchmarkMetrics(
            strategy_name=name,
            horizon=horizon,
            status="INSUFFICIENT_DATA",
            insufficiency_reason=(
                f"Matured observations={n_obs} (min {min_samples}), "
                f"unique_dates={n_dates} (min {min_unique_dates})."
            ),
            matured_observations=n_obs,
            pending_observations=pending_cnt,
            unique_observation_dates=n_dates,
            unique_symbols=n_syms,
        )

    daily = sub.groupby("entry_date", sort=True)[[gross_col, net_col]].mean()
    dg = daily[gross_col].values
    dn = daily[net_col].values

    gross_cum = round(float((np.prod(1.0 + dg / 100.0) - 1.0) * 100.0), 2)
    net_cum = round(float((np.prod(1.0 + dn / 100.0) - 1.0) * 100.0), 2)
    eq_curve = np.cumprod(1.0 + dn / 100.0)
    peak = np.maximum.accumulate(eq_curve)
    max_dd = round(float(((eq_curve / np.maximum(peak, 1e-9)) - 1.0).min() * 100.0), 2)
    std_d = float(np.std(dn, ddof=1)) if len(dn) > 1 else 0.0
    sharpe = round(float((np.mean(dn) / std_d) * math.sqrt(252.0)), 3) if std_d > 1e-9 else 0.0
    wins_cnt = int((dn > 0).sum())
    win_rate = round(wins_cnt / len(dn) * 100.0, 2)
    gross_w = float(dn[dn > 0].sum())
    gross_l = float(abs(dn[dn < 0].sum()))
    pf = round(gross_w / gross_l, 3) if gross_l > 1e-9 else (2.5 if gross_w > 0 else 0.0)

    return StrategyBenchmarkMetrics(
        strategy_name=name,
        horizon=horizon,
        status="OK",
        matured_observations=n_obs,
        pending_observations=pending_cnt,
        unique_observation_dates=n_dates,
        unique_symbols=n_syms,
        long_signals_count=n_obs,
        buy_recommendations_count=n_obs,
        gross_return_pct=gross_cum,
        net_return_pct=net_cum,
        daily_portfolio_sharpe=sharpe,
        max_drawdown_pct=max_dd,
        win_rate_pct=win_rate,
        win_rate_ci95=wilson_ci(wins_cnt, len(dn)),
        profit_factor=pf,
    )


def run_benchmark_comparison(
    snapshots_df: pd.DataFrame,
    equities_df: pd.DataFrame,
    indices_df: pd.DataFrame | None = None,
    min_samples: int = MIN_EVAL_OBSERVATIONS,
    min_unique_dates: int = MIN_EVAL_UNIQUE_DATES,
    cost_pct: float = DEFAULT_CNC_ROUNDTRIP_COST_PCT,
) -> dict[str, Any]:
    """Run full horizon-aligned benchmark & statistical evaluation."""
    canonical_df = deduplicate_daily_canonical_snapshots(snapshots_df)
    aligned_df = build_aligned_outcomes(canonical_df, equities_df, indices_df=indices_df, cost_pct=cost_pct)

    horizons_out: dict[str, Any] = {}
    for horizon in ("1D", "5D", "10D"):
        h_df = aligned_df[aligned_df["horizon"] == horizon].copy() if not aligned_df.empty else pd.DataFrame()
        matured = h_df[h_df["status"] == "MATURED"].copy() if not h_df.empty else pd.DataFrame()
        pending_cnt = int((h_df["status"] == "PENDING").sum()) if not h_df.empty else 0

        n_mat = len(matured)
        n_dates = int(matured["observation_date"].nunique()) if n_mat > 0 else 0
        n_syms = int(matured["symbol"].nunique()) if n_mat > 0 else 0

        strategies: dict[str, StrategyBenchmarkMetrics] = {
            "bharat_alpha": _evaluate_prob_strategy(
                "bharat_alpha", horizon, matured, pending_cnt, "model_prob_up", "model_recommendation", min_samples, min_unique_dates
            ),
            "naive_momentum": _evaluate_prob_strategy(
                "naive_momentum", horizon, matured, pending_cnt, "bench_prob_up", None, min_samples, min_unique_dates
            ),
            "equal_weight_universe": _evaluate_passive_benchmark(
                "equal_weight_universe", horizon, matured, pending_cnt, "gross_return_pct", "net_return_pct", min_samples, min_unique_dates
            ),
            "nifty_50_buy_hold": _evaluate_passive_benchmark(
                "nifty_50_buy_hold", horizon, matured, pending_cnt, "nifty_gross_return_pct", "nifty_net_return_pct", min_samples, min_unique_dates
            ),
            "cash_no_trade": StrategyBenchmarkMetrics(
                strategy_name="cash_no_trade",
                horizon=horizon,
                status="OK",
                matured_observations=n_mat,
                pending_observations=pending_cnt,
                unique_observation_dates=n_dates,
                unique_symbols=n_syms,
                gross_return_pct=0.0,
                net_return_pct=0.0,
                daily_portfolio_sharpe=0.0,
                max_drawdown_pct=0.0,
            ),
        }

        h_status = "INSUFFICIENT_DATA" if (n_mat < min_samples or n_dates < min_unique_dates) else strategies["bharat_alpha"].status
        report = HorizonBenchmarkReport(
            horizon=horizon,
            min_observations_required=min_samples,
            min_unique_dates_required=min_unique_dates,
            matured_observations=n_mat,
            pending_observations=pending_cnt,
            unique_observation_dates=n_dates,
            unique_symbols=n_syms,
            entry_window_start=str(matured["entry_date"].min()) if n_mat > 0 else None,
            entry_window_end=str(matured["entry_date"].max()) if n_mat > 0 else None,
            exit_window_end=str(matured["exit_date"].max()) if n_mat > 0 else None,
            cost_assumption_pct=cost_pct,
            status=h_status,
            strategies=strategies,
        )
        horizons_out[horizon] = asdict(report)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_version": "1.0.0",
        "cost_assumption_pct": cost_pct,
        "min_observations_required": min_samples,
        "min_unique_dates_required": min_unique_dates,
        "raw_snapshots_count": int(len(snapshots_df)),
        "canonical_daily_snapshots_count": int(len(canonical_df)),
        "horizons": horizons_out,
    }


def main(base_dir: Path | None = None) -> dict[str, Any]:
    root = base_dir or Path(__file__).resolve().parents[2]
    snaps_p = root / "data" / "predictions" / "forward_predictions.parquet"
    eq_p = root / "data" / "processed" / "equities_daily.parquet"
    idx_p = root / "data" / "processed" / "indices_daily.parquet"

    snaps_df = pd.read_parquet(snaps_p) if snaps_p.exists() else pd.DataFrame()
    eq_df = pd.read_parquet(eq_p) if eq_p.exists() else pd.DataFrame()
    idx_df = pd.read_parquet(idx_p) if idx_p.exists() else pd.DataFrame()

    result = run_benchmark_comparison(snaps_df, eq_df, indices_df=idx_df)
    out_path = root / "data" / "backtests" / "benchmark_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
