"""Remediated Prediction Chronology & Benchmark Engine (Read-Only Consumer).

Completely decoupled from frozen prediction logic (`backend/prediction/`,
`backend/features/`, `backend/news_intel/`). Never mutates original `data/` files.

Key guarantees:
  1. Strict prediction-before-entry enforcement:
     `entry_open_ts > prediction_timestamp`, where NSE session open is `09:15 IST`.
  2. Session-aware next eligible NSE session execution:
     - `PRE_MARKET` (`< 09:15 IST` on trading day T) -> eligible at `T 09:15 IST`
     - `INTRADAY` (`09:15-15:30 IST` on T) -> eligible at `T_next 09:15 IST`
     - `POST_MARKET` (`> 15:30 IST` on T) -> eligible at `T_next 09:15 IST`
     - Holidays / weekends -> next NSE trading day `09:15 IST`
  3. Centralised `normalize_benchmark_symbol()` (`NIFTY 50` / `NIFTY_50` -> `NIFTY_50`)
     and independent `1D`, `5D`, `10D` benchmark calculations (`NIFTY_50`,
     `naive_momentum`, `equal_weight`, `cash_no_trade`).
  4. Trading-calendar-aware outcome maturity + explicit `UNAVAILABLE` handling for
     missing/non-positive prices and missing dates; no metric reported without
     verified observation count and valid benchmark comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from backend.config import IST_TZ, NSE_TRADING_HOLIDAYS

DEFAULT_CNC_ROUNDTRIP_COST_PCT = 0.347
MIN_EVAL_OBSERVATIONS = 30
MIN_EVAL_UNIQUE_DATES = 10

NSE_OPEN_TIME = time(9, 15, 0)
NSE_CLOSE_TIME = time(15, 30, 0)

_SESSION_PRIORITY = {"POST_MARKET": 3, "INTRADAY": 2, "PRE_MARKET": 1}

_BENCHMARK_ALIASES: dict[str, str] = {
    "NIFTY 50": "NIFTY_50",
    "NIFTY_50": "NIFTY_50",
    "NIFTY50": "NIFTY_50",
    "^NSEI": "NIFTY_50",
    "NIFTY 500": "NIFTY_500",
    "NIFTY_500": "NIFTY_500",
    "NIFTY500": "NIFTY_500",
    "NIFTY BANK": "NIFTY_BANK",
    "NIFTY_BANK": "NIFTY_BANK",
    "BANKNIFTY": "NIFTY_BANK",
}


def normalize_benchmark_symbol(symbol: Any) -> str:
    """Centralised benchmark symbol normalizer (e.g. `NIFTY 50` / `NIFTY_50` -> `NIFTY_50`)."""
    raw = str(symbol or "").strip().upper()
    if not raw or raw in ("NAN", "NONE", "NAT"):
        return ""
    if raw in _BENCHMARK_ALIASES:
        return _BENCHMARK_ALIASES[raw]
    collapsed = re.sub(r"[\s\-]+", "_", raw)
    return _BENCHMARK_ALIASES.get(collapsed, collapsed)


def parse_ist_timestamp(val: Any) -> datetime | None:
    """Parse an ISO timestamp into Asia/Kolkata timezone, returning None if invalid."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST_TZ)
        return dt.astimezone(IST_TZ)
    except Exception:
        return None


def parse_trade_date(val: Any) -> date | None:
    """Parse YYYY-MM-DD (or ISO timestamp prefix) into a `date`, or None if invalid."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()[:10]
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def is_nse_trading_day_cal(d: date, holidays: set[str] | None = None) -> bool:
    """Return True if `d` is a weekday and not in the NSE trading holiday set."""
    if d.weekday() >= 5:
        return False
    hols = NSE_TRADING_HOLIDAYS if holidays is None else holidays
    return d.strftime("%Y-%m-%d") not in hols


def next_nse_trading_day(d: date, holidays: set[str] | None = None) -> date:
    """Return the first NSE trading day strictly after `d`."""
    cur = d + timedelta(days=1)
    while not is_nse_trading_day_cal(cur, holidays):
        cur += timedelta(days=1)
    return cur


def add_nse_trading_days(start_date: date, n_days: int, holidays: set[str] | None = None) -> date:
    """Advance `n_days` NSE trading sessions from `start_date`."""
    cur = start_date
    for _ in range(max(0, n_days)):
        cur = next_nse_trading_day(cur, holidays)
    return cur


def classify_nse_session(ts_ist: datetime) -> str:
    """Classify an IST datetime into PRE_MARKET (<09:15), INTRADAY (09:15-15:30), or POST_MARKET (>15:30)."""
    t = ts_ist.time()
    if t < NSE_OPEN_TIME:
        return "PRE_MARKET"
    if t <= NSE_CLOSE_TIME:
        return "INTRADAY"
    return "POST_MARKET"


def entry_open_timestamp_ist(entry_date: date) -> datetime:
    """Return the exact NSE session open timestamp (`09:15:00+05:30`) for `entry_date`."""
    return datetime.combine(entry_date, NSE_OPEN_TIME, tzinfo=IST_TZ)


def resolve_eligible_entry_session(
    prediction_timestamp: Any,
    observation_date: Any = None,
    holidays: set[str] | None = None,
) -> tuple[date | None, datetime | None, str]:
    """Compute the earliest eligible NSE entry date and `09:15 IST` open timestamp.

    Rules:
      - Unparseable/missing `prediction_timestamp` -> `(None, None, "UNAVAILABLE_TIMESTAMP")`
      - Unparseable `observation_date` (if supplied) -> `(None, None, "UNAVAILABLE_DATE")`
      - Holiday or weekend -> next NSE trading day `09:15 IST`
      - Trading day `T`, `PRE_MARKET` (`< 09:15 IST`) -> `T 09:15 IST`
      - Trading day `T`, `INTRADAY` (`09:15-15:30 IST`) or `POST_MARKET` (`> 15:30 IST`)
        -> `T_next 09:15 IST`
      - Strict invariant: `entry_open_ts > prediction_timestamp` is always enforced.
    """
    pred_dt = parse_ist_timestamp(prediction_timestamp)
    if pred_dt is None:
        return None, None, "UNAVAILABLE_TIMESTAMP"

    obs_d: date | None = None
    if observation_date is not None and not (isinstance(observation_date, float) and pd.isna(observation_date)):
        obs_str = str(observation_date).strip()
        if obs_str and obs_str.lower() not in ("nan", "nat", "none"):
            obs_d = parse_trade_date(obs_str)
            if obs_d is None:
                return None, None, "UNAVAILABLE_DATE"

    pred_d = pred_dt.date()
    session = classify_nse_session(pred_dt)

    if not is_nse_trading_day_cal(pred_d, holidays):
        eligible_d = next_nse_trading_day(pred_d, holidays)
    elif session == "PRE_MARKET":
        eligible_d = pred_d
    else:
        eligible_d = next_nse_trading_day(pred_d, holidays)

    if obs_d is not None and eligible_d < obs_d:
        eligible_d = obs_d if is_nse_trading_day_cal(obs_d, holidays) else next_nse_trading_day(obs_d, holidays)

    entry_ts = entry_open_timestamp_ist(eligible_d)
    while entry_ts <= pred_dt:
        eligible_d = next_nse_trading_day(eligible_d, holidays)
        entry_ts = entry_open_timestamp_ist(eligible_d)

    return eligible_d, entry_ts, session


def enforce_prediction_before_entry(
    prediction_timestamp: Any,
    entry_date: Any,
) -> tuple[bool, float | None]:
    """Verify strict `entry_open_ts > prediction_timestamp` invariant.

    Returns `(is_valid, lead_minutes)`. `lead_minutes > 0` indicates valid chronology;
    `<= 0` indicates a look-ahead violation.
    """
    pred_dt = parse_ist_timestamp(prediction_timestamp)
    entry_d = parse_trade_date(entry_date)
    if pred_dt is None or entry_d is None:
        return False, None
    entry_ts = entry_open_timestamp_ist(entry_d)
    delta_min = round((entry_ts - pred_dt).total_seconds() / 60.0, 4)
    return entry_ts > pred_dt, delta_min


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
    status: str  # "OK" | "WARNING" | "INSUFFICIENT_DATA" | "UNAVAILABLE"
    insufficiency_reason: str | None = None
    matured_observations: int = 0
    pending_observations: int = 0
    unavailable_observations: int = 0
    unique_observation_dates: int = 0
    unique_symbols: int = 0
    valid_benchmark_comparisons: int = 0
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
    unavailable_observations: int
    unique_observation_dates: int
    unique_symbols: int
    valid_nifty50_comparisons: int
    entry_window_start: str | None
    entry_window_end: str | None
    exit_window_end: str | None
    cost_assumption_pct: float
    status: str
    status_reason: str | None = None
    excess_return_vs_nifty50_pct: float | None = None
    excess_return_vs_equal_weight_pct: float | None = None
    excess_return_vs_cash_pct: float | None = None
    strategies: dict[str, StrategyBenchmarkMetrics] = field(default_factory=dict)


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
    """Select exactly one canonical snapshot per (symbol, observation_date)."""
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
    df["_ts_sort"] = df["prediction_timestamp"].astype(str) if "prediction_timestamp" in df.columns else ""

    sort_cols = ["observation_date", "symbol", "_sess_prio", "_ts_sort"]
    if "prediction_id" in df.columns:
        sort_cols.append("prediction_id")

    return (
        df.sort_values(sort_cols)
        .drop_duplicates(subset=["symbol", "observation_date"], keep="last")
        .drop(columns=["_sess_prio", "_ts_sort"])
        .reset_index(drop=True)
    )


def _is_valid_price(val: Any) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f) and f > 0.0
    except (TypeError, ValueError):
        return False


def _compute_ca_adj_factor(entry_row: pd.Series, exit_row: pd.Series) -> float:
    ac_in = entry_row.get("adj_close")
    ac_out = exit_row.get("adj_close")
    ec = entry_row.get("close")
    xc = exit_row.get("close")
    if _is_valid_price(ac_in) and _is_valid_price(ac_out) and _is_valid_price(ec) and _is_valid_price(xc):
        r_in = float(ac_in) / float(ec)
        r_out = float(ac_out) / float(xc)
        if r_in > 0 and abs((r_out / r_in) - 1.0) > 1e-4:
            return float(r_out / r_in)
    return 1.0


def build_chronology_aligned_outcomes(
    snapshots_df: pd.DataFrame,
    equities_df: pd.DataFrame,
    indices_df: pd.DataFrame | None = None,
    cost_pct: float = DEFAULT_CNC_ROUNDTRIP_COST_PCT,
    holidays: set[str] | None = None,
    as_of_date: date | str | None = None,
) -> pd.DataFrame:
    """Attach chronology-verified, trading-calendar-aware 1D/5D/10D outcomes and NIFTY_50 returns.

    - Enforces `entry_open_ts > prediction_timestamp` (where `entry_open_ts` is `09:15 IST`).
    - Resolves `PRE_MARKET`, `INTRADAY`, `POST_MARKET`, and holiday/weekend execution sessions.
    - Marks missing dates or missing/non-positive prices explicitly as `UNAVAILABLE`.
    - Marks future/unmatured horizons as `PENDING`.
    """
    if snapshots_df.empty:
        return pd.DataFrame()

    # Build equity lookup: (symbol, date_str) -> row
    eq_map: dict[tuple[str, str], pd.Series] = {}
    latest_market_date: date | None = None
    if equities_df is not None and not equities_df.empty:
        eq = equities_df.copy()
        eq["symbol"] = eq["symbol"].astype(str).str.upper().str.strip()
        eq["observation_date"] = eq["observation_date"].astype(str).str.strip().str[:10]
        eq = eq.sort_values(["symbol", "observation_date"]).drop_duplicates(
            subset=["symbol", "observation_date"], keep="last"
        )
        for _, r in eq.iterrows():
            d_parsed = parse_trade_date(r["observation_date"])
            if d_parsed is not None:
                eq_map[(str(r["symbol"]), d_parsed.isoformat())] = r
                if latest_market_date is None or d_parsed > latest_market_date:
                    latest_market_date = d_parsed

    ref_date = parse_trade_date(as_of_date) if as_of_date is not None else latest_market_date

    # Build normalised NIFTY_50 lookup: date_str -> (open, close)
    n50_map: dict[str, tuple[float | None, float | None]] = {}
    if indices_df is not None and not indices_df.empty and "index_code" in indices_df.columns:
        idx_norm = indices_df["index_code"].apply(normalize_benchmark_symbol)
        n50 = indices_df[idx_norm == "NIFTY_50"].copy()
        n50["observation_date"] = n50["observation_date"].astype(str).str.strip().str[:10]
        n50 = n50.sort_values("observation_date").drop_duplicates(subset=["observation_date"], keep="last")
        for _, r in n50.iterrows():
            op = float(r["open"]) if "open" in n50.columns and _is_valid_price(r.get("open")) else None
            cl = float(r["close"]) if "close" in n50.columns and _is_valid_price(r.get("close")) else None
            n50_map[str(r["observation_date"])] = (op, cl)

    rows: list[dict[str, Any]] = []
    for _, snap in snapshots_df.iterrows():
        pred_id = str(snap.get("prediction_id") or "")
        sym = str(snap.get("symbol") or "").upper().strip()
        obs_dt_raw = snap.get("observation_date")
        obs_dt_parsed = parse_trade_date(obs_dt_raw)
        obs_dt_str = obs_dt_parsed.isoformat() if obs_dt_parsed else str(obs_dt_raw or "").strip()
        pred_ts_raw = snap.get("prediction_timestamp")

        entry_d, entry_open_ts, resolved_session = resolve_eligible_entry_session(
            pred_ts_raw, observation_date=obs_dt_raw, holidays=holidays
        )

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
                "observation_date": obs_dt_str,
                "prediction_timestamp": str(pred_ts_raw or ""),
                "resolved_session": resolved_session,
                "horizon": horizon,
                "model_prob_up": float(p_model) if pd.notna(p_model) else np.nan,
                "model_recommendation": rec_model,
                "bench_prob_up": float(p_bench) if pd.notna(p_bench) else np.nan,
                "status": "UNAVAILABLE",
                "status_reason": None,
                "entry_date": None,
                "entry_open_timestamp": None,
                "exit_date": None,
                "chronology_valid": False,
                "lead_time_minutes": None,
                "gross_return_pct": np.nan,
                "net_return_pct": np.nan,
                "nifty_gross_return_pct": np.nan,
                "nifty_net_return_pct": np.nan,
                "nifty_status": "UNAVAILABLE",
            }

            if entry_d is None or entry_open_ts is None:
                base_rec["status"] = "UNAVAILABLE"
                base_rec["status_reason"] = f"Invalid chronology inputs ({resolved_session})."
                rows.append(base_rec)
                continue

            chrono_ok, lead_mins = enforce_prediction_before_entry(pred_ts_raw, entry_d)
            base_rec["chronology_valid"] = chrono_ok
            base_rec["lead_time_minutes"] = lead_mins
            if not chrono_ok:
                base_rec["status"] = "UNAVAILABLE"
                base_rec["status_reason"] = "Chronology violation: entry_open_ts <= prediction_timestamp."
                rows.append(base_rec)
                continue

            exit_d = add_nse_trading_days(entry_d, hold_n - 1, holidays)
            entry_str = entry_d.isoformat()
            exit_str = exit_d.isoformat()
            base_rec["entry_date"] = entry_str
            base_rec["entry_open_timestamp"] = entry_open_ts.isoformat()
            base_rec["exit_date"] = exit_str

            if ref_date is None or ref_date < exit_d:
                base_rec["status"] = "PENDING"
                base_rec["status_reason"] = f"Awaiting NSE trading session maturity (exit_date={exit_str})."
                rows.append(base_rec)
                continue

            entry_bar = eq_map.get((sym, entry_str))
            exit_bar = eq_map.get((sym, exit_str))
            if entry_bar is None or exit_bar is None:
                missing_parts = []
                if entry_bar is None:
                    missing_parts.append(f"entry_date={entry_str}")
                if exit_bar is None:
                    missing_parts.append(f"exit_date={exit_str}")
                base_rec["status"] = "UNAVAILABLE"
                base_rec["status_reason"] = f"Missing market bar for {sym}: {', '.join(missing_parts)}."
                rows.append(base_rec)
                continue

            entry_open = entry_bar.get("open")
            exit_close = exit_bar.get("close")
            if not _is_valid_price(entry_open) or not _is_valid_price(exit_close):
                base_rec["status"] = "UNAVAILABLE"
                base_rec["status_reason"] = f"Missing or non-positive price for {sym} on [{entry_str}, {exit_str}]."
                rows.append(base_rec)
                continue

            adj = _compute_ca_adj_factor(entry_bar, exit_bar)
            gross_ret = ((float(exit_close) * adj) / float(entry_open) - 1.0) * 100.0
            net_ret = gross_ret - cost_pct

            n_gross = np.nan
            n_net = np.nan
            n_status = "UNAVAILABLE"
            if entry_str in n50_map and exit_str in n50_map:
                n_op, n_cl_in = n50_map[entry_str]
                _, n_cl_out = n50_map[exit_str]
                n_in = n_op if (n_op is not None and n_op > 0) else n_cl_in
                if n_in is not None and n_in > 0 and n_cl_out is not None and n_cl_out > 0:
                    n_gross = ((n_cl_out / n_in) - 1.0) * 100.0
                    n_net = n_gross - cost_pct
                    n_status = "MATURED"

            base_rec.update(
                {
                    "status": "MATURED",
                    "status_reason": None,
                    "gross_return_pct": float(gross_ret),
                    "net_return_pct": float(net_ret),
                    "nifty_gross_return_pct": float(n_gross) if pd.notna(n_gross) else np.nan,
                    "nifty_net_return_pct": float(n_net) if pd.notna(n_net) else np.nan,
                    "nifty_status": n_status,
                }
            )
            rows.append(base_rec)

    return pd.DataFrame(rows)


def _insufficient_or_unavailable_metrics(
    name: str,
    horizon: str,
    status: str,
    reason: str,
    n_obs: int,
    pending_cnt: int,
    unavail_cnt: int,
    n_dates: int,
    n_syms: int,
    valid_bench_cnt: int,
) -> StrategyBenchmarkMetrics:
    return StrategyBenchmarkMetrics(
        strategy_name=name,
        horizon=horizon,
        status=status,
        insufficiency_reason=reason,
        matured_observations=n_obs,
        pending_observations=pending_cnt,
        unavailable_observations=unavail_cnt,
        unique_observation_dates=n_dates,
        unique_symbols=n_syms,
        valid_benchmark_comparisons=valid_bench_cnt,
    )


def _evaluate_prob_strategy(
    name: str,
    horizon: str,
    matured_df: pd.DataFrame,
    pending_cnt: int,
    unavail_cnt: int,
    prob_col: str,
    rec_col: str | None,
    min_samples: int,
    min_unique_dates: int,
    has_valid_benchmark: bool,
    valid_bench_cnt: int,
) -> StrategyBenchmarkMetrics:
    sub = matured_df.dropna(subset=[prob_col, "gross_return_pct", "net_return_pct"]).copy()
    n_obs = len(sub)
    n_dates = int(sub["observation_date"].nunique()) if n_obs > 0 else 0
    n_syms = int(sub["symbol"].nunique()) if n_obs > 0 else 0

    if n_obs < min_samples or n_dates < min_unique_dates:
        return _insufficient_or_unavailable_metrics(
            name,
            horizon,
            "INSUFFICIENT_DATA",
            f"Matured observations={n_obs} (min {min_samples}), unique_dates={n_dates} (min {min_unique_dates}).",
            n_obs,
            pending_cnt,
            unavail_cnt,
            n_dates,
            n_syms,
            valid_bench_cnt,
        )

    if not has_valid_benchmark:
        return _insufficient_or_unavailable_metrics(
            name,
            horizon,
            "UNAVAILABLE",
            f"Valid NIFTY_50 benchmark comparison unavailable (valid_comparisons={valid_bench_cnt}).",
            n_obs,
            pending_cnt,
            unavail_cnt,
            n_dates,
            n_syms,
            valid_bench_cnt,
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
        unavailable_observations=unavail_cnt,
        unique_observation_dates=n_dates,
        unique_symbols=n_syms,
        valid_benchmark_comparisons=valid_bench_cnt,
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
    unavail_cnt: int,
    gross_col: str,
    net_col: str,
    min_samples: int,
    min_unique_dates: int,
    has_valid_benchmark: bool,
    valid_bench_cnt: int,
) -> StrategyBenchmarkMetrics:
    sub = matured_df.dropna(subset=[gross_col, net_col]).copy()
    n_obs = len(sub)
    n_dates = int(sub["observation_date"].nunique()) if n_obs > 0 else 0
    n_syms = int(sub["symbol"].nunique()) if n_obs > 0 else 0

    if n_obs < min_samples or n_dates < min_unique_dates:
        status = "UNAVAILABLE" if (len(matured_df) >= min_samples and n_obs == 0) else "INSUFFICIENT_DATA"
        return _insufficient_or_unavailable_metrics(
            name,
            horizon,
            status,
            f"Valid observations={n_obs} (min {min_samples}), unique_dates={n_dates} (min {min_unique_dates}).",
            n_obs,
            pending_cnt,
            unavail_cnt,
            n_dates,
            n_syms,
            valid_bench_cnt,
        )

    if not has_valid_benchmark:
        return _insufficient_or_unavailable_metrics(
            name,
            horizon,
            "UNAVAILABLE",
            f"Valid NIFTY_50 benchmark comparison unavailable (valid_comparisons={valid_bench_cnt}).",
            n_obs,
            pending_cnt,
            unavail_cnt,
            n_dates,
            n_syms,
            valid_bench_cnt,
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
        unavailable_observations=unavail_cnt,
        unique_observation_dates=n_dates,
        unique_symbols=n_syms,
        valid_benchmark_comparisons=valid_bench_cnt,
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


def run_chronology_benchmark(
    snapshots_df: pd.DataFrame,
    equities_df: pd.DataFrame,
    indices_df: pd.DataFrame | None = None,
    min_samples: int = MIN_EVAL_OBSERVATIONS,
    min_unique_dates: int = MIN_EVAL_UNIQUE_DATES,
    cost_pct: float = DEFAULT_CNC_ROUNDTRIP_COST_PCT,
    holidays: set[str] | None = None,
    as_of_date: date | str | None = None,
    deduplicate_daily: bool = True,
) -> dict[str, Any]:
    """Run chronology-verified, horizon-independent (1D/5D/10D) benchmark evaluation."""
    work_snaps = deduplicate_daily_canonical_snapshots(snapshots_df) if deduplicate_daily else snapshots_df.copy()
    aligned_df = build_chronology_aligned_outcomes(
        work_snaps,
        equities_df,
        indices_df=indices_df,
        cost_pct=cost_pct,
        holidays=holidays,
        as_of_date=as_of_date,
    )

    horizons_out: dict[str, Any] = {}
    for horizon in ("1D", "5D", "10D"):
        h_df = aligned_df[aligned_df["horizon"] == horizon].copy() if not aligned_df.empty else pd.DataFrame()
        matured = h_df[h_df["status"] == "MATURED"].copy() if not h_df.empty else pd.DataFrame()
        pending_cnt = int((h_df["status"] == "PENDING").sum()) if not h_df.empty else 0
        unavail_cnt = int((h_df["status"] == "UNAVAILABLE").sum()) if not h_df.empty else 0

        n_mat = len(matured)
        n_dates = int(matured["observation_date"].nunique()) if n_mat > 0 else 0
        n_syms = int(matured["symbol"].nunique()) if n_mat > 0 else 0

        nifty_valid = matured.dropna(subset=["nifty_gross_return_pct", "nifty_net_return_pct"]) if n_mat > 0 else pd.DataFrame()
        n_nifty_valid = len(nifty_valid)
        n_nifty_dates = int(nifty_valid["observation_date"].nunique()) if n_nifty_valid > 0 else 0
        has_valid_bench = (n_nifty_valid >= min_samples) and (n_nifty_dates >= min_unique_dates)

        strategies: dict[str, StrategyBenchmarkMetrics] = {
            "bharat_alpha": _evaluate_prob_strategy(
                "bharat_alpha",
                horizon,
                matured,
                pending_cnt,
                unavail_cnt,
                "model_prob_up",
                "model_recommendation",
                min_samples,
                min_unique_dates,
                has_valid_bench,
                n_nifty_valid,
            ),
            "naive_momentum": _evaluate_prob_strategy(
                "naive_momentum",
                horizon,
                matured,
                pending_cnt,
                unavail_cnt,
                "bench_prob_up",
                None,
                min_samples,
                min_unique_dates,
                has_valid_bench,
                n_nifty_valid,
            ),
            "equal_weight": _evaluate_passive_benchmark(
                "equal_weight",
                horizon,
                matured,
                pending_cnt,
                unavail_cnt,
                "gross_return_pct",
                "net_return_pct",
                min_samples,
                min_unique_dates,
                has_valid_bench,
                n_nifty_valid,
            ),
            "NIFTY_50": _evaluate_passive_benchmark(
                "NIFTY_50",
                horizon,
                matured,
                pending_cnt,
                unavail_cnt,
                "nifty_gross_return_pct",
                "nifty_net_return_pct",
                min_samples,
                min_unique_dates,
                has_valid_bench,
                n_nifty_valid,
            ),
            "cash_no_trade": (
                StrategyBenchmarkMetrics(
                    strategy_name="cash_no_trade",
                    horizon=horizon,
                    status="OK",
                    matured_observations=n_mat,
                    pending_observations=pending_cnt,
                    unavailable_observations=unavail_cnt,
                    unique_observation_dates=n_dates,
                    unique_symbols=n_syms,
                    valid_benchmark_comparisons=n_nifty_valid,
                    gross_return_pct=0.0,
                    net_return_pct=0.0,
                    daily_portfolio_sharpe=0.0,
                    max_drawdown_pct=0.0,
                )
                if (n_mat >= min_samples and n_dates >= min_unique_dates and has_valid_bench)
                else _insufficient_or_unavailable_metrics(
                    "cash_no_trade",
                    horizon,
                    "INSUFFICIENT_DATA" if (n_mat < min_samples or n_dates < min_unique_dates) else "UNAVAILABLE",
                    (
                        f"Matured observations={n_mat} (min {min_samples}), unique_dates={n_dates} (min {min_unique_dates})."
                        if (n_mat < min_samples or n_dates < min_unique_dates)
                        else f"Valid NIFTY_50 benchmark comparison unavailable (valid_comparisons={n_nifty_valid})."
                    ),
                    n_mat,
                    pending_cnt,
                    unavail_cnt,
                    n_dates,
                    n_syms,
                    n_nifty_valid,
                )
            ),
        }

        if n_mat < min_samples or n_dates < min_unique_dates:
            h_status = "INSUFFICIENT_DATA"
            h_reason = f"Matured observations={n_mat} (min {min_samples}), unique_dates={n_dates} (min {min_unique_dates})."
        elif not has_valid_bench:
            h_status = "UNAVAILABLE"
            h_reason = f"Valid NIFTY_50 benchmark comparison unavailable (valid_comparisons={n_nifty_valid})."
        else:
            h_status = strategies["bharat_alpha"].status
            h_reason = strategies["bharat_alpha"].insufficiency_reason

        ba_net = strategies["bharat_alpha"].net_return_pct
        n50_net = strategies["NIFTY_50"].net_return_pct
        ew_net = strategies["equal_weight"].net_return_pct
        cash_net = strategies["cash_no_trade"].net_return_pct

        report = HorizonBenchmarkReport(
            horizon=horizon,
            min_observations_required=min_samples,
            min_unique_dates_required=min_unique_dates,
            matured_observations=n_mat,
            pending_observations=pending_cnt,
            unavailable_observations=unavail_cnt,
            unique_observation_dates=n_dates,
            unique_symbols=n_syms,
            valid_nifty50_comparisons=n_nifty_valid,
            entry_window_start=str(matured["entry_date"].min()) if n_mat > 0 else None,
            entry_window_end=str(matured["entry_date"].max()) if n_mat > 0 else None,
            exit_window_end=str(matured["exit_date"].max()) if n_mat > 0 else None,
            cost_assumption_pct=cost_pct,
            status=h_status,
            status_reason=h_reason,
            excess_return_vs_nifty50_pct=round(ba_net - n50_net, 2) if (ba_net is not None and n50_net is not None) else None,
            excess_return_vs_equal_weight_pct=round(ba_net - ew_net, 2) if (ba_net is not None and ew_net is not None) else None,
            excess_return_vs_cash_pct=round(ba_net - cash_net, 2) if (ba_net is not None and cash_net is not None) else None,
            strategies=strategies,
        )
        horizons_out[horizon] = asdict(report)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_version": "2.0.0-remediated",
        "cost_assumption_pct": cost_pct,
        "min_observations_required": min_samples,
        "min_unique_dates_required": min_unique_dates,
        "raw_snapshots_count": int(len(snapshots_df)),
        "canonical_snapshots_count": int(len(work_snaps)),
        "horizons": horizons_out,
    }
