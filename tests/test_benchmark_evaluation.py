"""Regression tests for independent benchmark & statistical evaluation engine."""

from __future__ import annotations

import pandas as pd

from backend.backtesting.benchmark_eval import (
    build_aligned_outcomes,
    deduplicate_daily_canonical_snapshots,
    run_benchmark_comparison,
    wilson_ci,
)


def _make_market_data(n_days: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-10-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    eq_rows = []
    for sym, base in (("RELIANCE", 1000.0), ("TCS", 2000.0)):
        for i, d in enumerate(dates):
            op = base + i * 10.0
            cl = op + 5.0 + (i % 3) * 2.0
            eq_rows.append(
                {
                    "symbol": sym,
                    "observation_date": d,
                    "open": op,
                    "close": cl,
                    "adj_close": cl,
                }
            )
    idx_rows = [
        {
            "index_code": "NIFTY_50",
            "observation_date": d,
            "open": 24000.0 + i * 25.0,
            "close": 24015.0 + i * 25.0,
        }
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(eq_rows), pd.DataFrame(idx_rows)


def _make_snapshots(dates: list[str], symbols: tuple[str, ...] = ("RELIANCE", "TCS")) -> pd.DataFrame:
    rows = []
    for d in dates:
        for s in symbols:
            rows.append(
                {
                    "prediction_id": f"fwd_{s}_{d}_POST_MARKET",
                    "symbol": s,
                    "observation_date": d,
                    "prediction_timestamp": f"{d}T18:00:00+05:30",
                    "nse_session": "POST_MARKET",
                    "is_duplicate_run": False,
                    "prob_1d_up": 0.62,
                    "prob_5d_up": 0.58,
                    "prob_10d_up": 0.55,
                    "recommendation_1d": "BUY",
                    "recommendation_5d": "BUY",
                    "recommendation_10d": "HOLD",
                    "benchmark_prob_1d_up": 0.48,
                    "benchmark_prob_5d_up": 0.60,
                    "benchmark_prob_10d_up": 0.52,
                }
            )
    return pd.DataFrame(rows)


def test_nifty50_underscore_lookup_and_exact_window_return() -> None:
    eq_df, idx_df = _make_market_data(n_days=10)
    dates = sorted(eq_df["observation_date"].unique().tolist())
    snaps = _make_snapshots([dates[0]], symbols=("RELIANCE",))

    aligned = build_aligned_outcomes(snaps, eq_df, indices_df=idx_df, cost_pct=0.347)
    r1 = aligned[aligned["horizon"] == "1D"].iloc[0]
    r5 = aligned[aligned["horizon"] == "5D"].iloc[0]

    assert r1["status"] == "MATURED"
    assert r1["entry_date"] == dates[1]
    assert r1["exit_date"] == dates[1]
    expected_n50_1d = ((24040.0 / 24025.0) - 1.0) * 100.0
    assert abs(r1["nifty_gross_return_pct"] - expected_n50_1d) < 1e-6
    assert abs(r1["nifty_net_return_pct"] - (expected_n50_1d - 0.347)) < 1e-6

    assert r5["status"] == "MATURED"
    assert r5["entry_date"] == dates[1]
    assert r5["exit_date"] == dates[5]
    expected_n50_5d = ((24140.0 / 24025.0) - 1.0) * 100.0
    assert abs(r5["nifty_gross_return_pct"] - expected_n50_5d) < 1e-6


def test_lookahead_exclusion_when_prediction_timestamp_is_next_day() -> None:
    eq_df, idx_df = _make_market_data(n_days=4)
    dates = sorted(eq_df["observation_date"].unique().tolist())
    # Snapshot claims observation_date=dates[0] but was actually generated post-close on dates[2]
    snaps = _make_snapshots([dates[0]], symbols=("RELIANCE",))
    snaps["prediction_timestamp"] = f"{dates[2]}T18:00:00+05:30"

    aligned = build_aligned_outcomes(snaps, eq_df, indices_df=idx_df)
    r1 = aligned[aligned["horizon"] == "1D"].iloc[0]
    assert r1["status"] == "MATURED"
    # Entry must be strictly after dates[2], i.e. dates[3], NOT dates[1]!
    assert r1["entry_date"] == dates[3]


def test_session_deduplication_prevents_double_counting() -> None:
    eq_df, _ = _make_market_data(n_days=5)
    d0 = sorted(eq_df["observation_date"].unique().tolist())[0]
    s_pre = _make_snapshots([d0], symbols=("RELIANCE",))
    s_pre["prediction_id"] = f"fwd_RELIANCE_{d0}_PRE"
    s_pre["nse_session"] = "PRE_MARKET"
    s_pre["prediction_timestamp"] = f"{d0}T08:45:00+05:30"

    s_post = _make_snapshots([d0], symbols=("RELIANCE",))
    s_dup = _make_snapshots([d0], symbols=("RELIANCE",))
    s_dup["prediction_id"] = f"fwd_RELIANCE_{d0}_DUP"
    s_dup["is_duplicate_run"] = True

    combined = pd.concat([s_pre, s_post, s_dup], ignore_index=True)
    deduped = deduplicate_daily_canonical_snapshots(combined)
    assert len(deduped) == 1
    assert deduped.iloc[0]["nse_session"] == "POST_MARKET"


def test_horizon_aligned_benchmark_never_uses_1d_for_5d() -> None:
    eq_df, idx_df = _make_market_data(n_days=25)
    dates = sorted(eq_df["observation_date"].unique().tolist())
    snaps = _make_snapshots(dates[:12], symbols=("RELIANCE", "TCS"))
    # Set 1D benchmark < 0.50 (0 long signals) while 5D benchmark >= 0.50 (all long signals)
    snaps["benchmark_prob_1d_up"] = 0.42
    snaps["benchmark_prob_5d_up"] = 0.65

    res = run_benchmark_comparison(
        snaps, eq_df, indices_df=idx_df, min_samples=10, min_unique_dates=5, cost_pct=0.347
    )
    mom_1d = res["horizons"]["1D"]["strategies"]["naive_momentum"]
    mom_5d = res["horizons"]["5D"]["strategies"]["naive_momentum"]
    assert mom_1d["long_signals_count"] == 0
    assert mom_5d["long_signals_count"] == mom_5d["matured_observations"]
    assert mom_5d["net_return_pct"] is not None


def test_insufficient_data_gating_and_confidence_intervals() -> None:
    eq_df, idx_df = _make_market_data(n_days=20)
    dates = sorted(eq_df["observation_date"].unique().tolist())
    # Only 2 unique dates -> 4 observations: must gate as INSUFFICIENT_DATA
    snaps_small = _make_snapshots(dates[:2], symbols=("RELIANCE", "TCS"))
    res_small = run_benchmark_comparison(
        snaps_small, eq_df, indices_df=idx_df, min_samples=10, min_unique_dates=5
    )
    ba_small = res_small["horizons"]["1D"]["strategies"]["bharat_alpha"]
    assert res_small["horizons"]["1D"]["status"] == "INSUFFICIENT_DATA"
    assert ba_small["status"] == "INSUFFICIENT_DATA"
    assert ba_small["directional_accuracy_pct"] is None
    assert ba_small["daily_portfolio_sharpe"] is None

    # 10 unique dates -> 20 observations: meets thresholds, returns metrics + CIs
    snaps_ok = _make_snapshots(dates[:10], symbols=("RELIANCE", "TCS"))
    res_ok = run_benchmark_comparison(
        snaps_ok, eq_df, indices_df=idx_df, min_samples=15, min_unique_dates=8
    )
    ba_ok = res_ok["horizons"]["1D"]["strategies"]["bharat_alpha"]
    n50_ok = res_ok["horizons"]["1D"]["strategies"]["nifty_50_buy_hold"]
    assert ba_ok["status"] in ("OK", "WARNING")
    assert ba_ok["directional_accuracy_ci95"]["method"] == "wilson_95"
    assert ba_ok["brier_score_ci95"]["method"] == "bootstrap_95"
    assert n50_ok["status"] == "OK"
    assert n50_ok["net_return_pct"] is not None

    ci = wilson_ci(15, 20)
    assert ci.low is not None and ci.high is not None and 0.0 <= ci.low < 75.0 < ci.high <= 100.0
