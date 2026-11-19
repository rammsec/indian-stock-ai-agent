"""Non-vacuous tests for Workstream 2: Prediction Chronology & Benchmark Engine."""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd
import pytest

from backend.backtesting.chronology_benchmark import (
    add_nse_trading_days,
    build_chronology_aligned_outcomes,
    enforce_prediction_before_entry,
    normalize_benchmark_symbol,
    resolve_eligible_entry_session,
    run_chronology_benchmark,
)


def _make_nse_calendar_market_data(
    start_date: str = "2026-10-01",
    n_trading_days: int = 25,
    index_code: str = "NIFTY_50",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build synthetic NSE-holiday-aware equity and index bars."""
    dates: list[str] = []
    cur = date.fromisoformat(start_date)
    while len(dates) < n_trading_days:
        if cur.weekday() < 5 and cur.isoformat() not in {"2026-10-02", "2026-10-20"}:
            dates.append(cur.isoformat())
        cur = date.fromordinal(cur.toordinal() + 1)

    eq_rows = []
    for sym, base in (("RELIANCE", 1000.0), ("TCS", 2000.0)):
        for i, d in enumerate(dates):
            op = base + i * 10.0
            cl = op + 8.0 + (i % 3) * 2.0
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
            "index_code": index_code,
            "observation_date": d,
            "open": 24000.0 + i * 20.0,
            "close": 24015.0 + i * 20.0,
        }
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(eq_rows), pd.DataFrame(idx_rows), dates


def test_prediction_timestamp_strictly_before_entry_open() -> None:
    # Exact 09:15:00 IST on trading day T -> NOT strictly before 09:15, must roll to T_next
    ok_exact, delta_exact = enforce_prediction_before_entry("2026-11-17T09:15:00+05:30", "2026-11-17")
    assert ok_exact is False
    assert delta_exact == 0.0

    # 10:25:29 IST on 2026-11-17 with entry_date=2026-11-17 -> look-ahead violation
    ok_late, delta_late = enforce_prediction_before_entry("2026-11-17T10:25:29+05:30", "2026-11-17")
    assert ok_late is False
    assert delta_late is not None and delta_late < 0

    # Resolved entry for 2026-11-17T10:25:29+05:30 must be 2026-11-18 09:15 IST
    entry_d, entry_ts, sess = resolve_eligible_entry_session(
        "2026-11-17T10:25:29+05:30", observation_date="2026-11-16"
    )
    assert sess == "INTRADAY"
    assert entry_d == date(2026, 11, 18)
    assert entry_ts is not None and entry_ts.isoformat() == "2026-11-18T09:15:00+05:30"
    ok_resolved, lead_mins = enforce_prediction_before_entry("2026-11-17T10:25:29+05:30", entry_d.isoformat())
    assert ok_resolved is True
    assert lead_mins is not None and lead_mins > 0


def test_pre_market_intraday_and_post_market_sessions() -> None:
    # PRE_MARKET (< 09:15 IST) on trading day Tue 2026-11-17 -> eligible same day 2026-11-17 09:15 IST
    d_pre, ts_pre, s_pre = resolve_eligible_entry_session("2026-11-17T09:14:59+05:30")
    assert s_pre == "PRE_MARKET"
    assert d_pre == date(2026, 11, 17)
    assert ts_pre is not None and ts_pre.hour == 9 and ts_pre.minute == 15

    # INTRADAY (09:15 - 15:30 IST inclusive) on Tue 2026-11-17 -> next trading day Wed 2026-11-18 09:15 IST
    d_intra_start, _, s_intra_start = resolve_eligible_entry_session("2026-11-17T09:15:00+05:30")
    assert s_intra_start == "INTRADAY"
    assert d_intra_start == date(2026, 11, 18)

    d_intra_end, _, s_intra_end = resolve_eligible_entry_session("2026-11-17T15:30:00+05:30")
    assert s_intra_end == "INTRADAY"
    assert d_intra_end == date(2026, 11, 18)

    # POST_MARKET (> 15:30 IST) on Tue 2026-11-17 -> next trading day Wed 2026-11-18 09:15 IST
    d_post, _, s_post = resolve_eligible_entry_session("2026-11-17T15:30:01+05:30")
    assert s_post == "POST_MARKET"
    assert d_post == date(2026, 11, 18)

    # UTC timestamp equivalent to 08:30 IST (03:00 UTC) -> PRE_MARKET same day
    d_utc, _, s_utc = resolve_eligible_entry_session("2026-11-17T03:00:00Z")
    assert s_utc == "PRE_MARKET"
    assert d_utc == date(2026, 11, 17)


def test_nse_holidays_and_weekends_roll_forward() -> None:
    # 2026-10-02 is Gandhi Jayanti (NSE holiday) followed by Sat 2026-10-03 and Sun 2026-10-04
    # Even a PRE_MARKET prediction on 2026-10-02 must roll to Mon 2026-10-05 09:15 IST
    d_hol, ts_hol, _ = resolve_eligible_entry_session("2026-10-02T08:30:00+05:30")
    assert d_hol == date(2026, 10, 5)
    assert ts_hol is not None and ts_hol.isoformat() == "2026-10-05T09:15:00+05:30"

    # Saturday prediction -> Monday 2026-10-05 09:15 IST
    d_sat, _, _ = resolve_eligible_entry_session("2026-10-03T11:00:00+05:30")
    assert d_sat == date(2026, 10, 5)

    # POST_MARKET on Thu 2026-10-01 -> skips Fri 2026-10-02 (holiday) & weekend -> Mon 2026-10-05
    d_thu_post, _, _ = resolve_eligible_entry_session("2026-10-01T18:00:00+05:30")
    assert d_thu_post == date(2026, 10, 5)

    # 5D trading calendar advance from 2026-10-01 (4 steps) skips 2026-10-02 & weekend:
    # 2026-10-01 (1), 2026-10-05 (2), 2026-10-06 (3), 2026-10-07 (4), 2026-10-08 (5)
    exit_5d = add_nse_trading_days(date(2026, 10, 1), 4)
    assert exit_5d == date(2026, 10, 8)


def test_missing_and_invalid_price_data_marked_unavailable() -> None:
    eq_df, idx_df, dates = _make_nse_calendar_market_data(n_trading_days=15)

    # Corrupt prices: non-positive open on dates[1] for RELIANCE, NaN close on dates[2] for TCS
    eq_df.loc[(eq_df["symbol"] == "RELIANCE") & (eq_df["observation_date"] == dates[1]), "open"] = -10.0
    eq_df.loc[(eq_df["symbol"] == "TCS") & (eq_df["observation_date"] == dates[2]), "close"] = np.nan
    # Drop bar for RELIANCE on dates[3] (missing date)
    eq_df = eq_df[~((eq_df["symbol"] == "RELIANCE") & (eq_df["observation_date"] == dates[3]))].copy()

    snaps = pd.DataFrame(
        [
            {
                "prediction_id": "p_rel_neg",
                "symbol": "RELIANCE",
                "observation_date": dates[0],
                "prediction_timestamp": f"{dates[0]}T18:00:00+05:30",
                "prob_1d_up": 0.65,
                "prob_5d_up": 0.65,
                "prob_10d_up": 0.65,
                "benchmark_prob_1d_up": 0.55,
                "benchmark_prob_5d_up": 0.55,
                "benchmark_prob_10d_up": 0.55,
            },
            {
                "prediction_id": "p_tcs_nan",
                "symbol": "TCS",
                "observation_date": dates[1],
                "prediction_timestamp": f"{dates[1]}T18:00:00+05:30",
                "prob_1d_up": 0.65,
                "prob_5d_up": 0.65,
                "prob_10d_up": 0.65,
                "benchmark_prob_1d_up": 0.55,
                "benchmark_prob_5d_up": 0.55,
                "benchmark_prob_10d_up": 0.55,
            },
            {
                "prediction_id": "p_rel_missing_bar",
                "symbol": "RELIANCE",
                "observation_date": dates[2],
                "prediction_timestamp": f"{dates[2]}T18:00:00+05:30",
                "prob_1d_up": 0.65,
                "prob_5d_up": 0.65,
                "prob_10d_up": 0.65,
                "benchmark_prob_1d_up": 0.55,
                "benchmark_prob_5d_up": 0.55,
                "benchmark_prob_10d_up": 0.55,
            },
            {
                "prediction_id": "p_bad_ts",
                "symbol": "TCS",
                "observation_date": dates[0],
                "prediction_timestamp": "not-a-timestamp",
                "prob_1d_up": 0.65,
            },
        ]
    )

    aligned = build_chronology_aligned_outcomes(snaps, eq_df, indices_df=idx_df)

    r_neg = aligned[(aligned["prediction_id"] == "p_rel_neg") & (aligned["horizon"] == "1D")].iloc[0]
    assert r_neg["status"] == "UNAVAILABLE"
    assert "non-positive" in r_neg["status_reason"].lower()

    r_nan = aligned[(aligned["prediction_id"] == "p_tcs_nan") & (aligned["horizon"] == "1D")].iloc[0]
    assert r_nan["status"] == "UNAVAILABLE"

    r_miss = aligned[(aligned["prediction_id"] == "p_rel_missing_bar") & (aligned["horizon"] == "1D")].iloc[0]
    assert r_miss["status"] == "UNAVAILABLE"
    assert "missing market bar" in r_miss["status_reason"].lower()

    r_bad_ts = aligned[(aligned["prediction_id"] == "p_bad_ts") & (aligned["horizon"] == "1D")].iloc[0]
    assert r_bad_ts["status"] == "UNAVAILABLE"


def test_benchmark_symbol_normalisation() -> None:
    assert normalize_benchmark_symbol("NIFTY 50") == "NIFTY_50"
    assert normalize_benchmark_symbol("NIFTY_50") == "NIFTY_50"
    assert normalize_benchmark_symbol(" nifty 50 ") == "NIFTY_50"
    assert normalize_benchmark_symbol("NIFTY50") == "NIFTY_50"
    assert normalize_benchmark_symbol("^NSEI") == "NIFTY_50"
    assert normalize_benchmark_symbol("NIFTY-50") == "NIFTY_50"

    # End-to-end verification when indices_df uses "NIFTY 50" (with space)
    eq_df, idx_df, dates = _make_nse_calendar_market_data(n_trading_days=15, index_code="NIFTY 50")
    snaps = pd.DataFrame(
        [
            {
                "prediction_id": "p1",
                "symbol": "RELIANCE",
                "observation_date": dates[0],
                "prediction_timestamp": f"{dates[0]}T18:00:00+05:30",
                "prob_1d_up": 0.70,
                "prob_5d_up": 0.70,
                "prob_10d_up": 0.70,
            }
        ]
    )
    aligned = build_chronology_aligned_outcomes(snaps, eq_df, indices_df=idx_df)
    r1 = aligned[aligned["horizon"] == "1D"].iloc[0]
    assert r1["status"] == "MATURED"
    assert r1["nifty_status"] == "MATURED"
    assert math.isfinite(r1["nifty_gross_return_pct"])


def test_1d_5d_10d_outcome_maturity_and_benchmark_gating() -> None:
    # 12 trading days -> dates[0..11]
    eq_df, idx_df, dates = _make_nse_calendar_market_data(n_trading_days=12)

    # Snapshot 1: POST_MARKET on dates[0] -> entry dates[1]
    #   1D exit dates[1] (MATURED), 5D exit dates[5] (MATURED), 10D exit dates[10] (MATURED)
    # Snapshot 2: POST_MARKET on dates[5] -> entry dates[6]
    #   1D exit dates[6] (MATURED), 5D exit dates[10] (MATURED), 10D exit dates[15] (> dates[11] -> PENDING!)
    snaps = pd.DataFrame(
        [
            {
                "prediction_id": "s1",
                "symbol": "RELIANCE",
                "observation_date": dates[0],
                "prediction_timestamp": f"{dates[0]}T18:00:00+05:30",
                "prob_1d_up": 0.65,
                "prob_5d_up": 0.65,
                "prob_10d_up": 0.65,
                "recommendation_1d": "BUY",
                "recommendation_5d": "BUY",
                "recommendation_10d": "BUY",
                "benchmark_prob_1d_up": 0.55,
                "benchmark_prob_5d_up": 0.55,
                "benchmark_prob_10d_up": 0.55,
            },
            {
                "prediction_id": "s2",
                "symbol": "TCS",
                "observation_date": dates[5],
                "prediction_timestamp": f"{dates[5]}T18:00:00+05:30",
                "prob_1d_up": 0.65,
                "prob_5d_up": 0.65,
                "prob_10d_up": 0.65,
                "recommendation_1d": "BUY",
                "recommendation_5d": "BUY",
                "recommendation_10d": "BUY",
                "benchmark_prob_1d_up": 0.55,
                "benchmark_prob_5d_up": 0.55,
                "benchmark_prob_10d_up": 0.55,
            },
        ]
    )

    aligned = build_chronology_aligned_outcomes(snaps, eq_df, indices_df=idx_df)
    s2_1d = aligned[(aligned["prediction_id"] == "s2") & (aligned["horizon"] == "1D")].iloc[0]
    s2_5d = aligned[(aligned["prediction_id"] == "s2") & (aligned["horizon"] == "5D")].iloc[0]
    s2_10d = aligned[(aligned["prediction_id"] == "s2") & (aligned["horizon"] == "10D")].iloc[0]

    assert s2_1d["status"] == "MATURED"
    assert s2_1d["entry_date"] == dates[6] and s2_1d["exit_date"] == dates[6]
    assert s2_5d["status"] == "MATURED"
    assert s2_5d["entry_date"] == dates[6] and s2_5d["exit_date"] == dates[10]
    assert s2_10d["status"] == "PENDING"

    # With default min_samples=30, all horizons and ALL strategies (including cash_no_trade) must be INSUFFICIENT_DATA
    rep_gated = run_chronology_benchmark(snaps, eq_df, indices_df=idx_df)
    for h in ("1D", "5D", "10D"):
        assert rep_gated["horizons"][h]["status"] == "INSUFFICIENT_DATA"
        for strat_name in ("bharat_alpha", "naive_momentum", "equal_weight", "NIFTY_50", "cash_no_trade"):
            s = rep_gated["horizons"][h]["strategies"][strat_name]
            assert s["status"] == "INSUFFICIENT_DATA"
            assert s["net_return_pct"] is None

    # With min_samples=2, min_unique_dates=2, 1D & 5D mature (n=2), 10D has n=1 (INSUFFICIENT_DATA)
    rep_low = run_chronology_benchmark(snaps, eq_df, indices_df=idx_df, min_samples=2, min_unique_dates=2)
    assert rep_low["horizons"]["1D"]["status"] in ("OK", "WARNING")
    assert rep_low["horizons"]["5D"]["status"] in ("OK", "WARNING")
    assert rep_low["horizons"]["10D"]["status"] == "INSUFFICIENT_DATA"
    assert rep_low["horizons"]["1D"]["excess_return_vs_nifty50_pct"] is not None
    assert rep_low["horizons"]["1D"]["excess_return_vs_cash_pct"] is not None

    # Without NIFTY_50 index data, valid benchmark comparison is missing -> UNAVAILABLE & no metrics reported
    rep_no_idx = run_chronology_benchmark(snaps, eq_df, indices_df=pd.DataFrame(), min_samples=2, min_unique_dates=2)
    assert rep_no_idx["horizons"]["1D"]["status"] == "UNAVAILABLE"
    assert rep_no_idx["horizons"]["1D"]["strategies"]["bharat_alpha"]["net_return_pct"] is None
    assert rep_no_idx["horizons"]["1D"]["strategies"]["NIFTY_50"]["status"] == "UNAVAILABLE"
