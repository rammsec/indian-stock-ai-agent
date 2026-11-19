"""Regression tests for the isolated Paper Portfolio & Risk Simulation Audit module."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.backtesting.portfolio_audit import (
    STARTING_CAPITAL_INR,
    run_audit_from_data_dir,
    run_paper_portfolio_audit,
)


def _make_eq(symbols: tuple[str, ...], dates: list[str], base_price: float = 100.0) -> pd.DataFrame:
    rows = []
    for s_idx, sym in enumerate(symbols):
        px = base_price * (1.0 + 0.1 * s_idx)
        for i, dt in enumerate(dates):
            o = px + i * 1.0
            c = o + 2.0
            rows.append(
                {
                    "symbol": sym,
                    "observation_date": dt,
                    "open": o,
                    "close": c,
                    "adj_close": c,
                }
            )
    return pd.DataFrame(rows)


def _make_snap(
    pid: str,
    sym: str,
    obs_date: str,
    pred_ts: str,
    rec_1d: str = "NO TRADE",
    rec_5d: str = "NO TRADE",
    rec_10d: str = "NO TRADE",
    nse_session: str = "POST_MARKET",
    is_dup: bool = False,
    mom_5d: float | None = None,
    cost_pct: float = 0.347,
) -> dict:
    return {
        "prediction_id": pid,
        "symbol": sym,
        "observation_date": obs_date,
        "prediction_timestamp": pred_ts,
        "nse_session": nse_session,
        "is_duplicate_run": is_dup,
        "recommendation_1d": rec_1d,
        "recommendation_5d": rec_5d,
        "recommendation_10d": rec_10d,
        "benchmark_prob_1d_up": 0.45,
        "benchmark_prob_5d_up": mom_5d,
        "benchmark_prob_10d_up": None,
        "metadata": f'{{"roundtrip_delivery_cost_pct": {cost_pct}}}',
    }


def test_rejection_classification_and_lookahead_and_same_day_dup():
    dates = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"]
    eq = _make_eq(("BIRLACABLE", "ITC", "TCS"), dates)

    snaps = pd.DataFrame(
        [
            # Duplicate retry -> filtered before simulation
            _make_snap("dup1", "BIRLACABLE", "2026-11-16", "2026-11-16T19:00:00+05:30", rec_10d="BUY", is_dup=True),
            # Valid post-market HOLD on 2026-11-16 -> REJECTED_NON_BUY_SIGNAL
            _make_snap("p_bc_pm", "BIRLACABLE", "2026-11-16", "2026-11-16T19:23:00+05:30", rec_10d="HOLD"),
            # Intraday run generated AFTER 2026-11-17 09:15 IST -> REJECTED_LOOKAHEAD
            _make_snap("p_bc_intra", "BIRLACABLE", "2026-11-16", "2026-11-17T10:25:00+05:30", rec_10d="BUY", nse_session="INTRADAY"),
            # Two valid canonical sessions on same date for ITC -> second is REJECTED_SAME_DAY_DUPLICATE
            _make_snap("p_itc_1", "ITC", "2026-11-16", "2026-11-16T18:00:00+05:30", rec_5d="BUY", nse_session="POST_MARKET"),
            _make_snap("p_itc_2", "ITC", "2026-11-16", "2026-11-17T08:30:00+05:30", rec_5d="BUY", nse_session="PRE_MARKET"),
            # Consecutive day BUY on ITC while 5D position open -> REJECTED_OVERLAPPING_POSITION
            _make_snap("p_itc_d2", "ITC", "2026-11-17", "2026-11-17T18:00:00+05:30", rec_5d="BUY"),
            # UNAVAILABLE model for TCS -> REJECTED_UNAVAILABLE_MODEL
            _make_snap("p_tcs_unavail", "TCS", "2026-11-16", "2026-11-16T18:00:00+05:30", rec_5d="UNAVAILABLE"),
        ]
    )

    rep = run_paper_portfolio_audit(snaps, equities_df=eq, audit_mode="strict_canonical")
    assert rep.total_snapshots == 7
    assert rep.duplicate_run_snapshots == 1
    assert rep.canonical_snapshots == 6
    assert rep.lookahead_flagged_snapshots == 1

    s10 = rep.strategies["10D"]
    assert s10.entries_executed == 0
    assert s10.rejection_counts["REJECTED_LOOKAHEAD"] == 1
    assert s10.rejection_counts["REJECTED_NON_BUY_SIGNAL"] >= 1

    s5 = rep.strategies["5D"]
    assert s5.entries_executed == 1
    assert s5.rejection_counts["REJECTED_SAME_DAY_DUPLICATE"] == 1
    assert s5.rejection_counts["REJECTED_OVERLAPPING_POSITION"] == 1
    assert s5.rejection_counts["REJECTED_UNAVAILABLE_MODEL"] == 1


def test_cash_and_mtm_conservation_and_cost_deduction():
    # 7 trading days so 5D signal on day 0 enters day 1 open and exits day 5 close
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    eq = _make_eq(("RELIANCE", "TCS"), dates, base_price=100.0)

    snaps = pd.DataFrame(
        [
            _make_snap("r0", "RELIANCE", dates[0], f"{dates[0]}T18:00:00+05:30", rec_5d="BUY", rec_10d="BUY", cost_pct=0.50),
            _make_snap("t0", "TCS", dates[0], f"{dates[0]}T18:00:00+05:30", rec_5d="BUY", cost_pct=0.40),
        ]
    )

    rep = run_paper_portfolio_audit(snaps, equities_df=eq, audit_mode="strict_canonical")

    # 5D matured both positions by dates[5]
    s5 = rep.strategies["5D"]
    assert s5.entries_executed == 2
    assert s5.matured_positions_count == 2
    assert s5.open_positions_count == 0
    assert s5.accounting_invariant_ok is True
    assert s5.cash_conservation_error_inr == 0.0
    assert s5.total_cost_inr == pytest.approx(900.0, abs=0.05)  # 100k * 0.5% + 100k * 0.4%
    assert s5.ending_capital_inr == pytest.approx(STARTING_CAPITAL_INR + s5.realized_pnl_inr, abs=0.05)
    assert s5.cash_inr == pytest.approx(s5.ending_capital_inr, abs=0.05)

    # 10D RELIANCE still open -> unrealized_pnl tracked, accrued_exit_cost_est reported
    s10 = rep.strategies["10D"]
    assert s10.entries_executed == 1
    assert s10.open_positions_count == 1
    assert s10.accounting_invariant_ok is True
    assert s10.cash_inr == pytest.approx(900_000.0, abs=0.01)
    assert s10.invested_cost_inr == pytest.approx(100_000.0, abs=0.01)
    assert s10.accrued_exit_cost_est_inr == pytest.approx(500.0, abs=0.01)
    assert s10.ending_capital_inr == pytest.approx(s10.cash_inr + s10.invested_mtm_inr, abs=0.01)


def test_nifty50_index_code_alignment_and_momentum_benchmark():
    dates = ["2026-11-16", "2026-11-17", "2026-11-18"]
    eq = _make_eq(("ITC",), dates)
    idx = pd.DataFrame(
        {
            "index_code": ["NIFTY_50", "NIFTY_50", "NIFTY_50"],
            "observation_date": dates,
            "open": [25200.0, 25000.0, 25100.0],
            "close": [25250.0, 25200.0, 25500.0],  # +2.0% from 25000 open on 2026-11-17
        }
    )
    snaps = pd.DataFrame(
        [
            _make_snap("i0", "ITC", "2026-11-16", "2026-11-16T18:00:00+05:30", mom_5d=0.60),
        ]
    )

    rep = run_paper_portfolio_audit(snaps, equities_df=eq, indices_df=idx, audit_mode="strict_canonical")
    nb = rep.benchmarks["nifty_50_buy_hold"]
    assert nb["status"] == "ACTIVE"
    assert nb["index_code_matched"] == "NIFTY_50"
    assert nb["entry_date"] == "2026-11-17"
    assert nb["entry_price"] == 25000.0
    # 200k cash + 800k * (1 - 0.0005) * (25500 / 25000) = 200k + 815,592 = 1,015,592 (+1.56%)
    assert nb["net_return_pct"] == pytest.approx(1.56, abs=0.01)

    m5 = rep.benchmarks["naive_momentum_5d"]
    assert m5["status"] == "ACTIVE"
    assert m5["entries_executed"] == 1


def test_missing_price_holiday_and_max_exposure_edge_cases():
    # Include a holiday gap between 2026-11-06 and 2026-11-11 (2026-11-09 holiday skipped)
    dates = ["2026-11-06", "2026-11-11", "2026-11-12"]
    symbols = tuple(f"SYM{i}" for i in range(10))
    eq = _make_eq(symbols, dates)
    # Invalidate open price on 2026-11-11 for SYM0
    eq.loc[(eq["symbol"] == "SYM0") & (eq["observation_date"] == "2026-11-11"), "open"] = 0.0
    # Remove SYM9 entirely to test missing observation bar
    eq = eq[eq["symbol"] != "SYM9"].reset_index(drop=True)

    # All 10 symbols signal BUY on 2026-11-06 -> SYM0 invalid price, SYM9 missing bar, SYM1..8 = 8 positions = 80% cap;
    # plus add SYM10 for max exposure rejection
    eq_sym10 = _make_eq(("SYM10",), dates)
    eq = pd.concat([eq, eq_sym10], ignore_index=True)

    snaps = pd.DataFrame(
        [
            _make_snap(f"p_{s}", s, "2026-11-06", "2026-11-06T18:00:00+05:30", rec_5d="BUY")
            for s in (*symbols, "SYM10")
        ]
    )

    rep = run_paper_portfolio_audit(snaps, equities_df=eq, audit_mode="strict_canonical")
    s5 = rep.strategies["5D"]
    assert s5.entries_executed == 8
    assert s5.peak_exposure_cost_pct <= 80.05
    assert s5.rejection_counts["REJECTED_INVALID_ENTRY_PRICE"] == 1
    assert s5.rejection_counts["REJECTED_MISSING_OBSERVATION_BAR"] == 1
    assert s5.rejection_counts["REJECTED_MAX_EXPOSURE"] == 1
    assert s5.accounting_invariant_ok is True


def test_persisted_repo_data_audit_reproducibility():
    rep_strict = run_audit_from_data_dir("data", audit_mode="strict_canonical")
    rep_replay = run_audit_from_data_dir("data", audit_mode="production_replay")

    assert rep_strict.total_snapshots == 65
    assert rep_strict.canonical_snapshots == 30
    assert rep_strict.duplicate_run_snapshots == 35
    assert rep_strict.lookahead_flagged_snapshots == 10

    assert rep_strict.strategies["10D"].entries_executed == 0
    assert rep_replay.strategies["10D"].entries_executed == 1
    assert rep_replay.strategies["10D"].ending_capital_inr == pytest.approx(1000782.68, abs=0.05)

    assert rep_strict.benchmarks["nifty_50_buy_hold"]["status"] == "ACTIVE"
    assert rep_strict.benchmarks["nifty_50_buy_hold"]["index_code_matched"] == "NIFTY_50"
    assert rep_strict.persistence_audit is not None
    assert rep_strict.persistence_audit.dropped_by_truncation_count == 91
    assert rep_strict.persistence_audit.duplicate_run_rows_in_parquet == 84
