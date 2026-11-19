"""Non-vacuous remediation tests for the Event-Ledger Paper Portfolio Simulator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from backend.backtesting.event_ledger_portfolio import (
    LEDGER_VERSION,
    STARTING_CAPITAL_INR,
    dataframe_to_events,
    events_to_dataframe,
    normalize_index_code,
    persist_ledger_parquet,
    replay_portfolio_from_events,
    replay_strategy_from_events,
    run_event_ledger_from_data_dir,
    run_event_ledger_portfolio,
    verify_ledger_chain,
)


def _make_eq(
    symbols: tuple[str, ...],
    dates: list[str],
    price_overrides: dict[tuple[str, str], tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    overrides = price_overrides or {}
    for s_idx, sym in enumerate(symbols):
        base = 100.0 + 20.0 * s_idx
        for i, dt in enumerate(dates):
            o = base + i * 2.0
            c = o + 3.0
            ac = c
            if (sym, dt) in overrides:
                o, c, ac = overrides[(sym, dt)]
            rows.append(
                {
                    "symbol": sym,
                    "observation_date": dt,
                    "open": o,
                    "close": c,
                    "adj_close": ac,
                }
            )
    return pd.DataFrame(rows)


def _make_snap(
    pid: str,
    sym: str,
    obs_date: str,
    pred_ts: str | None,
    rec_1d: str = "NO TRADE",
    rec_5d: str = "NO TRADE",
    rec_10d: str = "NO TRADE",
    nse_session: str = "POST_MARKET",
    is_dup: bool = False,
    mom_1d: float | None = 0.45,
    mom_5d: float | None = None,
    mom_10d: float | None = None,
    cost_pct: float = 0.50,
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
        "benchmark_prob_1d_up": mom_1d,
        "benchmark_prob_5d_up": mom_5d,
        "benchmark_prob_10d_up": mom_10d,
        "metadata": f'{{"roundtrip_delivery_cost_pct": {cost_pct}}}',
    }


def test_lookahead_and_duplicate_run_and_same_day_rejection():
    dates = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"]
    eq = _make_eq(("BIRLACABLE", "ITC", "TCS"), dates)

    snaps = pd.DataFrame(
        [
            # 1. Retry/duplicate run -> REJECTED_DUPLICATE_RUN
            _make_snap("dup1", "BIRLACABLE", "2026-11-16", "2026-11-16T19:00:00+05:30", rec_10d="BUY", is_dup=True),
            # 2. Valid post-close HOLD -> REJECTED_NON_BUY_SIGNAL
            _make_snap("bc_pm", "BIRLACABLE", "2026-11-16", "2026-11-16T19:23:00+05:30", rec_10d="HOLD"),
            # 3. Intraday run AFTER 2026-11-17 09:15 IST -> REJECTED_LOOKAHEAD
            _make_snap("bc_la", "BIRLACABLE", "2026-11-16", "2026-11-17T10:25:00+05:30", rec_10d="BUY", nse_session="INTRADAY"),
            # 4. Missing timestamp -> REJECTED_LOOKAHEAD
            _make_snap("tcs_nots", "TCS", "2026-11-16", None, rec_5d="BUY"),
            # 5. Two valid canonical sessions on same date for ITC -> earliest kept, second REJECTED_SAME_DAY_DUPLICATE
            _make_snap("itc_pm", "ITC", "2026-11-16", "2026-11-16T18:00:00+05:30", rec_5d="BUY", nse_session="POST_MARKET"),
            _make_snap("itc_pre", "ITC", "2026-11-16", "2026-11-17T08:30:00+05:30", rec_5d="BUY", nse_session="PRE_MARKET"),
            # 6. Next day BUY while 5D ITC still open -> REJECTED_OVERLAPPING_POSITION
            _make_snap("itc_d2", "ITC", "2026-11-17", "2026-11-17T18:00:00+05:30", rec_5d="BUY"),
        ]
    )

    rep = run_event_ledger_portfolio(snaps, equities_df=eq, strict_canonical=True)
    assert rep.ledger_version == LEDGER_VERSION
    assert rep.total_input_snapshots == 7
    assert rep.duplicate_run_snapshots == 1
    assert rep.canonical_snapshots == 6
    assert rep.lookahead_rejected_snapshots == 2
    assert rep.same_day_duplicate_snapshots == 1

    s10 = rep.strategies["10D"]
    assert s10.entries_executed == 0
    assert s10.rejection_counts["REJECTED_DUPLICATE_RUN"] == 1
    assert s10.rejection_counts["REJECTED_LOOKAHEAD"] == 2
    assert s10.rejection_counts["REJECTED_NON_BUY_SIGNAL"] >= 1

    s5 = rep.strategies["5D"]
    assert s5.entries_executed == 1
    assert s5.rejection_counts["REJECTED_SAME_DAY_DUPLICATE"] == 1
    assert s5.rejection_counts["REJECTED_OVERLAPPING_POSITION"] == 1
    assert s5.rejection_counts["REJECTED_LOOKAHEAD"] == 2
    assert s5.rejection_counts["REJECTED_DUPLICATE_RUN"] == 1


def test_portfolio_cash_mtm_and_accrued_cost_reconciliation():
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    # RELIANCE: entry day1 open=100, day1 close=95 (-5%), day5 close=110 (+10%)
    # TCS: entry day1 open=200, day1 close=204 (+2%), day5 close=190 (-5%)
    overrides = {
        ("RELIANCE", dates[1]): (100.0, 95.0, 95.0),
        ("RELIANCE", dates[5]): (108.0, 110.0, 110.0),
        ("TCS", dates[1]): (200.0, 204.0, 204.0),
        ("TCS", dates[5]): (192.0, 190.0, 190.0),
    }
    eq = _make_eq(("RELIANCE", "TCS"), dates, price_overrides=overrides)
    snaps = pd.DataFrame(
        [
            _make_snap("r0", "RELIANCE", dates[0], f"{dates[0]}T18:00:00+05:30", rec_1d="BUY", rec_5d="BUY", rec_10d="BUY", cost_pct=0.50),
            _make_snap("t0", "TCS", dates[0], f"{dates[0]}T18:00:00+05:30", rec_1d="BUY", rec_5d="BUY", cost_pct=0.40),
        ]
    )

    rep = run_event_ledger_portfolio(snaps, equities_df=eq, strict_canonical=True)

    # 1D: enters day1 open, exits day1 close (same day)
    s1 = rep.strategies["1D"]
    assert s1.entries_executed == 2
    assert s1.matured_positions_count == 2
    assert s1.open_positions_count == 0
    assert s1.accrued_exit_cost_inr == 0.0
    # RELIANCE: 100k * (-5% - 0.5%) = -5,500; TCS: 100k * (+2% - 0.4%) = +1,600 -> net -3,900
    assert s1.realized_pnl_inr == pytest.approx(-3900.0, abs=0.05)
    assert s1.total_cost_inr == pytest.approx(900.0, abs=0.05)
    assert s1.ending_equity_inr == pytest.approx(STARTING_CAPITAL_INR - 3900.0, abs=0.05)
    assert s1.cash_inr == pytest.approx(s1.ending_equity_inr, abs=0.05)

    # 5D: enters day1 open, MTM on days 1..4, exits day5 close
    s5 = rep.strategies["5D"]
    assert s5.entries_executed == 2
    assert s5.matured_positions_count == 2
    assert s5.open_positions_count == 0
    assert s5.accrued_exit_cost_inr == 0.0
    # RELIANCE: +10% - 0.5% = +9,500; TCS: -5% - 0.4% = -5,400 -> net +4,100
    assert s5.realized_pnl_inr == pytest.approx(4100.0, abs=0.05)
    assert s5.total_cost_inr == pytest.approx(900.0, abs=0.05)
    assert s5.accounting_invariant_ok is True
    assert s5.max_reconciliation_error_inr == 0.0

    # Check MTM event on day1 for 5D before exit: RELIANCE MTM=95k, TCS MTM=102k, accrued_cost=900
    mtm_day1 = [e for e in s5.events if e.action == "POSITION_MTM" and e.event_date == dates[1]]
    assert len(mtm_day1) == 2
    eod_mtm = mtm_day1[-1]
    assert eod_mtm.cash_after_inr == pytest.approx(800_000.0, abs=0.01)
    assert eod_mtm.invested_cost_after_inr == pytest.approx(200_000.0, abs=0.01)
    assert eod_mtm.mtm_value_after_inr == pytest.approx(197_000.0, abs=0.01)
    assert eod_mtm.total_equity_after_inr == pytest.approx(997_000.0, abs=0.01)
    assert eod_mtm.accrued_exit_cost_after_inr == pytest.approx(900.0, abs=0.01)
    assert eod_mtm.net_liquidation_equity_after_inr == pytest.approx(996_100.0, abs=0.01)

    # 10D: RELIANCE stays open at end of dates (7 bars < 10D), tracks MTM + accrued CNC exit cost
    s10 = rep.strategies["10D"]
    assert s10.entries_executed == 1
    assert s10.open_positions_count == 1
    assert s10.cash_inr == pytest.approx(900_000.0, abs=0.01)
    assert s10.invested_cost_inr == pytest.approx(100_000.0, abs=0.01)
    assert s10.accrued_exit_cost_inr == pytest.approx(500.0, abs=0.01)
    assert s10.ending_equity_inr == pytest.approx(s10.cash_inr + s10.mtm_value_inr, abs=0.01)
    assert s10.net_liquidation_equity_inr == pytest.approx(s10.ending_equity_inr - 500.0, abs=0.01)

    # Every single event across all horizons satisfies cash + MTM == total_equity
    for ev in rep.all_events:
        assert ev.cash_after_inr + ev.mtm_value_after_inr == pytest.approx(ev.total_equity_after_inr, abs=1e-6)
        assert ev.total_equity_after_inr - ev.accrued_exit_cost_after_inr == pytest.approx(
            ev.net_liquidation_equity_after_inr, abs=1e-6
        )


def test_ledger_replay_reproducibility_zero_truncation_and_tamper_detection(tmp_path: Path):
    # Generate > 60 events across 12 trading days & 3 symbols to prove zero [-60:] truncation
    dates = [f"2026-01-{d:02d}" for d in range(5, 21)]
    syms = ("RELIANCE", "TCS", "ITC")
    eq = _make_eq(syms, dates)
    rows = []
    for i, dt in enumerate(dates[:10]):
        for sym in syms:
            rows.append(
                _make_snap(
                    f"p_{sym}_{i}",
                    sym,
                    dt,
                    f"{dt}T18:00:00+05:30",
                    rec_1d="BUY" if i % 2 == 0 else "HOLD",
                    rec_5d="BUY" if i % 3 == 0 else "NO TRADE",
                    rec_10d="BUY" if i == 0 else "NO TRADE",
                    cost_pct=0.45,
                )
            )
    snaps = pd.DataFrame(rows)
    rep = run_event_ledger_portfolio(snaps, equities_df=eq, strict_canonical=True)

    # Zero truncation check (far more than 60 events)
    assert len(rep.all_events) > 60
    assert rep.reconciliation_checks["zero_truncation_events_count"] == len(rep.all_events)
    assert rep.reconciliation_checks["replay_exact_match"] is True

    # Persist to Parquet and reload without losing a single event
    ledger_path = tmp_path / "event_ledger.parquet"
    written = persist_ledger_parquet(rep.all_events, ledger_path)
    assert written == len(rep.all_events)

    loaded_df = pd.read_parquet(ledger_path)
    assert len(loaded_df) == len(rep.all_events)

    replayed_all = replay_portfolio_from_events(loaded_df)
    for h in ("1D", "5D", "10D"):
        orig = rep.strategies[h]
        rep_h = replayed_all[h]
        assert len(rep_h.events) == len(orig.events)
        assert rep_h.ending_equity_inr == orig.ending_equity_inr
        assert rep_h.net_liquidation_equity_inr == orig.net_liquidation_equity_inr
        assert rep_h.cash_inr == orig.cash_inr
        assert rep_h.invested_cost_inr == orig.invested_cost_inr
        assert rep_h.mtm_value_inr == orig.mtm_value_inr
        assert rep_h.realized_pnl_inr == orig.realized_pnl_inr
        assert rep_h.unrealized_pnl_inr == orig.unrealized_pnl_inr
        assert rep_h.accrued_exit_cost_inr == orig.accrued_exit_cost_inr
        assert rep_h.max_drawdown_pct == orig.max_drawdown_pct
        assert rep_h.peak_exposure_cost_pct == orig.peak_exposure_cost_pct
        assert rep_h.peak_exposure_mtm_pct == orig.peak_exposure_mtm_pct
        assert rep_h.daily_equity_curve == orig.daily_equity_curve
        assert rep_h.ledger_chain_valid is True

    # Tamper detection: mutating cash_after_inr breaks both hash chain & invariant
    ev_copy = list(rep.strategies["5D"].events)
    ev_copy[0] = replace(ev_copy[0], cash_after_inr=ev_copy[0].cash_after_inr + 1234.0)
    ok, errs = verify_ledger_chain(ev_copy)
    assert ok is False
    assert any("Tampered event_hash" in e for e in errs)
    with pytest.raises(ValueError, match="Ledger verification failed"):
        replay_strategy_from_events(ev_copy)


def test_aligned_benchmarks_and_index_normalization():
    dates = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"]
    eq = _make_eq(
        ("ITC", "RELIANCE"),
        dates,
        price_overrides={
            ("ITC", "2026-11-17"): (100.0, 102.0, 102.0),
            ("ITC", "2026-11-18"): (102.0, 105.0, 105.0),
            ("ITC", "2026-11-19"): (105.0, 104.0, 104.0),
        },
    )
    idx = pd.DataFrame(
        {
            "index_code": ["NIFTY_50", "NIFTY_50", "NIFTY_50", "NIFTY_50"],
            "observation_date": dates,
            "open": [25100.0, 25000.0, 25200.0, 25300.0],
            "close": [25150.0, 25250.0, 25500.0, 25500.0],
        }
    )
    snaps = pd.DataFrame(
        [
            _make_snap("itc_16", "ITC", "2026-11-16", "2026-11-16T18:00:00+05:30", rec_1d="BUY", rec_5d="BUY", mom_5d=0.60, cost_pct=0.35),
        ]
    )

    assert normalize_index_code("NIFTY 50") == "NIFTY_50"
    assert normalize_index_code("nifty_50") == "NIFTY_50"

    rep = run_event_ledger_portfolio(snaps, equities_df=eq, indices_df=idx, strict_canonical=True)

    # Buy-and-hold enters at 2026-11-17 open (25000) and ends at 2026-11-19 close (25500) -> +2.0% * 80% - cost
    nb = rep.benchmarks["nifty_50_buy_hold"]
    assert nb["status"] == "ACTIVE"
    assert nb["index_code_normalized"] == "NIFTY_50"
    assert nb["entry_date"] == "2026-11-17"
    assert nb["end_date"] == "2026-11-19"
    assert nb["entry_price"] == 25000.0
    # 200k + 800k * (1 - 0.0005) * (25500 / 25000) = 200k + 815,592 = 1,015,592 (+1.56%)
    assert nb["ending_equity_inr"] == pytest.approx(1_015_592.0, abs=0.05)
    assert nb["net_return_pct"] == pytest.approx(1.56, abs=0.01)

    # Matched window on 1D replicates exact 100k lot on 2026-11-17 open (25000) -> close (25250) = +1.0% - 0.05% = +0.95% on 100k -> +950 INR
    nm1 = rep.benchmarks["nifty_50_matched_windows"]["1D"]
    assert nm1["status"] == "ACTIVE"
    assert nm1["matched_entries"] == 1
    assert nm1["ending_equity_inr"] == pytest.approx(1_000_950.0, abs=0.05)

    # Naive momentum 5D uses identical rules and enters ITC on 2026-11-17
    m5 = rep.benchmarks["naive_momentum_5d"]
    assert m5["status"] == "ACTIVE"
    assert m5["entries_executed"] == 1
    assert m5["accounting_invariant_ok"] is True
    assert m5["ledger_chain_valid"] is True

    # Cash no-trade aligned on 2026-11-17 .. 2026-11-19
    cnt = rep.benchmarks["cash_no_trade"]
    assert cnt["entry_date"] == "2026-11-17"
    assert cnt["end_date"] == "2026-11-19"
    assert cnt["net_return_pct"] == 0.0
    assert cnt["ending_equity_inr"] == STARTING_CAPITAL_INR


def test_persisted_repo_data_reconciliation_and_replay():
    rep_strict = run_event_ledger_from_data_dir("data", strict_canonical=True)
    rep_replay = run_event_ledger_from_data_dir("data", strict_canonical=False)

    assert rep_strict.total_input_snapshots == 65
    assert rep_strict.duplicate_run_snapshots == 35
    assert rep_strict.canonical_snapshots == 30
    assert rep_strict.lookahead_rejected_snapshots == 15

    # Strict mode rejects the lookahead BIRLACABLE 10D INTRADAY BUY -> 0 entries
    assert rep_strict.strategies["10D"].entries_executed == 0
    assert rep_strict.strategies["10D"].ending_equity_inr == STARTING_CAPITAL_INR

    # Replay mode (lookahead check disabled) reproduces exact ₹1,000,782.68 MTM + ₹515.40 accrued exit cost
    s10_r = rep_replay.strategies["10D"]
    assert s10_r.entries_executed == 1
    assert s10_r.ending_equity_inr == pytest.approx(1_000_782.68, abs=0.05)
    assert s10_r.accrued_exit_cost_inr == pytest.approx(515.40, abs=0.05)
    assert s10_r.net_liquidation_equity_inr == pytest.approx(1_000_267.28, abs=0.05)
    assert s10_r.accounting_invariant_ok is True
    assert s10_r.ledger_chain_valid is True

    # Full zero-truncation ledger checks
    assert rep_strict.reconciliation_checks["zero_truncation_events_count"] == 375
    assert rep_strict.reconciliation_checks["replay_exact_match"] is True
    assert rep_replay.reconciliation_checks["zero_truncation_events_count"] == 362
    assert rep_replay.reconciliation_checks["replay_exact_match"] is True

    # NIFTY_50 benchmark resolved and aligned
    nb = rep_strict.benchmarks["nifty_50_buy_hold"]
    assert nb["status"] == "ACTIVE"
    assert nb["index_code_matched"] == "NIFTY_50"
    assert nb["net_return_pct"] == pytest.approx(-0.20, abs=0.01)
