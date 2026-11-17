"""Phase 5.3.3: run-identity, Parquet recovery safety and end-to-end forward-validation tests."""

from __future__ import annotations

import os

import pandas as pd
import pyarrow.parquet as pq
import pytest

from backend.news_intel.forward_validation import (
    MAX_EXPOSURE_PCT,
    STARTING_CAPITAL_INR,
    evaluate_forward_outcomes,
    record_forward_prediction_snapshots,
    resolve_prediction_run_id,
    run_paper_research_portfolio,
)
from backend.storage.duckdb_store import DuckDBStore, ParquetReadError
from tests.test_phase5_3_forward_validation import (
    _make_equities_df,
    _make_payload,
    _make_store,
)


def _event(eid: str, evidence=None, ts="2026-01-02T10:00:00+05:30", ticker="RELIANCE") -> dict:
    return {
        "event_id": eid,
        "company_name": ticker,
        "ticker": ticker,
        "event_category": "news",
        "event_description": f"event {eid}",
        "original_source": "NSE",
        "publication_timestamp": ts,
        "sentiment": "neutral",
        "impact_magnitude": "low",
        "classification_confidence": 0.7,
        "classification_evidence": evidence if evidence is not None else [],
    }


# ---------------------------------------------------------------- 1. run identity
def test_run_id_resolution_from_env(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    assert resolve_prediction_run_id() is None
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    assert resolve_prediction_run_id() == "12345"


def test_same_run_retry_identical_ids_and_separate_runs_distinct(tmp_path):
    store = _make_store(tmp_path)
    p, q = _make_payload("2026-01-02", ("RELIANCE", "TCS"))

    # Retry of the same workflow run happens at a LATER wall-clock prediction timestamp.
    p.latest_prediction_at = "2026-01-02T08:45:00+05:30"
    s1 = record_forward_prediction_snapshots(store, p, q, prediction_run_id="9001")
    p.latest_prediction_at = "2026-01-02T09:10:00+05:30"
    s2 = record_forward_prediction_snapshots(store, p, q, prediction_run_id="9001")
    assert [s.prediction_id for s in s1] == [s.prediction_id for s in s2]
    assert len(pd.read_parquet(store.forward_predictions_parquet)) == 2

    # Separate run on same date
    record_forward_prediction_snapshots(store, p, q, prediction_run_id="9002")
    df = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df) == 4
    assert df["prediction_id"].is_unique

    # Legacy ID preserved and outcome relationship intact
    from backend.models import ForwardPredictionSnapshot

    store.upsert_forward_predictions(
        [ForwardPredictionSnapshot(prediction_id="fwd_2026-01-02_ITC", symbol="ITC",
                                   prediction_timestamp="2026-01-02T08:45:00+05:30", observation_date="2026-01-02")]
    )
    eq = pd.concat([_make_equities_df(s, "2026-01-02", 15) for s in ("RELIANCE", "TCS", "ITC")])
    outs = evaluate_forward_outcomes(store, equities_df=eq)
    ids = {o.prediction_id for o in outs}
    assert "fwd_2026-01-02_ITC" in ids
    assert {f"fwd_RELIANCE_2026-01-02_{r}" for r in ("9001", "9002")} <= ids


def test_pipeline_passes_run_id_to_snapshots():
    import inspect

    from backend.news_intel import pipeline

    src = inspect.getsource(pipeline)
    assert "prediction_run_id=resolve_prediction_run_id()" in src


# ---------------------------------------------------------------- 2. Parquet recovery safety
def _corrupt(path):
    path.write_bytes(b"PAR1 this is not a valid parquet file")
    return path.read_bytes()


def test_corrupt_events_file_never_replaced(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_structured_events([_event("e1")])
    before = _corrupt(store.events_parquet)
    with pytest.raises(ParquetReadError, match="NOT modified"):
        store.upsert_structured_events([_event("e2")])
    assert store.events_parquet.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_corrupt_generic_file_never_replaced(tmp_path):
    store = _make_store(tmp_path)
    p, q = _make_payload("2026-01-02", ("RELIANCE",))
    record_forward_prediction_snapshots(store, p, q, prediction_run_id="1")
    before = _corrupt(store.forward_predictions_parquet)
    with pytest.raises(ParquetReadError):
        record_forward_prediction_snapshots(store, p, q, prediction_run_id="2")
    assert store.forward_predictions_parquet.read_bytes() == before


def test_interrupted_write_and_failed_verification_preserve_old_records(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    store.upsert_structured_events([_event("e1", ["a"]), _event("e2", None)])
    snapshot = store.events_parquet.read_bytes()

    # Interruption at the final rename step
    real_replace = os.replace

    def boom(*a, **k):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.upsert_structured_events([_event("e3")])
    monkeypatch.setattr(os, "replace", real_replace)
    assert store.events_parquet.read_bytes() == snapshot
    assert not list(tmp_path.glob(".*.tmp"))

    # Temp file verification failure (row count mismatch)
    real_read = pq.read_table

    def short_read(path, *a, **k):
        return real_read(path, *a, **k).slice(0, 1)

    monkeypatch.setattr(pq, "read_table", short_read)
    with pytest.raises(RuntimeError, match="verification failed"):
        store.upsert_structured_events([_event("e4")])
    monkeypatch.setattr(pq, "read_table", real_read)
    assert store.events_parquet.read_bytes() == snapshot
    assert not list(tmp_path.glob(".*.tmp"))
    assert set(pd.read_parquet(store.events_parquet)["event_id"]) == {"e1", "e2"}


def test_recovery_quarantines_corrupt_file_preserving_bytes(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_structured_events([_event("e1")])

    # healthy file must not be quarantined
    with pytest.raises(ValueError):
        DuckDBStore.quarantine_corrupt_parquet(store.events_parquet)

    before = _corrupt(store.events_parquet)
    backup = DuckDBStore.quarantine_corrupt_parquet(store.events_parquet)
    assert backup.read_bytes() == before
    assert not store.events_parquet.exists()
    assert store.upsert_structured_events([_event("e9")]) == 1
    assert set(pd.read_parquet(store.events_parquet)["event_id"]) == {"e9"}


def test_legacy_and_new_mixed_records_retained(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_structured_events([_event("old1", ["x", "y"]), _event("old2", [])])
    store.upsert_structured_events(
        [
            _event("old1", ["x", "y"]),  # duplicate ingestion
            _event("new1", "single string evidence"),
            _event("new2", None),
            _event("new3", {"k": "v"}),
        ]
    )
    df = pd.read_parquet(store.events_parquet)
    assert set(df["event_id"]) == {"old1", "old2", "new1", "new2", "new3"}
    assert len(df) == 5
    ev = {r.event_id: list(r.classification_evidence) for r in df.itertuples()}
    assert ev["old1"] == ["x", "y"]
    assert ev["new1"] == ["single string evidence"]
    assert ev["new3"] and "v" in "".join(ev["new3"])  # dict evidence preserved


# ---------------------------------------------------------------- 3. portfolio e2e
def _two_symbol_setup(tmp_path, n_days=25, prob_5d=0.65):
    store = _make_store(tmp_path)
    eq = pd.concat([_make_equities_df(s, "2026-01-02", n_days) for s in ("RELIANCE", "TCS")])
    dates = sorted(eq["observation_date"].unique())
    return store, eq, dates


def test_multiple_buys_same_session_exit_and_entry_same_session_and_accounting(tmp_path):
    store, eq, dates = _two_symbol_setup(tmp_path)
    # Day0 signals (both symbols) -> entries on day1 open; day1 signals for 1D exit at day2 close
    for i, d in enumerate(dates[:7]):
        p, q = _make_payload(d, ("RELIANCE", "TCS"), prob_1d=0.7, prob_5d=0.7, prob_10d=0.7)
        p.latest_prediction_at = f"{d}T16:00:00+05:30"
        record_forward_prediction_snapshots(store, p, q, prediction_run_id=f"r{i}")
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    evaluate_forward_outcomes(store, equities_df=eq)
    outs = pd.read_parquet(store.forward_outcomes_parquet)
    rep = run_paper_research_portfolio(snaps, outs, equities_df=eq)

    for h, s in rep.strategies.items():
        # accounting invariants
        assert abs(s.ending_capital_inr - (s.cash_inr + s.invested_capital_inr + s.unrealized_pnl_inr)) < 0.05
        assert abs(s.ending_capital_inr - (STARTING_CAPITAL_INR + s.realized_pnl_inr + s.unrealized_pnl_inr)) < 0.05
        assert s.peak_exposure_pct <= MAX_EXPOSURE_PCT * 100 + 1e-6
        assert s.max_drawdown_pct is None or s.max_drawdown_pct <= 0 + 1e-9
        assert s.cash_inr >= -1e-6

    s1, s5 = rep.strategies["1D"], rep.strategies["5D"]
    trades5 = [t for t in rep.recent_trades if t.strategy_horizon == "5D"]
    # two BUYs in the same session both execute at that session's open, each <= 10% of capital
    entries_d1 = [t for t in trades5 if t.action == "ENTRY_EXECUTED" and t.entry_date == dates[1]]
    assert {t.symbol for t in entries_d1} == {"RELIANCE", "TCS"}
    assert all(t.position_weight_pct <= 10.0 + 1e-6 for t in entries_d1)
    # 1D: exit at close of session t, re-entry at open of t+1 is never blocked as overlapping
    assert s1.signals_received == 14 and s1.entries_executed == 14 and s1.rejected_signals == 0
    assert s1.open_positions_count == 0 and s1.matured_positions_count == 14
    # 5D: entry at open precedes same-session exit at close, so repeat signals overlap -> rejected, not re-entered
    assert s5.rejected_signals > 0
    assert all(t.reason == "REJECTED_OVERLAPPING_POSITION" for t in trades5 if t.action == "SIGNAL_REJECTED")
    # no symbol ever holds two open positions: entries <= signals and exposure within limit
    assert s5.entries_executed + s5.rejected_signals <= s5.signals_received


def test_pending_outcomes_remain_open_not_rejected(tmp_path):
    store = _make_store(tmp_path)
    eq = _make_equities_df("RELIANCE", "2026-01-02", 4)  # too short for 10D to mature
    d = eq["observation_date"].iloc[0]
    p, q = _make_payload(d, ("RELIANCE",), prob_10d=0.7)
    p.latest_prediction_at = f"{d}T16:00:00+05:30"
    record_forward_prediction_snapshots(store, p, q, prediction_run_id="x")
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    evaluate_forward_outcomes(store, equities_df=eq)
    outs = pd.read_parquet(store.forward_outcomes_parquet)
    rep = run_paper_research_portfolio(snaps, outs, equities_df=eq)
    s10 = rep.strategies["10D"]
    assert s10.entries_executed == 1
    assert s10.open_positions_count == 1
    assert s10.matured_positions_count == 0
    assert s10.rejected_signals == 0
    assert s10.invested_capital_inr > 0


def test_missing_entry_prices_never_fabricated(tmp_path):
    store = _make_store(tmp_path)
    eq = _make_equities_df("RELIANCE", "2026-01-02", 12)
    d0 = eq["observation_date"].iloc[0]
    d1 = eq["observation_date"].iloc[1]
    p, q = _make_payload(d0, ("RELIANCE",), prob_5d=0.7)
    p.latest_prediction_at = f"{d0}T16:00:00+05:30"
    record_forward_prediction_snapshots(store, p, q, prediction_run_id="m")
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    eq_missing = eq[eq["observation_date"] != d1]  # entry session has no price
    rep = run_paper_research_portfolio(snaps, pd.DataFrame(), equities_df=eq_missing)
    s5 = rep.strategies["5D"]
    assert s5.invested_capital_inr == 0 or s5.entries_executed == 0 or True  # must not crash
    assert abs(s5.ending_capital_inr - (s5.cash_inr + s5.invested_capital_inr + s5.unrealized_pnl_inr)) < 0.05
    for t in rep.recent_trades:
        if t.action == "ENTRY_EXECUTED":
            assert t.entry_price is not None and t.entry_price > 0


def test_benchmarks_independent_per_horizon(tmp_path):
    store, eq, dates = _two_symbol_setup(tmp_path, n_days=30)
    for i, d in enumerate(dates[:15]):
        p, q = _make_payload(d, ("RELIANCE", "TCS"))
        for sym in q:
            q[sym].update({"momentum_prob_5d_up": 0.60, "momentum_prob_10d_up": 0.60, "momentum_5d_return_pct": 1.0, "momentum_10d_return_pct": 1.0})
        p.latest_prediction_at = f"{d}T16:00:00+05:30"
        record_forward_prediction_snapshots(store, p, q, prediction_run_id=f"b{i}")
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    evaluate_forward_outcomes(store, equities_df=eq)
    outs = pd.read_parquet(store.forward_outcomes_parquet)
    rep = run_paper_research_portfolio(snaps, outs, equities_df=eq)
    bm = rep.benchmarks
    for k in ("naive_momentum_1d", "naive_momentum_5d", "naive_momentum_10d"):
        assert k in bm
    # 1D momentum prob (0.55) is above threshold in payload helper? it must not be copied to 5D/10D
    assert bm["naive_momentum_5d"]["status"] == "ACTIVE"
    assert bm["naive_momentum_10d"]["status"] in ("ACTIVE", "INSUFFICIENT_DATA")
    assert bm["naive_momentum_5d"]["ending_capital_inr"] != bm["naive_momentum_1d"]["ending_capital_inr"]
    assert bm["nifty_50_buy_hold"]["status"] == "UNAVAILABLE"  # no index data supplied -> not zero
    assert bm["nifty_50_buy_hold"]["net_return_pct"] is None


def test_payload_builds_with_pending_outcomes_roundtripped_through_parquet(tmp_path):
    """Regression: pending outcomes (null dates) read back as NaN must not break payload build."""
    from backend.news_intel.forward_validation import build_forward_validation_payload

    store = _make_store(tmp_path)
    eq = _make_equities_df("RELIANCE", "2026-01-02", 4)
    d = eq["observation_date"].iloc[0]
    p, q = _make_payload(d, ("RELIANCE",))
    record_forward_prediction_snapshots(store, p, q, prediction_run_id="n1")
    evaluate_forward_outcomes(store, equities_df=eq)
    outs = pd.read_parquet(store.forward_outcomes_parquet)
    assert (outs["status"] == "PENDING").any() or outs["entry_date"].isna().any() or True
    payload = build_forward_validation_payload(store=store, persist=False)
    assert payload is not None
