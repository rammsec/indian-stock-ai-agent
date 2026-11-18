"""Workstream C — Automated Regression Tests for Data Integrity, Audit & Lineage.

Covers all 14 critical failure scenarios:
1. Multiple predictions for the same stock and trading session.
2. Legitimate retries versus independent prediction runs.
3. Missing and invalid timestamps.
4. NSE holidays and weekends.
5. Missing entry and exit prices.
6. Outcomes that have not matured.
7. Outcomes already finalised.
8. Missing benchmark data.
9. Duplicate news events.
10. Incorrect company attribution.
11. Historical news incorrectly labelled prospective.
12. Portfolio cash and exposure consistency.
13. Realised and unrealised P&L calculations.
14. Look-ahead bias prevention.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from backend.audit.data_quality_audit import (
    audit_benchmark_integrity,
    audit_news_integrity,
    audit_outcome_integrity,
    audit_prediction_integrity,
    run_full_data_quality_audit,
)
from backend.config import IST_TZ, is_nse_trading_day
from backend.models import (
    MonitoringSnapshot,
    MultiHorizonSignalSummary,
    Phase5IntelligencePayload,
    StockIntelligenceItem,
)
from backend.news_intel.collector import RawNewsItem
from backend.news_intel.extractor import extract_structured_events_with_quarantine
from backend.news_intel.features import compute_pit_news_features
from backend.news_intel.forward_validation import (
    STARTING_CAPITAL_INR,
    build_forward_validation_payload,
    deduplicate_forward_predictions,
    evaluate_forward_outcomes,
    make_prediction_id,
    record_forward_prediction_snapshots,
    run_paper_research_portfolio,
)
from backend.storage.duckdb_store import DuckDBStore


def _make_store(tmp_path: Path) -> DuckDBStore:
    return DuckDBStore(
        db_path=tmp_path / "test.duckdb",
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
    )


def _raw_news(
    symbol: str,
    title: str,
    published: str,
    url: str = "https://example.com/news",
    source: str = "TestFeed",
    event_category: str = "company_news",
    collection_timestamp: str = "2026-11-16T10:05:00+05:30",
) -> RawNewsItem:
    return RawNewsItem(
        symbol=symbol,
        company_name="Reliance Industries Ltd" if symbol == "RELIANCE" else symbol,
        event_category=event_category,
        title=title,
        source=source,
        published=published,
        url=url,
        metadata={"collection_timestamp": collection_timestamp},
    )


def _make_payload(obs_date: str = "2026-11-16", ts: str = "2026-11-16T18:00:00+05:30", rec: str = "BUY") -> Phase5IntelligencePayload:
    h1 = MultiHorizonSignalSummary(
        horizon="1D",
        prob_positive=0.64,
        quant_prob_positive=0.62,
        expected_gross_return_pct=1.20,
        expected_net_return_pct=0.85,
        quant_signal="BULLISH",
        news_signal="POSITIVE",
        cross_stock_signal="NEUTRAL",
        combined_signal="BULLISH",
        confidence=72.0,
        uncertainty=0.28,
        decision=rec,  # type: ignore[arg-type]
        is_model_generated=True,
        data_freshness="Current",
    )
    h5 = h1.model_copy(update={"horizon": "5D", "expected_net_return_pct": 1.9})
    h10 = h1.model_copy(update={"horizon": "10D", "expected_net_return_pct": 3.1})
    stock = StockIntelligenceItem(
        symbol="RELIANCE",
        company_name="Reliance Industries Ltd",
        sector="Energy",
        observation_date=obs_date,
        horizon_outlooks={"1D": h1, "5D": h5, "10D": h10},
    )
    return Phase5IntelligencePayload(
        generated_at=ts,
        last_news_collection_at=ts,
        last_quant_update_at=ts,
        latest_prediction_at=ts,
        pilot_universe=["RELIANCE"],
        stocks={"RELIANCE": stock},
        monitoring=MonitoringSnapshot(evaluated_at=ts),
    )


# 1. Multiple predictions for the same stock and trading session
def test_multiple_predictions_same_stock_and_session(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}

    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T16:00:00+05:30"), quant,
        prediction_run_id="run_1", nse_session="POST_MARKET",
    )
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T16:15:00+05:30"), quant,
        prediction_run_id="run_2", nse_session="POST_MARKET",
    )
    stats = deduplicate_forward_predictions(store)
    assert stats["total"] == 2
    assert stats["canonical"] == 1
    assert stats["duplicates"] == 1

    snaps = pd.read_parquet(store.forward_predictions_parquet)
    audit = audit_prediction_integrity(snaps)
    assert audit.duplicate_records == 1
    assert audit.details["canonical_snapshots"] == 1
    assert not audit.critical_failures


# 2. Legitimate retries versus independent prediction runs
def test_legitimate_retries_vs_independent_runs(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}

    # Retry with same GITHUB_RUN_ID -> same prediction_id, idempotent upsert
    id_a = make_prediction_id("RELIANCE", "2026-11-16", run_id="1001", nse_session="POST_MARKET")
    id_retry = make_prediction_id("RELIANCE", "2026-11-16", run_id="1001", nse_session="POST_MARKET")
    assert id_a == id_retry

    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30"), quant,
        prediction_run_id="1001", nse_session="POST_MARKET",
    )
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:05:00+05:30"), quant,
        prediction_run_id="1001", nse_session="POST_MARKET",
    )
    # Independent run in PRE_MARKET -> distinct canonical prediction_id
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T09:00:00+05:30"), quant,
        prediction_run_id="1000", nse_session="PRE_MARKET",
    )
    stats = deduplicate_forward_predictions(store)
    assert stats["total"] == 2
    assert stats["canonical"] == 2
    assert stats["duplicates"] == 0


# 3. Missing and invalid timestamps
def test_missing_and_invalid_timestamps() -> None:
    snaps = pd.DataFrame([
        {
            "prediction_id": "fwd_RELIANCE_2026-11-16_1",
            "symbol": "RELIANCE",
            "observation_date": "bad-date",
            "prediction_timestamp": None,
            "nse_session": "POST_MARKET",
            "prediction_run_id": "1",
            "is_duplicate_run": False,
            "canonical_prediction_id": "fwd_RELIANCE_2026-11-16_1",
        }
    ])
    pred_audit = audit_prediction_integrity(snaps)
    assert pred_audit.invalid_records == 1
    assert any("missing/invalid timestamps" in f for f in pred_audit.critical_failures)

    # Raw news with missing/invalid publication timestamp must be quarantined
    raw = [
        _raw_news(
            symbol="RELIANCE",
            title="Reliance Industries reports quarterly growth",
            published="",
            url="https://example.com/1",
            collection_timestamp="2026-11-16T10:00:00+05:30",
        )
    ]
    events, quarantined = extract_structured_events_with_quarantine(raw)
    assert len(events) == 0
    assert len(quarantined) == 1


# 4. NSE holidays and weekends
def test_nse_holidays_and_weekends() -> None:
    # 2026-11-24 is an NSE holiday (Guru Nanak Jayanti) in config.py; 2026-11-21 is Saturday
    assert is_nse_trading_day(date(2026, 11, 21)) is False
    assert is_nse_trading_day(date(2026, 11, 24)) is False
    assert is_nse_trading_day(date(2026, 11, 18)) is True

    snaps = pd.DataFrame([
        {
            "prediction_id": "fwd_RELIANCE_2026-11-21_1",
            "symbol": "RELIANCE",
            "observation_date": "2026-11-21",
            "prediction_timestamp": "2026-11-21T10:00:00+05:30",
            "nse_session": "INTRADAY",
            "prediction_run_id": "1",
            "is_duplicate_run": False,
            "canonical_prediction_id": "fwd_RELIANCE_2026-11-21_1",
        },
        {
            "prediction_id": "fwd_RELIANCE_2026-11-24_2",
            "symbol": "RELIANCE",
            "observation_date": "2026-11-24",
            "prediction_timestamp": "2026-11-24T10:00:00+05:30",
            "nse_session": "INTRADAY",
            "prediction_run_id": "2",
            "is_duplicate_run": False,
            "canonical_prediction_id": "fwd_RELIANCE_2026-11-24_2",
        },
    ])
    audit = audit_prediction_integrity(snaps)
    assert audit.invalid_records == 2
    assert any("NSE holidays/weekends" in f for f in audit.critical_failures)


# 5. Missing entry and exit prices
def test_missing_entry_and_exit_prices(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    eq = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1200.0, "close": 1210.0, "adj_close": 1210.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 0.0, "close": None, "adj_close": None},
    ])
    outs = evaluate_forward_outcomes(store, equities_df=eq)
    o_1d = [o for o in outs if o.horizon == "1D"][0]
    assert o_1d.status == "INVALID"
    assert "non-positive" in o_1d.status_reason.lower()

    # Also verify audit detects a corrupted MATURED row with missing price
    bad_out = pd.DataFrame([{
        "prediction_id": "fwd_RELIANCE_2026-11-16_1",
        "symbol": "RELIANCE",
        "observation_date": "2026-11-16",
        "horizon": "1D",
        "status": "MATURED",
        "entry_date": "2026-11-17",
        "exit_date": "2026-11-17",
        "entry_price": 0.0,
        "exit_price": None,
    }])
    out_audit = audit_outcome_integrity(pd.read_parquet(store.forward_predictions_parquet), bad_out, equities_df=eq)
    assert out_audit.invalid_records == 1
    assert any("missing or non-positive price" in f for f in out_audit.critical_failures)


# 6. Outcomes that have not matured
def test_outcomes_not_matured(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    # Only 2 sessions after 2026-11-16 -> 1D matures, 5D & 10D stay PENDING
    eq = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1200.0, "close": 1210.0, "adj_close": 1210.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1215.0, "close": 1230.0, "adj_close": 1230.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-18", "open": 1230.0, "close": 1240.0, "adj_close": 1240.0},
    ])
    outs = evaluate_forward_outcomes(store, equities_df=eq)
    by_h = {o.horizon: o for o in outs}
    assert by_h["1D"].status == "MATURED"
    assert by_h["5D"].status == "PENDING"
    assert by_h["10D"].status == "PENDING"


# 7. Outcomes already finalised
def test_outcomes_already_finalised_are_immutable(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    eq1 = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1200.0, "close": 1210.0, "adj_close": 1210.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1200.0, "close": 1224.0, "adj_close": 1224.0},
    ])
    outs1 = evaluate_forward_outcomes(store, equities_df=eq1)
    assert any(o.horizon == "1D" and o.status == "MATURED" for o in outs1)

    # Even if equity prices change later, 1D is finalized and never re-evaluated
    eq2 = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1200.0, "close": 1210.0, "adj_close": 1210.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1200.0, "close": 9999.0, "adj_close": 9999.0},
    ])
    outs2 = evaluate_forward_outcomes(store, equities_df=eq2)
    assert all(o.horizon != "1D" for o in outs2)
    stored = pd.read_parquet(store.forward_outcomes_parquet)
    row_1d = stored[stored["horizon"] == "1D"].iloc[0]
    assert float(row_1d["exit_price"]) == 1224.0


# 8. Missing benchmark data
def test_missing_benchmark_data() -> None:
    snaps = pd.DataFrame([{"prediction_id": "p1"}])
    outs = pd.DataFrame([{
        "prediction_id": "p1", "horizon": "1D", "status": "MATURED",
        "entry_date": "2026-11-17", "exit_date": "2026-11-17", "benchmark_return_pct": None,
    }])
    # Empty indices -> critical failure
    b1 = audit_benchmark_integrity(snaps, outs, pd.DataFrame())
    assert b1.critical_failures

    # Indices with NIFTY_50 (underscore) mismatch -> warning
    idx_df = pd.DataFrame([{"index_code": "NIFTY_50", "observation_date": "2026-11-17", "open": 24000.0, "close": 24100.0}])
    b2 = audit_benchmark_integrity(snaps, outs, idx_df)
    assert any("NIFTY_50" in w for w in b2.warnings)


# 9. Duplicate news events
def test_duplicate_news_events() -> None:
    raw = [
        _raw_news(
            symbol="RELIANCE",
            title="Reliance Industries wins major telecom order from partner",
            source="SourceA",
            url="https://example.com/a",
            published="2026-11-16T10:00:00+05:30",
            collection_timestamp="2026-11-16T10:05:00+05:30",
        ),
        _raw_news(
            symbol="RELIANCE",
            title="Reliance Industries wins major telecom order from partner - SourceB",
            source="SourceB",
            url="https://example.com/b",
            published="2026-11-16T10:15:00+05:30",
            collection_timestamp="2026-11-16T10:20:00+05:30",
        ),
    ]
    events, _ = extract_structured_events_with_quarantine(raw)
    assert len(events) == 2
    assert sum(1 for e in events if e.is_duplicate_report) == 1
    ev_df = pd.DataFrame([e.model_dump() for e in events])
    n_audit = audit_news_integrity(ev_df)
    assert n_audit.duplicate_records == 1


# 10. Incorrect company attribution
def test_incorrect_company_attribution() -> None:
    ev_df = pd.DataFrame([
        {
            "event_id": "evt_fp_1",
            "ticker": "ITC",
            "event_category": "company_news",
            "event_description": "ITC Holdings CEO says data center growth cuts Midwest transmission rates",
            "publication_timestamp": "2026-11-16T10:00:00+05:30",
            "collection_timestamp": "2026-11-16T10:05:00+05:30",
            "is_duplicate_report": False,
        },
        {
            "event_id": "evt_mis_2",
            "ticker": "TATAELXSI",
            "event_category": "company_news",
            "event_description": "TCS wins multi-year European cloud transformation deal",
            "publication_timestamp": "2026-11-16T10:00:00+05:30",
            "collection_timestamp": "2026-11-16T10:05:00+05:30",
            "is_duplicate_report": False,
        },
    ])
    n_audit = audit_news_integrity(ev_df)
    assert n_audit.invalid_records == 2
    assert len(n_audit.critical_failures) == 2


# 11. Historical news incorrectly labelled prospective
def test_historical_news_vs_prospective_separation() -> None:
    ev_df = pd.DataFrame([
        {
            "event_id": "evt_hist_no_coll",
            "ticker": "RELIANCE",
            "event_category": "company_news",
            "event_description": "Reliance Industries expands retail footprint",
            "publication_timestamp": "2026-10-20T10:00:00+05:30",
            "collection_timestamp": None,
            "is_duplicate_report": False,
        },
        {
            "event_id": "evt_future_pub",
            "ticker": "RELIANCE",
            "event_category": "company_news",
            "event_description": "Reliance Industries announces Q3 schedule",
            "publication_timestamp": "2026-11-17T12:00:00+05:30",
            "collection_timestamp": "2026-11-17T10:00:00+05:30",  # pub > coll!
            "is_duplicate_report": False,
        },
    ])
    n_audit = audit_news_integrity(ev_df, baseline_start_date="2026-11-16")
    assert n_audit.details["historical_events"] == 1
    assert n_audit.invalid_records == 1
    assert any("publication_timestamp" in f and "collection_timestamp" in f for f in n_audit.critical_failures)


# 12. Portfolio cash and exposure consistency
def test_portfolio_cash_and_exposure_consistency(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30", rec="BUY"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    # 5D position opens on 2026-11-17 and stays open on 2026-11-18
    eq = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1000.0, "close": 1000.0, "adj_close": 1000.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1000.0, "close": 1050.0, "adj_close": 1050.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-18", "open": 1050.0, "close": 1100.0, "adj_close": 1100.0},
    ])
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    outs = pd.DataFrame()
    report = run_paper_research_portfolio(snaps, outs, equities_df=eq)
    s5 = report.strategies["5D"]
    assert s5.open_positions_count == 1
    assert s5.cash_inr + s5.invested_capital_inr == pytest.approx(STARTING_CAPITAL_INR, abs=0.05)
    assert s5.ending_capital_inr == pytest.approx(s5.cash_inr + s5.invested_capital_inr + s5.unrealized_pnl_inr, abs=0.05)
    assert s5.peak_exposure_pct <= 80.0


# 13. Realised and unrealised P&L calculations
def test_realised_and_unrealised_pnl_calculations(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    quant = {
        "RELIANCE": {
            "observation_date": "2026-11-16",
            "market_regime": "Bull",
            "roundtrip_delivery_cost_pct": 0.347,
        }
    }
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30", rec="BUY"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    eq = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1000.0, "close": 1000.0, "adj_close": 1000.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1000.0, "close": 1100.0, "adj_close": 1100.0},
    ])
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    report = run_paper_research_portfolio(snaps, pd.DataFrame(), equities_df=eq)

    # 1D enters at 1000 and exits at 1100 (+10% gross - 0.347% cost = +9.653% net on 100,000 INR = +9,653 INR)
    s1 = report.strategies["1D"]
    assert s1.matured_positions_count == 1
    assert s1.realized_pnl_inr == pytest.approx(9653.0, abs=1.0)
    assert s1.unrealized_pnl_inr == 0.0

    # 5D enters at 1000 and remains open at 1100 close (+10% unrealised on 100,000 INR = +10,000 INR)
    s5 = report.strategies["5D"]
    assert s5.open_positions_count == 1
    assert s5.realized_pnl_inr == 0.0
    assert s5.unrealized_pnl_inr == pytest.approx(10000.0, abs=1.0)


# 14. Look-ahead bias prevention
def test_lookahead_bias_prevention(tmp_path: Path) -> None:
    # A) PIT news features must strictly exclude events published after as_of_ts
    raw = [
        _raw_news(
            symbol="RELIANCE",
            title="Reliance wins major contract",
            source="PastFeed",
            url="https://example.com/past",
            published="2026-11-16T10:00:00+05:30",
            collection_timestamp="2026-11-16T10:05:00+05:30",
        ),
        _raw_news(
            symbol="RELIANCE",
            title="Reliance profit plunges on future shock",
            source="FutureFeed",
            url="https://example.com/future",
            published="2026-11-17T10:00:00+05:30",
            collection_timestamp="2026-11-17T10:05:00+05:30",
        ),
    ]
    events, _ = extract_structured_events_with_quarantine(raw)
    as_of = datetime(2026, 11, 16, 15, 30, tzinfo=IST_TZ)
    feats = compute_pit_news_features(events, as_of_ts=as_of)
    assert feats["news_volume_1d"] == 1.0
    assert feats["news_company_sentiment_1d"] > 0.0

    # B) Paper portfolio must never enter on observation_date t (only t+1 open)
    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30", rec="BUY"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    eq_only_t = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1000.0, "close": 1050.0, "adj_close": 1050.0},
    ])
    snaps = pd.read_parquet(store.forward_predictions_parquet)
    report = run_paper_research_portfolio(snaps, pd.DataFrame(), equities_df=eq_only_t)
    assert report.strategies["1D"].entries_executed == 0


# End-to-end full audit smoke test on isolated temp store
def test_full_audit_report_structure(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    report = run_full_data_quality_audit(store=store)
    assert report.overall_status in {"PASS", "WARN", "FAIL"}
    assert report.total_records_inspected >= 0


def test_reporting_source_of_truth_envelope(tmp_path: Path) -> None:
    from backend.audit.reporting_truth import build_reporting_source_of_truth

    store = _make_store(tmp_path)
    quant = {"RELIANCE": {"observation_date": "2026-11-16", "market_regime": "Bull"}}
    record_forward_prediction_snapshots(
        store, _make_payload("2026-11-16", "2026-11-16T18:00:00+05:30", rec="BUY"), quant,
        prediction_run_id="1", nse_session="POST_MARKET",
    )
    eq = pd.DataFrame([
        {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1000.0, "close": 1000.0, "adj_close": 1000.0},
        {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1000.0, "close": 1020.0, "adj_close": 1020.0},
    ])
    evaluate_forward_outcomes(store, equities_df=eq)
    truth = build_reporting_source_of_truth(store=store)
    rec_1d = truth.horizon_records["1D"]
    assert rec_1d.model_version == "5.3.0-pilot"
    assert rec_1d.horizon == "1D"
    assert rec_1d.evaluation_period.start_date == "2026-11-16"
    assert rec_1d.data_population == "pilot_5_canonical_snapshots"
    assert rec_1d.canonical_predictions_count == 1
    assert rec_1d.matured_outcomes_count == 1
    assert rec_1d.pending_outcomes_count == 0
    assert rec_1d.observation_classification == "PROSPECTIVE"
    assert rec_1d.generated_at
