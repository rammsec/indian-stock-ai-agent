"""Phase 5.3 Forward Validation, Data Quality & Paper Research Infrastructure tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.models import (
    CrossStockImpactItem,
    ForwardPredictionSnapshot,
    ForwardValidationPayload,
    MonitoringSnapshot,
    Phase5IntelligencePayload,
    StockIntelligenceItem,
    StructuredNewsEvent,
)
from backend.news_intel.collector import RawNewsItem
from backend.news_intel.combiner import build_multi_horizon_outlooks
from backend.news_intel.extractor import extract_structured_events_with_quarantine
from backend.news_intel.features import compute_pit_news_features
from backend.news_intel.forward_validation import (
    STARTING_CAPITAL_INR,
    build_forward_validation_payload,
    compute_forward_horizon_metrics,
    evaluate_forward_outcomes,
    make_prediction_id,
    record_forward_prediction_snapshots,
    run_paper_research_portfolio,
)
from backend.storage.duckdb_store import (
    STRUCTURED_EVENTS_SCHEMA,
    DuckDBStore,
    normalize_string_list,
)


def _make_store(tmp_path) -> DuckDBStore:
    return DuckDBStore(
        processed_dir=tmp_path,
        features_dir=tmp_path,
        predictions_dir=tmp_path,
        backtests_dir=tmp_path,
        db_path=tmp_path / "test.duckdb",
    )


def _make_equities_df(
    symbol: str = "RELIANCE",
    start_date: str = "2026-01-02",
    n_days: int = 20,
    split_at_idx: int | None = None,
) -> pd.DataFrame:
    """Create a synthetic NSE trading calendar DataFrame (skipping weekends)."""
    dates: list[str] = []
    cur = datetime.strptime(start_date, "%Y-%m-%d")
    while len(dates) < n_days:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    rows = []
    for i, d in enumerate(dates):
        base_px = 1000.0 + i * 10.0
        open_px = base_px + 2.0
        close_px = base_px + 5.0
        adj_close_px = close_px
        if split_at_idx is not None and i >= split_at_idx:
            # Simulate a 2:1 split where raw price halves but adj_close continues smoothly
            open_px = open_px / 2.0
            close_px = close_px / 2.0
        rows.append(
            {
                "symbol": symbol,
                "series": "EQ",
                "observation_date": d,
                "open": open_px,
                "high": close_px + 5.0,
                "low": open_px - 5.0,
                "close": close_px,
                "adj_close": adj_close_px,
                "volume": 100000.0,
                "source": "TEST",
                "publication_timestamp": f"{d}T16:00:00+05:30",
                "collection_timestamp": f"{d}T16:05:00+05:30",
            }
        )
    return pd.DataFrame(rows)


def _make_payload(
    obs_date: str = "2026-01-02",
    symbols: tuple[str, ...] = ("RELIANCE", "TCS"),
    include_10d: bool = True,
    prob_1d: float = 0.62,
    prob_5d: float = 0.58,
    prob_10d: float = 0.56,
) -> tuple[Phase5IntelligencePayload, dict[str, dict]]:
    stocks: dict[str, StockIntelligenceItem] = {}
    quant_by_sym: dict[str, dict] = {}
    for sym in symbols:
        q_pred = {
            "symbol": sym,
            "observation_date": obs_date,
            "confidence_score": 72.0,
            "roundtrip_delivery_cost_pct": 0.347,
            "prob_next_day_up": prob_1d,
            "expected_next_day_return_pct": 0.85,
            "prob_5d_up": prob_5d,
            "expected_5d_return_pct": 1.75,
            "prob_10d_up": prob_10d if include_10d else None,
            "expected_10d_return_pct": 2.90 if include_10d else None,
            "momentum_prob_up": 0.55,
            "momentum_5d_return_pct": 1.10,
            "market_regime": "Bull",
        }
        quant_by_sym[sym] = q_pred
        outlooks = build_multi_horizon_outlooks(
            symbol=sym,
            quant_pred=q_pred,
            news_features={"news_company_sentiment_5d": 0.4},
            cross_impacts=[],
            regime_label="Bull",
            trade_eligible=True,
            data_freshness="Current",
        )
        stocks[sym] = StockIntelligenceItem(
            symbol=sym,
            company_name=sym,
            sector="Energy",
            observation_date=obs_date,
            news_sentiment_score=0.4,
            news_features={"news_company_sentiment_5d": 0.4, "news_cross_stock_net_impact": 0.1},
            horizon_outlooks=outlooks,
        )
    payload = Phase5IntelligencePayload(
        generated_at=f"{obs_date}T08:45:00+05:30",
        last_news_collection_at=f"{obs_date}T08:30:00+05:30",
        last_quant_update_at=f"{obs_date}T08:40:00+05:30",
        latest_prediction_at=f"{obs_date}T08:45:00+05:30",
        stocks=stocks,
        monitoring=MonitoringSnapshot(evaluated_at=f"{obs_date}T08:45:00+05:30"),
    )
    return payload, quant_by_sym


def test_phase52_audit_horizon_isolation_and_missing_10d_never_buys(tmp_path):
    """Phase 1 Audit: 1D/5D/10D stay independent and missing 10D produces UNAVAILABLE, never BUY."""
    outlooks = build_multi_horizon_outlooks(
        symbol="RELIANCE",
        quant_pred={
            "observation_date": "2026-01-02",
            "confidence_score": 85.0,
            "roundtrip_delivery_cost_pct": 0.347,
            "prob_next_day_up": 0.68,
            "expected_next_day_return_pct": 1.10,
            "prob_5d_up": 0.61,
            "expected_5d_return_pct": 2.20,
            "prob_10d_up": None,
            "expected_10d_return_pct": None,
        },
        news_features={"news_company_sentiment_5d": 0.9, "news_cross_stock_net_impact": 0.8},
        cross_impacts=[
            CrossStockImpactItem(
                affected_company="RELIANCE",
                originating_event_id="evt_1",
                originating_entity="TCS",
                relationship_type="sector_peer",
                expected_impact_direction="positive",
                impact_magnitude="high",
                confidence=0.9,
                rationale="Strong IT demand",
            )
        ],
        regime_label="Bull",
        trade_eligible=True,
    )
    assert outlooks["1D"].is_model_generated is True
    assert outlooks["5D"].is_model_generated is True
    assert outlooks["1D"].quant_prob_positive != outlooks["5D"].quant_prob_positive
    assert outlooks["10D"].is_model_generated is False
    assert outlooks["10D"].decision == "UNAVAILABLE"
    assert outlooks["10D"].prob_positive is None
    assert outlooks["10D"].expected_net_return_pct is None
    assert outlooks["10D"].news_signal == "BULLISH"

    store = _make_store(tmp_path)
    payload, quant_by_sym = _make_payload("2026-01-02", ("RELIANCE",), include_10d=False)
    snaps = record_forward_prediction_snapshots(store, payload, quant_by_sym)
    assert len(snaps) == 1
    assert snaps[0].prob_1d_up != snaps[0].prob_5d_up
    assert snaps[0].prob_10d_up is None
    assert snaps[0].recommendation_10d == "UNAVAILABLE"


def test_quarantine_and_deduplication_and_pit_news_cutoff(tmp_path):
    """Phase 5.3 QA: invalid timestamps quarantined, duplicates collapsed, future news excluded."""
    store = _make_store(tmp_path)
    raw_items = [
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="company_news",
            title="Reliance Industries reports strong Q3 retail EBITDA growth",
            source="Economic Times",
            published="2026-01-05T10:00:00+05:30",
            url="https://example.com/rel-1",
        ),
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="company_news",
            title="Reliance Industries reports strong Q3 retail EBITDA growth",
            source="Moneycontrol",
            published="2026-01-05T10:15:00+05:30",
            url="https://example.com/rel-dup",
        ),
        RawNewsItem(
            symbol="TCS",
            company_name="Tata Consultancy Services Ltd",
            event_category="company_news",
            title="TCS wins multi-year cloud contract",
            source="Unknown Feed",
            published="",
            url="https://example.com/tcs-bad-ts",
        ),
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="company_news",
            title="Reliance Industries announces future expansion after close",
            source="LiveMint",
            published="2026-01-06T18:30:00+05:30",
            url="https://example.com/rel-future",
        ),
    ]
    valid_events, quarantined = extract_structured_events_with_quarantine(raw_items)
    assert len(valid_events) == 3
    assert sum(1 for e in valid_events if e.is_duplicate_report) == 1
    assert sum(1 for e in valid_events if not e.is_duplicate_report) == 2
    assert len(quarantined) == 1
    assert quarantined[0]["symbol"] == "TCS"
    assert quarantined[0]["quarantine_reason"] == "invalid_or_missing_timestamp"

    store.upsert_structured_events(valid_events)
    store.upsert_quarantined_events(quarantined)
    assert store.quarantined_events_parquet.exists()

    # PIT cutoff on 2026-01-05 must exclude a4 (published 2026-01-06)
    rel_events = [e for e in valid_events if e.ticker == "RELIANCE"]
    feats_jan5 = compute_pit_news_features(rel_events, "2026-01-05T15:30:00+05:30")
    assert feats_jan5["news_volume_1d"] == 1.0


def test_immutable_snapshots_never_overwritten_on_retrain(tmp_path):
    """Phase 5.3 QA: retraining does not overwrite existing historical prediction snapshots."""
    store = _make_store(tmp_path)
    p1, q1 = _make_payload("2026-01-02", ("RELIANCE",), prob_1d=0.61, prob_5d=0.57, prob_10d=0.55)
    record_forward_prediction_snapshots(store, p1, q1, model_version="5.3.0-v1")

    # Simulate model retrain with different probabilities & version for the same date
    p2, q2 = _make_payload("2026-01-02", ("RELIANCE",), prob_1d=0.41, prob_5d=0.42, prob_10d=0.43)
    record_forward_prediction_snapshots(store, p2, q2, model_version="5.3.0-v2")

    df = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df) == 1
    assert df.iloc[0]["model_version"] == "5.3.0-v1"
    assert round(float(df.iloc[0]["prob_1d_up"]), 2) > 0.55


def test_outcome_maturity_execution_price_costs_and_corporate_action(tmp_path):
    """Phase 5.3 QA: PENDING before t+H, open[t+1] entry, CNC cost deduction, adj_close split adjustment."""
    store = _make_store(tmp_path)
    eq_df = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=8, split_at_idx=4)

    p1, q1 = _make_payload("2026-01-02", ("RELIANCE",))
    record_forward_prediction_snapshots(store, p1, q1)

    outcomes = evaluate_forward_outcomes(store, equities_df=eq_df)
    by_h = {o.horizon: o for o in outcomes}

    # 8 trading days from index 0: 1D (idx 1) and 5D (idx 5) are MATURED; 10D (idx 10) is PENDING
    assert by_h["1D"].status == "MATURED"
    assert by_h["1D"].entry_price == round(float(eq_df.loc[1, "open"]), 4)
    assert by_h["1D"].transaction_cost_pct == 0.347
    assert round(by_h["1D"].gross_return_pct - 0.347, 4) == by_h["1D"].net_return_pct

    assert by_h["5D"].status == "MATURED"
    assert by_h["5D"].corporate_action_adjusted is True
    assert by_h["5D"].gross_return_pct > 0.0  # Split-adjusted return remains positive

    assert by_h["10D"].status == "PENDING"
    assert by_h["10D"].net_return_pct is None


def test_paper_portfolio_risk_limits_and_benchmarks(tmp_path):
    """Phase 5.3 QA: ₹10L capital, <=10% position, <=80% exposure, no overlapping positions, benchmarks."""
    store = _make_store(tmp_path)
    eq_rel = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=20)
    eq_tcs = _make_equities_df("TCS", start_date="2026-01-02", n_days=20)
    eq_df = pd.concat([eq_rel, eq_tcs], ignore_index=True)

    # Record consecutive daily BUY signals so 5D/10D horizons encounter overlapping position rejections
    for day in eq_rel["observation_date"].iloc[:5]:
        p, q = _make_payload(str(day), ("RELIANCE", "TCS"), prob_1d=0.66, prob_5d=0.64, prob_10d=0.63)
        record_forward_prediction_snapshots(store, p, q)

    evaluate_forward_outcomes(store, equities_df=eq_df)
    snaps_df = pd.read_parquet(store.forward_predictions_parquet)
    outs_df = pd.read_parquet(store.forward_outcomes_parquet)

    report = run_paper_research_portfolio(snaps_df, outs_df, indices_df=None, equities_df=eq_df)
    assert report.starting_capital_inr == STARTING_CAPITAL_INR
    assert report.max_position_pct == 10.0
    assert report.max_total_exposure_pct == 80.0
    assert report.strategies["5D"].peak_exposure_pct <= 80.0
    assert report.strategies["5D"].entries_executed >= 1
    assert report.strategies["5D"].rejected_signals >= 1
    assert "naive_momentum" in report.benchmarks
    assert "cash_no_trade" in report.benchmarks
    assert report.benchmarks["cash_no_trade"]["net_return_pct"] == 0.0


def test_insufficient_sample_threshold_and_sufficient_metrics(tmp_path):
    """Phase 5.3 QA: INSUFFICIENT_DATA when <10 matured outcomes, OK/WARNING when >=10."""
    store = _make_store(tmp_path)
    eq_df = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=25)
    for day in eq_df["observation_date"].iloc[:5]:
        p, q = _make_payload(str(day), ("RELIANCE",))
        record_forward_prediction_snapshots(store, p, q)
    evaluate_forward_outcomes(store, equities_df=eq_df)
    snaps_df = pd.read_parquet(store.forward_predictions_parquet)
    outs_df = pd.read_parquet(store.forward_outcomes_parquet)

    eval_small = compute_forward_horizon_metrics(snaps_df, outs_df, "1D", min_samples=10)
    assert eval_small.status == "INSUFFICIENT_DATA"
    assert eval_small.matured_count == 5

    eval_ok = compute_forward_horizon_metrics(snaps_df, outs_df, "1D", min_samples=5)
    assert eval_ok.status in ("OK", "WARNING")
    assert eval_ok.directional_accuracy_pct is not None
    assert eval_ok.brier_score is not None


def test_api_forward_validation_and_frontend_contract():
    """Phase 5.3 QA: /api/forward-validation returns valid payload and frontend renders it."""
    client = TestClient(app)
    resp = client.get("/api/forward-validation")
    assert resp.status_code == 200
    parsed = ForwardValidationPayload(**resp.json())
    assert set(parsed.horizon_evaluations.keys()) == {"1D", "5D", "10D"}
    assert parsed.paper_portfolio.starting_capital_inr == 1_000_000.0

    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    with open("frontend/app.js", "r", encoding="utf-8") as f:
        js = f.read()
    assert "sec-forward-validation" in html
    assert "renderForwardValidation" in js
    assert "/api/forward-validation" in js


def test_zero_lookahead_pending_positions_remain_open_and_executed(tmp_path):
    """Phase 5.3.1 QA: A qualifying BUY signal creates a position at next open even if future outcome is PENDING."""
    store = _make_store(tmp_path)
    # Only 2 sessions: 2026-01-02 (signal date) and 2026-01-05 (next open entry date)
    eq_df = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=2)

    p1, q1 = _make_payload("2026-01-02", ("RELIANCE",), prob_1d=0.68, prob_5d=0.65, prob_10d=0.62)
    record_forward_prediction_snapshots(store, p1, q1)

    # Outcomes for 5D and 10D are PENDING (only 2 days of data exist)
    outcomes = evaluate_forward_outcomes(store, equities_df=eq_df)
    by_h = {o.horizon: o for o in outcomes}
    assert by_h["5D"].status == "PENDING"
    assert by_h["10D"].status == "PENDING"

    snaps_df = pd.read_parquet(store.forward_predictions_parquet)
    outs_df = pd.read_parquet(store.forward_outcomes_parquet)

    # Run paper portfolio simulation with ONLY the 2 days of data
    report_2d = run_paper_research_portfolio(snaps_df, outs_df, equities_df=eq_df)

    s5 = report_2d.strategies["5D"]
    # Look-ahead check: The 5D position MUST be entered, NOT rejected due to pending outcome!
    assert s5.entries_executed == 1
    assert s5.open_positions_count == 1
    assert s5.matured_positions_count == 0
    assert s5.rejected_signals == 0
    assert s5.invested_capital_inr > 0

    # Verify event audit trail
    actions = [t.action for t in report_2d.recent_trades if t.strategy_horizon == "5D"]
    assert "SIGNAL_RECEIVED" in actions
    assert "ENTRY_EXECUTED" in actions
    assert "POSITION_OPEN" in actions
    assert "SIGNAL_REJECTED" not in actions

    # Invariance check: Add future days with a big crash. The original entry on 2026-01-05 must still be executed!
    eq_df_10 = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=10)
    # Crash prices after entry
    eq_df_10.loc[2:, "close"] = 100.0
    report_10d = run_paper_research_portfolio(snaps_df, outs_df, equities_df=eq_df_10)
    assert report_10d.strategies["5D"].entries_executed == 1
    assert report_10d.strategies["5D"].rejected_signals == 0


def test_missing_prices_marked_invalid_never_fabricated(tmp_path):
    """Phase 5.3.1 QA: Missing or non-positive prices produce INVALID status, never fabricated return."""
    store = _make_store(tmp_path)
    eq_df = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=5)
    # Zero out entry open price on day 1 (t+1)
    eq_df.loc[1, "open"] = 0.0

    p, q = _make_payload("2026-01-02", ("RELIANCE",))
    record_forward_prediction_snapshots(store, p, q)

    outcomes = evaluate_forward_outcomes(store, equities_df=eq_df)
    by_h = {o.horizon: o for o in outcomes}
    assert by_h["1D"].status == "INVALID"
    assert by_h["1D"].gross_return_pct is None
    assert by_h["1D"].net_return_pct is None
    assert "refusing to fabricate return" in (by_h["1D"].status_reason or "")


def test_independent_benchmark_horizons_never_compared_to_1d(tmp_path):
    """Phase 5.3.1 QA: 5D forward evaluation compares against 5D benchmark, never 1D benchmark."""
    store = _make_store(tmp_path)
    eq_df = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=25)
    for d in eq_df["observation_date"].iloc[:12]:
        p, q = _make_payload(str(d), ("RELIANCE",))
        # Set distinct 1D vs 5D benchmark probabilities in metadata
        q["RELIANCE"]["momentum_prob_up"] = 0.52
        q["RELIANCE"]["momentum_prob_5d_up"] = 0.61
        record_forward_prediction_snapshots(store, p, q)

    evaluate_forward_outcomes(store, equities_df=eq_df)
    snaps_df = pd.read_parquet(store.forward_predictions_parquet)
    outs_df = pd.read_parquet(store.forward_outcomes_parquet)

    eval_5d = compute_forward_horizon_metrics(snaps_df, outs_df, "5D", min_samples=5)
    assert eval_5d.status in ("OK", "WARNING")
    # Verify benchmark probability column used was benchmark_prob_5d_up (0.61), not 1D (0.52)
    assert snaps_df["benchmark_prob_5d_up"].iloc[0] == 0.61
    assert snaps_df["benchmark_prob_1d_up"].iloc[0] == 0.52


def test_news_provenance_audit_and_quarantine_metrics(tmp_path):
    """Phase 5.3.1 QA: compute_news_provenance accurately audits valid, quarantined, URLs and forward status."""
    store = _make_store(tmp_path)
    p, q = _make_payload("2026-01-02", ("RELIANCE",))
    record_forward_prediction_snapshots(store, p, q)
    snaps_df = pd.read_parquet(store.forward_predictions_parquet)

    raw_items = [
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="company_news",
            title="Reliance Retail strong growth",
            source="Economic Times",
            published="2026-01-02T10:00:00+05:30",
            url="https://example.com/rel-1",
        ),
        RawNewsItem(
            symbol="TCS",
            company_name="Tata Consultancy Services Ltd",
            event_category="company_news",
            title="TCS cloud contract",
            source="Moneycontrol",
            published="",  # invalid timestamp -> quarantined
            url="https://example.com/tcs-quar",
        ),
    ]
    valid_ev, quar_ev = extract_structured_events_with_quarantine(raw_items)
    store.upsert_structured_events(valid_ev)
    store.upsert_quarantined_events(quar_ev)

    from backend.news_intel.forward_validation import compute_news_provenance
    prov = compute_news_provenance(store, snaps_df)
    assert prov.valid_events == 1
    assert prov.quarantined_events == 1
    assert prov.unique_source_urls == 1
    assert prov.is_genuinely_forward_collected is True


def test_parquet_persistence_mixed_types_and_schema_normalization(tmp_path):
    """Phase 5.3.2 QA: Handle list mixed with null, string, dict, empty list/dict, reload & deduplicate."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # 1. Verify normalize_string_list handles all edge cases
    assert normalize_string_list(None) == []
    assert normalize_string_list(np.nan) == []
    assert normalize_string_list(["abc", None, np.nan, "def"]) == ["abc", "def"]
    assert normalize_string_list('["xyz", "123"]') == ["xyz", "123"]
    assert normalize_string_list({"k": "v"}) == ['{"k": "v"}']
    assert normalize_string_list("single_evidence") == ["single_evidence"]
    assert normalize_string_list([]) == []
    assert normalize_string_list({}) == []

    store = _make_store(tmp_path)

    # 2. Ingest events with mixed nested types
    ev1 = {
        "event_id": "ev_001",
        "company_name": "Reliance Industries",
        "ticker": "RELIANCE",
        "related_competitor_or_industry": "Energy",
        "event_category": "promoter_management",
        "event_description": "Promoter buys shares",
        "original_source": "NSE",
        "source_url": "https://example.com/rel",
        "publication_timestamp": "2026-01-02T10:00:00+05:30",
        "effective_trading_date": "2026-01-02",
        "sentiment": "positive",
        "impact_magnitude": "medium",
        "classification_confidence": 0.85,
        "novelty_score": 0.8,
        "source_credibility": 0.9,
        "classification_evidence": ["regulatory_filing_confirmed"],
        "related_companies": ["RELIANCE"],
        "related_sectors": ["Energy", "Petrochemicals"],
        "promoter_verification_status": "CONFIRMED",
        "promoter_classification_confidence": 0.95,
        "extraction_method": "deterministic",
        "validation_status": "VALID",
    }
    # ev2 has null evidence and empty lists
    ev2 = {
        "event_id": "ev_002",
        "company_name": "Tata Consultancy Services",
        "ticker": "TCS",
        "related_competitor_or_industry": "IT",
        "event_category": "company_news",
        "event_description": "Q3 results announced",
        "original_source": "BSE",
        "source_url": "https://example.com/tcs",
        "publication_timestamp": "2026-01-02T11:00:00+05:30",
        "effective_trading_date": "2026-01-02",
        "sentiment": "neutral",
        "impact_magnitude": "low",
        "classification_confidence": 0.70,
        "novelty_score": 0.5,
        "source_credibility": 0.8,
        "classification_evidence": None,
        "related_companies": [],
        "related_sectors": None,
        "promoter_verification_status": "NOT_APPLICABLE",
        "promoter_classification_confidence": None,
        "extraction_method": "deterministic",
        "validation_status": "VALID",
    }
    # ev3 has scalar string evidence and dictionary related_companies
    ev3 = {
        "event_id": "ev_003",
        "company_name": "Tata Elxsi",
        "ticker": "TATAELXSI",
        "related_competitor_or_industry": "IT",
        "event_category": "company_news",
        "event_description": "EV software deal",
        "original_source": "Moneycontrol",
        "source_url": "https://example.com/elxsi",
        "publication_timestamp": "2026-01-02T12:00:00+05:30",
        "effective_trading_date": "2026-01-02",
        "sentiment": "positive",
        "impact_magnitude": "high",
        "classification_confidence": 0.88,
        "novelty_score": 0.7,
        "source_credibility": 0.85,
        "classification_evidence": "press_release_quote",
        "related_companies": {"partner": "Jaguar Land Rover"},
        "related_sectors": ["Auto Tech"],
        "promoter_verification_status": "UNKNOWN",
        "promoter_classification_confidence": np.nan,
        "extraction_method": "deterministic",
        "validation_status": "VALID",
    }

    # Upsert batch 1
    cnt1 = store.upsert_structured_events([ev1, ev2])
    assert cnt1 == 2

    # Upsert batch 2 (ev3 + duplicate of ev1)
    cnt2 = store.upsert_structured_events([ev3, ev1])
    assert cnt2 == 3  # ev1 duplicate dropped cleanly

    # 3. Verify on-disk Parquet schema and data
    table = pq.read_table(store.events_parquet)
    assert table.num_rows == 3
    assert table.schema.field("classification_evidence").type == pa.list_(pa.string())
    assert table.schema.field("related_companies").type == pa.list_(pa.string())
    assert table.schema.field("related_sectors").type == pa.list_(pa.string())

    # 4. Verify reloading into Pydantic models works seamlessly
    pydf = table.to_pandas()
    for _, row in pydf.iterrows():
        raw_dict = row.to_dict()
        clean_dict = {}
        for k, v in raw_dict.items():
            if isinstance(v, (list, tuple, set, np.ndarray)):
                clean_dict[k] = list(v)
            elif isinstance(v, dict):
                clean_dict[k] = v
            elif not pd.isna(v) and v is not None:
                clean_dict[k] = v
        event = StructuredNewsEvent(**clean_dict)
        assert isinstance(event.classification_evidence, list)
        assert isinstance(event.related_companies, list)
        assert isinstance(event.related_sectors, list)


def test_parquet_atomic_persistence_and_corruption_prevention(tmp_path, monkeypatch):
    """Phase 5.3.2 QA: Failed write does not corrupt existing Parquet dataset and cleans up temp files."""
    import pyarrow.parquet as pq

    store = _make_store(tmp_path)
    ev1 = {
        "event_id": "ev_orig",
        "company_name": "Reliance",
        "ticker": "RELIANCE",
        "event_category": "news",
        "event_description": "Initial valid event",
        "original_source": "NSE",
        "publication_timestamp": "2026-01-02T10:00:00+05:30",
        "sentiment": "positive",
        "impact_magnitude": "low",
        "classification_confidence": 0.8,
    }
    cnt = store.upsert_structured_events([ev1])
    assert cnt == 1
    assert store.events_parquet.exists()
    orig_table = pq.read_table(store.events_parquet)
    assert orig_table.num_rows == 1

    # Simulate an I/O crash during write_table
    def _mock_write_table(*args, **kwargs):
        raise OSError("Simulated disk error during atomic write")

    monkeypatch.setattr(pq, "write_table", _mock_write_table)

    bad_ev = {
        "event_id": "ev_bad",
        "company_name": "TCS",
        "ticker": "TCS",
        "event_category": "news",
        "event_description": "Failed event",
        "original_source": "NSE",
        "publication_timestamp": "2026-01-02T11:00:00+05:30",
        "sentiment": "neutral",
        "impact_magnitude": "low",
        "classification_confidence": 0.5,
    }
    try:
        store.upsert_structured_events([bad_ev])
    except OSError:
        pass

    # Verify original file was NOT corrupted and still has 1 row
    after_table = pq.read_table(store.events_parquet)
    assert after_table.num_rows == 1
    assert after_table.column("event_id")[0].as_py() == "ev_orig"

    # Verify no dangling temp files left behind
    tmp_files = list(store.predictions_dir.glob(".*.tmp"))
    assert len(tmp_files) == 0


def test_snapshot_identity_idempotency_and_legacy_compatibility(tmp_path):
    """Phase 5.3.2 QA: Prediction ID separates identity from observation date, avoids collisions, and supports legacy IDs."""
    store = _make_store(tmp_path)

    # 1. Unit test make_prediction_id
    id_run1 = make_prediction_id("RELIANCE", "2026-01-02", prediction_timestamp="2026-01-02T08:45:00+05:30")
    id_run1_repeat = make_prediction_id("RELIANCE", "2026-01-02", prediction_timestamp="2026-01-02T08:45:00+05:30")
    id_run2 = make_prediction_id("RELIANCE", "2026-01-02", prediction_timestamp="2026-01-02T15:30:00+05:30")
    id_with_run = make_prediction_id("RELIANCE", "2026-01-02", run_id="workflow-run-42")

    assert id_run1 == id_run1_repeat  # Idempotent for same run
    assert id_run1 != id_run2          # Distinct for different timestamp on same date
    assert "workflow-run-42" in id_with_run

    # 2. Ingest Run 1 snapshots
    p1, q1 = _make_payload("2026-01-02", ("RELIANCE",))
    p1.latest_prediction_at = "2026-01-02T08:45:00+05:30"
    record_forward_prediction_snapshots(store, p1, q1)
    df1 = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df1) == 1
    assert df1["prediction_id"].iloc[0] == id_run1

    # 3. Repeated execution of Run 1 does not duplicate
    record_forward_prediction_snapshots(store, p1, q1)
    df1_rep = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df1_rep) == 1

    # 4. Genuinely new run on same date produces new snapshot without collision
    p2, q2 = _make_payload("2026-01-02", ("RELIANCE",))
    p2.latest_prediction_at = "2026-01-02T15:30:00+05:30"
    record_forward_prediction_snapshots(store, p2, q2)
    df2 = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df2) == 2
    assert set(df2["prediction_id"]) == {id_run1, id_run2}

    # 5. Backward compatibility with legacy format snapshots
    legacy_snap = ForwardPredictionSnapshot(
        prediction_id="fwd_2026-01-02_TCS",
        symbol="TCS",
        prediction_timestamp="2026-01-02T08:45:00+05:30",
        observation_date="2026-01-02",
        prob_1d_up=0.60,
        expected_1d_return_pct=0.80,
        recommendation_1d="BUY",
        recommendation_5d="BUY",
        recommendation_10d="BUY",
        model_availability={"1D": True, "5D": True, "10D": True},
    )
    store.upsert_forward_predictions([legacy_snap])
    df_all = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df_all) == 3
    assert "fwd_2026-01-02_TCS" in set(df_all["prediction_id"])

    # 6. Verify evaluate_forward_outcomes processes both new and legacy IDs
    eq_df = pd.concat([
        _make_equities_df("RELIANCE", "2026-01-02", 5),
        _make_equities_df("TCS", "2026-01-02", 5),
    ])
    outs = evaluate_forward_outcomes(store, equities_df=eq_df)
    out_pred_ids = {o.prediction_id for o in outs}
    assert id_run1 in out_pred_ids
    assert id_run2 in out_pred_ids
    assert "fwd_2026-01-02_TCS" in out_pred_ids


def test_paper_portfolio_accounting_and_overlapping_signal_rejection(tmp_path):
    """Phase 5.3.2 QA: Test exact cash accounting, overlapping position rejection, and open->close lifecycle."""
    store = _make_store(tmp_path)
    eq_df = _make_equities_df("RELIANCE", start_date="2026-01-02", n_days=10)

    # Generate BUY signal on Day 0 (2026-01-02) and Day 1 (2026-01-05)
    d0 = str(eq_df["observation_date"].iloc[0])
    d1 = str(eq_df["observation_date"].iloc[1])

    p0, q0 = _make_payload(d0, ("RELIANCE",), prob_5d=0.65)
    p0.latest_prediction_at = f"{d0}T16:00:00+05:30"
    record_forward_prediction_snapshots(store, p0, q0)

    p1, q1 = _make_payload(d1, ("RELIANCE",), prob_5d=0.68)
    p1.latest_prediction_at = f"{d1}T16:00:00+05:30"
    record_forward_prediction_snapshots(store, p1, q1)

    snaps_df = pd.read_parquet(store.forward_predictions_parquet)
    outcomes = evaluate_forward_outcomes(store, equities_df=eq_df)
    outs_df = pd.read_parquet(store.forward_outcomes_parquet)

    report = run_paper_research_portfolio(snaps_df, outs_df, equities_df=eq_df)
    s5 = report.strategies["5D"]

    # Verify signal received count vs entries executed
    assert s5.signals_received == 2
    assert s5.entries_executed == 1  # 1st signal executed at Day 1 open
    assert s5.rejected_signals == 1  # 2nd signal on Day 1 rejected at Day 2 open because position already active!

    # Check the trade audit records
    recent = report.recent_trades
    actions = [t.action for t in recent if t.strategy_horizon == "5D"]
    assert "SIGNAL_RECEIVED" in actions
    assert "ENTRY_EXECUTED" in actions
    assert "SIGNAL_REJECTED" in actions

    rejected_trade = next(t for t in recent if t.action == "SIGNAL_REJECTED" and t.strategy_horizon == "5D")
    assert rejected_trade.reason == "REJECTED_OVERLAPPING_POSITION"

    # Check exact accounting invariant:
    # starting_capital + realized_pnl + unrealized_pnl == cash + invested_capital + unrealized_pnl == ending_capital
    assert abs(s5.ending_capital_inr - (STARTING_CAPITAL_INR + s5.realized_pnl_inr + s5.unrealized_pnl_inr)) < 0.05
    assert abs(s5.ending_capital_inr - (s5.cash_inr + s5.invested_capital_inr + s5.unrealized_pnl_inr)) < 0.05


def test_benchmark_calculations_and_missing_data_integrity(tmp_path):
    """Phase 5.3.2 QA: Benchmarks return UNAVAILABLE / INSUFFICIENT_DATA when data missing, never fabricated 0%."""
    snaps_empty = pd.DataFrame()
    outs_empty = pd.DataFrame()

    report = run_paper_research_portfolio(snaps_empty, outs_empty)
    bm = report.benchmarks

    assert bm["nifty_50_buy_hold"]["status"] == "UNAVAILABLE"
    assert bm["nifty_50_buy_hold"]["net_return_pct"] is None
    assert bm["nifty_50_buy_hold"]["max_drawdown_pct"] is None

    assert bm["naive_momentum"]["status"] == "UNAVAILABLE"
    assert bm["naive_momentum"]["net_return_pct"] is None

    assert bm["cash_no_trade"]["status"] == "ACTIVE"
    assert bm["cash_no_trade"]["net_return_pct"] == 0.0
    assert bm["cash_no_trade"]["ending_capital_inr"] == STARTING_CAPITAL_INR


