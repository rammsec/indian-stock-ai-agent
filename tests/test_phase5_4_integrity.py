"""Phase 5.4 Reliability, Prediction Identity, Outcome Maturity & News Quality tests."""

from __future__ import annotations

import pandas as pd

from backend.models import ForwardPredictionSnapshot
from backend.news_intel.collector import RawNewsItem
from backend.news_intel.extractor import extract_structured_events_with_quarantine
from backend.news_intel.forward_validation import (
    build_forward_validation_payload,
    deduplicate_forward_predictions,
    evaluate_forward_outcomes,
    record_forward_prediction_snapshots,
    resolve_nse_session,
)
from tests.test_phase5_3_forward_validation import (
    _make_equities_df,
    _make_payload,
    _make_store,
)


def _make_indices_df(start: str, n: int) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame(
        {
            "index_code": ["NIFTY 50"] * n,
            "observation_date": dates,
            "open": [24000.0 + i * 20.0 for i in range(n)],
            "close": [24010.0 + i * 20.0 for i in range(n)],
        }
    )


def test_session_idempotency_and_pre_vs_post_market_distinction(tmp_path) -> None:
    store = _make_store(tmp_path)
    assert resolve_nse_session("2026-01-02T08:50:00+05:30") == "PRE_MARKET"
    assert resolve_nse_session("2026-01-02T11:30:00+05:30") == "INTRADAY"
    assert resolve_nse_session("2026-01-02T18:30:00+05:30") == "POST_MARKET"

    p1, q1 = _make_payload("2026-01-02", ("RELIANCE", "TCS"))
    p1.latest_prediction_at = "2026-01-02T08:50:00+05:30"
    s1 = record_forward_prediction_snapshots(store, p1, q1, nse_session="PRE_MARKET")

    # Retry in same PRE_MARKET session at later timestamp -> idempotent
    p1.latest_prediction_at = "2026-01-02T09:05:00+05:30"
    s1_retry = record_forward_prediction_snapshots(store, p1, q1, nse_session="PRE_MARKET")
    assert [x.prediction_id for x in s1] == [x.prediction_id for x in s1_retry]
    assert len(pd.read_parquet(store.forward_predictions_parquet)) == 2

    # Genuinely new POST_MARKET run on same observation_date -> distinct records
    p2, q2 = _make_payload("2026-01-02", ("RELIANCE", "TCS"))
    p2.latest_prediction_at = "2026-01-02T18:30:00+05:30"
    s2 = record_forward_prediction_snapshots(store, p2, q2, nse_session="POST_MARKET")
    assert [x.prediction_id for x in s2] != [x.prediction_id for x in s1]
    assert len(pd.read_parquet(store.forward_predictions_parquet)) == 4


def test_non_destructive_deduplication_preserves_history_and_excludes_from_eval(tmp_path) -> None:
    store = _make_store(tmp_path)
    # Simulate 3 historical duplicate runs on the same observation_date + INTRADAY session
    for ts in ("2026-01-02T10:00:00+05:30", "2026-01-02T10:05:00+05:30", "2026-01-02T10:10:00+05:30"):
        p, q = _make_payload("2026-01-02", ("RELIANCE",))
        p.latest_prediction_at = ts
        record_forward_prediction_snapshots(store, p, q)

    assert len(pd.read_parquet(store.forward_predictions_parquet)) == 3
    stats = deduplicate_forward_predictions(store)
    assert stats == {"total": 3, "canonical": 1, "duplicates": 2}

    df = pd.read_parquet(store.forward_predictions_parquet)
    assert len(df) == 3  # Historical rows preserved, never deleted
    assert int(df["is_duplicate_run"].sum()) == 2

    eq = _make_equities_df("RELIANCE", "2026-01-02", 15)
    idx = _make_indices_df("2026-01-02", 15)
    evaluate_forward_outcomes(store, equities_df=eq, indices_df=idx)
    payload = build_forward_validation_payload(store=store, equities_df=eq, indices_df=idx, persist=True)

    assert payload.total_immutable_snapshots == 3
    assert payload.canonical_snapshots_count == 1
    assert payload.duplicate_snapshots_count == 2
    assert payload.matured_outcomes_total == 3  # 1 canonical * 3 horizons (1D, 5D, 10D)
    assert any("retry/duplicate" in w for w in payload.data_quality_warnings)


def test_outcomes_immutable_once_matured_and_holiday_entry_and_benchmark_attached(tmp_path) -> None:
    store = _make_store(tmp_path)
    # Observation date is a holiday/missing bar (2026-01-01), market bars start 2026-01-02
    store.upsert_forward_predictions(
        [
            ForwardPredictionSnapshot(
                prediction_id="fwd_RELIANCE_2026-01-01_run1",
                symbol="RELIANCE",
                prediction_timestamp="2026-01-01T18:00:00+05:30",
                observation_date="2026-01-01",
                prob_1d_up=0.65,
                recommendation_1d="BUY",
            ),
            ForwardPredictionSnapshot(
                prediction_id="fwd_MISSING_2026-01-01_run1",
                symbol="MISSING_SYM",
                prediction_timestamp="2026-01-01T18:00:00+05:30",
                observation_date="2026-01-01",
            ),
        ]
    )

    eq = _make_equities_df("RELIANCE", "2026-01-02", 12)
    idx = _make_indices_df("2026-01-02", 12)

    first = evaluate_forward_outcomes(store, equities_df=eq, indices_df=idx)
    rel_1d = [o for o in first if o.symbol == "RELIANCE" and o.horizon == "1D"][0]
    assert rel_1d.status == "MATURED"
    assert rel_1d.entry_date > "2026-01-01"
    assert rel_1d.benchmark_return_pct is not None

    miss_1d = [o for o in first if o.symbol == "MISSING_SYM" and o.horizon == "1D"][0]
    assert miss_1d.status == "UNAVAILABLE"
    assert miss_1d.net_return_pct is None

    # Second call must NOT re-evaluate already-MATURED / UNAVAILABLE outcomes
    second = evaluate_forward_outcomes(store, equities_df=eq, indices_df=idx)
    assert second == []


def test_news_entity_isolation_and_generic_headline_quarantine() -> None:
    pub = "Tue, 17 Nov 2026 09:00:00 GMT"
    items = [
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="promoter_management",
            title="Britannia CEO plays down Reliance's pricing threat in India biscuit market - Reuters",
            source="Reuters",
            published=pub,
            url="https://example.com/britannia-reliance",
        ),
        RawNewsItem(
            symbol="ITC",
            company_name="ITC Ltd",
            event_category="promoter_management",
            title="ITC Holdings CEO says data center growth is helping cut Midwest transmission rates 13% - CNBC",
            source="CNBC",
            published=pub,
            url="https://example.com/itc-holdings",
        ),
        RawNewsItem(
            symbol="ITC",
            company_name="ITC Ltd",
            event_category="competitor",
            title="Godfrey Phillips India Ltd. Share Price Today - Godfrey Phillips India Ltd. Stock Price Live NSE/BSE - CNBC TV18",
            source="CNBC TV18",
            published=pub,
            url="https://example.com/godfrey-price",
        ),
        RawNewsItem(
            symbol="RELIANCE",
            company_name="Reliance Industries Ltd",
            event_category="competitor",
            title="Q2 Results, Nov 5: Bharti Airtel, LIC, Titan, Mahindra & Mahindra, others to post earnings - Upstox",
            source="Upstox",
            published=pub,
            url="https://example.com/q2-roundup",
        ),
    ]
    events, quarantined = extract_structured_events_with_quarantine(items, enable_gemini=False)

    # Britannia CEO must NOT be promoter_management for RELIANCE
    rel_evts = [e for e in events if e.ticker == "RELIANCE"]
    assert all(e.event_category != "promoter_management" for e in rel_evts)

    # ITC Holdings, Godfrey Phillips Share Price Today, and Q2 roundup must be quarantined
    q_reasons = {q["quarantine_reason"] for q in quarantined}
    assert "irrelevant_entity_contamination" in q_reasons
    assert "generic_price_headline" in q_reasons
