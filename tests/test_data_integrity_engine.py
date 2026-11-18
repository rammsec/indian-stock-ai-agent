"""Tests for the BharatAlpha AI Data Integrity Engine & Provenance Classification."""

from __future__ import annotations

import pandas as pd

from backend.audit import (
    check_benchmark_integrity,
    check_forward_outcomes,
    check_forward_predictions,
    check_news_integrity,
    classify_forward_predictions,
    classify_legacy_prediction_log,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 1. Genuinely prospective POST_MARKET on 2026-11-16 -> entry 2026-11-17 09:15 IST
            {
                "prediction_id": "fwd_2026-11-16_RELIANCE",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-16T19:23:27+05:30",
                "nse_session": "POST_MARKET",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "prob_1d_up": 0.55,
                "prob_5d_up": 0.58,
                "prob_10d_up": 0.60,
                "model_availability": '{"1D": true, "5D": true, "10D": true}',
                "is_duplicate_run": False,
            },
            # 2. Look-ahead contaminated INTRADAY on 2026-11-17 10:25 IST (after 09:15 open, before 15:30 close)
            {
                "prediction_id": "fwd_RELIANCE_2026-11-16_INTRADAY",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T10:25:29+05:30",
                "nse_session": "INTRADAY",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "prob_1d_up": 0.54,
                "prob_5d_up": 0.57,
                "prob_10d_up": 0.59,
                "model_availability": '{"1D": true, "5D": true, "10D": true}',
                "is_duplicate_run": False,
            },
            # 3. Duplicate run flagged True
            {
                "prediction_id": "fwd_RELIANCE_2026-11-16_DUP",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T11:06:10+05:30",
                "nse_session": "INTRADAY",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "prob_1d_up": 0.54,
                "prob_5d_up": 0.57,
                "prob_10d_up": 0.59,
                "model_availability": '{"1D": true, "5D": true, "10D": true}',
                "is_duplicate_run": True,
            },
            # 4. Genuinely prospective PRE_MARKET on 2026-11-17 -> entry 2026-11-18 09:15 IST (PENDING)
            {
                "prediction_id": "fwd_RELIANCE_2026-11-17_PRE",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-17",
                "prediction_timestamp": "2026-11-18T03:00:20+05:30",
                "nse_session": "PRE_MARKET",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "prob_1d_up": 0.62,
                "prob_5d_up": 0.61,
                "prob_10d_up": 0.63,
                "model_availability": '{"1D": true, "5D": true, "10D": true}',
                "is_duplicate_run": False,
            },
            # 5. Post-close POST_MARKET on 2026-11-17 -> 1D HISTORICAL (after 2026-11-18 15:30 close), 5D INVALID
            {
                "prediction_id": "fwd_RELIANCE_2026-11-17_POST",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-17",
                "prediction_timestamp": "2026-11-18T17:32:47+05:30",
                "nse_session": "POST_MARKET",
                "model_version": "5.3.0-pilot",
                "feature_set_version": "237+17",
                "prob_1d_up": 0.62,
                "prob_5d_up": 0.61,
                "prob_10d_up": 0.63,
                "model_availability": '{"1D": true, "5D": true, "10D": true}',
                "is_duplicate_run": False,
            },
        ]
    )


def _sample_outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_id": "fwd_2026-11-16_RELIANCE",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-16",
                "horizon": "1D",
                "status": "MATURED",
                "entry_date": "2026-11-17",
                "entry_price": 1339.0,
                "exit_date": "2026-11-17",
                "exit_price": 1367.0,
                "gross_return_pct": 2.0911,
                "transaction_cost_pct": 0.3535,
                "net_return_pct": 1.7376,
                "benchmark_return_pct": 0.5,
                "corporate_action_adjusted": False,
            },
            {
                "prediction_id": "fwd_RELIANCE_2026-11-16_INTRADAY",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-16",
                "horizon": "1D",
                "status": "MATURED",
                "entry_date": "2026-11-17",
                "entry_price": 1339.0,
                "exit_date": "2026-11-17",
                "exit_price": 1367.0,
                "gross_return_pct": 2.0911,
                "transaction_cost_pct": 0.3535,
                "net_return_pct": 1.7376,
                "benchmark_return_pct": 0.5,
                "corporate_action_adjusted": False,
            },
            {
                "prediction_id": "fwd_RELIANCE_2026-11-17_PRE",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-17",
                "horizon": "1D",
                "status": "PENDING",
            },
            {
                "prediction_id": "fwd_RELIANCE_2026-11-17_POST",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-17",
                "horizon": "1D",
                "status": "PENDING",
            },
        ]
    )


def test_classification_post_event_and_counts() -> None:
    preds = _sample_predictions()
    outs = _sample_outcomes()
    records, sums, metrics = classify_forward_predictions(preds, outs)

    by_id_1d = {r.prediction_id: r for r in records if r.horizon == "1D"}
    assert by_id_1d["fwd_2026-11-16_RELIANCE"].reporting_class == "MATURED"
    assert by_id_1d["fwd_RELIANCE_2026-11-16_INTRADAY"].reporting_class == "INVALID"
    assert by_id_1d["fwd_RELIANCE_2026-11-16_DUP"].reporting_class == "DUPLICATE"
    assert by_id_1d["fwd_RELIANCE_2026-11-17_PRE"].reporting_class == "PENDING"
    assert by_id_1d["fwd_RELIANCE_2026-11-17_POST"].reporting_class == "HISTORICAL"

    by_id_5d = {r.prediction_id: r for r in records if r.horizon == "5D"}
    assert by_id_5d["fwd_RELIANCE_2026-11-17_POST"].reporting_class == "INVALID"

    s1 = sums["1D"]
    assert s1.total_stored == 5
    assert s1.unique_canonical == 4
    assert s1.prospective == 2
    assert s1.matured == 1
    assert s1.pending == 1
    assert s1.historical == 1
    assert s1.invalid == 1
    assert s1.duplicate == 1

    assert metrics["1D"].n_matured_prospective == 1
    assert metrics["1D"].directional_accuracy_pct == 100.0
    assert metrics["1D"].statistically_sufficient is False


def test_classification_duplicate_target_session() -> None:
    preds = pd.DataFrame(
        [
            {
                "prediction_id": "fwd_early",
                "symbol": "TCS",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-16T19:00:00+05:30",
                "nse_session": "POST_MARKET",
                "prob_1d_up": 0.6,
                "is_duplicate_run": False,
            },
            {
                "prediction_id": "fwd_late",
                "symbol": "TCS",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T08:45:00+05:30",
                "nse_session": "PRE_MARKET",
                "prob_1d_up": 0.61,
                "is_duplicate_run": False,
            },
        ]
    )
    records, sums, _ = classify_forward_predictions(preds, None, horizons=("1D",))
    by_id = {r.prediction_id: r for r in records}
    assert by_id["fwd_early"].provenance_class == "PROSPECTIVE"
    assert by_id["fwd_late"].provenance_class == "DUPLICATE"
    assert sums["1D"].prospective == 1
    assert sums["1D"].duplicate == 1


def test_quarantine_leak_detected() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "ev1",
                "ticker": "ITC",
                "source_url": "https://example.com/bad",
                "event_category": "company_news",
                "event_description": "ITC hotel demerger update",
                "publication_timestamp": "2026-11-17T10:00:00+05:30",
                "collection_timestamp": "2026-11-17T10:05:00+05:30",
                "sentiment": "neutral",
                "classification_confidence": 0.8,
                "novelty_score": 0.5,
                "source_credibility": 0.7,
                "is_duplicate_report": False,
            }
        ]
    )
    quarantined = pd.DataFrame(
        [
            {
                "symbol": "ITC",
                "url": "https://example.com/bad",
                "quarantine_reason": "irrelevant_entity_contamination",
            }
        ]
    )
    res = check_news_integrity(events, quarantined)
    leak_check = next(c for c in res.checks if c.check_id == "NEWS_QUARANTINE_LEAK")
    assert leak_check.passed is False
    assert leak_check.affected_records == 1


def test_benchmark_key_mismatch_detected() -> None:
    indices = pd.DataFrame(
        [
            {
                "index_code": "NIFTY_50",
                "observation_date": "2026-11-17",
                "open": 24000.0,
                "close": 24100.0,
            }
        ]
    )
    res = check_benchmark_integrity(indices)
    chk = next(c for c in res.checks if c.check_id == "BENCH_KEY_MISMATCH")
    assert chk.passed is False
    assert chk.severity == "ERROR"


def test_finalised_outcome_modification_detected() -> None:
    preds = _sample_predictions().iloc[:1]
    equities = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "observation_date": "2026-11-16", "open": 1330.0, "close": 1335.0},
            {"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1339.0, "close": 1367.0},
        ]
    )
    ref_out = _sample_outcomes().iloc[:1].copy()
    cur_out = ref_out.copy()
    cur_out.loc[cur_out.index[0], "exit_price"] = 1399.0

    res = check_forward_outcomes(cur_out, preds, equities, reference_outcomes_df=ref_out)
    chk = next(c for c in res.checks if c.check_id == "OUT_MODIFIED_AFTER_FINALISATION")
    assert chk.passed is False
    assert chk.severity == "CRITICAL"


def test_probability_out_of_range_detected() -> None:
    preds = _sample_predictions().iloc[:1].copy()
    preds.loc[preds.index[0], "prob_1d_up"] = 1.25
    res = check_forward_predictions(preds)
    chk = next(c for c in res.checks if c.check_id == "PRED_PROB_RANGE")
    assert chk.passed is False
    assert chk.severity == "CRITICAL"


def test_legacy_prediction_log_classification() -> None:
    df = pd.DataFrame(
        [
            {"prediction_id": "p1", "symbol": "RELIANCE", "observation_date": "2026-11-16", "model_version": "3.0.0", "prob_next_day_up": 0.6, "realized_next_day_return_pct": 1.0},
            {"prediction_id": "p2", "symbol": "RELIANCE", "observation_date": "2026-11-16", "model_version": "3.2.0", "prob_next_day_up": 0.6, "realized_next_day_return_pct": 1.0},
            {"prediction_id": "p3", "symbol": "TCS", "observation_date": "2026-11-17", "model_version": "3.2.0", "prob_next_day_up": 0.4, "realized_next_day_return_pct": None},
        ]
    )
    s = classify_legacy_prediction_log(df)
    assert s.total_stored == 3
    assert s.unique_canonical_symbol_date == 2
    assert s.duplicate == 1
    assert s.evaluated == 2
    assert s.pending == 1
    assert s.deduped_1d_accuracy_pct == 100.0
