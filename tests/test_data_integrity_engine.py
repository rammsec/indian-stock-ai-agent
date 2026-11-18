"""Comprehensive Regression Tests for the BharatAlpha AI Data Integrity Engine & Provenance Classification."""

from __future__ import annotations

import pandas as pd

from backend.audit import (
    check_benchmark_integrity,
    check_forward_outcomes,
    check_forward_predictions,
    check_legacy_prediction_log,
    check_news_integrity,
    check_portfolio_integrity,
    check_published_reports,
    classify_forward_predictions,
    classify_legacy_prediction_log,
    classify_news_events,
    run_data_integrity_audit,
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
    assert by_id_1d["fwd_2026-11-16_RELIANCE"].has_conflict is False
    assert by_id_1d["fwd_RELIANCE_2026-11-16_INTRADAY"].reporting_class == "INVALID"
    assert by_id_1d["fwd_RELIANCE_2026-11-16_INTRADAY"].has_conflict is True
    assert by_id_1d["fwd_RELIANCE_2026-11-16_DUP"].reporting_class == "DUPLICATE"
    assert by_id_1d["fwd_RELIANCE_2026-11-17_PRE"].reporting_class == "PENDING"
    assert by_id_1d["fwd_RELIANCE_2026-11-17_POST"].reporting_class == "HISTORICAL"
    assert by_id_1d["fwd_RELIANCE_2026-11-17_POST"].has_conflict is True

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
    assert s1.conflicting_records == 2

    assert metrics["1D"].n_matured_prospective == 1
    assert metrics["1D"].directional_accuracy_pct == 100.0
    assert metrics["1D"].statistically_sufficient is False


def test_classification_boundary_exactly_at_entry_open_and_holiday() -> None:
    """Boundary: 09:14:59 IST is PROSPECTIVE; 09:15:00 IST is INVALID; 2026-11-23 skips 2026-11-24 holiday to 2026-11-25."""
    preds = pd.DataFrame(
        [
            {
                "prediction_id": "fwd_pre_open_boundary",
                "symbol": "RELIANCE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T09:14:59+05:30",
                "nse_session": "PRE_MARKET",
                "prob_1d_up": 0.55,
                "is_duplicate_run": False,
            },
            {
                "prediction_id": "fwd_at_open_boundary",
                "symbol": "TCS",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T09:15:00+05:30",
                "nse_session": "INTRADAY",
                "prob_1d_up": 0.55,
                "is_duplicate_run": False,
            },
            {
                "prediction_id": "fwd_holiday_skip",
                "symbol": "ITC",
                "observation_date": "2026-11-23",  # 2026-11-24 is in NSE_TRADING_HOLIDAYS -> next session is 2026-11-25
                "prediction_timestamp": "2026-11-24T14:00:00+05:30",
                "nse_session": "POST_MARKET",
                "prob_1d_up": 0.52,
                "is_duplicate_run": False,
            },
        ]
    )
    records, _, _ = classify_forward_predictions(preds, None, horizons=("1D",))
    by_id = {r.prediction_id: r for r in records}
    assert by_id["fwd_pre_open_boundary"].provenance_class == "PROSPECTIVE"
    assert by_id["fwd_at_open_boundary"].provenance_class == "INVALID"
    assert by_id["fwd_holiday_skip"].target_entry_date == "2026-11-25"
    assert by_id["fwd_holiday_skip"].provenance_class == "PROSPECTIVE"


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
    assert by_id["fwd_late"].has_conflict is True
    assert sums["1D"].prospective == 1
    assert sums["1D"].duplicate == 1


def test_news_collection_time_leakage_classification() -> None:
    """Article published before snapshot but collected after snapshot must be classified HISTORICAL with conflict."""
    preds = pd.DataFrame(
        [
            {
                "prediction_id": "fwd_2026-11-16_BIRLACABLE",
                "symbol": "BIRLACABLE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-16T19:23:27+05:30",
                "is_duplicate_run": False,
            }
        ]
    )
    events = pd.DataFrame(
        [
            # Published Nov 16 04:12, but collected Nov 17 10:19 (after Nov 16 19:23 snapshot)
            {
                "event_id": "c378197f3aa255ca",
                "ticker": "BIRLACABLE",
                "source_url": "https://example.com/late-collected",
                "publication_timestamp": "2026-11-16T04:12:00+05:30",
                "collection_timestamp": "2026-11-17T10:19:13+05:30",
                "effective_trading_date": "2026-11-16",
                "is_duplicate_report": False,
            },
            # Genuinely prospective: collected after earliest snapshot and published after earliest snapshot
            {
                "event_id": "prospective_ev",
                "ticker": "BIRLACABLE",
                "source_url": "https://example.com/prospective",
                "publication_timestamp": "2026-11-17T08:00:00+05:30",
                "collection_timestamp": "2026-11-17T08:05:00+05:30",
                "effective_trading_date": "2026-11-17",
                "is_duplicate_report": False,
            },
            # Missing collection_timestamp -> HISTORICAL backfill
            {
                "event_id": "backfill_ev",
                "ticker": "BIRLACABLE",
                "source_url": "https://example.com/backfill",
                "publication_timestamp": "2026-11-05T08:30:00+05:30",
                "collection_timestamp": None,
                "effective_trading_date": "2026-11-05",
                "is_duplicate_report": False,
            },
        ]
    )
    recs, summary = classify_news_events(events, None, preds)
    by_id = {r.event_id: r for r in recs}
    assert by_id["c378197f3aa255ca"].classification == "HISTORICAL"
    assert by_id["c378197f3aa255ca"].has_collection_leakage_conflict is True
    assert by_id["prospective_ev"].classification == "PROSPECTIVE"
    assert by_id["backfill_ev"].classification == "HISTORICAL"
    assert summary.historical == 2
    assert summary.prospective == 1
    assert summary.collection_leakage_conflicts == 1


def test_quarantine_leak_and_entity_misattribution_detected() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "ev_leak",
                "ticker": "ITC",
                "source_url": "https://example.com/bad",
                "original_source": "Example",
                "event_category": "company_news",
                "event_description": "ITC hotel demerger update",
                "facts_summary": "ITC update",
                "publication_timestamp": "2026-11-17T10:00:00+05:30",
                "collection_timestamp": "2026-11-17T10:05:00+05:30",
                "effective_trading_date": "2026-11-17",
                "sentiment": "neutral",
                "classification_confidence": 0.8,
                "novelty_score": 0.5,
                "source_credibility": 0.7,
                "is_duplicate_report": False,
            },
            {
                "event_id": "0f6f95d76eeab2bf",
                "ticker": "TATAELXSI",
                "source_url": "https://example.com/tcs-only",
                "original_source": "Example",
                "event_category": "company_news",
                "event_description": "TCS wins major European cloud deal",
                "facts_summary": "Tata Consultancy Services expands deal pipeline",
                "publication_timestamp": "2026-11-17T10:00:00+05:30",
                "collection_timestamp": "2026-11-17T10:05:00+05:30",
                "effective_trading_date": "2026-11-17",
                "sentiment": "positive",
                "classification_confidence": 0.8,
                "novelty_score": 0.5,
                "source_credibility": 0.7,
                "is_duplicate_report": False,
            },
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
    mis_check = next(c for c in res.checks if c.check_id == "NEWS_ENTITY_MISATTRIBUTION")
    assert leak_check.passed is False
    assert leak_check.severity == "HIGH"
    assert leak_check.affected_records == 1
    assert mis_check.passed is False
    assert mis_check.affected_records == 1


def test_benchmark_key_mismatch_and_valid_match() -> None:
    indices_bad = pd.DataFrame(
        [{"index_code": "NIFTY_50", "observation_date": "2026-11-17", "open": 24000.0, "close": 24100.0}]
    )
    res_bad = check_benchmark_integrity(indices_bad)
    chk_bad = next(c for c in res_bad.checks if c.check_id == "BENCH_KEY_MISMATCH")
    assert chk_bad.passed is False
    assert chk_bad.severity == "HIGH"

    indices_ok = pd.DataFrame(
        [{"index_code": "NIFTY 50", "observation_date": "2026-11-17", "open": 24000.0, "close": 24100.0}]
    )
    res_ok = check_benchmark_integrity(indices_ok)
    chk_ok = next(c for c in res_ok.checks if c.check_id == "BENCH_KEY_MISMATCH")
    assert chk_ok.passed is True


def test_finalised_outcome_immutability_and_invalid_prices() -> None:
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
    chk_mod = next(c for c in res.checks if c.check_id == "OUT_MODIFIED_AFTER_FINALISATION")
    assert chk_mod.passed is False
    assert chk_mod.severity == "CRITICAL"

    neg_out = ref_out.copy()
    neg_out.loc[neg_out.index[0], "entry_price"] = -10.0
    res_neg = check_forward_outcomes(neg_out, preds, equities)
    chk_px = next(c for c in res_neg.checks if c.check_id == "OUT_INVALID_PRICES")
    assert chk_px.passed is False
    assert chk_px.severity == "CRITICAL"


def test_probability_out_of_range_detected() -> None:
    preds = _sample_predictions().iloc[:1].copy()
    preds.loc[preds.index[0], "prob_1d_up"] = 1.25
    res = check_forward_predictions(preds)
    chk = next(c for c in res.checks if c.check_id == "PRED_PROB_RANGE")
    assert chk.passed is False
    assert chk.severity == "CRITICAL"


def test_portfolio_integrity_valid_and_broken_reconciliation() -> None:
    preds = pd.DataFrame(
        [
            {
                "prediction_id": "fwd_BIRLACABLE_2026-11-16_20261117T102529",
                "symbol": "BIRLACABLE",
                "observation_date": "2026-11-16",
                "prediction_timestamp": "2026-11-17T10:25:29+05:30",
                "is_duplicate_run": False,
            }
        ]
    )
    equities = pd.DataFrame(
        [
            {"symbol": "BIRLACABLE", "observation_date": "2026-11-16", "open": 322.0, "close": 309.0},
            {"symbol": "BIRLACABLE", "observation_date": "2026-11-17", "open": 300.25, "close": 294.30},
        ]
    )
    fv_payload = {
        "paper_portfolio": {
            "starting_capital_inr": 1000000.0,
            "max_position_pct": 10.0,
            "max_total_exposure_pct": 80.0,
            "strategies": {
                "10D": {
                    "starting_capital_inr": 1000000.0,
                    "ending_capital_inr": 998018.32,
                    "cash_inr": 900000.0,
                    "invested_capital_inr": 100000.0,
                    "realized_pnl_inr": 0.0,
                    "unrealized_pnl_inr": -1981.68,
                    "peak_exposure_pct": 10.02,
                    "open_positions_count": 1,
                }
            },
        }
    }
    trades_df = pd.DataFrame(
        [
            {
                "strategy_horizon": "10D",
                "prediction_id": "fwd_BIRLACABLE_2026-11-16_20261117T102529",
                "symbol": "BIRLACABLE",
                "signal_date": "2026-11-16",
                "event_date": "2026-11-17",
                "action": "ENTRY_EXECUTED",
                "entry_date": "2026-11-17",
                "entry_price": 300.25,
                "allocated_capital_inr": 100000.0,
                "position_weight_pct": 10.0,
            },
            {
                "strategy_horizon": "10D",
                "prediction_id": "fwd_BIRLACABLE_2026-11-16_20261117T102529",
                "symbol": "BIRLACABLE",
                "signal_date": "2026-11-16",
                "event_date": "2026-11-17",
                "action": "POSITION_OPEN",
                "entry_date": "2026-11-17",
                "entry_price": 300.25,
                "allocated_capital_inr": 100000.0,
                "position_weight_pct": 10.0,
                "unrealized_pnl_inr": -1981.68,
            },
        ]
    )
    res = check_portfolio_integrity(fv_payload, trades_df, preds, equities)
    by_id = {c.check_id: c for c in res.checks}
    assert by_id["PORT_CASH_INVESTED_RECONCILIATION"].passed is True
    assert by_id["PORT_EQUITY_RECONCILIATION"].passed is True
    assert by_id["PORT_UNREALIZED_PNL_MATH"].passed is True
    assert by_id["PORT_LOOKAHEAD_CONTAMINATED_ENTRY"].passed is False
    assert by_id["PORT_LOOKAHEAD_CONTAMINATED_ENTRY"].severity == "HIGH"

    # Deliberately break cash reconciliation, allocation limit, and duplicate execution
    bad_payload = {
        "paper_portfolio": {
            "starting_capital_inr": 1000000.0,
            "max_position_pct": 10.0,
            "max_total_exposure_pct": 80.0,
            "strategies": {
                "10D": {
                    "starting_capital_inr": 1000000.0,
                    "ending_capital_inr": 950000.0,
                    "cash_inr": 800000.0,
                    "invested_capital_inr": 100000.0,
                    "realized_pnl_inr": 0.0,
                    "unrealized_pnl_inr": -1981.68,
                    "peak_exposure_pct": 85.0,
                    "open_positions_count": 1,
                }
            },
        }
    }
    bad_trades = pd.concat([trades_df.iloc[:1], trades_df.iloc[:1]], ignore_index=True)
    bad_trades.loc[0, "position_weight_pct"] = 15.0
    res_bad = check_portfolio_integrity(bad_payload, bad_trades, preds, equities)
    bad_by_id = {c.check_id: c for c in res_bad.checks}
    assert bad_by_id["PORT_CASH_INVESTED_RECONCILIATION"].passed is False
    assert bad_by_id["PORT_EQUITY_RECONCILIATION"].passed is False
    assert bad_by_id["PORT_EXPOSURE_LIMIT"].passed is False
    assert bad_by_id["PORT_POSITION_ALLOCATION_LIMIT"].passed is False
    assert bad_by_id["PORT_DUPLICATE_ENTRIES"].passed is False


def test_published_reports_consistency_checks() -> None:
    fv = {"model_version": "5.3.0-pilot", "matured_outcomes_total": 10}
    p5 = {
        "generated_at": "2026-11-18T17:32:47+05:30",
        "model_version": "5.0.0-pilot",
        "monitoring": {"evaluated_outcomes_count": 60},
        "stocks": {"RELIANCE": {"symbol": "RELIANCE", "observation_date": "2026-11-18"}},
    }
    lp = {"generated_at": "2026-11-18T03:00:07+05:30"}
    preds = _sample_predictions()
    equities = pd.DataFrame([{"symbol": "RELIANCE", "observation_date": "2026-11-17", "open": 1339.0, "close": 1367.0}])

    res = check_published_reports(fv, fv, p5, p5, lp, preds, equities)
    by_id = {c.check_id: c for c in res.checks}
    assert by_id["REP_DOCS_DATA_SYNC"].passed is True
    assert by_id["REP_MODEL_VERSION_MISMATCH"].passed is False
    assert by_id["REP_OUTCOME_POPULATION_MISMATCH"].passed is False
    assert by_id["REP_MONITORING_TRUNCATION"].passed is False
    assert by_id["REP_OBS_DATE_MISMATCH"].passed is False
    assert by_id["REP_LATEST_PRED_STALE"].passed is False
    assert by_id["REP_SESSION_WORKFLOW_CONSISTENCY"].passed is False


def test_legacy_prediction_log_classification_and_checks() -> None:
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

    res = check_legacy_prediction_log(df)
    by_id = {c.check_id: c for c in res.checks}
    assert by_id["LOG_DUPLICATE_SYMBOL_DATE"].passed is False
    assert by_id["LOG_MIXED_MODEL_VERSIONS"].passed is False


def test_full_repo_data_integrity_report_schema() -> None:
    report = run_data_integrity_audit()
    assert report.overall_status == "FAIL"
    assert report.total_records_inspected > 1000
    assert report.prospective_records == 10
    assert report.historical_records == 5
    assert report.duplicate_records == 35
    assert report.invalid_records == 5
    assert report.matured_outcomes == 5
    assert report.pending_outcomes == 25
    assert report.high_findings_count >= 4
    assert len(report.individual_check_results) >= 70
