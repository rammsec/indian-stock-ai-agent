"""Tests for immutable prediction logging, deferred outcome matching, and clean-runner recovery."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.models import QuantitativePredictionRecord
from backend.storage.duckdb_store import DuckDBStore


def _make_record(pred_id: str, prob: float = 0.56, obs_date: str = "2025-06-02") -> QuantitativePredictionRecord:
    return QuantitativePredictionRecord(
        prediction_id=pred_id,
        symbol="RELIANCE",
        observation_date=obs_date,
        target_session_date="2025-06-03",
        generated_at="2025-06-02T13:00:00+00:00",
        model_name="ml_ensemble",
        model_version="3.2.0",
        market_regime="Weak Bull",
        prob_next_day_up=prob,
        prob_lgbm_up=prob,
        prob_xgb_up=prob,
        prob_linear_up=0.52,
        expected_next_day_return_pct=0.45,
        expected_5d_return_pct=1.10,
        downside_risk_mae_pct=-1.20,
        expected_upside_mfe_pct=1.80,
        reward_to_risk_ratio=1.5,
        model_agreement_score=0.88,
        confidence_score=65.0,
        roundtrip_delivery_cost_pct=0.35,
        roundtrip_intraday_cost_pct=0.20,
        net_expected_1d_return_pct=0.25,
        net_expected_5d_return_pct=0.75,
        recommended_action="BUY / ACCUMULATE",
        abstention_reason=None,
        top_bullish_drivers=[],
        top_bearish_drivers=[],
        feature_snapshot={"pm_ret_5d_pct": 1.25},
    )


def test_prediction_records_are_immutable_and_preserve_versions(tmp_path: Path) -> None:
    store = DuckDBStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        db_path=tmp_path / "bharat.duckdb",
    )
    r1 = _make_record("RELIANCE_2025-06-02_ml_ensemble_3.2.0_run1", prob=0.56)
    r1_dup = _make_record("RELIANCE_2025-06-02_ml_ensemble_3.2.0_run1", prob=0.99)  # same id must NOT overwrite
    r2 = _make_record("RELIANCE_2025-06-02_ml_ensemble_3.2.0_run2", prob=0.58)  # different run_id preserved

    store.upsert_predictions([r1])
    store.upsert_predictions([r1_dup, r2])

    df = pd.read_parquet(store.predictions_parquet)
    assert len(df) == 2
    row1 = df[df["prediction_id"] == "RELIANCE_2025-06-02_ml_ensemble_3.2.0_run1"].iloc[0]
    assert abs(float(row1["prob_next_day_up"]) - 0.56) < 1e-6


def test_realized_outcomes_match_only_after_target_session_closes(tmp_path: Path) -> None:
    store = DuckDBStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        db_path=tmp_path / "bharat.duckdb",
    )
    rec = _make_record("RELIANCE_2025-06-02_ml_ensemble_3.2.0_run1", prob=0.56, obs_date="2025-06-02")
    store.upsert_predictions([rec])

    # Before 2025-06-03 closes, equities_df only has 2025-06-02 -> realized_next_day_return_pct stays None
    eq_before = pd.DataFrame([{"symbol": "RELIANCE", "observation_date": "2025-06-02", "close": 100.0}])
    assert store.evaluate_realized_outcomes(eq_before) == 0
    df0 = pd.read_parquet(store.predictions_parquet)
    assert pd.isna(df0.loc[0, "realized_next_day_return_pct"])

    # After 2025-06-03 closes at 102.0 -> realized_next_day_return_pct = +2.0%, prob unchanged
    eq_after = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "observation_date": "2025-06-02", "close": 100.0},
            {"symbol": "RELIANCE", "observation_date": "2025-06-03", "close": 102.0},
        ]
    )
    assert store.evaluate_realized_outcomes(eq_after) == 1
    df1 = pd.read_parquet(store.predictions_parquet)
    assert abs(float(df1.loc[0, "realized_next_day_return_pct"]) - 2.0) < 1e-4
    assert abs(float(df1.loc[0, "prob_next_day_up"]) - 0.56) < 1e-6
    assert pd.isna(df1.loc[0, "realized_5d_return_pct"])
