"""Contract tests for published JSON schemas and frontend zero-fallback integrity."""

from __future__ import annotations

import json
from pathlib import Path

from backend.models import BacktestComparisonSummary, QuantitativePredictionRecord

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_published_backtest_summary_matches_pydantic_schema() -> None:
    bt_path = ROOT_DIR / "docs" / "data" / "backtest_summary.json"
    assert bt_path.exists()
    data = json.loads(bt_path.read_text())
    summary = BacktestComparisonSummary.model_validate(data)
    assert summary.purge_gap_days >= 5
    assert summary.embargo_sessions >= 1
    assert "ml_ensemble" in summary.models
    assert "naive_momentum" in summary.models


def test_published_latest_predictions_match_pydantic_schema() -> None:
    pred_path = ROOT_DIR / "docs" / "data" / "latest_predictions.json"
    assert pred_path.exists()
    data = json.loads(pred_path.read_text())
    assert "generated_at" in data
    assert isinstance(data["predictions"], list) and len(data["predictions"]) > 0
    for item in data["predictions"]:
        rec = QuantitativePredictionRecord.model_validate(item)
        assert rec.target_session_date is not None
        assert len(rec.feature_snapshot) > 0


def test_frontend_contains_no_hardcoded_fallback_metrics() -> None:
    forbidden = [
        "Walk-Forward OOS Net Sharpe: +0.783 (+19.75% Net)",
        "0.545 : 0.495",
        "latest_delivery_pct: 52.0",
        "institutional_conviction_score: 58.0",
        "nifty_50_buy_hold_return_pct ?? -1.43",
        "cumulative_net_return_pct ?? 19.75",
        'p.observation_date || "2026-11-16"',
    ]
    for rel in ("frontend/app.js", "frontend/index.html", "docs/index.html"):
        text = (ROOT_DIR / rel).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"Found forbidden hardcoded fallback `{needle}` in {rel}"


def test_frontend_table_headers_match_rendered_fields() -> None:
    for rel in ("frontend/index.html", "docs/index.html"):
        html = (ROOT_DIR / rel).read_text(encoding="utf-8")
        assert "<th>Brier / ROC AUC</th>" in html
        assert "<th>Obs. Date</th>" in html
        assert "<th>LGBM P(Up)</th>" in html
