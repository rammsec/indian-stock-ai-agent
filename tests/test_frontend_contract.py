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


def test_frontend_has_three_report_views_and_default_quick_summary() -> None:
    for rel in ("frontend/index.html", "docs/index.html"):
        html = (ROOT_DIR / rel).read_text(encoding="utf-8")
        assert 'id="rr-root"' in html
        assert 'data-rr-view="quick"' in html
        assert 'data-rr-view="detailed"' in html
        assert 'data-rr-view="technical"' in html
        assert 'id="rr-pane-quick" class="rr-pane active"' in html
        assert 'id="rr-pane-detailed" class="rr-pane"' in html
        assert 'id="rr-pane-technical" class="rr-pane"' in html

    docs_html = (ROOT_DIR / "docs" / "index.html").read_text(encoding="utf-8")
    assert "renderResearchReport" in docs_html
    assert "initReportViews" in docs_html


def test_report_renderer_real_data_and_integrity_via_node() -> None:
    import subprocess

    script = r"""
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const rep = require(path.resolve("frontend/report.js"));

const expectedSections = [
  "Executive Summary",
  "Business Overview",
  "Financial Health",
  "Revenue and Profitability Trends",
  "Balance Sheet and Cash Flow",
  "Valuation",
  "Promoter and Institutional Ownership",
  "Competitive Position",
  "News and Market Intelligence",
  "Technical Analysis",
  "Quantitative Outlook \u2013 1D, 5D and 10D",
  "Key Risks",
  "What Investors Should Monitor",
  "Final Research View",
  "Sources and Data Freshness",
];
assert.deepStrictEqual(rep.RR_SECTIONS, expectedSections);

const load = (p) => JSON.parse(fs.readFileSync(path.resolve("docs/data", p), "utf8").replace(/\bNaN\b/g, "null"));
const bundle = load("RELIANCE.json");
const p5 = load("phase5_intelligence.json");
const backtest = load("backtest_summary.json");
const forward = load("forward_validation.json");

const ctx = { bundle, p5, backtest, forward, now: "2026-11-17T03:50:00Z" };
const assess = rep.rrAssessment(ctx);
assert(rep.RR_ASSESSMENTS.includes(assess.label), `Unexpected assessment: ${assess.label}`);

const views = rep.rrBuildViews(ctx);
assert(views.quick.includes(bundle.overview.name));
assert(views.quick.includes("NSE: RELIANCE"));
assert(views.quick.includes("2026-11-17"));
assert(views.quick.includes("Key positives"));
assert(views.quick.includes("Key concerns"));
assert(views.quick.includes("Important upcoming catalysts"));
assert(views.quick.includes("Data freshness:"));
assert(views.quick.includes("not investment advice"));

// Detailed view must contain sections 2..15 in exact order
let lastIdx = views.detailed.indexOf('data-rr-block="executive-summary"');
assert(lastIdx >= 0);
for (let i = 2; i <= 15; i++) {
  const marker = `id="rr-sec-${i}"`;
  const idx = views.detailed.indexOf(marker);
  assert(idx > lastIdx, `Missing or out-of-order section ${i}`);
  lastIdx = idx;
}

// Phase 5 news grouping & generic price headline exclusion
for (const grp of ["Company", "Promoter / Management", "Industry", "Competitor", "Regulatory", "Global Macro"]) {
  assert(views.detailed.includes(`<h4>${grp}`), `Missing news group ${grp}`);
}
assert(views.detailed.includes("Verified fact:"));
assert(views.detailed.includes("Model interpretation:"));
assert(rep.rrIsGenericPriceHeadline("Reliance Industries Share Price - Live NSE: RELIANCE Stock Price & Chart - Upstox"));

// Technical appendix separates historical backtest from live forward validation
assert(views.technical.includes("Historical walk-forward backtest"));
assert(views.technical.includes("Live forward validation"));
assert(views.technical.includes("INSUFFICIENT DATA"));

// Edge case 1: Missing 10D horizon renders UNAVAILABLE, never 0.00%
const missing10d = JSON.parse(JSON.stringify(bundle));
missing10d.quantitative_prediction.prob_10d_up = null;
missing10d.quantitative_prediction.expected_10d_return_pct = null;
const missingP5 = JSON.parse(JSON.stringify(p5));
delete missingP5.stocks.RELIANCE.horizon_outlooks["10D"];
const hz = rep.rrHorizons({ bundle: missing10d, p5: missingP5 });
const h10 = hz.find((x) => x.h === "10D");
assert.strictEqual(h10.available, false);
assert.strictEqual(h10.prob, null);
const viewsMissing = rep.rrBuildViews({ bundle: missing10d, p5: missingP5, backtest, forward, now: "2026-11-17T03:50:00Z" });
assert(viewsMissing.detailed.includes("UNAVAILABLE"));

// Edge case 2: Low data quality forces Insufficient Data assessment
const lowDq = JSON.parse(JSON.stringify(bundle));
lowDq.data_quality.quality_score = 35;
assert.strictEqual(rep.rrAssessment({ bundle: lowDq, p5 }).label, "Insufficient Data");

// Edge case 3: Zero-sentinel cash flow renders UNAVAILABLE, never ₹0 Cr
assert(views.detailed.includes("Cash-flow figures are reported as zero by the source and are shown as UNAVAILABLE"));
console.log("NODE_REPORT_OK");
"""
    res = subprocess.run(
        ["node", "-e", script],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "NODE_REPORT_OK" in res.stdout

