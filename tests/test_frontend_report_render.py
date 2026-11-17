"""Frontend Executive Research Report rendering & edge-case tests (via Node.js)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_node(script: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_report_renders_real_published_stocks_and_paper_portfolio() -> None:
    script = r"""
    const fs = require('fs');
    const r = require('./frontend/report.js');
    const p5 = JSON.parse(fs.readFileSync('docs/data/phase5_intelligence.json', 'utf8'));
    const bt = JSON.parse(fs.readFileSync('docs/data/backtest_summary.json', 'utf8'));
    const fv = JSON.parse(fs.readFileSync('docs/data/forward_validation.json', 'utf8'));
    const out = {};
    for (const sym of ['RELIANCE', 'TCS', 'TATAELXSI', 'HDFCBANK']) {
      const bundle = JSON.parse(fs.readFileSync('docs/data/' + sym + '.json', 'utf8'));
      const ctx = { bundle, p5, backtest: bt, forward: fv, now: '2026-11-17T12:00:00Z' };
      const views = r.rrBuildViews(ctx);
      out[sym] = {
        assessment: r.rrAssessment(ctx).label,
        hasPaperPortfolio: views.technical.includes('Paper portfolio'),
        hasBacktest: views.technical.includes('Historical walk-forward backtest'),
        hasForward: views.technical.includes('Live forward validation'),
        hasUndefined: Object.values(views).some((h) => /\bundefined\b/.test(h)),
        hasNaNToken: Object.values(views).some((h) => />\s*NaN\s*</.test(h)),
      };
    }
    console.log(JSON.stringify(out));
    """
    res = _run_node(script)
    for sym in ("RELIANCE", "TCS", "TATAELXSI", "HDFCBANK"):
        assert res[sym]["assessment"] in {
            "Positive", "Cautiously Positive", "Neutral", "Cautiously Negative", "Negative", "Insufficient Data"
        }
        assert res[sym]["hasPaperPortfolio"] is True
        assert res[sym]["hasBacktest"] is True
        assert res[sym]["hasForward"] is True
        assert res[sym]["hasUndefined"] is False
        assert res[sym]["hasNaNToken"] is False


def test_report_missing_data_low_quality_stale_and_unsafe_urls() -> None:
    script = r"""
    const r = require('./frontend/report.js');
    const sparseBundle = {
      overview: { symbol: 'TESTCO', company_name: 'Test Co Ltd', sector: 'IT' },
      last_5_annual_reports: [],
      last_5_quarterly_reports: [],
      cash_and_balance_sheet: {},
      stockholders: {},
      technicals: {},
      next_day_guidance: {},
      quantitative_prediction: { observation_date: '2026-10-01' },
      data_quality: { quality_score: 35, warnings: ['Missing financials'] },
      news: [{ title: 'Test Co wins deal', source: 'Test', published: '2026-10-01', url: 'javascript:alert(1)' }],
    };
    const ctx = { bundle: sparseBundle, p5: null, backtest: null, forward: null, now: '2026-11-17T12:00:00Z' };
    const assess = r.rrAssessment(ctx);
    const fresh = r.rrFreshness(ctx);
    const views = r.rrBuildViews(ctx);
    console.log(JSON.stringify({
      label: assess.label,
      stale: fresh.stale,
      hasUnavailable: views.detailed.includes('UNAVAILABLE'),
      hasUnsafeJsUrl: views.detailed.includes('javascript:alert'),
      hasBacktestUnavailable: views.technical.includes('Backtest summary UNAVAILABLE'),
      hasForwardUnavailable: views.technical.includes('Forward validation data UNAVAILABLE'),
    }));
    """
    res = _run_node(script)
    assert res["label"] == "Insufficient Data"
    assert res["stale"] is True
    assert res["hasUnavailable"] is True
    assert res["hasUnsafeJsUrl"] is False
    assert res["hasBacktestUnavailable"] is True
    assert res["hasForwardUnavailable"] is True
