"""Node/DOM contract tests for `frontend/app.js` against published `docs/data/*.json` schemas."""

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


def test_app_regime_and_data_quality_contract() -> None:
    script = r"""
    const fs = require('fs');
    function makeEl(id) {
      return { id, textContent: '', innerHTML: '', style: {}, className: '', children: [] };
    }
    const elements = {};
    const getEl = (id) => {
      if (!elements[id]) elements[id] = makeEl(id);
      return elements[id];
    };
    global.document = {
      getElementById: getEl,
      querySelectorAll: () => [],
      addEventListener: () => {},
    };
    const app = require('./frontend/app.js');
    const regime = JSON.parse(fs.readFileSync('docs/data/regime.json', 'utf8'));
    const reliance = JSON.parse(fs.readFileSync('docs/data/RELIANCE.json', 'utf8'));
    const bel = JSON.parse(fs.readFileSync('docs/data/BEL.json', 'utf8'));

    // 1. Standalone regime banner render from regime.json
    app.updateRegimeBannerDOM(regime, null);
    const afterRegime = {
      regimeBadge: getEl('mr-regime-badge').textContent,
      niftyMetric: getEl('mr-nifty-metric').innerHTML,
      vixMetric: getEl('mr-vix-metric').innerHTML,
      breadthMetric: getEl('mr-breadth-metric').innerHTML,
      gateOnNull: getEl('mr-gate-badge').textContent,
    };

    // 2. Render RELIANCE (has data_quality + SHAP drivers)
    app.renderQuantAndCorporateIntelSection(reliance);
    const afterReliance = {
      gate: getEl('mr-gate-badge').textContent,
      bullDrivers: getEl('qp-bull-drivers').innerHTML,
      bearDrivers: getEl('qp-bear-drivers').innerHTML,
      modelSub: getEl('qp-model-sub').textContent,
    };

    // 3. Render BEL (data_quality is null -> must reset gate badge to UNAVAILABLE, not leak RELIANCE)
    app.renderQuantAndCorporateIntelSection(bel);
    const afterBel = {
      gate: getEl('mr-gate-badge').textContent,
      modelSub: getEl('qp-model-sub').textContent,
    };

    console.log(JSON.stringify({
      regime,
      afterRegime,
      afterReliance,
      afterBel,
    }));
    """
    res = _run_node(script)
    reg = res["regime"]
    ar = res["afterRegime"]
    assert reg["regime"].upper() in ar["regimeBadge"]
    assert f"{reg['india_vix_close']:.2f}" in ar["vixMetric"]
    assert f"{reg['advance_count']}▲ / {reg['decline_count']}▼" in ar["breadthMetric"]
    assert "UNAVAILABLE" in ar["gateOnNull"]

    rel = res["afterReliance"]
    assert "ELIGIBLE" in rel["gate"]
    assert "SHAP]" in rel["bullDrivers"]
    assert "bps]" not in rel["bullDrivers"]
    assert "SHAP]" in rel["bearDrivers"]
    assert "bps]" not in rel["bearDrivers"]

    bel = res["afterBel"]
    assert bel["gate"] == "Data Quality: UNAVAILABLE"
    assert "ELIGIBLE" not in bel["gate"]


def test_app_backtest_experiments_and_forward_validation_contract() -> None:
    script = r"""
    const fs = require('fs');
    function makeEl(id) {
      return { id, textContent: '', innerHTML: '', style: {}, className: '', children: [], querySelectorAll: () => [] };
    }
    const elements = {};
    const getEl = (id) => {
      if (!elements[id]) elements[id] = makeEl(id);
      return elements[id];
    };
    global.document = {
      getElementById: getEl,
      querySelectorAll: () => [],
      addEventListener: () => {},
    };
    const app = require('./frontend/app.js');
    const bt = JSON.parse(fs.readFileSync('docs/data/backtest_summary.json', 'utf8'));
    const preds = JSON.parse(fs.readFileSync('docs/data/latest_predictions.json', 'utf8'));
    const exp = JSON.parse(fs.readFileSync('docs/data/phase5_experiments.json', 'utf8'));
    const p5 = JSON.parse(fs.readFileSync('docs/data/phase5_intelligence.json', 'utf8'));
    const fv = JSON.parse(fs.readFileSync('docs/data/forward_validation.json', 'utf8'));

    app.renderWalkForwardBacktestTable(bt);
    app.renderAllStockPredictionsMatrix(preds);
    app.renderPhase5ExperimentsTable(exp);
    app.renderPhase5Intelligence(p5);
    app.renderForwardValidation(fv);

    console.log(JSON.stringify({
      btHeading: getEl('quant-bt-heading').textContent,
      btFoldsExpected: bt.total_walk_forward_folds,
      predCountBadge: getEl('quant-pred-count-badge').textContent,
      predCountExpected: (preds.predictions || []).length,
      predTbody: getEl('tbody-quant-predictions').innerHTML,
      expPitNote: getEl('p5-experiments-pit-note').innerHTML,
      expTbody: getEl('tbody-p5-experiments').innerHTML,
      p5MonKpis: getEl('p5-monitoring-kpis').innerHTML,
      fvDistinction: getEl('fv-distinction-note').innerHTML,
      fvProvenance: getEl('fv-news-provenance-banner').innerHTML,
      fvKpis: getEl('fv-kpis').innerHTML,
      fvHorizons: getEl('tbody-fv-horizons').innerHTML,
      fvPaper: getEl('tbody-fv-paper').innerHTML,
    }));
    """
    res = _run_node(script)
    assert f"({res['btFoldsExpected']} Expanding Folds" in res["btHeading"]
    assert "7 Expanding Folds" not in res["btHeading"]
    assert res["predCountBadge"] == str(res["predCountExpected"])
    assert "SHAP]" in res["predTbody"]
    assert "HISTORICAL PROXY" in res["expPitNote"]
    assert "latest_feature_date" not in res["expPitNote"]
    assert "2026-11-16" in res["expPitNote"]
    assert "PROXY" in res["expTbody"]
    assert "All-Universe Log" in res["p5MonKpis"]
    assert "PROSPECTIVE PILOT" in res["fvDistinction"]
    assert "PRE-BASELINE NEWS ARCHIVE" in res["fvProvenance"]
    assert "Post-Cutoff Forward News" in res["fvKpis"]
    assert "Canonical Snapshots (Stored)" in res["fvKpis"]
    assert "30 (65)" in res["fvKpis"]
    assert "INSUFFICIENT_DATA" in res["fvHorizons"]
    assert "Paper ML 10D" in res["fvPaper"]


def test_build_github_pages_has_no_fabricated_live_fallbacks() -> None:
    src = (ROOT / "scripts" / "build_github_pages.py").read_text(encoding="utf-8")
    forbidden = [
        "rd_pct_of_sales: 1.5",
        "profit_cagr_5y_pct: 14.2",
        "piotroski_score: 7",
        "return_1y_pct: 12.4",
        "rsi_14: 58.5",
        "macd_hist: 1.2",
        "atr_pct: 1.8",
        "volume_Surge_ratio: 1.15",
        "promoters_pct: 50",
    ]
    for needle in forbidden:
        assert needle not in src, f"Found fabricated fallback `{needle}` in build_github_pages.py"
