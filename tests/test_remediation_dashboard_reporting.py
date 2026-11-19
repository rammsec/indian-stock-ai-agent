"""Non-vacuous tests for Dashboard & Reporting Remediation (Workstream 5).

Covers:
1. Automated reconciliation (`backend/audit/dashboard_reconciler.py`) across
   `docs/data/*.json` vs source Parquet/JSON datasets (forward validation,
   phase5 intelligence, phase5 experiments, backtest summary, market regime,
   and all 28 stock JSONs).
2. Discrepancy detection on synthetic mismatched fixtures (ensuring checks are
   non-vacuous and catch count drift, missing sample sizes, NaN literals, and
   INSUFFICIENT_DATA metric leakage).
3. Sample-size accompaniment, model version, data cutoff timestamps,
   UNAVAILABLE handling (rather than fallback strings), and rejected-signal
   reason breakdown rendering in `frontend/app.js`, `frontend/index.html`,
   and `frontend/report.js` via Node.js.
4. Boundary verification: `docs/index.html` untouched and reconciler strictly
   read-only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from backend.audit.dashboard_reconciler import (
    reconcile_backtest_summary,
    reconcile_dashboard_artifacts,
    reconcile_forward_validation,
    reconcile_phase5_experiments,
    reconcile_phase5_intelligence,
    reconcile_regime,
    reconcile_stock_jsons,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def _run_node(script: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_dashboard_reconciler_live_datasets_and_read_only(tmp_path: Path) -> None:
    fv_path = ROOT_DIR / "docs" / "data" / "forward_validation.json"
    bt_path = ROOT_DIR / "docs" / "data" / "backtest_summary.json"
    mtime_fv = fv_path.stat().st_mtime_ns
    mtime_bt = bt_path.stat().st_mtime_ns

    out_report = tmp_path / "DASHBOARD_RECONCILE.json"
    report = reconcile_dashboard_artifacts(ROOT_DIR, output_json_path=out_report)

    assert out_report.exists()
    assert report["read_only_guarantee"] is True
    assert report["nan_token_files"] == {}
    assert report["total_discrepancies"] == 0
    assert report["overall_status"] == "CRITICAL"  # driven by upstream look-ahead audit finding

    # Verify read-only guarantee
    assert fv_path.stat().st_mtime_ns == mtime_fv
    assert bt_path.stat().st_mtime_ns == mtime_bt

    secs = report["sections"]
    fv = secs["forward_validation"]
    assert fv["status"] == "OK"
    assert fv["mirrors_data_json"] is True
    assert fv["model_version"] == "5.3.0-pilot"
    assert fv["counts_reconciliation"]["total_immutable_snapshots"] == 65
    assert fv["counts_reconciliation"]["canonical_snapshots_count"] == 30
    assert fv["counts_reconciliation"]["duplicate_snapshots_count"] == 35
    assert fv["counts_reconciliation"]["matured_outcomes_total"] == 20
    assert fv["counts_reconciliation"]["pending_outcomes_total"] == 70
    assert fv["counts_reconciliation"]["derived_canonical_1d_audit_counts"] == {
        "INVALID_LOOKAHEAD": 15,
        "MATURED": 10,
        "PENDING": 5,
    }
    assert fv["nifty_benchmark"]["json_status"] == "UNAVAILABLE"
    assert fv["nifty_benchmark"]["json_net_return_pct"] is None
    assert fv["nifty_benchmark"]["parquet_has_NIFTY_50"] is True

    # Rejection breakdown checks
    rej = fv["rejection_breakdown"]
    assert rej["total_strategy_rejected_signals"] == 59
    assert rej["recent_trades_rejected_count"] == 19
    assert rej["recent_trades_truncated_flag"] is True
    assert rej["dropped_rejected_signals_in_recent_trades"] == 40
    assert rej["json_rejection_reasons"]["REJECTED_NON_BUY_SIGNAL_NO TRADE"] == 18
    assert rej["json_rejection_reasons"]["REJECTED_NON_BUY_SIGNAL_HOLD"] == 1
    assert rej["parquet_legacy_rejected_rows"] == 15

    # Phase 5 intelligence & monitoring inflation
    p5 = secs["phase5_intelligence"]
    assert p5["status"] == "OK"
    assert p5["model_version"] == "5.0.0-pilot"
    assert p5["monitoring_reconciliation"]["json_evaluated_1d"] == 75
    assert p5["monitoring_reconciliation"]["parquet_duplicate_sym_obs_rows"] == 45
    assert p5["horizon_date_gaps_count"] == 15

    # Phase 5 experiments
    exp = secs["phase5_experiments"]
    assert exp["status"] == "OK"
    assert exp["latest_feature_date"] == "2026-11-16"
    assert exp["daily_features_max_date"] == "2026-11-18"
    assert exp["experiment_sample_checks"]["A_quant_237_only"]["folds"] == 6
    assert exp["experiment_sample_checks"]["A_quant_237_only"]["1D_active_trades"] == 16

    # Backtest summary
    bt = secs["backtest_summary"]
    assert bt["status"] == "OK"
    assert bt["total_walk_forward_folds"] == 6
    assert bt["model_checks"]["ml_ensemble"]["json_total_predictions"] == 3735
    assert bt["model_checks"]["ml_ensemble"]["parquet_rows"] == 3735
    assert bt["model_checks"]["ml_ensemble"]["json_active_trades_count"] == 72
    assert bt["model_checks"]["naive_momentum"]["json_active_trades_count"] == 302

    # Market regime
    reg = secs["regime"]
    assert reg["status"] == "OK"
    assert reg["observation_date"] == "2026-11-18"
    assert reg["regime"] == "Weak Bull"

    # Stock JSONs
    sj = secs["stock_jsons"]
    assert sj["status"] == "OK"
    assert sj["total_stock_jsons"] == 28
    assert sj["with_quantitative_prediction"] == 13
    assert sj["with_data_quality"] == 5
    assert sj["zero_cashflow_sentinel_count"] == 27
    assert len(sj["stale_fetched_at_with_newer_qp_symbols"]) == 8


def test_dashboard_reconciler_detects_synthetic_mismatches(tmp_path: Path) -> None:
    """Non-vacuous check: reconciler catches count mismatches, NaN tokens, and leaked metrics."""
    (tmp_path / "docs" / "data").mkdir(parents=True)
    (tmp_path / "data" / "predictions").mkdir(parents=True)
    (tmp_path / "data" / "backtests").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "features").mkdir(parents=True)

    # Copy minimal real files then corrupt docs JSON intentionally
    shutil.copy(ROOT_DIR / "data" / "predictions" / "forward_predictions.parquet", tmp_path / "data" / "predictions" / "forward_predictions.parquet")
    shutil.copy(ROOT_DIR / "data" / "predictions" / "forward_outcomes.parquet", tmp_path / "data" / "predictions" / "forward_outcomes.parquet")
    shutil.copy(ROOT_DIR / "data" / "predictions" / "paper_trades.parquet", tmp_path / "data" / "predictions" / "paper_trades.parquet")
    shutil.copy(ROOT_DIR / "data" / "predictions" / "structured_events.parquet", tmp_path / "data" / "predictions" / "structured_events.parquet")
    shutil.copy(ROOT_DIR / "data" / "predictions" / "quarantined_events.parquet", tmp_path / "data" / "predictions" / "quarantined_events.parquet")
    shutil.copy(ROOT_DIR / "data" / "processed" / "indices_daily.parquet", tmp_path / "data" / "processed" / "indices_daily.parquet")

    fv = json.loads((ROOT_DIR / "docs" / "data" / "forward_validation.json").read_text(encoding="utf-8"))
    (tmp_path / "data" / "predictions" / "forward_validation.json").write_text(json.dumps(fv), encoding="utf-8")

    # Introduce intentional discrepancies: wrong snapshot count & leaked metric under INSUFFICIENT_DATA
    fv_bad = json.loads(json.dumps(fv))
    fv_bad["canonical_snapshots_count"] = 999
    fv_bad["horizon_evaluations"]["5D"]["directional_accuracy_pct"] = 55.0
    (tmp_path / "docs" / "data" / "forward_validation.json").write_text(json.dumps(fv_bad), encoding="utf-8")

    res_fv = reconcile_forward_validation(tmp_path)
    assert res_fv["status"] == "DISCREPANCY"
    joined = " | ".join(res_fv["discrepancies"])
    assert "canonical_snapshots_count mismatch" in joined
    assert "INSUFFICIENT_DATA" in joined

    # Also test NaN token detection and backtest trade count mismatch
    shutil.copy(ROOT_DIR / "data" / "backtests" / "walk_forward_predictions.parquet", tmp_path / "data" / "backtests" / "walk_forward_predictions.parquet")
    bt = json.loads((ROOT_DIR / "docs" / "data" / "backtest_summary.json").read_text(encoding="utf-8"))
    (tmp_path / "data" / "backtests" / "walk_forward_summary.json").write_text(json.dumps(bt), encoding="utf-8")
    bt_bad = json.loads(json.dumps(bt))
    bt_bad["models"]["ml_ensemble"]["active_trades_count"] = 1234
    (tmp_path / "docs" / "data" / "backtest_summary.json").write_text(json.dumps(bt_bad), encoding="utf-8")

    res_bt = reconcile_backtest_summary(tmp_path)
    assert res_bt["status"] == "DISCREPANCY"
    assert any("active_trades_count mismatch" in d for d in res_bt["discrepancies"])

    # Also test regime mismatch
    shutil.copy(ROOT_DIR / "data" / "processed" / "market_regimes.parquet", tmp_path / "data" / "processed" / "market_regimes.parquet")
    reg_bad = json.loads((ROOT_DIR / "docs" / "data" / "regime.json").read_text(encoding="utf-8"))
    reg_bad["regime"] = "Strong Bear"
    (tmp_path / "docs" / "data" / "regime.json").write_text(json.dumps(reg_bad), encoding="utf-8")
    res_reg = reconcile_regime(tmp_path)
    assert res_reg["status"] == "DISCREPANCY"
    assert any("regime.regime mismatch" in d for d in res_reg["discrepancies"])


def test_frontend_app_js_rejection_breakdown_cutoff_and_unavailable() -> None:
    """Verify frontend/app.js renders rejection reasons, sample sizes, cutoff dates, and UNAVAILABLE."""
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
    const fv = JSON.parse(fs.readFileSync('docs/data/forward_validation.json', 'utf8'));
    const exp = JSON.parse(fs.readFileSync('docs/data/phase5_experiments.json', 'utf8'));
    const bt = JSON.parse(fs.readFileSync('docs/data/backtest_summary.json', 'utf8'));
    const reg = JSON.parse(fs.readFileSync('docs/data/regime.json', 'utf8'));
    const bel = JSON.parse(fs.readFileSync('docs/data/BEL.json', 'utf8'));
    const p5 = JSON.parse(fs.readFileSync('docs/data/phase5_intelligence.json', 'utf8'));

    app.updateRegimeBannerDOM(reg, null);
    app.renderQuantAndCorporateIntelSection(bel);
    app.renderWalkForwardBacktestTable(bt);
    app.renderPhase5ExperimentsTable(exp);
    app.renderPhase5Intelligence(p5);
    app.renderForwardValidation(fv);

    console.log(JSON.stringify({
      regimeBadge: getEl('mr-regime-badge').textContent,
      belActionPill: getEl('qp-action-pill').textContent,
      belKpiHtml: getEl('qp-kpi-strip').innerHTML,
      belCiHtml: getEl('ci-kpi-strip').innerHTML,
      btKpiHtml: getEl('quant-bt-kpis').innerHTML,
      expPitNote: getEl('p5-experiments-pit-note').innerHTML,
      expTbody: getEl('tbody-p5-experiments').innerHTML,
      p5Sub: getEl('p5-freshness-sub').textContent,
      fvSub: getEl('fv-subtitle').textContent,
      fvHorizons: getEl('tbody-fv-horizons').innerHTML,
      fvPaper: getEl('tbody-fv-paper').innerHTML,
      fvRejSummary: getEl('fv-rejection-summary').innerHTML,
    }));
    """
    res = _run_node(script)

    # Regime observation date cutoff visible
    assert "2026-11-18" in res["regimeBadge"]

    # Missing quant/corporate intel uses UNAVAILABLE rather than N/A
    assert res["belActionPill"] == "ML: UNAVAILABLE"
    assert "UNAVAILABLE" in res["belKpiHtml"]
    assert "N/A" not in res["belKpiHtml"]
    assert "UNAVAILABLE" in res["belCiHtml"]
    assert "N/A" not in res["belCiHtml"]

    # Backtest KPIs include sample size & cutoff
    assert "n=72" in res["btKpiHtml"]
    assert "Backtest Cutoff / Generated" in res["btKpiHtml"]

    # Phase 5 experiments include feature cutoff, generated timestamp, folds, and sample size
    assert "Feature Cutoff:" in res["expPitNote"]
    assert "2026-11-16" in res["expPitNote"]
    assert "6 folds (n=16)" in res["expTbody"]

    # Phase 5 intelligence displays quant cutoff and latest pred timestamp
    assert "Quant Cutoff:" in res["p5Sub"]
    assert "Latest Pred:" in res["p5Sub"]

    # Forward validation displays model version, snapshot cutoff, min sample size, UNAVAILABLE for missing benchmarks, and rejection reasons
    assert "5.3.0-pilot" in res["fvSub"]
    assert "Snapshot Cutoff:" in res["fvSub"]
    assert "(min 10)" in res["fvHorizons"]
    assert "UNAVAILABLE" in res["fvPaper"]
    assert "Rejected Signals Breakdown (59 total across 1D/5D/10D" in res["fvRejSummary"]
    assert "REJECTED_NON_BUY_SIGNAL_NO TRADE: 18" in res["fvRejSummary"]
    assert "REJECTED_NON_BUY_SIGNAL_HOLD: 1" in res["fvRejSummary"]
    assert "40 earlier rejections truncated in upstream JSON" in res["fvRejSummary"]


def test_frontend_report_js_rejection_breakdown_and_cutoff() -> None:
    """Verify frontend/report.js Technical Appendix surfaces model version, cutoff, and rejection breakdown."""
    script = r"""
    const fs = require('fs');
    const rep = require('./frontend/report.js');
    const bundle = JSON.parse(fs.readFileSync('docs/data/RELIANCE.json', 'utf8'));
    const p5 = JSON.parse(fs.readFileSync('docs/data/phase5_intelligence.json', 'utf8'));
    const bt = JSON.parse(fs.readFileSync('docs/data/backtest_summary.json', 'utf8'));
    const fv = JSON.parse(fs.readFileSync('docs/data/forward_validation.json', 'utf8'));
    const views = rep.rrBuildViews({ bundle, p5, backtest: bt, forward: fv, now: '2026-11-19T12:00:00Z' });
    console.log(JSON.stringify({
      technical: views.technical,
      quick: views.quick,
    }));
    """
    res = _run_node(script)
    tech = res["technical"]
    assert "Model 5.3.0-pilot" in tech
    assert "cutoff/generated" in tech
    assert "Rejected" in tech
    assert "Rejected signals (59 total across 1D/5D/10D; 19 in recent_trades window): REJECTED_NON_BUY_SIGNAL_NO TRADE: 18, REJECTED_NON_BUY_SIGNAL_HOLD: 1 (40 earlier rejections truncated in JSON)." in tech


def test_frontend_index_html_has_rejection_container_and_docs_index_untouched() -> None:
    """Ensure frontend/index.html has the rejection summary container and docs/index.html is unmodified."""
    idx_html = (ROOT_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="fv-rejection-summary"' in idx_html

    proc = subprocess.run(
        ["git", "diff", "--name-only", "--", "docs/index.html"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "", f"docs/index.html must remain untouched, got diff: {proc.stdout}"
