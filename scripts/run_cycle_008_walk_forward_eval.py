#!/usr/bin/env python3
"""CYCLE-008 — Historical Out-of-Sample Evaluation & Strategy Validation CLI Runner.

Executes ``Cycle008WalkForwardEvaluator`` strictly offline under
``OfflineExecutionGuard``, audits all committed historical data and holdout
partitions, runs the 5 required strategy evaluations without lookahead or
synthetic candles, and atomically writes:
- data/lab/cycle_008_walk_forward_eval_report.json
- docs/research/CYCLE_008_WALK_FORWARD_EVAL_REPORT.md
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trader.evaluation.walk_forward import Cycle008Config, Cycle008WalkForwardEvaluator
from trader.paper.cycle006b import write_report_atomic

DEFAULT_JSON_REPORT = Path("data/lab/cycle_008_walk_forward_eval_report.json")
DEFAULT_MD_REPORT = Path("docs/research/CYCLE_008_WALK_FORWARD_EVAL_REPORT.md")


def _write_text_atomic(content: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(
        f"{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def _fmt_opt(val: Any, fmt: str = ".4f", suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{float(val):{fmt}}{suffix}"


def _render_markdown_report(report: dict[str, Any]) -> str:
    p1 = report["phase1_data_readiness"]
    h1 = p1["intraday_1h_audit"]
    s1_arch = p1["stage1_historical_archives"]
    holdout = p1["holdout_registry_audit"]
    p2 = report["phase2_walk_forward_splits"]
    leak = report["data_leakage_audit"]
    strats = report["strategies"]
    exec_prov = report["execution_provenance"]
    reg_int = report["holdout_registry_integrity"]

    lines = [
        "# CYCLE-008 — Historical Out-of-Sample Evaluation & Strategy Validation Report",
        "",
        f"- **Cycle ID**: `{report['cycle_id']}`",
        f"- **Phase**: `{report['phase']}`",
        f"- **Overall Classification**: **`{report['overall_classification']}`**",
        f"- **Confidence Calibration Status**: `{report['confidence_calibration_status']}`",
        f"- **Qualification Allowed**: `{str(report['qualification_allowed']).lower()}`",
        f"- **Unseen Evaluation Status**: `{report['unseen_evaluation_status']}`",
        f"- **Live Trading Enabled**: `{str(report['live_trading_enabled']).lower()}`",
        f"- **Code Commit SHA**: `{report['code_commit_sha']}`",
        f"- **Completed At (IST)**: `{report['completed_at_ist']}`",
        f"- **Output Report SHA-256**: `{report['output_report_sha256']}`",
        "",
        "> **Governance & Evidence Notice**:",
        f"> Classification is **`{report['overall_classification']}`**.",
        f"> Reasons: {', '.join(f'`{r}`' for r in report['insufficient_evidence_reasons'])}.",
        f"> `{report['confidence_calibration_status']}` — {report['confidence_calibration_note']}",
        "",
        "## 1. Phase 1 — Historical Data Readiness & Provenance Audit",
        "",
        f"- **1h Archive Provenance**: `{h1['provenance_status']}` (`verified={h1['verified']}`, Yahoo Finance — never upgraded to `VERIFIED`)",
        f"- **Common 1h Trading Days**: `{h1['common_trading_days_count']}` (`{h1['common_date_min']}` .. `{h1['common_date_max']}`)",
        f"- **Committed 15m Archive**: `{p1['intraday_15m_audit']['classification']}` (never fabricated/substituted)",
        f"- **Committed 5m Archive**: `{p1['intraday_5m_audit']['classification']}` (never fabricated/substituted)",
        f"- **Stage 1 Archives**: `prediction_log`=`{s1_arch['prediction_log']['observation_date_count']}` dates (`{s1_arch['prediction_log']['observation_dates']}`), "
        f"`forward_predictions`=`{s1_arch['forward_predictions']['observation_date_count']}` dates, "
        f"`market_regimes`=`{s1_arch['market_regimes']['observation_date_count']}` dates, "
        f"`structured_events`=`{s1_arch['structured_events']['row_count']}` rows "
        f"(`collection_timestamp` completeness: `{s1_arch['structured_events']['collection_timestamp_completeness_pct']}%` — "
        f"`{s1_arch['structured_events']['collection_timestamp_present_count']}/{s1_arch['structured_events']['row_count']}`)",
        f"- **Holdout Registry (`sealed_holdout_registry.json`)**: `{holdout['exposed_count']} EXPOSED`, "
        f"`{holdout['available_count']} AVAILABLE`, `{holdout['sealed_count']} SEALED` (SHA-256: `{holdout['sha256']}`)",
        f"- **Date Partitioning**: `{holdout['non_holdout_date_count']}` non-holdout trading days (`{holdout['non_holdout_date_min']}` .. `{holdout['non_holdout_date_max']}`) vs "
        f"`{holdout['exposed_holdout_date_count']}` exposed holdout trading days (`{holdout['exposed_holdout_date_min']}` .. `{holdout['exposed_holdout_date_max']}`, strictly forbidden for unseen claims).",
        "",
        "### Per-Symbol 1h Archive Integrity",
        "",
        "| Symbol | Rows | Days | Date Range | Duplicates | NaNs | OHLC Violations | Timezone | >15% Jumps |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]

    for sym, meta in h1["per_symbol"].items():
        lines.append(
            f"| `{sym}` | {meta['row_count']} | {meta['trading_days']} | "
            f"`{meta['date_min']}`..`{meta['date_max']}` | {meta['duplicate_timestamps']} | "
            f"{meta['nan_count']} | {meta['ohlc_invariant_violations']} | "
            f"`{meta['timezone']}` | {meta['corporate_action_jump_count']} |"
        )

    lines.extend(
        [
            "",
            "## 2. Phase 2 — Chronological Walk-Forward Splits & Data Leakage Audit",
            "",
            "| Split | Days | Date Min | Date Max |",
            "|---|---:|---|---|",
        ]
    )
    for split_name in ("train", "validation", "dev_test_oos"):
        sp = p2["splits"][split_name]
        lines.append(
            f"| `{split_name}` | {sp['date_count']} | `{sp['date_min']}` | `{sp['date_max']}` |"
        )

    lines.extend(
        [
            "",
            f"- **Leakage Audit Passed**: `{str(leak['passed']).lower()}`",
            f"- **Exposed Holdout Dates in Walk-Forward**: `{leak['exposed_holdout_dates_in_walk_forward']}`",
            f"- **Lookahead Bar Violations**: `{leak['lookahead_bar_violations']}` (checked `{leak['evaluated_symbol_sessions_checked']}` symbol-sessions with `max_feature_ts < as_of_ist`)",
            f"- **Parameter Tuning on `dev_test_oos`**: `{leak['parameter_tuning_on_dev_test_oos']}`",
            f"- **Synthetic Candles / Fabricated Timeframes**: `{leak['synthetic_candles_used']}` / `{leak['fabricated_timeframes']}`",
            "",
            "## 3. Phase 3 — Non-Holdout Walk-Forward Strategy Comparison (`dev_test_oos`)",
            "",
            "| Strategy | Trades | Win Rate | Dir Acc | Net Return % | Net P&L (₹) | Friction (₹) | Profit Factor | Max DD % | Sharpe | Sortino | Avg MAE % | Avg MFE % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for sid in (
        "independent_candle_timing_agent",
        "simple_technical_baseline",
        "no_trade_baseline",
    ):
        m = strats[sid]["dev_test_oos"]["metrics"]
        cb = m["cost_breakdown"]
        lines.append(
            f"| `{sid}` | {m['total_trades']} | {_fmt_opt(m['win_rate'])} | "
            f"{_fmt_opt(m['directional_accuracy'])} | {cb['net_return_pct']:+.4f}% | "
            f"{cb['net_pnl_inr']:+.2f} | {cb['total_friction_inr']:.2f} | "
            f"{_fmt_opt(m['profit_factor'])} | {m['max_drawdown_pct']:.4f}% | "
            f"{_fmt_opt(m['sharpe_ratio'])} | {_fmt_opt(m['sortino_ratio'])} | "
            f"{_fmt_opt(m['avg_mae_pct'], '+.4f', '%')} | {_fmt_opt(m['avg_mfe_pct'], '+.4f', '%')} |"
        )

    lines.extend(
        [
            "",
            f"### Full Non-Holdout Walk-Forward Summary ({p2['non_holdout_total_days']} Trading Days)",
            "",
            "| Strategy | Trades | Win Rate | Dir Acc | Net Return % | Net P&L (₹) | Friction (₹) | Profit Factor | Max DD % | Sharpe | Sortino |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for sid in (
        "independent_candle_timing_agent",
        "simple_technical_baseline",
        "no_trade_baseline",
    ):
        m = strats[sid]["full_non_holdout"]["metrics"]
        cb = m["cost_breakdown"]
        lines.append(
            f"| `{sid}` | {m['total_trades']} | {_fmt_opt(m['win_rate'])} | "
            f"{_fmt_opt(m['directional_accuracy'])} | {cb['net_return_pct']:+.4f}% | "
            f"{cb['net_pnl_inr']:+.2f} | {cb['total_friction_inr']:.2f} | "
            f"{_fmt_opt(m['profit_factor'])} | {m['max_drawdown_pct']:.4f}% | "
            f"{_fmt_opt(m['sharpe_ratio'])} | {_fmt_opt(m['sortino_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Archived Historical Diagnostics (`EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY`)",
            "",
            "> These runs cover only the 3 committed archive sessions (`2026-11-17`, `2026-11-18`, `2026-11-19`) inside the exposed holdout window. Verdict: **`INSUFFICIENT_EVIDENCE`**.",
            "",
            "| Strategy | Scope | Verdict | Completed / Requested | Trades | Dir Acc | Net P&L (₹) | Friction (₹) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for sid in (
        "central_directional_agent_archived_diagnostic",
        "combined_cycle007_pipeline_archived_diagnostic",
    ):
        s = strats[sid]
        m = s["metrics"]
        cb = m["cost_breakdown"]
        lines.append(
            f"| `{sid}` | `{s['evaluation_scope']}` | **`{s['verdict']}`** | "
            f"{s['sessions_completed']}/{len(s['sessions_requested'])} | "
            f"{m['total_trades']} | {_fmt_opt(m['directional_accuracy'])} | "
            f"{cb['net_pnl_inr']:+.2f} | {cb['total_friction_inr']:.2f} |"
        )

    dirty_paths = exec_prov.get("dirty_paths_at_execution", [])
    dirty_rendered = ", ".join(f"`{p}`" for p in dirty_paths) if dirty_paths else "`[]` (clean)"

    lines.extend(
        [
            "",
            "## 5. Safety, Isolation, Holdout Integrity & Execution Provenance",
            "",
            f"- `external_api_calls`: `{report['external_api_calls']}` | `broker_connections`: `{report['broker_connections']}` | `paid_api_cost`: `{report['paid_api_cost']}`",
            f"- `synthetic_candles_used`: `{report['synthetic_candles_used']}` | `visual_ai_enabled`: `{str(report['visual_ai_enabled']).lower()}`",
            f"- `holdout_registry_unchanged`: `{str(reg_int['unchanged']).lower()}` (`before={reg_int['sha256_before']}`, `after={reg_int['sha256_after']}`)",
            f"- `implementation_commit_sha`: `{exec_prov['implementation_commit_sha']}`",
            f"- `working_tree_clean_at_execution`: `{str(exec_prov['working_tree_clean_at_execution']).lower()}`",
            f"- `dirty_paths_at_execution`: {dirty_rendered}",
            f"- `report_commit_distinction`: {exec_prov['report_commit_distinction']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_cycle_008_eval(
    *,
    json_out: Path = DEFAULT_JSON_REPORT,
    md_out: Path = DEFAULT_MD_REPORT,
    config: Cycle008Config | None = None,
) -> dict[str, Any]:
    evaluator = Cycle008WalkForwardEvaluator(config or Cycle008Config())
    report = evaluator.run()
    write_report_atomic(report, json_out)
    _write_text_atomic(_render_markdown_report(report), md_out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CYCLE-008 historical walk-forward evaluation & strategy validation."
    )
    parser.add_argument(
        "--json-out",
        default=str(DEFAULT_JSON_REPORT),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--md-out",
        default=str(DEFAULT_MD_REPORT),
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    report = run_cycle_008_eval(
        json_out=Path(args.json_out),
        md_out=Path(args.md_out),
    )
    print(
        f"CYCLE-008 Evaluation: status={report['run_status']}, "
        f"classification={report['overall_classification']}, "
        f"leakage_passed={report['data_leakage_audit']['passed']}, "
        f"holdout_unchanged={report['holdout_registry_integrity']['unchanged']}, "
        f"external_api_calls={report['external_api_calls']}"
    )
    return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
