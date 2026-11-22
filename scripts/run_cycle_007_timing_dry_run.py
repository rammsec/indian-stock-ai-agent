#!/usr/bin/env python3
"""CYCLE-007 — Directional Prediction & Intelligent Entry Timing (Numerical Phase) CLI Runner.

Reconstructs the genuine 2026-11-19 pre-cutoff historical snapshot into
/tmp/cycle_007_snapshot, executes Cycle007Pipeline against committed
data/intraday/*_1h.parquet under OfflineExecutionGuard, and atomically writes:
- data/lab/cycle_007_timing_dry_run_report.json
- docs/research/CYCLE_007_TIMING_DRY_RUN_REPORT.md

Zero network/API calls, zero paid LLM/visual AI calls, zero live broker
connections, zero synthetic candles, and zero holdout registry mutations.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trader.config import LAB_DEFAULT_UNIVERSE
from trader.paper.cycle006b import DEFAULT_CACHE_DIR, write_report_atomic
from trader.timing.cycle007_pipeline import Cycle007Config, Cycle007Pipeline

DEFAULT_SESSION_DATE = "2026-11-19"
DEFAULT_SNAPSHOT_DIR = Path("/tmp/cycle_007_snapshot")
DEFAULT_JSON_REPORT = Path("data/lab/cycle_007_timing_dry_run_report.json")
DEFAULT_MD_REPORT = Path("docs/research/CYCLE_007_TIMING_DRY_RUN_REPORT.md")


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


def _render_markdown_report(report: dict[str, Any]) -> str:
    prov = report.get("market_data_provenance", {})
    tf_prov = prov.get("timeframe_provenance", {})
    port = report.get("portfolio_results", {})
    costs = report.get("cost_accounting", {})
    ledger = report.get("ledger_reconciliation", {})
    holdout = report.get("holdout_registry_integrity", {})
    per_sym = report.get("per_symbol_decisions", [])
    exec_prov = report.get("execution_provenance", {})
    conf_status = report.get("confidence_calibration_status", "UNCALIBRATED")
    conf_note = report.get(
        "confidence_calibration_note",
        "Heuristic directional confidence score; not a calibrated probability of a profitable trade.",
    )

    lines = [
        "# CYCLE-007 — Directional Prediction & Intelligent Entry Timing (Numerical Phase) Dry-Run Report",
        "",
        f"- **Cycle ID**: `{report['cycle_id']}`",
        f"- **Phase**: `{report['phase']}`",
        f"- **Run Status**: **`{report['run_status']}`**",
        f"- **Session Date**: `{report['session_date']}` (as-of `{report['as_of_ist']}`)",
        f"- **Market Regime**: `{report.get('market_regime')}`",
        f"- **Confidence Calibration Status**: `{conf_status}`",
        f"- **Code Commit SHA**: `{report.get('code_commit_sha')}`",
        f"- **Completed At (IST)**: `{report.get('completed_at_ist')}`",
        f"- **Output Report SHA-256**: `{report.get('output_report_sha256')}`",
        "",
        "> **Governance & Scope Notice**: Numerical-only historical paper-trading dry run.",
        "> `qualification_allowed=false`, `unseen_evaluation_status=NOT_EVALUATED`,",
        "> `live_trading_enabled=false`, `visual_ai_enabled=false`, `synthetic_candles_used=0`,",
        "> `external_api_calls=0`, `paid_api_cost=0.0`.",
        f"> `confidence_calibration_status: {conf_status}` — {conf_note}",
        "",
        "## 1. Per-Symbol Stage 1 (Central Directional) + Stage 2 (Candle Timing) + Risk Decisions",
        "",
        "| Symbol | Stage 1 Direction | Prob Up | Exp Ret % | Stage 2 Candle Bias | PA Score | Timing Quality | Patterns | Stage 2 Decision | Alignment | Risk Gate |",
        "|---|---|---:|---:|---|---:|---:|---|---|---|---|",
    ]

    for row in per_sym:
        s1 = row["stage1_central"]
        s2 = row["stage2_timing"]
        rk = row["risk_decision"]
        pats = ", ".join(s2.get("detected_patterns") or []) or "NONE"
        lines.append(
            f"| `{row['symbol']}` | `{s1['direction']}` | {s1['prob_next_day_up']:.4f} | "
            f"{s1['expected_next_day_return_pct']:+.4f}% | `{s2['candle_bias']}` | "
            f"{s2['price_action_score']:+.4f} | {s2['timing_quality_score']:.4f} | "
            f"`{pats}` | **`{s2['decision']}`** | `{s2['alignment_status']}` | `{rk['status']}` |"
        )

    dirty_paths = exec_prov.get("dirty_paths_at_execution", [])
    dirty_paths_rendered = ", ".join(f"`{p}`" for p in dirty_paths) if dirty_paths else "`[]` (clean)"

    lines.extend(
        [
            "",
            "## 2. Timeframe Provenance (Honest Classification)",
            "",
            "| Timeframe | Classification | Notes |",
            "|---|---|---|",
            f"| **1h** | `{tf_prov.get('1h', {}).get('classification', prov.get('provenance_status', 'UNKNOWN'))}` | Committed `data/intraday/*_1h.parquet` (Yahoo Finance; never upgraded to VERIFIED) |",
            f"| **15m** | `{tf_prov.get('15m', {}).get('classification', 'UNAVAILABLE')}` | Not committed in repo; never fabricated |",
            f"| **5m** | `{tf_prov.get('5m', {}).get('classification', 'UNAVAILABLE')}` | Not committed in repo; never fabricated |",
            "",
            "## 3. Paper Portfolio & Ledger Reconciliation",
            "",
            f"- **Initial Capital**: `₹{port.get('initial_capital_inr', 0.0):,.2f}`",
            f"- **Ending Cash / Equity**: `₹{port.get('ending_cash_inr', 0.0):,.2f}` / `₹{port.get('ending_equity_inr', 0.0):,.2f}`",
            f"- **Net P&L**: `₹{port.get('net_pnl_inr', 0.0):,.2f}` (`{port.get('net_return_pct', 0.0):.4f}%`)",
            f"- **Transaction Friction**: `₹{costs.get('total_friction_inr', 0.0):,.2f}` (brokerage `₹{costs.get('brokerage_inr', 0.0):.2f}`, statutory `₹{costs.get('statutory_charges_inr', 0.0):.2f}`, slippage `₹{costs.get('slippage_inr', 0.0):.2f}`)",
            f"- **Orders Created / Filled**: `{report.get('order_count', 0)}` / `{report.get('filled_order_count', 0)}`",
            f"- **Open / Closed Positions**: `{report.get('open_positions_count', 0)}` / `{report.get('closed_trades_count', 0)}`",
            f"- **Ledger Events**: `{ledger.get('ledger_event_count', 0)}` | **SHA-256 Chain Valid**: `{ledger.get('sha256_chain_valid')}` | **Equity Reconciles**: `{ledger.get('cash_plus_invested_matches_equity')}` (`diff={ledger.get('equity_reconciliation_diff_inr', 0.0)}`)",
            "",
            "## 4. Safety, Isolation & Holdout Integrity Attestation",
            "",
            f"- `qualification_allowed`: `{str(report.get('qualification_allowed')).lower()}`",
            f"- `unseen_evaluation_status`: `{report.get('unseen_evaluation_status')}`",
            f"- `live_trading_enabled`: `{str(report.get('live_trading_enabled')).lower()}`",
            f"- `visual_ai_enabled`: `{str(report.get('visual_ai_enabled')).lower()}`",
            f"- `synthetic_candles_used`: `{report.get('synthetic_candles_used')}`",
            f"- `external_api_calls`: `{report.get('external_api_calls')}` | `broker_connections`: `{report.get('broker_connections')}` | `paid_api_cost`: `{report.get('paid_api_cost')}`",
            f"- `holdout_registry_unchanged`: `{str(holdout.get('unchanged')).lower()}` (`before={holdout.get('sha256_before')}`, `after={holdout.get('sha256_after')}`)",
            "",
            "## 5. Execution Provenance",
            "",
            f"- `implementation_commit_sha`: `{exec_prov.get('implementation_commit_sha', report.get('code_commit_sha'))}`",
            f"- `working_tree_clean_at_execution`: `{str(exec_prov.get('working_tree_clean_at_execution')).lower()}`",
            f"- `dirty_paths_at_execution`: {dirty_paths_rendered}",
            f"- `report_commit_distinction`: {exec_prov.get('report_commit_distinction', '')}",
            f"- `confidence_calibration_status`: `{conf_status}` — {conf_note}",
            "",
        ]
    )
    return "\n".join(lines)


def run_cycle_007_dry_run(
    *,
    session_date: date,
    snapshot_dir: Path,
    json_out: Path,
    md_out: Path,
) -> dict[str, Any]:
    cfg = Cycle007Config(
        session_date=session_date,
        symbols=LAB_DEFAULT_UNIVERSE,
        cache_dir=DEFAULT_CACHE_DIR,
        snapshot_dir=snapshot_dir,
    )
    pipeline = Cycle007Pipeline(cfg)
    report = pipeline.run()

    write_report_atomic(report, json_out)
    _write_text_atomic(_render_markdown_report(report), md_out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CYCLE-007 numerical timing dry run.")
    parser.add_argument("--session-date", default=DEFAULT_SESSION_DATE, help="Session date (YYYY-MM-DD)")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR), help="Directory for reconstructed snapshot")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_REPORT), help="Output JSON report path")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_REPORT), help="Output Markdown report path")
    args = parser.parse_args()

    report = run_cycle_007_dry_run(
        session_date=date.fromisoformat(args.session_date),
        snapshot_dir=Path(args.snapshot_dir),
        json_out=Path(args.json_out),
        md_out=Path(args.md_out),
    )
    print(
        f"CYCLE-007 Dry Run: status={report['run_status']}, "
        f"orders={report['order_count']}, "
        f"ledger_valid={report['ledger_reconciliation']['sha256_chain_valid']}, "
        f"holdout_unchanged={report['holdout_registry_integrity']['unchanged']}"
    )
    return 0 if report["run_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
