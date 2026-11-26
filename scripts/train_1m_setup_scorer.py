#!/usr/bin/env python3
"""CYCLE-022: Train & Evaluate Leak-Free Walk-Forward 1m ML Setup Scorer.

Trains a pure-NumPy L2-regularized Logistic Regression + Platt probability calibrator
strictly on chronological Train dates (`max(train_dates) < min(test_dates)`), evaluates
honest unseen OOS Brier/Log-Loss/ROC-AUC and Calibrated vs Uncalibrated OOS Net P&L,
and writes `models/setup_scorer_1m.json` with SHA-256 weight provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trader.data.loader_1m import DEFAULT_1M_DIR, DEFAULT_SYMBOLS_1M
from trader.intelligence.ml_setup_scorer import train_walkforward_setup_scorer


def main() -> int:
    parser = argparse.ArgumentParser(description="CYCLE-022 Leak-Free 1m ML Setup Scorer Trainer")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_1M_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS_1M))
    parser.add_argument(
        "--profile",
        choices=("BALANCED_ALL", "ORB_BREAKOUT", "SUPERTREND_ADX", "LIQUIDITY_REVERSAL"),
        default="ORB_BREAKOUT",
    )
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--l2-reg", type=float, default=2.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--model-out", type=Path, default=Path("models/setup_scorer_1m.json"))
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_1M_DIR / "cycle022_ml_calibration_report.json",
    )
    args = parser.parse_args()

    model, report = train_walkforward_setup_scorer(
        symbols=args.symbols,
        base_dir=args.data_dir,
        strategy_profile=args.profile,
        initial_capital_inr=args.capital,
        train_ratio=args.train_ratio,
        l2_reg=args.l2_reg,
        max_days=args.max_days,
    )

    model.save_json(args.model_out)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cal = report["oos_calibration"]
    un = report["oos_unfiltered"]
    fl = report["oos_ml_filtered"]
    print(
        f"CYCLE-022 ML Setup Scorer Complete | Profile: {args.profile} | Status: {model.calibration_status}\n"
        f"  Train: {model.train_date_range[0]}..{model.train_date_range[1]} ({model.train_samples} trades) | "
        f"OOS Test: {model.oos_test_date_range[0]}..{model.oos_test_date_range[1]} ({model.oos_samples} trades)\n"
        f"  OOS Calibration: Brier={cal['brier_score']:.5f} (Baseline={cal['baseline_brier']:.5f}), "
        f"LogLoss={cal['log_loss']:.5f}, ROC-AUC={cal['roc_auc']:.5f}, Train-Selected Threshold={model.recommended_threshold:.2f}\n"
        f"  OOS Unfiltered : {un['total_trades']:3d} tr, WR {un['win_rate']*100:5.1f}%, PF {un['profit_factor']:5.2f}, "
        f"Net ₹{un['total_net_pnl_inr']:+9.2f} (Fric ₹{un['total_friction_inr']:8.2f})\n"
        f"  OOS ML-Filtered: {fl['total_trades']:3d} tr, WR {fl['win_rate']*100:5.1f}%, PF {fl['profit_factor']:5.2f}, "
        f"Net ₹{fl['total_net_pnl_inr']:+9.2f} (Fric ₹{fl['total_friction_inr']:8.2f}, Blocked={fl['blocked_by_ml']})\n"
        f"  Saved model -> {args.model_out} (sha256={model.weights_sha256[:16]}...)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
