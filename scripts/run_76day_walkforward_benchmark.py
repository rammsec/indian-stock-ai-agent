#!/usr/bin/env python3
"""CYCLE-020: Run Full 76-Day x 5-Stock Strictly Chronological Walk-Forward Benchmark.

Evaluates all 4 strategy profiles across all 5 genuine NSE 1-minute stocks (`TCS`, `RELIANCE`,
`ITC`, `TATAELXSI`, `BIRLACABLE`) on a strict 70% Train / 30% unseen Out-of-Sample Test split
(`max(train_dates) < min(test_dates)`) across ₹50,000 and ₹2,00,000 capital tiers.

Strictly paper-only (`LIVE_TRADING_ENABLED = False`). Reports honest net-of-NSE-tax metrics.
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
from trader.evaluation.walkforward_optimizer import (
    ALL_STRATEGY_PROFILES,
    run_walkforward_optimization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CYCLE-020 76-Day Walk-Forward 1m Profit Optimizer")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_1M_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS_1M))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--capitals", nargs="+", type=float, default=[50_000.0, 200_000.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_1M_DIR / "cycle020_76day_walkforward_report.json",
    )
    args = parser.parse_args()

    report = run_walkforward_optimization(
        symbols=args.symbols,
        base_dir=args.data_dir,
        profiles=ALL_STRATEGY_PROFILES,
        capital_tiers=args.capitals,
        train_ratio=args.train_ratio,
        max_days=args.max_days,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"CYCLE-020 Walk-Forward Benchmark Complete | "
        f"Train: {report['train_date_range'][0]}..{report['train_date_range'][1]} ({report['train_days']}d, {report['train_sessions']} sessions) | "
        f"OOS Test: {report['oos_test_date_range'][0]}..{report['oos_test_date_range'][1]} ({report['oos_test_days']}d, {report['oos_test_sessions']} sessions)"
    )
    for cap_key, tier in report["capital_tiers"].items():
        print(f"\n=== Capital Tier: {cap_key} (Train-Selected Profile: {tier['train_selected_profile']}) ===")
        for prof, pdata in tier["profiles"].items():
            tr = pdata["train"]
            te = pdata["oos_test"]
            print(
                f"  {prof:18s} | "
                f"TRAIN: {tr['total_trades']:3d} tr, WR {tr['win_rate']*100:5.1f}%, PF {tr['profit_factor']:5.2f}, Net ₹{tr['total_net_pnl_inr']:+8.2f} (Fric ₹{tr['total_friction_inr']:6.2f}) | "
                f"OOS: {te['total_trades']:3d} tr, WR {te['win_rate']*100:5.1f}%, PF {te['profit_factor']:5.2f}, Net ₹{te['total_net_pnl_inr']:+8.2f} (Fric ₹{te['total_friction_inr']:6.2f})"
            )
    print(f"\nSaved report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
