#!/usr/bin/env python3
"""CYCLE-023: Run 76-Day x 5-Stock Multi-Timeframe (1m vs 5m vs 15m) Walk-Forward Benchmark.

Causally resamples genuine 1m NSE data into 5m and 15m NSE-aligned bars and benchmarks
1m vs 5m vs 15m across all 5 stocks & 4 strategy profiles on a strict 70/30 chronological split.
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
from trader.evaluation.mtf_walkforward_optimizer import run_mtf_walkforward_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="CYCLE-023 1m vs 5m vs 15m Walk-Forward Benchmark")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_1M_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS_1M))
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_1M_DIR / "cycle023_mtf_76day_report.json",
    )
    args = parser.parse_args()

    report = run_mtf_walkforward_comparison(
        symbols=args.symbols,
        base_dir=args.data_dir,
        timeframes_min=(1, 5, 15),
        initial_capital_inr=args.capital,
        train_ratio=args.train_ratio,
        max_days=args.max_days,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"CYCLE-023 Multi-Timeframe Walk-Forward Complete | Capital: ₹{args.capital:,.0f} | "
        f"Train: {report['train_date_range'][0]}..{report['train_date_range'][1]} ({report['train_days']}d) | "
        f"OOS Test: {report['oos_test_date_range'][0]}..{report['oos_test_date_range'][1]} ({report['oos_test_days']}d)"
    )
    for tf_label, tf_data in report["timeframes"].items():
        print(f"\n=== Timeframe: {tf_label} (Train-Selected: {tf_data['train_selected_profile']}) ===")
        for prof, pdata in tf_data["profiles"].items():
            tr = pdata["train"]
            te = pdata["oos_test"]
            print(
                f"  {prof:18s} | "
                f"TRAIN: {tr['total_trades']:3d} tr, WR {tr['win_rate']*100:5.1f}%, PF {tr['profit_factor']:5.2f}, Net ₹{tr['total_net_pnl_inr']:+9.2f} (Fric ₹{tr['total_friction_inr']:8.2f}) | "
                f"OOS: {te['total_trades']:3d} tr, WR {te['win_rate']*100:5.1f}%, PF {te['profit_factor']:5.2f}, Net ₹{te['total_net_pnl_inr']:+9.2f} (Fric ₹{te['total_friction_inr']:8.2f})"
            )
    sel = report["train_selected_overall"]
    if sel:
        oos = sel["oos_test"]
        print(
            f"\n>>> Train-Selected Overall Winner: {sel['timeframe']} / {sel['profile']} -> "
            f"Unseen OOS: {oos['total_trades']} trades, WR {oos['win_rate']*100:.1f}%, "
            f"PF {oos['profit_factor']:.2f}, Gross ₹{oos['total_gross_pnl_inr']:+.2f}, "
            f"Fric ₹{oos['total_friction_inr']:.2f}, Net ₹{oos['total_net_pnl_inr']:+.2f}"
        )
    print(f"Saved report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
