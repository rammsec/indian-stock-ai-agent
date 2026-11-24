#!/usr/bin/env python3
"""CYCLE-011 CLI: Run Chronological Walk-Forward 1m NSE Calibration & OOS Evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trader.authority import PermissionType
from trader.data.loader_1m import DEFAULT_1M_DIR, DEFAULT_SYMBOLS_1M, load_1m_sessions
from trader.evaluation.walkforward_1m import run_walkforward_1m_calibration
from trader.timing.central_directional_agent import DirectionalBias


def main() -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward 1m NSE calibration on genuine Parquet archives.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS_1M))
    parser.add_argument("--data-dir", default=str(DEFAULT_1M_DIR))
    parser.add_argument("--max-days", type=int, default=15, help="Most recent N trading days per symbol to evaluate.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--direction", choices=("BULLISH", "BEARISH"), default="BULLISH")
    parser.add_argument("--output-json", default="data/intraday_1m/walkforward_calibration_report.json")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bundles = []
    for sym in symbols:
        try:
            bundles.extend(load_1m_sessions(sym, base_dir=args.data_dir, max_days=args.max_days))
        except FileNotFoundError as exc:
            print(f"[walkforward_1m] SKIP {sym}: {exc}")

    if not bundles:
        raise RuntimeError(f"No 1m session bundles loaded from {args.data_dir}. Run scripts/fetch_free_nse_1m_data.py first.")

    report = run_walkforward_1m_calibration(
        bundles,
        direction=DirectionalBias[args.direction],
        human_permission=PermissionType.ALLOW_BOTH,
        train_ratio=args.train_ratio,
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
