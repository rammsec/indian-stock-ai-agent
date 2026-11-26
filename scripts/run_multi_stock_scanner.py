#!/usr/bin/env python3
"""CYCLE-024: CLI Multi-Stock Intraday Opportunity Scanner across 5 NSE Equities."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trader.config import IST_TZ
from trader.data.loader_1m import DEFAULT_1M_DIR, DEFAULT_SYMBOLS_1M, load_1m_sessions
from trader.intelligence.ml_setup_scorer import CausalSetupScorerModel
from trader.intelligence.multi_stock_scanner import scan_multi_stock_opportunities


def main() -> int:
    parser = argparse.ArgumentParser(description="CYCLE-024 Multi-Stock NSE Intraday Opportunity Scanner")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_1M_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS_1M))
    parser.add_argument("--date", type=str, default=None, help="Trading date YYYY-MM-DD (default: latest)")
    parser.add_argument("--time-ist", type=str, default="10:30", help="Causal scan time HH:MM IST")
    parser.add_argument("--ml-model", type=Path, default=Path("models/setup_scorer_1m.json"))
    args = parser.parse_args()

    bundles_by_sym = {}
    target_date = None
    for sym in args.symbols:
        sessions = load_1m_sessions(sym, base_dir=args.data_dir, max_days=10)
        if not sessions:
            continue
        chosen = None
        if args.date:
            for s in sessions:
                if s.trading_date.isoformat() == args.date:
                    chosen = s
                    break
        if chosen is None:
            chosen = sessions[-1]
        bundles_by_sym[sym] = chosen
        target_date = chosen.trading_date

    if not bundles_by_sym or target_date is None:
        print("No genuine 1m sessions found.", file=sys.stderr)
        return 1

    hh, mm = [int(x) for x in args.time_ist.split(":")]
    as_of = datetime(target_date.year, target_date.month, target_date.day, hh, mm, tzinfo=IST_TZ)

    ml_fn = None
    if args.ml_model and args.ml_model.exists():
        ml_fn = CausalSetupScorerModel.load_json(args.ml_model).predict_proba

    result = scan_multi_stock_opportunities(bundles_by_sym, as_of_ist=as_of, ml_scorer=ml_fn)
    print(f"CYCLE-024 Multi-Stock Scan @ {result.as_of_ist} (LIVE_TRADING_ENABLED=False)")
    for row in result.ranked_opportunities:
        adx_str = f"{row.adx_14:4.1f}" if row.adx_14 is not None else "  --"
        ml_str = f"ML={row.ml_win_prob:.2f}" if row.ml_win_prob is not None else "ML=--"
        print(
            f"  #{row.rank} {row.symbol:10s} | Score {row.opportunity_score:5.1f} | {row.action_tag:11s} | "
            f"AUTO={row.auto_direction:7s} | ADX={adx_str} | ORB={row.orb_state:17s} | {ml_str} | LTP=₹{row.last_close:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
