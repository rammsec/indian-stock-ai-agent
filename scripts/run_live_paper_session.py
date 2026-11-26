#!/usr/bin/env python3
"""CYCLE-021: CLI for Live Market-Hours Read-Only Paper-Trading Runner.

Streams live (or offline JSONL replay) NSE ticks strictly READ-ONLY, builds causal 1m bars,
runs paper-only trading with AUTO live market direction + Chandelier trailing stop + NSE taxes,
and simultaneously archives partitioned Parquet files + `live_paper_state.json`.

HARD SAFETY INVARIANTS:
- LIVE_TRADING_ENABLED = False
- READ_ONLY_MODE = True
- Zero broker order placement or live execution routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trader.data.tick_recorder import (
    AngelOneSmartApiReadOnlyAdapter,
    OfflineJsonlTickAdapter,
    ReadOnlyTickAdapter,
    ZerodhaKiteReadOnlyAdapter,
)
from trader.timing.live_paper_runner import (
    LIVE_TRADING_ENABLED,
    READ_ONLY_MODE,
    LivePaperConfig,
    LivePaperTradingRunner,
    assert_paper_only_live_runner,
)


def _build_adapter(provider: str, jsonl_file: Path | None) -> ReadOnlyTickAdapter:
    if provider == "zerodha":
        return ZerodhaKiteReadOnlyAdapter()
    if provider == "angel_one":
        return AngelOneSmartApiReadOnlyAdapter()
    if provider == "offline_jsonl":
        if jsonl_file is None:
            raise ValueError("--jsonl-file is required when --provider=offline_jsonl")
        return OfflineJsonlTickAdapter(jsonl_path=jsonl_file)
    raise ValueError(f"Unsupported provider: {provider!r}")


def main() -> int:
    assert_paper_only_live_runner()
    parser = argparse.ArgumentParser(
        description="CYCLE-021 Live Market-Hours Read-Only Paper-Trading Runner (LIVE_TRADING_ENABLED=False)"
    )
    parser.add_argument(
        "--provider",
        choices=("zerodha", "angel_one", "offline_jsonl"),
        required=True,
    )
    parser.add_argument("--symbols", nargs="+", default=["TCS", "RELIANCE", "ITC"])
    parser.add_argument(
        "--profile",
        choices=("BALANCED_ALL", "ORB_BREAKOUT", "SUPERTREND_ADX", "LIQUIDITY_REVERSAL"),
        default="ORB_BREAKOUT",
    )
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/live_paper"))
    parser.add_argument("--jsonl-file", type=Path, default=None)
    parser.add_argument("--ml-model", type=Path, default=None, help="Optional CYCLE-022 setup_scorer_1m.json")
    parser.add_argument("--min-ml-prob", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate read-only adapter config and exit")
    args = parser.parse_args()

    adapter = _build_adapter(args.provider, args.jsonl_file)
    if args.dry_run:
        adapter.validate_credentials()
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "provider": adapter.provider_name,
                    "symbols": [s.strip().upper() for s in args.symbols],
                    "profile": args.profile,
                    "capital_inr": args.capital,
                    "live_trading_enabled": LIVE_TRADING_ENABLED,
                    "read_only_mode": READ_ONLY_MODE,
                },
                indent=2,
            )
        )
        return 0

    ml_scorer_fn = None
    if args.ml_model is not None:
        from trader.intelligence.ml_setup_scorer import CausalSetupScorerModel

        model = CausalSetupScorerModel.load_json(args.ml_model)
        ml_scorer_fn = model.predict_proba

    cfg = LivePaperConfig(
        strategy_profile=args.profile,
        initial_capital_inr=args.capital,
        min_ml_win_prob=args.min_ml_prob,
    )
    runner = LivePaperTradingRunner(
        adapter=adapter,
        output_dir=args.output_dir,
        config=cfg,
        ml_scorer=ml_scorer_fn,
    )
    summary = runner.run(args.symbols)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
