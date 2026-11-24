#!/usr/bin/env python3
"""CYCLE-010 Phase 5 CLI: Read-Only Local NSE Tick/Seconds Recorder.

Records genuine multi-day NSE tick data during regular market hours (09:15-15:30 IST)
and causally resamples into 1s, 5s, 1m, 5m, and 15m Parquet archives for training
the BharatAlpha Continuous Candle Agent.

SAFETY:
- Strictly READ-ONLY (LIVE_TRADING_ENABLED = False, READ_ONLY_MODE = True).
- Zero broker order routing or execution.
- Credentials are read strictly from local environment variables (.env) and never logged.

Examples:
  # Dry-run credential & guardrail check
  python scripts/record_nse_ticks.py --provider zerodha --symbols TCS,RELIANCE --dry-run

  # Record from offline JSONL tick stream into a Parquet archive
  python scripts/record_nse_ticks.py --provider offline_jsonl --jsonl-input ticks.jsonl --symbols TCS --output-dir data/ticks

  # Live read-only Zerodha KiteTicker recording (requires KITE_API_KEY & KITE_ACCESS_TOKEN)
  python scripts/record_nse_ticks.py --provider zerodha --symbols TCS,RELIANCE,ITC --token-map tokens.json
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from trader.data.tick_recorder import (
    LIVE_TRADING_ENABLED,
    READ_ONLY_MODE,
    SUPPORTED_TIMEFRAMES,
    AngelOneSmartApiReadOnlyAdapter,
    OfflineJsonlTickAdapter,
    ReadOnlyNseTickRecorder,
    ReadOnlyTickAdapter,
    TimeframeName,
    ZerodhaKiteReadOnlyAdapter,
    assert_read_only_recorder_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BharatAlpha Read-Only NSE Tick/Seconds Recorder & Multi-Timeframe Parquet Archiver",
    )
    parser.add_argument(
        "--provider",
        choices=("zerodha", "angel_one", "offline_jsonl"),
        required=True,
        help="Read-only tick feed provider.",
    )
    parser.add_argument(
        "--symbols",
        default="TCS,RELIANCE,ITC,TATAELXSI,BIRLACABLE",
        help="Comma-separated NSE symbols to record.",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(SUPPORTED_TIMEFRAMES),
        help="Comma-separated target timeframes (subset of 1s,5s,1m,5m,15m).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ticks",
        help="Output root directory for partitioned Parquet archives.",
    )
    parser.add_argument(
        "--token-map",
        default=None,
        help="Optional JSON file mapping SYMBOL -> broker instrument token (for zerodha/angel_one).",
    )
    parser.add_argument(
        "--jsonl-input",
        default=None,
        help="Path to JSONL tick file (required when --provider offline_jsonl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate read-only guardrails, arguments, and credentials without connecting.",
    )
    return parser.parse_args(argv)


def _load_token_map(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Token map JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def build_adapter(args: argparse.Namespace) -> ReadOnlyTickAdapter:
    if args.provider == "zerodha":
        raw_map = _load_token_map(args.token_map)
        token_map = {str(k).upper(): int(v) for k, v in raw_map.items()}
        return ZerodhaKiteReadOnlyAdapter(symbol_to_token=token_map)
    if args.provider == "angel_one":
        raw_map = _load_token_map(args.token_map)
        token_map = {str(k).upper(): str(v) for k, v in raw_map.items()}
        return AngelOneSmartApiReadOnlyAdapter(symbol_to_token=token_map)
    if args.provider == "offline_jsonl":
        if not args.jsonl_input:
            raise ValueError("--jsonl-input is required when --provider offline_jsonl")
        return OfflineJsonlTickAdapter(args.jsonl_input)
    raise ValueError(f"Unsupported provider: {args.provider}")


def main(argv: list[str] | None = None) -> int:
    assert_read_only_recorder_config(
        live_trading_enabled=LIVE_TRADING_ENABLED,
        read_only=READ_ONLY_MODE,
    )
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes: list[TimeframeName] = [
        tf.strip() for tf in args.timeframes.split(",") if tf.strip()  # type: ignore[misc]
    ]
    for tf in timeframes:
        if tf not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe {tf!r}; expected subset of {SUPPORTED_TIMEFRAMES}")

    adapter = build_adapter(args)
    if args.dry_run:
        adapter.validate_credentials()
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "live_trading_enabled": LIVE_TRADING_ENABLED,
                    "read_only": READ_ONLY_MODE,
                    "provider": adapter.provider_name,
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "output_dir": str(Path(args.output_dir)),
                },
                indent=2,
            )
        )
        return 0

    recorder = ReadOnlyNseTickRecorder(
        adapter=adapter,
        output_dir=args.output_dir,
        timeframes=timeframes,
    )

    def _graceful_stop(signum: int, _frame: Any) -> None:
        print(f"\n[record_nse_ticks] Received signal {signum}; stopping read-only feed cleanly...")
        adapter.stop()

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    summary = recorder.run(symbols)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "live_trading_enabled": LIVE_TRADING_ENABLED,
                "read_only": READ_ONLY_MODE,
                "provider": summary.provider,
                "symbols": summary.symbols,
                "total_ticks_accepted": summary.total_ticks_accepted,
                "out_of_order_ticks": summary.out_of_order_ticks,
                "rejected_outside_hours_ticks": summary.rejected_outside_hours_ticks,
                "bars_written_by_timeframe": summary.bars_written_by_timeframe,
                "written_files": summary.written_files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
