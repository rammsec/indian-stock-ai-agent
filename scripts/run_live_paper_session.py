#!/usr/bin/env python3
"""CYCLE-021 / REMED-01: CLI for Governed Live Market-Hours Read-Only Paper-Trading Runner.

Streams live (or offline JSONL replay) NSE ticks strictly READ-ONLY, builds causal 1m bars,
runs paper-only trading governed by `MandatoryExecutionGate` + `DeterministicRiskEngine` +
`IntradayPaperPortfolio`, and simultaneously archives partitioned Parquet files + `live_paper_state.json`.

HARD SAFETY INVARIANTS:
- LIVE_TRADING_ENABLED = False
- READ_ONLY_MODE = True
- Zero broker order placement or live execution routing.
- Explicit `--token-map` required for Zerodha / Angel One adapters (fails closed if missing).
- Explicit `--human-permission` required for new entries (fail-closed if omitted).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trader.authority.instructions import (
    AuthorityInstruction,
    AuthoritySource,
    PermissionType,
)
from trader.config import IST_TZ
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


def parse_token_map(raw: str | None) -> dict[str, Any]:
    """Parse `--token-map` from either a JSON file path or an inline JSON string."""
    if not raw:
        return {}
    p = Path(raw)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--token-map must decode to a JSON object mapping symbol -> token")
    return {str(k).strip().upper(): v for k, v in data.items()}


def build_adapter(
    provider: str,
    jsonl_file: Path | None,
    token_map: dict[str, Any] | None = None,
) -> ReadOnlyTickAdapter:
    """Construct the read-only market-data adapter, failing closed when live token maps are missing."""
    tmap = token_map or {}
    if provider == "zerodha":
        if not tmap:
            raise ValueError(
                "--token-map is required when --provider=zerodha (e.g. '{\"TCS\": 2953217}' or path to JSON)"
            )
        int_map = {k: int(v) for k, v in tmap.items()}
        return ZerodhaKiteReadOnlyAdapter(symbol_to_token=int_map)
    if provider == "angel_one":
        if not tmap:
            raise ValueError(
                "--token-map is required when --provider=angel_one (e.g. '{\"TCS\": \"11536\"}' or path to JSON)"
            )
        str_map = {k: str(v) for k, v in tmap.items()}
        return AngelOneSmartApiReadOnlyAdapter(symbol_to_token=str_map)
    if provider == "offline_jsonl":
        if jsonl_file is None:
            raise ValueError("--jsonl-file is required when --provider=offline_jsonl")
        return OfflineJsonlTickAdapter(jsonl_path=jsonl_file)
    raise ValueError(f"Unsupported provider: {provider!r}")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="REMED-01 Governed Live Market-Hours Read-Only Paper-Trading Runner (LIVE_TRADING_ENABLED=False)"
    )
    parser.add_argument(
        "--provider",
        choices=("zerodha", "angel_one", "offline_jsonl"),
        required=True,
    )
    parser.add_argument("--symbols", nargs="+", default=["TCS", "RELIANCE", "ITC"])
    parser.add_argument(
        "--token-map",
        type=str,
        default=None,
        help="JSON string or path to JSON file mapping NSE symbol -> broker instrument token (required for zerodha/angel_one)",
    )
    parser.add_argument(
        "--human-permission",
        choices=("ALLOW_BOTH", "ALLOW_LONG", "ALLOW_SHORT", "NO_NEW_ENTRIES", "EXIT_ONLY", "PAUSE", "STOP_ALL", "NONE"),
        default="NO_NEW_ENTRIES",
        help="Explicit Human Authority permission (default: NO_NEW_ENTRIES fail-safe; NONE tests fail-closed missing authority)",
    )
    parser.add_argument(
        "--central-direction",
        choices=("AUTO", "BULLISH", "BEARISH", "NEUTRAL"),
        default="AUTO",
        help="Central Agent direction mode (AUTO yields to explicit BULLISH/BEARISH/NEUTRAL)",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    assert_paper_only_live_runner()
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    token_map = parse_token_map(args.token_map)
    adapter = build_adapter(args.provider, args.jsonl_file, token_map=token_map)
    clean_symbols = [s.strip().upper() for s in args.symbols if s.strip()]

    if args.provider in ("zerodha", "angel_one"):
        missing = [s for s in clean_symbols if s not in token_map]
        if missing:
            raise ValueError(f"--token-map missing instrument tokens for requested symbols: {missing}")

    if args.dry_run:
        adapter.validate_credentials()
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "provider": adapter.provider_name,
                    "symbols": clean_symbols,
                    "token_map_count": len(token_map),
                    "human_permission": args.human_permission,
                    "central_direction": args.central_direction,
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

    now_ist = datetime.now(IST_TZ)
    human_instr: AuthorityInstruction | None = None
    if args.human_permission != "NONE":
        human_instr = AuthorityInstruction(
            instruction_id=f"HUMAN-CLI-{uuid.uuid4().hex[:8]}",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType(args.human_permission),
            valid_from=now_ist - timedelta(hours=24),
            expires_at=now_ist + timedelta(hours=24),
            reason=f"Explicit CLI Human Authority: {args.human_permission}",
            version="1.0",
        )

    central_instr: AuthorityInstruction | None = None
    if args.central_direction != "AUTO":
        c_perm = (
            PermissionType.ALLOW_LONG
            if args.central_direction == "BULLISH"
            else (
                PermissionType.ALLOW_SHORT
                if args.central_direction == "BEARISH"
                else PermissionType.NO_NEW_ENTRIES
            )
        )
        central_instr = AuthorityInstruction(
            instruction_id=f"CENTRAL-CLI-{uuid.uuid4().hex[:8]}",
            authority=AuthoritySource.CENTRAL_AGENT,
            scope="*",
            permission=c_perm,
            valid_from=now_ist - timedelta(hours=24),
            expires_at=now_ist + timedelta(hours=24),
            reason=f"Explicit CLI Central Direction: {args.central_direction}",
            version="1.0",
        )

    cfg = LivePaperConfig(
        strategy_profile=args.profile,
        initial_capital_inr=args.capital,
        auto_direction=(args.central_direction == "AUTO"),
        min_ml_win_prob=args.min_ml_prob,
    )
    runner = LivePaperTradingRunner(
        adapter=adapter,
        output_dir=args.output_dir,
        config=cfg,
        human_instruction=human_instr,
        central_instruction=central_instr,
        ml_scorer=ml_scorer_fn,
    )

    def _graceful_stop(signum: int, _frame: Any) -> None:
        adapter.stop()

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    summary = runner.run(clean_symbols)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
