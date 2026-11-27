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

from trader.authority.gate import MandatoryExecutionGate
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
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine
from trader.timing.autonomous_session_controller import (
    AutonomousLongOnlySessionController,
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
    parser.add_argument(
        "--autonomous-long-only",
        action="store_true",
        help="Enable AutonomousLongOnlySessionController (Stage A preflight -> 09:15 NO_NEW_ENTRIES -> 09:30 ALLOW_LONG -> 15:15 EOD square-off)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run Stage A preflight verification, print JSON result without secrets, and exit 0 (PASS) or 2 (FAIL)",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=None,
        help="Optional single-instance PID lock file path to prevent duplicate runner processes",
    )
    return parser


def _scan_jsonl_time_bounds(jsonl_file: Path | None) -> tuple[datetime | None, datetime | None]:
    """Scan an offline JSONL tick file for its earliest and latest IST timestamps."""
    if jsonl_file is None or not jsonl_file.exists():
        return None, None
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    with jsonl_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts_raw = rec.get("timestamp_ist") if isinstance(rec, dict) else None
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_raw))
            except ValueError:
                continue
            ts = ts.replace(tzinfo=IST_TZ) if ts.tzinfo is None else ts.astimezone(IST_TZ)
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts
    return min_ts, max_ts


def main(argv: list[str] | None = None) -> int:
    assert_paper_only_live_runner()
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.autonomous_long_only and args.human_permission in ("ALLOW_SHORT", "ALLOW_BOTH"):
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "error": (
                        f"--autonomous-long-only NEVER permits --human-permission {args.human_permission}"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    clean_symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    now_ist = datetime.now(IST_TZ)
    min_ts, max_ts = (
        _scan_jsonl_time_bounds(args.jsonl_file)
        if args.provider == "offline_jsonl"
        else (None, None)
    )
    session_date = min_ts.date() if min_ts is not None else now_ist.date()

    if args.preflight_only:
        portfolio = IntradayPaperPortfolio(initial_capital=float(args.capital))
        risk_engine = DeterministicRiskEngine()
        controller = AutonomousLongOnlySessionController(
            session_date=session_date,
            provider=args.provider,
            symbols=tuple(clean_symbols),
            output_dir=args.output_dir,
        )
        preflight = controller.run_preflight(
            risk_engine=risk_engine,
            portfolio=portfolio,
            lock_file=args.lock_file,
        )
        controller.release_lock()
        print(json.dumps(preflight.to_dict(), indent=2))
        return 0 if preflight.passed else 2

    token_map = parse_token_map(args.token_map)
    adapter = build_adapter(args.provider, args.jsonl_file, token_map=token_map)

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
                    "autonomous_long_only": bool(args.autonomous_long_only),
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

    valid_from = now_ist - timedelta(hours=24)
    expires_at = now_ist + timedelta(hours=24)
    if min_ts is not None and max_ts is not None:
        valid_from = min_ts - timedelta(hours=1)
        expires_at = max_ts + timedelta(hours=24)

    cfg = LivePaperConfig(
        strategy_profile=args.profile,
        initial_capital_inr=args.capital,
        auto_direction=(args.central_direction == "AUTO"),
        min_ml_win_prob=args.min_ml_prob,
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
            valid_from=valid_from,
            expires_at=expires_at,
            reason=f"Explicit CLI Central Direction: {args.central_direction}",
            version="1.0",
        )

    session_controller: AutonomousLongOnlySessionController | None = None
    portfolio: IntradayPaperPortfolio | None = None
    execution_gate: MandatoryExecutionGate | None = None
    human_instr: AuthorityInstruction | None = None

    if args.autonomous_long_only:
        portfolio = IntradayPaperPortfolio(initial_capital=float(cfg.initial_capital_inr))
        risk_engine = DeterministicRiskEngine(max_risk_per_trade_pct=cfg.risk_pct)
        execution_gate = MandatoryExecutionGate(risk_engine=risk_engine)
        session_controller = AutonomousLongOnlySessionController(
            session_date=session_date,
            provider=args.provider,
            symbols=tuple(clean_symbols),
            output_dir=args.output_dir,
        )
        preflight = session_controller.run_preflight(
            risk_engine=risk_engine,
            portfolio=portfolio,
            lock_file=args.lock_file,
        )
        if not preflight.passed:
            session_controller.release_lock()
            print(json.dumps(preflight.to_dict(), indent=2), file=sys.stderr)
            return 2
        obs_start_ts = min_ts if min_ts is not None else now_ist
        session_controller.start_observation(execution_gate, obs_start_ts)
    elif args.human_permission != "NONE":
        human_instr = AuthorityInstruction(
            instruction_id=f"HUMAN-CLI-{uuid.uuid4().hex[:8]}",
            authority=AuthoritySource.HUMAN,
            scope="*",
            permission=PermissionType(args.human_permission),
            valid_from=valid_from,
            expires_at=expires_at,
            reason=f"Explicit CLI Human Authority: {args.human_permission}",
            version="1.0",
        )

    runner = LivePaperTradingRunner(
        adapter=adapter,
        output_dir=args.output_dir,
        config=cfg,
        portfolio=portfolio,
        execution_gate=execution_gate,
        human_instruction=human_instr,
        central_instruction=central_instr,
        ml_scorer=ml_scorer_fn,
        session_controller=session_controller,
        long_only_standing_mode=bool(args.autonomous_long_only),
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
