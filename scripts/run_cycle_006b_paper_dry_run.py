"""CYCLE-006B Bounded Historical Paper-Trading Dry-Run CLI.

Executes Cycle006BPaperTradingRunner in strictly offline, read-only-source mode
and writes a versioned, timestamped JSON dry-run report via atomic replacement.
Never connects to brokers, never calls external APIs, and never mutates
sealed_holdout_registry.json or historical reports.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, LabRiskConfig
from trader.paper.cycle006b import (
    DEFAULT_CACHE_DIR,
    DEFAULT_INTELLIGENCE_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_REGIME_PATH,
    Cycle006BConfig,
    Cycle006BPaperTradingRunner,
    write_report_atomic,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CYCLE-006B bounded historical paper-trading dry run (no live orders / no external APIs)."
    )
    parser.add_argument(
        "--session-date",
        required=True,
        help="Historical NSE session date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(LAB_DEFAULT_UNIVERSE),
        help="Comma-separated NSE symbols (default: LAB_DEFAULT_UNIVERSE).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=LabRiskConfig().initial_capital_inr,
        help="Initial paper capital in INR (default: 50000.0).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Base slippage in basis points (default: 5.0).",
    )
    parser.add_argument(
        "--max-staleness-hours",
        type=float,
        default=72.0,
        help="Explicit maximum allowed staleness of inputs in hours (default: 72.0).",
    )
    parser.add_argument(
        "--max-input-spread-hours",
        type=float,
        default=48.0,
        help="Maximum allowed timestamp spread across input files in hours (default: 48.0).",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--intelligence-path", type=Path, default=DEFAULT_INTELLIGENCE_PATH)
    parser.add_argument("--regime-path", type=Path, default=DEFAULT_REGIME_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/cycle_006b_reports"),
        help="Directory for timestamped dry-run report output (default: /tmp/cycle_006b_reports).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output JSON path (overrides --output-dir).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if len(args.session_date.strip()) != 10:
            raise ValueError(f"Invalid --session-date {args.session_date!r}: expected YYYY-MM-DD")
        session_d = date.fromisoformat(args.session_date.strip())
        symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())

        cfg = Cycle006BConfig(
            session_date=session_d,
            symbols=symbols,
            initial_capital_inr=args.capital,
            slippage_bps=args.slippage_bps,
            max_staleness_hours=args.max_staleness_hours,
            max_input_spread_hours=args.max_input_spread_hours,
            cache_dir=args.cache_dir,
            predictions_path=args.predictions_path,
            intelligence_path=args.intelligence_path,
            regime_path=args.regime_path,
        )

        runner = Cycle006BPaperTradingRunner(cfg)
    except ValueError as exc:
        print(f"CYCLE-006B configuration error: {exc}", file=sys.stderr)
        return 2

    report = runner.run()

    if args.output_path is not None:
        out_path = args.output_path
    else:
        run_id = f"{datetime.now(tz=IST_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_path = args.output_dir / f"cycle_006b_dry_run_{session_d.isoformat()}_{run_id}.json"

    written = write_report_atomic(report, out_path)
    print(
        f"CYCLE-006B dry run finished: status={report['run_status']} | "
        f"orders={report['order_count']} | "
        f"ledger_valid={report['ledger_reconciliation']['sha256_chain_valid']} | "
        f"external_api_calls={report['external_api_calls']} | "
        f"report={written}"
    )
    return 0 if report["run_status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
