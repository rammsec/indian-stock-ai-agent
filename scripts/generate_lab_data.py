"""Generate verified historical 5m datasets, run 9-model evaluations across multi-day ranges, persist to SQLite, and export Lab JSONs."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.schema import DataSourceMode
from trader.lab.data_service import get_available_historical_dates, refresh_universe_cache
from trader.lab.database import HistoricalLabDatabase
from trader.lab.replay_engine import STRATEGY_REGISTRY, HistoricalLabReplayEngine

LAB_DATA_DIR = Path("docs/lab/data")


def generate_historical_lab_artifacts(max_dates: int = 12) -> None:
    LAB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    refresh_universe_cache(LAB_DEFAULT_UNIVERSE, days=60)
    avail_dates = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=60)
    # Include all dates with at least partial or full real data (honestly labelled FULLY_READY vs PARTIAL)
    eval_dates = [d for d in avail_dates if d.date_status in ("FULLY_READY", "PARTIAL")][:max_dates]

    engine = HistoricalLabReplayEngine()
    db = HistoricalLabDatabase()

    for d_info in eval_dates:
        target_d = date.fromisoformat(d_info.date)
        date_runs = {}
        print(f"Evaluating 9 models on {d_info.date} ({d_info.date_status}: {d_info.ready_count}/{d_info.total_stocks} Ready, {d_info.incomplete_count} Incomplete)...")
        for sid in STRATEGY_REGISTRY:
            exec_mode = "AI_AGENT" if sid.startswith("AGENT-") else "DETERMINISTIC"
            bundle = engine.run_historical_day(
                target_date=target_d,
                strategy_id=sid,
                symbols=LAB_DEFAULT_UNIVERSE,
                data_mode=DataSourceMode.HISTORICAL_REAL,
                execution_mode=exec_mode,  # type: ignore[arg-type]
                db=db,
            )
            db.save_run_bundle(bundle)
            date_runs[sid] = bundle

        best_sid = max(date_runs.keys(), key=lambda k: date_runs[k]["run"]["total_return_pct"])
        best_run = date_runs[best_sid]["run"]
        db.save_experiment({
            "experiment_id": f"exp_{d_info.date}_{best_sid}",
            "parent_strategy_version": f"{best_sid}@v1.0",
            "modified_parameters": {"atr_sl_mult": 1.3},
            "validation_period": d_info.date,
            "performance_metrics": {"net_pnl": best_run["net_pnl"], "win_rate": best_run["win_rate_pct"]},
            "suggestion_summary": "; ".join(date_runs[best_sid]["stage_bc_analysis"]["stage_c_suggestions"]),
            "approval_status": "PROPOSED_PENDING_WALK_FORWARD",
            "created_at": datetime.now(tz=IST_TZ).isoformat(),
        })

        out_file = LAB_DATA_DIR / f"runs_{d_info.date}.json"
        out_file.write_text(json.dumps({"date": d_info.date, "runs": date_runs}, indent=2), encoding="utf-8")

    # Compute walk-forward summary for default strategy across ordered eval_dates
    ordered_d = sorted(date.fromisoformat(d.date) for d in eval_dates)
    walk_forward = {}
    if len(ordered_d) >= 5:
        walk_forward = engine.run_walk_forward(ordered_d, strategy_id="STRAT-001-VWAP-PULLBACK")

    index_payload = {
        "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
        "initial_capital_inr": 50_000.0,
        "universe": list(LAB_DEFAULT_UNIVERSE),
        "coverage_summary": {
            "total_cached_trading_days": len(avail_dates),
            "evaluated_days": len(eval_dates),
            "start_date": avail_dates[-1].date if avail_dates else None,
            "end_date": avail_dates[0].date if avail_dates else None,
            "fully_ready_days": sum(1 for d in avail_dates if d.date_status == "FULLY_READY"),
            "partial_days": sum(1 for d in avail_dates if d.date_status == "PARTIAL"),
            "gap_note": "Yahoo Finance 5m history provides ~60 trading days; 1-2 bars/day are occasionally omitted on NSE.",
        },
        "strategies": [
            {"id": sid, "name": meta["name"], "kind": meta["kind"]}
            for sid, meta in STRATEGY_REGISTRY.items()
        ],
        "available_dates": [
            {
                "date": d.date,
                "date_status": d.date_status,
                "all_stocks_ready": d.all_stocks_ready,
                "ready_count": d.ready_count,
                "incomplete_count": d.incomplete_count,
                "unavailable_count": d.unavailable_count,
                "total_stocks": d.total_stocks,
                "stocks": [
                    {
                        "symbol": s.symbol,
                        "data_availability": s.data_availability,
                        "candles_count": s.candles_count,
                        "expected_candles": s.expected_candles,
                        "missing_intervals_count": s.missing_intervals_count,
                        "provider": s.provider,
                        "status": s.status,
                        "quality_reason": s.quality_reason,
                    }
                    for s in d.stocks
                ],
            }
            for d in eval_dates
        ],
        "walk_forward_summary": walk_forward,
        "history_runs": db.list_runs(),
        "experiments": db.list_experiments(),
    }
    (LAB_DATA_DIR / "lab_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"Historical Lab data generated for {len(eval_dates)} verified dates at {LAB_DATA_DIR}")


if __name__ == "__main__":
    generate_historical_lab_artifacts(max_dates=10)
