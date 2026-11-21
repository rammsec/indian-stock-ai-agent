"""Generate verified historical 5m datasets, run 9-model evaluations, persist to SQLite, and export Lab JSONs."""

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


def generate_historical_lab_artifacts(max_dates: int = 5) -> None:
    LAB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    refresh_universe_cache(LAB_DEFAULT_UNIVERSE, days=60)
    avail_dates = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=20)
    ready_dates = [d for d in avail_dates if d.all_stocks_ready][:max_dates]

    engine = HistoricalLabReplayEngine()
    db = HistoricalLabDatabase()

    for d_info in ready_dates:
        target_d = date.fromisoformat(d_info.date)
        date_runs = {}
        print(f"Evaluating 9 models on verified historical date {d_info.date}...")
        for sid in STRATEGY_REGISTRY:
            exec_mode = "AI_AGENT" if sid.startswith("AGENT-") else "DETERMINISTIC"
            bundle = engine.run_historical_day(
                target_date=target_d,
                strategy_id=sid,
                symbols=LAB_DEFAULT_UNIVERSE,
                data_mode=DataSourceMode.HISTORICAL_REAL,
                execution_mode=exec_mode,  # type: ignore[arg-type]
            )
            db.save_run_bundle(bundle)
            date_runs[sid] = bundle

        # Persist Stage C experiment suggestion for top strategy
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

    index_payload = {
        "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
        "initial_capital_inr": 50_000.0,
        "universe": list(LAB_DEFAULT_UNIVERSE),
        "strategies": [
            {"id": sid, "name": meta["name"], "kind": meta["kind"]}
            for sid, meta in STRATEGY_REGISTRY.items()
        ],
        "available_dates": [
            {
                "date": d.date,
                "all_stocks_ready": d.all_stocks_ready,
                "ready_count": d.ready_count,
                "total_stocks": d.total_stocks,
                "stocks": [
                    {
                        "symbol": s.symbol,
                        "data_availability": s.data_availability,
                        "candles_count": s.candles_count,
                        "missing_intervals_count": s.missing_intervals_count,
                        "provider": s.provider,
                        "status": s.status,
                    }
                    for s in d.stocks
                ],
            }
            for d in ready_dates
        ],
        "history_runs": db.list_runs(),
    }
    (LAB_DATA_DIR / "lab_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"Historical Lab data generated for {len(ready_dates)} verified dates at {LAB_DATA_DIR}")


if __name__ == "__main__":
    generate_historical_lab_artifacts(max_dates=5)
