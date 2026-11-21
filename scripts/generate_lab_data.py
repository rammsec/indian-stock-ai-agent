"""Generate Phase 8 verified 5m datasets, 35+ session multi-day evaluation, slippage sensitivity, and Lab JSONs."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.schema import DataSourceMode
from trader.lab.data_service import (
    compute_data_quality_analytics,
    filter_research_eligible_dates,
    get_available_historical_dates,
    refresh_universe_cache,
)
from trader.lab.database import HistoricalLabDatabase
from trader.lab.replay_engine import STRATEGY_REGISTRY, HistoricalLabReplayEngine

LAB_DATA_DIR = Path("docs/lab/data")


def generate_historical_lab_artifacts(
    replay_dates_count: int = 10,
    multi_day_sessions_target: int = 35,
) -> None:
    LAB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Use existing 59-day cache if already populated; only refresh if missing
    avail_dates = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=90)
    if not avail_dates:
        refresh_universe_cache(LAB_DEFAULT_UNIVERSE, days=60)
        avail_dates = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=90)

    eligible_dates = filter_research_eligible_dates(avail_dates, min_bars_per_stock=70)
    eval_dates = eligible_dates[:replay_dates_count]
    multi_dates = sorted([date.fromisoformat(d.date) for d in eligible_dates[:multi_day_sessions_target]])

    dq_analytics = compute_data_quality_analytics(LAB_DEFAULT_UNIVERSE, min_bars_for_research=70)
    engine = HistoricalLabReplayEngine()
    db = HistoricalLabDatabase()

    # 1. Generate bar-by-bar interactive replay JSONs for recent dates
    for d_info in eval_dates:
        target_d = date.fromisoformat(d_info.date)
        date_runs = {}
        print(f"Generating interactive replay for {d_info.date} ({d_info.date_status}: {d_info.ready_count}/{d_info.total_stocks} Ready)...")
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

        out_file = LAB_DATA_DIR / f"runs_{d_info.date}.json"
        out_file.write_text(json.dumps({"date": d_info.date, "runs": date_runs}, indent=2), encoding="utf-8")

    # 2. Multi-day (35+ sessions) evaluation across all 9 models
    multi_day_leaderboard = []
    if multi_dates:
        start_d, end_d = multi_dates[0], multi_dates[-1]
        print(f"\nRunning {len(multi_dates)}-session evaluation ({start_d} -> {end_d}) across 9 models...")
        for sid in STRATEGY_REGISTRY:
            exec_mode = "AI_AGENT" if sid.startswith("AGENT-") else "DETERMINISTIC"
            res_range = engine.run_date_range(
                start_date=start_d,
                end_date=end_d,
                strategy_id=sid,
                symbols=LAB_DEFAULT_UNIVERSE,
                data_mode=DataSourceMode.HISTORICAL_REAL,
                execution_mode=exec_mode,  # type: ignore[arg-type]
                db=db,
            )
            multi_day_leaderboard.append(res_range)
            print(
                f"  {sid:28s} | Sessions: {res_range['sessions_evaluated']} | "
                f"Net: {res_range['total_return_pct']:+.2f}% (Gross: {res_range['gross_return_pct']:+.2f}%) | "
                f"Alpha vs B&H: {res_range['alpha_vs_buy_and_hold_pct']:+.2f}% | "
                f"Trades: {res_range['total_trades']} | {res_range['statistical_sufficiency']}"
            )

    # 3. Slippage sensitivity (5, 10, 20 bps) & Walk-Forward on default strategy
    slippage_sensitivity = []
    walk_forward = {}
    if len(multi_dates) >= 5:
        slippage_sensitivity = engine.run_slippage_sensitivity(
            multi_dates[0], multi_dates[-1], strategy_id="STRAT-001-VWAP-PULLBACK", slippage_bps_list=(5.0, 10.0, 20.0)
        )
        walk_forward = engine.run_walk_forward(multi_dates, strategy_id="STRAT-001-VWAP-PULLBACK")

    index_payload = {
        "generated_at_ist": datetime.now(tz=IST_TZ).isoformat(),
        "initial_capital_inr": 50_000.0,
        "universe": list(LAB_DEFAULT_UNIVERSE),
        "data_quality_analytics": dq_analytics,
        "coverage_summary": {
            "total_cached_trading_days": dq_analytics["total_available_days"],
            "evaluated_days": len(multi_dates),
            "start_date": multi_dates[0].isoformat() if multi_dates else None,
            "end_date": multi_dates[-1].isoformat() if multi_dates else None,
            "fully_ready_days": dq_analytics["fully_ready_sessions"],
            "partial_days": dq_analytics["partial_sessions"],
            "gap_note": "Yahoo Finance 5m history provides ~59 trading days; 1-2 bars/day near 15:15-15:20 IST are omitted on 4 large-caps (disclosed as PARTIAL).",
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
        "multi_day_leaderboard": sorted(
            multi_day_leaderboard,
            key=lambda r: (r["total_return_pct"], r["sharpe_ratio"]),
            reverse=True,
        ),
        "slippage_sensitivity": slippage_sensitivity,
        "walk_forward_summary": walk_forward,
        "history_runs": db.list_runs(),
        "experiments": db.list_experiments(),
    }
    (LAB_DATA_DIR / "lab_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"\nPhase 8 Lab artifacts generated: {len(multi_dates)} sessions evaluated across 9 models.")


if __name__ == "__main__":
    generate_historical_lab_artifacts(replay_dates_count=10, multi_day_sessions_target=35)
