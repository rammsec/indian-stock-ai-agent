"""Phase 9: Walk-Forward Cost-Aware Strategy Experiment Runner (Train / Validation / Unseen Test)."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.schema import DataSourceMode
from trader.lab.data_service import filter_research_eligible_dates, get_available_historical_dates
from trader.lab.database import HistoricalLabDatabase
from trader.lab.replay_engine import STRATEGY_REGISTRY, HistoricalLabReplayEngine
from trader.strategies.cost_aware_wrapper import CostAwareFilterWrapper
from trader.strategies.ema_cross import EMACrossStrategy
from trader.strategies.orb_breakout import ORBBreakoutStrategy
from trader.strategies.vwap_pullback import VWAPPullbackStrategy


def run_phase9_walk_forward_experiments() -> None:
    avail = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=90)
    eligible = filter_research_eligible_dates(avail, min_bars_per_stock=70)
    ordered_dates = sorted(date.fromisoformat(d.date) for d in eligible)
    print(f"Loaded {len(ordered_dates)} research-eligible sessions ({ordered_dates[0]} -> {ordered_dates[-1]})")

    # Register v2.0 cost-aware candidates temporarily in STRATEGY_REGISTRY for replay evaluation
    candidates = [
        ("STRAT-002-ORB-BREAKOUT", ORBBreakoutStrategy, 0.010, "STRAT-002-ORB-V2"),
        ("STRAT-003-EMA-CROSS", EMACrossStrategy, 0.012, "STRAT-003-EMA-V2"),
        ("STRAT-001-VWAP-PULLBACK", VWAPPullbackStrategy, 0.012, "STRAT-001-VWAP-V2"),
    ]

    engine = HistoricalLabReplayEngine()
    db = HistoricalLabDatabase()
    experiment_rows = []

    for parent_id, cls_ref, min_move, exp_sid in candidates:
        STRATEGY_REGISTRY[exp_sid] = {
            "name": f"{STRATEGY_REGISTRY[parent_id]['name']} (v2.0 Cost-Aware)",
            "cls": lambda c=cls_ref, m=min_move: CostAwareFilterWrapper(c(), min_expected_move_pct=m),
            "kind": "STRATEGY",
        }

        wf_base = engine.run_walk_forward(ordered_dates, strategy_id=parent_id)
        wf_cand = engine.run_walk_forward(ordered_dates, strategy_id=exp_sid)

        val_improved = wf_cand["validation"]["total_return_pct"] > wf_base["validation"]["total_return_pct"]
        unseen_improved = wf_cand["unseen_test"]["total_return_pct"] > wf_base["unseen_test"]["total_return_pct"]
        approved = val_improved and unseen_improved and wf_cand["unseen_test"]["total_trades"] >= 3
        status = "APPROVED_FOR_PAPER" if approved else "REJECTED_NO_OUT_OF_SAMPLE_EDGE"

        print(
            f"\n[{parent_id} -> {exp_sid}] Status: {status}\n"
            f"  Baseline  | Train: {wf_base['train']['total_return_pct']:+.2f}% ({wf_base['train']['total_trades']} tr) | "
            f"Val: {wf_base['validation']['total_return_pct']:+.2f}% ({wf_base['validation']['total_trades']} tr) | "
            f"Unseen: {wf_base['unseen_test']['total_return_pct']:+.2f}% ({wf_base['unseen_test']['total_trades']} tr)\n"
            f"  CostAware | Train: {wf_cand['train']['total_return_pct']:+.2f}% ({wf_cand['train']['total_trades']} tr) | "
            f"Val: {wf_cand['validation']['total_return_pct']:+.2f}% ({wf_cand['validation']['total_trades']} tr) | "
            f"Unseen: {wf_cand['unseen_test']['total_return_pct']:+.2f}% ({wf_cand['unseen_test']['total_trades']} tr)"
        )

        exp_record = {
            "experiment_id": f"exp_phase9_{exp_sid}",
            "parent_strategy_version": f"{parent_id}@v1.0",
            "modified_parameters": {
                "min_expected_move_pct": min_move,
                "require_above_vwap": True,
                "require_ema_alignment": True,
                "entry_window": "09:30-14:00",
            },
            "validation_period": f"{wf_cand['validation']['start_date']} -> {wf_cand['validation']['end_date']}",
            "performance_metrics": {
                "baseline_train_pct": wf_base["train"]["total_return_pct"],
                "baseline_val_pct": wf_base["validation"]["total_return_pct"],
                "baseline_unseen_pct": wf_base["unseen_test"]["total_return_pct"],
                "candidate_train_pct": wf_cand["train"]["total_return_pct"],
                "candidate_val_pct": wf_cand["validation"]["total_return_pct"],
                "candidate_unseen_pct": wf_cand["unseen_test"]["total_return_pct"],
                "candidate_unseen_trades": wf_cand["unseen_test"]["total_trades"],
            },
            "suggestion_summary": (
                f"Cost-aware hurdle ({min_move*100:.1f}%) + VWAP/EMA gate: "
                f"Val {wf_base['validation']['total_return_pct']:+.2f}%->{wf_cand['validation']['total_return_pct']:+.2f}%, "
                f"Unseen {wf_base['unseen_test']['total_return_pct']:+.2f}%->{wf_cand['unseen_test']['total_return_pct']:+.2f}%"
            ),
            "approval_status": status,
            "created_at": datetime.now(tz=IST_TZ).isoformat(),
        }
        db.save_experiment(exp_record)
        experiment_rows.append(exp_record)

    # Update docs/lab/data/lab_index.json with Phase 9 experiments
    idx_path = Path("docs/lab/data/lab_index.json")
    if idx_path.exists():
        idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
        idx_data["experiments"] = db.list_experiments()
        idx_data["phase9_experiments"] = experiment_rows
        idx_path.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")
        print(f"\nUpdated {idx_path} with {len(experiment_rows)} Phase 9 experiments.")


if __name__ == "__main__":
    run_phase9_walk_forward_experiments()
