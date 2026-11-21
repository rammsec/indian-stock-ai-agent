"""Phase 10: Ticket-Size & Brokerage-Tier Sensitivity Study across 56 Verified Sessions."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import LAB_DEFAULT_UNIVERSE, CostConfig, LabRiskConfig
from trader.data.schema import DataSourceMode
from trader.lab.data_service import filter_research_eligible_dates, get_available_historical_dates
from trader.lab.replay_engine import STRATEGY_REGISTRY, HistoricalLabReplayEngine
from trader.strategies.cost_aware_wrapper import CostAwareFilterWrapper
from trader.strategies.vwap_pullback import VWAPPullbackStrategy


def run_phase10_brokerage_and_sizing_study() -> None:
    avail = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=90)
    eligible = filter_research_eligible_dates(avail, min_bars_per_stock=70)
    ordered_dates = sorted(date.fromisoformat(d.date) for d in eligible)
    start_d, end_d = ordered_dates[0], ordered_dates[-1]

    STRATEGY_REGISTRY["STRAT-001-VWAP-V2"] = {
        "name": "VWAP Pullback (v2.0 Cost-Aware)",
        "cls": lambda: CostAwareFilterWrapper(VWAPPullbackStrategy(), min_expected_move_pct=0.012),
        "kind": "STRATEGY",
    }

    configs = [
        ("A_Standard_20INR_3Pos", LabRiskConfig(max_open_positions=3, risk_per_trade_pct=0.005, max_risk_per_trade_inr=250.0), CostConfig(brokerage_pct=0.0003, max_brokerage_per_order_inr=20.0)),
        ("B_Concentrated_20INR_1Pos", LabRiskConfig(max_open_positions=1, risk_per_trade_pct=0.010, max_risk_per_trade_inr=500.0), CostConfig(brokerage_pct=0.0003, max_brokerage_per_order_inr=20.0)),
        ("C_ZeroBrokerage_1Pos", LabRiskConfig(max_open_positions=1, risk_per_trade_pct=0.010, max_risk_per_trade_inr=500.0), CostConfig(brokerage_pct=0.0, max_brokerage_per_order_inr=0.0)),
    ]

    strategies_to_test = [
        "STRAT-002-ORB-BREAKOUT",
        "STRAT-003-EMA-CROSS",
        "STRAT-001-VWAP-V2",
    ]

    study_rows = []
    for cfg_name, r_cfg, c_cfg in configs:
        eng = HistoricalLabReplayEngine(risk_config=r_cfg, cost_config=c_cfg)
        for sid in strategies_to_test:
            res = eng.run_date_range(start_d, end_d, strategy_id=sid, data_mode=DataSourceMode.HISTORICAL_REAL)
            row = {
                "config_name": cfg_name,
                "strategy_id": sid,
                "strategy_name": res["strategy_name"],
                "sessions": res["sessions_evaluated"],
                "gross_return_pct": res["gross_return_pct"],
                "net_return_pct": res["total_return_pct"],
                "alpha_vs_bh_pct": res["alpha_vs_buy_and_hold_pct"],
                "total_trades": res["total_trades"],
                "win_rate_pct": res["win_rate_pct"],
                "total_fees_inr": res["total_fees_inr"],
            }
            study_rows.append(row)
            print(
                f"{cfg_name:26s} | {sid:24s} | Gross: {row['gross_return_pct']:+6.2f}% | "
                f"Net: {row['net_return_pct']:+6.2f}% | Alpha: {row['alpha_vs_bh_pct']:+6.2f}% | "
                f"Trades: {row['total_trades']:3d} | Fees: ₹{row['total_fees_inr']:7.2f}"
            )

    idx_path = Path("docs/lab/data/lab_index.json")
    if idx_path.exists():
        idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
        idx_data["phase10_sizing_brokerage_study"] = study_rows
        idx_path.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")
        print(f"\nSaved {len(study_rows)} Phase 10 study rows to {idx_path}")


if __name__ == "__main__":
    run_phase10_brokerage_and_sizing_study()
