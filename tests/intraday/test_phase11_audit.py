"""Phase 11 Independent Audit, Qualification Framework, and Multi-Agent Research Tests."""

from datetime import date, datetime
import pandas as pd
import pytest

from trader.config import IST_TZ
from trader.data.loader import generate_synthetic_5m_session
from trader.data.schema import DataProvenance, DataSourceMode
from trader.lab.data_service import filter_research_eligible_dates, get_available_historical_dates
from trader.lab.replay_engine import HistoricalLabReplayEngine
from trader.portfolio.cost_calculator import calculate_intraday_costs
from trader.research.agents import (
    StatisticalValidationAgent,
    StrategyChallengerRedTeamAgent,
)
from trader.research.qualification import evaluate_seven_gates


def test_session_selection_reconciliation_parity():
    """Verify run_date_range evaluates ONLY explicitly passed eligible_dates (56, never 59)."""
    avail = get_available_historical_dates(max_dates=90)
    elig = filter_research_eligible_dates(avail, min_bars_per_stock=70)
    ordered = sorted(date.fromisoformat(d.date) for d in elig)

    eng = HistoricalLabReplayEngine()
    res = eng.run_date_range(
        ordered[0],
        ordered[-1],
        strategy_id="STRAT-002-ORB-BREAKOUT",
        eligible_dates=ordered,
    )
    assert res["sessions_evaluated"] == len(ordered)
    assert len(res["daily_runs"]) == len(ordered)


def test_archiver_completeness_protection(tmp_path, monkeypatch):
    """Verify daily 5m archiver NEVER overwrites a 75-bar session with an incomplete 50-bar fetch."""
    import scripts.archive_daily_5m as arch

    monkeypatch.setattr(arch, "CACHE_DIR", tmp_path)
    day1 = date(2026, 11, 20)
    full_75 = generate_synthetic_5m_session("RELIANCE", session_date=day1, num_bars=75)
    incomplete_50 = full_75[:50]

    pd.DataFrame([
        {"timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
        for c in full_75
    ]).to_parquet(tmp_path / "RELIANCE_NSE_5m.parquet", index=False)

    prov = DataProvenance(
        source_mode=DataSourceMode.HISTORICAL_REAL,
        source_name="MockOutage",
        ingested_at=datetime.now(tz=IST_TZ),
        symbol="RELIANCE",
        exchange="NSE",
        bars_count=50,
        start_timestamp=incomplete_50[0].timestamp,
        end_timestamp=incomplete_50[-1].timestamp,
        sha256_hash="0" * 64,
    )
    monkeypatch.setattr(arch, "load_historical_5m_candles", lambda **kw: (incomplete_50, prov))

    rep = arch.merge_and_archive_symbol("RELIANCE", days=5)
    assert rep["total_bars"] == 75
    assert rep["net_added"] == 0


def test_execution_and_accounting_audit_invariants():
    """Verify exact trade-by-trade gross PnL, fees, net PnL, and capital conservation."""
    eng = HistoricalLabReplayEngine()
    res = eng.run_historical_day(date(2026, 11, 18), strategy_id="STRAT-004-SUPERTREND")

    total_net_from_trades = sum(t["net_pnl"] for t in res["trades"])
    total_fees_from_trades = sum(t["fees"] for t in res["trades"])

    for t in res["trades"]:
        expected_gross = round((t["exit_price"] - t["entry_price"]) * t["quantity"], 2)
        assert abs(t["gross_pnl"] - expected_gross) <= 0.05
        assert abs((t["gross_pnl"] - t["fees"]) - t["net_pnl"]) <= 0.05

    assert abs(res["run"]["net_pnl"] - total_net_from_trades) <= 0.10
    assert abs(res["run"]["total_fees"] - total_fees_from_trades) <= 0.10
    assert abs((50_000.0 + res["run"]["net_pnl"]) - res["run"]["final_portfolio_value"]) <= 0.10


def test_seven_gate_qualification_and_red_team_veto():
    """Verify 7-Gate Qualification rejects negative/concentrated strategies and approves genuine qualified ones."""
    # Case 1: Single-day luck concentrated profit -> Red Team rejects
    rt = StrategyChallengerRedTeamAgent()
    bad_metrics = {
        "total_trades": 25,
        "total_return_pct": 4.0,
        "net_pnl_inr": 2000.0,
        "daily_runs": [{"net_pnl": 2500.0}, {"net_pnl": -500.0}],
    }
    wf_ok = {"unseen_test": {"total_return_pct": 1.0}}
    approved, reason = rt.challenge("TEST-LUCKY", bad_metrics, wf_ok)
    assert approved is False
    assert "single exceptional day" in reason

    # Case 2: All 7 gates passing -> QUALIFIED
    good_metrics = {
        "total_return_pct": 8.5,
        "net_pnl_inr": 4250.0,
        "alpha_vs_buy_and_hold_pct": 4.2,
        "max_drawdown_pct": 4.5,
        "profit_factor": 1.55,
        "sharpe_ratio": 1.45,
        "expectancy_inr": 110.0,
        "sessions_evaluated": 65,
        "total_trades": 40,
        "daily_runs": [{"net_pnl": 1000.0}, {"net_pnl": 1200.0}, {"net_pnl": 2050.0}],
    }
    stat = StatisticalValidationAgent()
    s_verdict, _ = stat.validate(good_metrics, wf_ok)
    assert s_verdict == "PASS"

    r_ok, r_reason = rt.challenge("TEST-GOOD", good_metrics, wf_ok)
    assert r_ok is True

    qual = evaluate_seven_gates(
        "TEST-GOOD", "Good Strategy", good_metrics, wf_ok["unseen_test"],
        {"net_return_pct": 6.2}, True, s_verdict, r_ok, r_reason,
        independent_holdout_valid=True,
    )
    assert qual.final_status == "QUALIFIED"
    assert qual.passed_gates_count == 7
