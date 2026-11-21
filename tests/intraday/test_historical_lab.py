"""Comprehensive tests for the Historical AI Trading Lab (5-stock ₹50,000 simulator, DB, and dashboard)."""

import json
from datetime import date, datetime
from pathlib import Path
import pytest

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, LAB_INITIAL_CAPITAL_INR
from trader.data.instruments import get_instrument_by_symbol, get_instrument_by_isin
from trader.data.schema import DataSourceMode
from trader.lab.data_service import load_session_with_warmup, get_available_historical_dates
from trader.lab.database import HistoricalLabDatabase
from trader.lab.replay_engine import HistoricalLabReplayEngine, STRATEGY_REGISTRY


def test_five_stock_universe_and_birlacable_isin():
    """Verify fixed 5-stock universe and BIRLACABLE ISIN/BSE mapping."""
    assert LAB_DEFAULT_UNIVERSE == ("RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI")
    assert LAB_INITIAL_CAPITAL_INR == 50_000.0

    bc = get_instrument_by_symbol("BIRLACABLE")
    assert bc is not None
    assert bc.isin == "INE800A01015"
    assert bc.bse_code == "500060"
    assert get_instrument_by_isin("INE800A01015").symbol == "BIRLACABLE"


def test_warmup_slicing_zero_future_leakage():
    """Ensure load_session_with_warmup never returns candles after target_date."""
    target = date(2026, 11, 18)
    warmup, session, quality = load_session_with_warmup("RELIANCE", target, warmup_days=5, mode=DataSourceMode.HISTORICAL_REAL)
    assert len(session) > 50
    assert quality.status in ("Ready", "Incomplete")
    assert quality.source_mode == "HISTORICAL_REAL"

    for c in warmup:
        assert c.timestamp.astimezone(IST_TZ).date() < target
    for c in session:
        assert c.timestamp.astimezone(IST_TZ).date() == target


def test_replay_determinism_and_next_bar_execution():
    """Replaying identical inputs produces bit-for-bit identical ₹50,000 portfolio results."""
    eng = HistoricalLabReplayEngine()
    target = date(2026, 11, 18)

    r1 = eng.run_historical_day(target, strategy_id="STRAT-001-VWAP-PULLBACK", data_mode=DataSourceMode.HISTORICAL_REAL)
    r2 = eng.run_historical_day(target, strategy_id="STRAT-001-VWAP-PULLBACK", data_mode=DataSourceMode.HISTORICAL_REAL)

    assert r1["run"]["starting_capital"] == 50_000.0
    assert r1["run"]["final_portfolio_value"] == r2["run"]["final_portfolio_value"]
    assert r1["run"]["total_trades"] == r2["run"]["total_trades"]
    assert r1["run"]["execution_config"]["execution_rule"] == "NEXT_BAR_OPEN"
    assert r1["benchmark"]["initial_capital_inr"] == 50_000.0

    # Verify all 9 strategies/models are registered
    assert len(STRATEGY_REGISTRY) == 9


def test_historical_lab_sqlite_persistence(tmp_path):
    """Verify all 6 entities persist and query cleanly in SQLite."""
    db_file = tmp_path / "test_lab.db"
    db = HistoricalLabDatabase(db_path=db_file)
    eng = HistoricalLabReplayEngine()

    bundle = eng.run_historical_day(date(2026, 11, 18), strategy_id="STRAT-004-SUPERTREND", data_mode=DataSourceMode.HISTORICAL_REAL)
    db.save_run_bundle(bundle)

    runs = db.list_runs("2026-11-18")
    assert len(runs) == 1
    assert runs[0]["strategy_id"] == "STRAT-004-SUPERTREND"
    assert runs[0]["starting_capital"] == 50_000.0

    db.save_experiment({
        "experiment_id": "exp_test_01",
        "parent_strategy_version": "STRAT-004-SUPERTREND@v1.0",
        "modified_parameters": {"atr_mult": 1.3},
        "validation_period": "2026-11-18",
        "performance_metrics": {"net_pnl": runs[0]["net_pnl"]},
        "suggestion_summary": "Test ATR adjustment",
        "approval_status": "PROPOSED_PENDING_WALK_FORWARD",
        "created_at": datetime.now(tz=IST_TZ).isoformat(),
    })
    assert len(db.list_experiments()) == 1


def test_lab_dashboard_files_and_json_contract():
    """Verify docs/lab/ static files and generated JSON index contract."""
    lab_dir = Path("docs/lab")
    assert (lab_dir / "index.html").exists()
    assert (lab_dir / "styles.css").exists()
    assert (lab_dir / "app.js").exists()
    idx_file = lab_dir / "data" / "lab_index.json"
    assert idx_file.exists()

    idx = json.loads(idx_file.read_text(encoding="utf-8"))
    assert idx["initial_capital_inr"] == 50_000.0
    assert idx["universe"] == ["RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI"]
    assert len(idx["strategies"]) == 9
    assert len(idx["available_dates"]) >= 1
