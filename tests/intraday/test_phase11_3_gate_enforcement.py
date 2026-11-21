"""Phase 11.3 Regression Tests: Qualification Gate Enforcement, Strict Provenance & Holdout Ownership."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE, is_nse_trading_day
from trader.data.provenance_1h import validate_1h_provenance
from trader.research.holdout import SealedHoldoutGuard
from trader.research.integrity_guard import ResearchIntegrityGuard
from trader.research.qualification import evaluate_seven_gates


def _good_metrics() -> dict:
    return {
        "total_return_pct": 12.0,
        "net_pnl_inr": 6000.0,
        "alpha_vs_buy_and_hold_pct": 5.0,
        "max_drawdown_pct": 4.0,
        "profit_factor": 1.8,
        "sharpe_ratio": 1.5,
        "expectancy_inr": 150.0,
        "sessions_evaluated": 100,
        "total_trades": 40,
    }


def test_qualification_fails_closed_on_missing_or_partial_provenance_and_missing_holdout() -> None:
    m = _good_metrics()
    # Missing provenance & missing holdout -> BLOCKED_INTEGRITY
    ev_missing = evaluate_seven_gates(
        "STRAT-X", "Test", m, {"total_return_pct": 4.0}, {"net_return_pct": 8.0},
        stat_validation_verdict="PASS", red_team_approved=True, red_team_reason="OK",
    )
    assert ev_missing.final_status == "BLOCKED_INTEGRITY"
    assert ev_missing.gates[0].passed is False

    # PARTIALLY_VERIFIED provenance -> BLOCKED_INTEGRITY even if data_provenance_verified=True passed erroneously
    ev_partial = evaluate_seven_gates(
        "STRAT-X", "Test", m, {"total_return_pct": 4.0}, {"net_return_pct": 8.0},
        data_provenance_verified=True,
        independent_holdout_valid=True,
        provenance_status="PARTIALLY_VERIFIED",
        stat_validation_verdict="PASS", red_team_approved=True, red_team_reason="OK",
    )
    assert ev_partial.final_status == "BLOCKED_INTEGRITY"
    assert ev_partial.provenance_status == "PARTIALLY_VERIFIED"


def test_provenance_1h_verified_is_false_when_partially_verified(tmp_path: Path) -> None:
    trading_days = [d for d in pd.date_range("2026-08-01", "2026-11-20", freq="D").date if is_nse_trading_day(d)][:60]
    for sym in LAB_DEFAULT_UNIVERSE:
        rows = []
        for idx_d, d in enumerate(trading_days):
            start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST_TZ)
            # Make 1 day incomplete (4 bars) for RELIANCE
            n_bars = 4 if (sym == "RELIANCE" and idx_d == 0) else 7
            for i in range(n_bars):
                ts = start + timedelta(hours=i)
                rows.append({"timestamp": ts, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000})
        pd.DataFrame(rows).to_parquet(tmp_path / f"{sym}_1h.parquet", index=False)

    verified, rep = validate_1h_provenance(LAB_DEFAULT_UNIVERSE, cache_dir=tmp_path, min_sessions_required=59)
    assert rep["final_provenance_status"] == "PARTIALLY_VERIFIED"
    assert verified is False
    assert rep["qualification_allowed"] is False
    assert rep["exploratory_research_allowed"] is True


def test_holdout_ownership_and_central_integrity_guard(tmp_path: Path) -> None:
    reg_path = tmp_path / "holdout.json"
    holdout = SealedHoldoutGuard(registry_path=reg_path)
    d_train = [date(2026, 11, 10), date(2026, 11, 11)]
    d_val = [date(2026, 11, 12)]
    d_unseen = [date(2026, 11, 13), date(2026, 11, 16)]

    # CYCLE-A reserves unseen dates
    seal_a = holdout.seal_dates("CYCLE-A", d_unseen, dataset_fingerprint="fp1")
    assert seal_a["ownership_valid"] is True

    # CYCLE-B tries to reserve or use CYCLE-A's sealed dates -> blocked!
    seal_b = holdout.seal_dates("CYCLE-B", d_unseen, dataset_fingerprint="fp1")
    assert seal_b["ownership_valid"] is False

    guard = ResearchIntegrityGuard(registry_path=reg_path)
    res_b = guard.validate(
        cycle_id="CYCLE-B",
        training_dates=d_train,
        validation_dates=d_val,
        unseen_dates=d_unseen,
        dataset_fingerprint="fp1",
        provenance_result={"final_provenance_status": "VERIFIED", "verified": True},
    )
    assert res_b["integrity_passed"] is False
    assert res_b["qualification_allowed"] is False
    assert res_b["ownership_valid"] is False
    assert res_b["holdout_status"] == "HOLDOUT_OWNERSHIP_VIOLATION"

    # CYCLE-A with VERIFIED provenance and valid splits -> allowed!
    res_a = guard.validate(
        cycle_id="CYCLE-A",
        training_dates=d_train,
        validation_dates=d_val,
        unseen_dates=d_unseen,
        dataset_fingerprint="fp1",
        provenance_result={"final_provenance_status": "VERIFIED", "verified": True},
    )
    assert res_a["integrity_passed"] is True
    assert res_a["qualification_allowed"] is True
    assert res_a["ownership_valid"] is True
    assert res_a["split_valid"] is True

    # Overlapping splits fail closed
    res_overlap = guard.validate(
        cycle_id="CYCLE-A",
        training_dates=d_train,
        validation_dates=[d_train[0]],
        unseen_dates=d_unseen,
        dataset_fingerprint="fp1",
        provenance_result={"final_provenance_status": "VERIFIED", "verified": True},
    )
    assert res_overlap["split_valid"] is False
    assert res_overlap["qualification_allowed"] is False

    # Corrupted registry fails closed
    reg_path.write_text("{corrupt json", encoding="utf-8")
    res_corrupt = guard.validate(
        cycle_id="CYCLE-A",
        training_dates=d_train,
        validation_dates=d_val,
        unseen_dates=d_unseen,
        dataset_fingerprint="fp1",
        provenance_result={"final_provenance_status": "VERIFIED", "verified": True},
    )
    assert res_corrupt["integrity_passed"] is False
    assert res_corrupt["qualification_allowed"] is False
