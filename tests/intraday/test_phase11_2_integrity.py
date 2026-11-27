"""Phase 11.2 Regression Tests: Holdout Lifecycle, 1H Provenance, Benchmark Separation & Archive Health."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from scripts.archive_daily_5m import evaluate_archive_health
from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.provenance_1h import validate_1h_provenance, validate_resampled_1h_consistency
from trader.data.resampler import resample_5m_candles
from trader.data.schema import Candle5M
from trader.research.holdout import SealedHoldoutGuard
from trader.research.qualification import evaluate_seven_gates, separate_strategies_and_benchmarks


def _write_1h_parquet(path: Path, dates: list[date], *, tz=IST_TZ, dup: bool = False, bad_ohlc: bool = False, neg_vol: bool = False, bars_per_day: int = 7) -> None:
    rows = []
    for d in dates:
        start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=tz)
        for i in range(bars_per_day):
            ts = start + timedelta(hours=i)
            o, h, l, c, v = 100.0, 102.0, 99.0, 101.0, 1000
            if bad_ohlc and i == 0 and d == dates[0]:
                h = 95.0  # Invalid: high < open/close
            if neg_vol and i == 0 and d == dates[0]:
                v = -50
            rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    if dup and rows:
        rows.append(dict(rows[0]))
    pd.DataFrame(rows).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# 1. Holdout Lifecycle Tests (9 cases)
# ---------------------------------------------------------------------------

def test_holdout_lifecycle_allocation_sealing_exposure_and_reuse(tmp_path: Path) -> None:
    reg_path = tmp_path / "holdout.json"
    guard = SealedHoldoutGuard(registry_path=reg_path)
    d1, d2, d3 = date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 20)

    # 1. First-time allocation & sealing
    assert guard.get_date_state(d1) == "AVAILABLE"
    seal_res = guard.seal_dates("CYCLE-T1", [d1, d2], dataset_fingerprint="fp_v1")
    assert seal_res["sealed_count"] == 2
    assert seal_res["independent_holdout_valid"] is True
    assert guard.get_date_state(d1) == "SEALED"

    # 2. Attempt to tune on sealed dates
    assert guard.check_tuning_leakage([d1, d3]) == ["2026-11-18"]
    split_viol = guard.validate_cycle_splits([d1], [d3], [date(2026, 11, 21)])
    assert split_viol["independent_holdout_valid"] is False
    assert split_viol["train_violations"] == ["2026-11-18"]

    # 3. First-time exposure
    exp1 = guard.record_exposure("CYCLE-T1", [d1, d2], dataset_fingerprint="fp_v1", strategy_ids=["S1"])
    assert exp1["leakage_detected"] is False
    assert exp1["independent_holdout_valid"] is True
    assert guard.get_date_state(d1) == "EXPOSED"

    # 4. Attempt to tune on exposed dates
    assert guard.check_tuning_leakage([d2]) == ["2026-11-19"]

    # 5. Reuse of an exposed date + duplicate exposure recording
    exp2 = guard.record_exposure("CYCLE-T2", [d1, d3], dataset_fingerprint="fp_v1", strategy_ids=["S2"])
    assert exp2["leakage_detected"] is True
    assert exp2["independent_holdout_valid"] is False
    assert exp2["previously_exposed_dates"] == 1
    assert exp2["reused_dates"] == ["2026-11-18"]
    assert exp2["status"] == "EXPOSED_HOLDOUT_REUSED"

    # 6. Dataset fingerprint mismatch
    exp3 = guard.record_exposure("CYCLE-T3", [d2], dataset_fingerprint="fp_DIFFERENT")
    assert exp3["leakage_detected"] is True
    assert exp3["fingerprint_mismatches"] == ["2026-11-19"]

    # 7. Registry persistence across repeated runs
    guard_reloaded = SealedHoldoutGuard(registry_path=reg_path)
    reg = guard_reloaded.load_registry()
    assert reg["records"]["2026-11-18"]["previous_exposure_count"] == 2
    assert reg["records"]["2026-11-18"]["state"] == "EXPOSED"

    # 8. Retire date
    guard_reloaded.retire_dates([d3], "End of validation window")
    assert guard_reloaded.get_date_state(d3) == "RETIRED"


# ---------------------------------------------------------------------------
# 2. 1-Hour Provenance & Resampling Tests (10 cases)
# ---------------------------------------------------------------------------

def test_1h_provenance_validator_and_failure_cases(tmp_path: Path) -> None:
    from trader.config import is_nse_trading_day
    # Build 60 valid NSE trading days
    trading_days = [d for d in pd.date_range("2026-08-01", "2026-11-20", freq="D").date if is_nse_trading_day(d)][:60]

    # 1. Valid dataset
    for sym in LAB_DEFAULT_UNIVERSE:
        _write_1h_parquet(tmp_path / f"{sym}_1h.parquet", trading_days)
    ok, rep = validate_1h_provenance(
        LAB_DEFAULT_UNIVERSE,
        cache_dir=tmp_path,
        min_sessions_required=60,
        enforce_provider_authority_cap=False,
    )
    assert ok is True
    assert rep["final_provenance_status"] in ("VERIFIED", "PARTIALLY_VERIFIED")
    orig_fp = rep["symbols"]["RELIANCE"]["dataset_fingerprint"]

    # 2. Hash / fingerprint change when data changes
    _write_1h_parquet(tmp_path / "RELIANCE_1h.parquet", trading_days[:-1])
    _, rep_mod = validate_1h_provenance(["RELIANCE"], cache_dir=tmp_path, min_sessions_required=50)
    assert rep_mod["symbols"]["RELIANCE"]["dataset_fingerprint"] != orig_fp

    # 3. Missing source file
    (tmp_path / "TCS_1h.parquet").unlink()
    ok_miss, rep_miss = validate_1h_provenance(LAB_DEFAULT_UNIVERSE, cache_dir=tmp_path)
    assert ok_miss is False
    assert rep_miss["final_provenance_status"] == "FAILED"
    assert rep_miss["symbols"]["TCS"]["final_provenance_status"] == "FAILED"

    # 4. Duplicate timestamps
    _write_1h_parquet(tmp_path / "TCS_1h.parquet", trading_days, dup=True)
    ok_dup, rep_dup = validate_1h_provenance(["TCS"], cache_dir=tmp_path)
    assert ok_dup is False
    assert rep_dup["symbols"]["TCS"]["duplicate_count"] == 1

    # 5. Invalid OHLC
    _write_1h_parquet(tmp_path / "TCS_1h.parquet", trading_days, bad_ohlc=True)
    ok_ohlc, rep_ohlc = validate_1h_provenance(["TCS"], cache_dir=tmp_path)
    assert ok_ohlc is False
    assert rep_ohlc["symbols"]["TCS"]["invalid_ohlc_count"] == 1

    # 6. Negative volume
    _write_1h_parquet(tmp_path / "TCS_1h.parquet", trading_days, neg_vol=True)
    ok_vol, rep_vol = validate_1h_provenance(["TCS"], cache_dir=tmp_path)
    assert ok_vol is False
    assert rep_vol["symbols"]["TCS"]["negative_volume_count"] == 1

    # 7. Incorrect timezone (naive timestamps)
    _write_1h_parquet(tmp_path / "TCS_1h.parquet", trading_days, tz=None)
    ok_tz, rep_tz = validate_1h_provenance(["TCS"], cache_dir=tmp_path)
    assert ok_tz is False
    assert "FAIL_NAIVE" in rep_tz["symbols"]["TCS"]["timezone_validation"]

    # 8. Missing session & incomplete session
    gapped_days = trading_days[:10] + trading_days[12:]
    _write_1h_parquet(tmp_path / "TCS_1h.parquet", gapped_days, bars_per_day=5)
    ok_inc, rep_inc = validate_1h_provenance(["TCS"], cache_dir=tmp_path, min_bars_per_session=6, min_sessions_required=1)
    assert rep_inc["symbols"]["TCS"]["incomplete_session_count"] > 0
    assert rep_inc["symbols"]["TCS"]["missing_interval_count"] > 0


def test_resampled_1h_consistency_validator() -> None:
    d = date(2026, 11, 16)
    start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST_TZ)
    c5m = [
        Candle5M(
            symbol="RELIANCE", exchange="NSE", timestamp=start + timedelta(minutes=5 * i),
            open=100.0 + i, high=102.0 + i, low=99.0 + i, close=101.0 + i, volume=100,
        )
        for i in range(12)
    ]
    r60 = resample_5m_candles(c5m, target_minutes=60, include_incomplete=False)
    ok, errs = validate_resampled_1h_consistency(c5m, r60)
    assert ok is True and errs == []

    # Tamper with resampled volume -> must fail
    tampered = list(r60)
    from dataclasses import replace
    tampered[0] = replace(tampered[0], volume=999999)
    ok_bad, errs_bad = validate_resampled_1h_consistency(c5m, tampered)
    assert ok_bad is False
    assert any("volume mismatch" in e for e in errs_bad)


# ---------------------------------------------------------------------------
# 3. Benchmark Separation & Qualification Tests (5 cases)
# ---------------------------------------------------------------------------

def test_benchmark_separation_and_qualification_invalidation() -> None:
    board = [
        {
            "strategy_id": "BENCHMARK-CASH-NO-TRADE",
            "strategy_name": "Cash / No-Trade Baseline",
            "sessions": 240,
            "net_return_pct": 0.0,
            "alpha_vs_bh_pct": 5.5,
            "final_status": "REJECTED",
        },
        {
            "strategy_id": "STRAT-017-60M-ORB-FILTERED",
            "strategy_name": "60m ORB",
            "sessions": 240,
            "net_return_pct": -4.65,
            "alpha_vs_bh_pct": 0.85,
            "final_status": "REJECTED",
        },
    ]
    sep = separate_strategies_and_benchmarks(board, period_label="2025-11-21 -> 2026-11-20")
    assert len(sep["active_strategy_leaderboard"]) == 1
    assert sep["active_strategy_leaderboard"][0]["strategy_id"] == "STRAT-017-60M-ORB-FILTERED"
    assert sep["best_active_strategy"]["strategy_id"] == "STRAT-017-60M-ORB-FILTERED"
    assert sep["qualified_active_strategies"] == 0
    bm_ids = {b["benchmark_id"] for b in sep["reference_benchmark_leaderboard"]}
    assert "BENCHMARK-CASH-NO-TRADE" in bm_ids
    assert "BENCHMARK-BUY-AND-HOLD" in bm_ids

    # Reused holdout invalidates even a positive strategy
    ev = evaluate_seven_gates(
        strategy_id="STRAT-TEST",
        strategy_name="Test",
        multi_day_metrics={"total_return_pct": 12.0, "net_pnl_inr": 6000.0, "alpha_vs_buy_and_hold_pct": 5.0,
                           "max_drawdown_pct": 4.0, "profit_factor": 1.8, "sharpe_ratio": 1.5, "expectancy_inr": 150.0,
                           "sessions_evaluated": 100, "total_trades": 40},
        unseen_metrics={"total_return_pct": 4.0},
        conservative_slippage_metrics={"net_return_pct": 8.0},
        data_provenance_verified=True,
        stat_validation_verdict="PASS",
        red_team_approved=True,
        red_team_reason="OK",
        independent_holdout_valid=False,
    )
    assert ev.final_status == "INVALIDATED"


# ---------------------------------------------------------------------------
# 4. Daily Archive Health Tests (7 cases)
# ---------------------------------------------------------------------------

def test_archive_health_all_seven_statuses() -> None:
    friday_close = datetime(2026, 11, 20, 16, 0, tzinfo=IST_TZ)
    saturday = datetime(2026, 11, 21, 12, 0, tzinfo=IST_TZ)
    prev = {"last_successful_run_ist": "2026-11-20T15:50:00+05:30", "latest_trading_timestamp_ist": "2026-11-20T15:25:00+05:30"}

    ok_syms = [{"symbol": s, "status": "OK", "latest_bar_ist": "2026-11-20T15:25:00+05:30"} for s in LAB_DEFAULT_UNIVERSE]

    # 1. New market data
    h_new = evaluate_archive_health(ok_syms, total_added=75, now_ist=friday_close, prev_health=prev)
    assert h_new["archive_status"] == "SUCCESS_NEW_DATA"

    # 2. Market holiday / closed
    h_closed = evaluate_archive_health(ok_syms, total_added=0, now_ist=saturday, prev_health=prev)
    assert h_closed["archive_status"] == "SUCCESS_MARKET_CLOSED"

    # 3. Missing expected trading session (1 trading day behind)
    old_1d = [{"symbol": s, "status": "OK", "latest_bar_ist": "2026-11-19T15:25:00+05:30"} for s in LAB_DEFAULT_UNIVERSE]
    h_miss = evaluate_archive_health(old_1d, total_added=0, now_ist=friday_close, prev_health=prev)
    assert h_miss["archive_status"] == "WARNING_EXPECTED_DATA_MISSING"

    # 4. Incomplete symbol retrieval
    partial = ok_syms[:-1] + [{"symbol": "TATAELXSI", "status": "FAIL", "latest_bar_ist": "2026-11-19T15:25:00+05:30"}]
    h_part = evaluate_archive_health(partial, total_added=0, now_ist=friday_close, prev_health=prev)
    assert h_part["archive_status"] == "WARNING_EXPECTED_DATA_MISSING"

    # 5. Provider failure preserves previous last_successful_run_ist
    fail_syms = [{"symbol": s, "status": "FAIL"} for s in LAB_DEFAULT_UNIVERSE]
    h_fail = evaluate_archive_health(fail_syms, total_added=0, now_ist=friday_close, prev_health=prev)
    assert h_fail["archive_status"] == "FAILED_PROVIDER"
    assert h_fail["last_successful_run_ist"] == prev["last_successful_run_ist"]

    # 6. Validation failure
    val_fail = ok_syms[:-1] + [{"symbol": "TATAELXSI", "status": "VALIDATION_FAILED"}]
    h_val = evaluate_archive_health(val_fail, total_added=0, now_ist=friday_close, prev_health=prev)
    assert h_val["archive_status"] == "FAILED_VALIDATION"

    # 7. Stale archive (> 2 trading days behind)
    stale = [{"symbol": s, "status": "OK", "latest_bar_ist": "2026-11-10T15:25:00+05:30"} for s in LAB_DEFAULT_UNIVERSE]
    h_stale = evaluate_archive_health(stale, total_added=0, now_ist=friday_close, prev_health=prev)
    assert h_stale["archive_status"] == "STALE_ARCHIVE"
