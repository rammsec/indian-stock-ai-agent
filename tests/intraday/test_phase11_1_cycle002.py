"""Phase 11.1 & CYCLE-002 Regression Tests: Resampling, Provenance, Holdout Guard, Cost Separation & Archive Health."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from scripts.archive_daily_5m import classify_archive_status
from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.data.resampler import resample_5m_candles, to_candle5m_compatible
from trader.data.schema import Candle5M, DataSourceMode
from trader.lab.data_service import filter_research_eligible_dates, get_available_historical_dates
from trader.lab.replay_engine import HistoricalLabReplayEngine
from trader.research.agents import ExecutionCostAgent, SoftwareQualityAgent
from trader.research.orchestrator import SealedHoldoutGuard, verify_dataset_provenance


def _make_5m_session(day: date, count: int = 75, base_px: float = 1000.0) -> list[Candle5M]:
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST_TZ)
    bars: list[Candle5M] = []
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        px = base_px + i * 0.5
        bars.append(
            Candle5M(
                symbol="RELIANCE",
                exchange="NSE",
                timestamp=ts,
                open=px,
                high=px + 2.0,
                low=px - 1.0,
                close=px + 1.0,
                volume=1000 + i * 10,
            )
        )
    return bars


def test_resampler_15m_and_30m_strict_alignment_and_no_cross_day_merge() -> None:
    d1 = date(2026, 11, 16)
    d2 = date(2026, 11, 17)
    # Day 1 has 75 bars (25 complete 15m bars, 12 complete 30m bars + 1 incomplete 15m remainder at 15:15)
    s1 = _make_5m_session(d1, count=75, base_px=1000.0)
    # Day 2 has 4 bars (1 complete 15m bar + 1 incomplete 15m bar)
    s2 = _make_5m_session(d2, count=4, base_px=1100.0)

    r15 = resample_5m_candles(s1 + s2, target_minutes=15, include_incomplete=False)
    assert len(r15) == 26  # 25 from d1 + 1 from d2
    assert r15[0].timestamp.strftime("%H:%M") == "09:15"
    assert r15[0].open == s1[0].open
    assert r15[0].high == max(b.high for b in s1[:3])
    assert r15[0].low == min(b.low for b in s1[:3])
    assert r15[0].close == s1[2].close
    assert r15[0].volume == sum(b.volume for b in s1[:3])
    assert all(b.is_complete and b.source_bar_count == 3 for b in r15)

    # Verify incomplete bars are flagged when include_incomplete=True
    r15_inc = resample_5m_candles(s1 + s2, target_minutes=15, include_incomplete=True)
    assert len(r15_inc) == 27
    assert r15_inc[-1].is_complete is False
    assert r15_inc[-1].source_bar_count == 1

    # Verify 30m resampling: 75 bars = 12 full 30m bars (09:15..14:45) + 1 NSE closing 15m bar (15:15..15:30, 3 x 5m)
    r30 = resample_5m_candles(s1, target_minutes=30, include_incomplete=False)
    assert len(r30) == 13
    assert r30[0].timestamp.strftime("%H:%M") == "09:15"
    assert r30[-2].timestamp.strftime("%H:%M") == "14:45"
    assert r30[-1].timestamp.strftime("%H:%M") == "15:15"
    assert r30[-1].source_bar_count == 3
    assert r30[-1].is_complete is True

    compat = to_candle5m_compatible(r30)
    assert len(compat) == 13
    assert compat[0].volume == r30[0].volume


def test_resampler_never_interpolates_missing_bars() -> None:
    d1 = date(2026, 11, 16)
    s1 = _make_5m_session(d1, count=9)  # 3 15m buckets: [0..2], [3..5], [6..8]
    # Remove one bar from middle bucket -> middle bucket only has 2 bars
    gapped = s1[:4] + s1[5:]
    r15 = resample_5m_candles(gapped, target_minutes=15, include_incomplete=False)
    assert len(r15) == 2
    assert [b.timestamp.strftime("%H:%M") for b in r15] == ["09:15", "09:45"]


def test_provenance_verification_and_sealed_holdout_guard(tmp_path: Path) -> None:
    ok_empty, rep_empty = verify_dataset_provenance([], LAB_DEFAULT_UNIVERSE)
    assert ok_empty is False
    assert rep_empty["verified"] is False

    avail = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=90)
    elig = filter_research_eligible_dates(avail, min_bars_per_stock=70)
    dates = sorted(date.fromisoformat(d.date) for d in elig)
    ok, rep = verify_dataset_provenance(dates[:3], LAB_DEFAULT_UNIVERSE)
    assert ok is True
    assert rep["total_duplicate_candles"] == 0
    assert rep["symbols"]["RELIANCE"]["exchange"] == "NSE"

    guard = SealedHoldoutGuard(registry_path=tmp_path / "sealed.json")
    guard.record_exposure("CYCLE-TEST", [date(2026, 11, 18), date(2026, 11, 19)], notes="unit test")
    leaks = guard.check_tuning_leakage([date(2026, 11, 17), date(2026, 11, 19)])
    assert leaks == ["2026-11-19"]


def test_dynamic_qa_and_execution_cost_audit_invariants() -> None:
    sq = SoftwareQualityAgent()
    fail_rep = sq.review_system(test_report={"tests_executed": 10, "tests_passed": 9, "tests_failed": 1, "tests_skipped": 0, "commit_sha": "abc12345"})
    assert fail_rep.status == "FAIL"

    pass_rep = sq.review_system(test_report={"tests_executed": 10, "tests_passed": 10, "tests_failed": 0, "tests_skipped": 0, "commit_sha": "abc12345"})
    assert pass_rep.status == "PASS"
    assert "abc12345" in pass_rep.latest_finding

    ea = ExecutionCostAgent()
    bad_lb = [{
        "strategy_id": "BAD-STRAT",
        "initial_capital_inr": 50000.0,
        "gross_pnl_inr": 100.0,
        "total_fees_inr": 40.0,
        "total_brokerage_inr": 40.0,
        "total_statutory_inr": 0.0,
        "final_equity_inr": 50999.0,  # Drift!
    }]
    res_bad = ea.audit_costs(bad_lb, [])
    assert res_bad.status == "FAIL"


def test_mtf_replay_and_slippage_sensitivity_cost_separation() -> None:
    avail = get_available_historical_dates(LAB_DEFAULT_UNIVERSE, max_dates=90)
    elig = filter_research_eligible_dates(avail, min_bars_per_stock=70)
    d = date.fromisoformat(elig[-1].date)

    engine = HistoricalLabReplayEngine()
    cash_res = engine.run_date_range(d, d, strategy_id="BENCHMARK-CASH-NO-TRADE", eligible_dates=[d])
    assert cash_res["total_trades"] == 0
    assert cash_res["total_return_pct"] == 0.0
    assert cash_res["timeframe_minutes"] == 30

    slip_rows = engine.run_slippage_sensitivity(
        d,
        d,
        strategy_id="STRAT-008-15M-RS-LEADER",
        slippage_bps_list=(5.0, 10.0, 20.0, 30.0),
        data_mode=DataSourceMode.HISTORICAL_REAL,
        eligible_dates=[d],
    )
    assert [r["slippage_bps"] for r in slip_rows] == [5.0, 10.0, 20.0, 30.0]
    for r in slip_rows:
        assert "total_brokerage_inr" in r
        assert "total_statutory_inr" in r
        assert "total_slippage_inr" in r


def test_archive_status_classification() -> None:
    assert classify_archive_status([{"status": "FAIL"}, {"status": "FAIL"}], 0) == "FAILED_RETRIEVAL"
    assert classify_archive_status([{"status": "OK"}, {"status": "FAIL"}], 10) == "PARTIAL_RETRIEVAL"
    assert classify_archive_status([{"status": "OK"}, {"status": "OK"}], 25) == "SUCCESS_NEW_DATA"
    assert classify_archive_status([{"status": "OK"}, {"status": "OK"}], 0) == "SUCCESS_NO_NEW_DATA"
