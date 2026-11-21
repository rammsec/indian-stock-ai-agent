"""Phase 11.3.2 Tests: Exploratory Holdout Consumption & Failure Recovery."""
from __future__ import annotations

import inspect
import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from trader.research.holdout import SealedHoldoutGuard
from trader.research.orchestrator import ChiefResearchOrchestrator
from trader.research.qualification import evaluate_seven_gates


def _make_guard(tmp_path: Path) -> SealedHoldoutGuard:
    reg_path = tmp_path / "sealed_holdout_registry.json"
    return SealedHoldoutGuard(registry_path=reg_path)


def _seed_1h_dates(n: int = 80) -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _dummy_backtest_result(sessions: int) -> dict:
    return {
        "strategy_id": "S",
        "strategy_name": "S",
        "sessions_evaluated": sessions,
        "total_trades": 0,
        "gross_return_pct": 0.0,
        "total_return_pct": 0.0,
        "net_pnl_inr": 0.0,
        "alpha_vs_buy_and_hold_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "profit_factor": 0.0,
        "expectancy_inr": 0.0,
        "total_fees_inr": 0.0,
        "total_brokerage_inr": 0.0,
        "total_statutory_inr": 0.0,
        "total_slippage_inr": 0.0,
        "win_rate_pct": 0.0,
        "daily_runs": [{"net_pnl": 0.0, "trades": 0} for _ in range(sessions)],
        "orders": [],
        "trades": [],
    }


_FAKE_TEST_REPORT = {"total": 10, "passed": 10, "failed": 0, "skipped": 0, "pass_rate_pct": 100.0}


def test_01_exploratory_with_unseen_consumes_holdout(tmp_path: Path) -> None:
    """Scenario 1: Exploratory evaluation reading unseen dates consumes them (SEALED -> EXPOSED)."""
    guard = _make_guard(tmp_path)
    dates = _seed_1h_dates(80)
    unseen = dates[64:]
    guard.seal_dates("CYCLE-004", unseen, dataset_fingerprint="fp-1")

    prog = guard.mark_evaluation_in_progress(
        "CYCLE-004",
        unseen,
        dataset_fingerprint="fp-1",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
    )
    assert prog["ok"] is True
    res = guard.complete_evaluation(
        "CYCLE-004",
        evaluation_id="eval-exp-1",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=unseen,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-1",
        evaluation_ref=str(tmp_path / "c4.json"),
    )
    assert res["ok"] is True
    assert res["holdout_state"] == "EXPOSED"
    assert res["holdout_consumed"] is True
    reg = guard.load_registry()
    for d in unseen:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"
        assert reg["records"][d.isoformat()]["owning_cycle_id"] == "CYCLE-004"


def test_02_exploratory_train_val_only_preserves_sealed(tmp_path: Path) -> None:
    """Scenario 2: Exploratory evaluation on train+val only preserves SEALED state."""
    guard = _make_guard(tmp_path)
    unseen = [date(2026, 3, 10), date(2026, 3, 11), date(2026, 3, 12)]
    seal_res = guard.seal_dates("CYCLE-004", unseen, dataset_fingerprint="fp-2")
    assert seal_res["sealed_count"] == 3

    reg = guard.load_registry()
    for d in unseen:
        rec = reg["records"][d.isoformat()]
        assert rec["state"] == "SEALED"
        assert rec.get("data_accessed", False) is False


def test_03_subsequent_run_cannot_claim_exploratory_consumed_dates(tmp_path: Path) -> None:
    """Scenario 3: Dates consumed by exploratory research cannot be re-sealed as untouched."""
    guard = _make_guard(tmp_path)
    unseen = [date(2026, 3, 10), date(2026, 3, 11)]
    guard.seal_dates("CYCLE-004", unseen, dataset_fingerprint="fp-3")
    guard.mark_evaluation_in_progress("CYCLE-004", unseen, dataset_fingerprint="fp-3", evaluation_type="EXPLORATORY_WITH_UNSEEN")
    guard.complete_evaluation(
        "CYCLE-004",
        evaluation_id="eval-exp-2",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=unseen,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-3",
    )

    re_seal = guard.seal_dates("CYCLE-005", unseen, dataset_fingerprint="fp-3")
    assert re_seal["sealed_count"] == 0
    assert len(re_seal["blocked_dates"]) == 2


def test_04_formal_qualification_transitions_sealed_to_exposed(tmp_path: Path) -> None:
    """Scenario 4: Formal qualification success transitions SEALED -> EXPOSED."""
    guard = _make_guard(tmp_path)
    unseen = [date(2026, 4, 1), date(2026, 4, 2)]
    guard.seal_dates("CYCLE-005", unseen, dataset_fingerprint="fp-4")
    prog = guard.mark_evaluation_in_progress(
        "CYCLE-005", unseen, dataset_fingerprint="fp-4", evaluation_type="FORMAL_QUALIFICATION"
    )
    assert prog["ok"] is True
    comp = guard.complete_evaluation(
        "CYCLE-005",
        evaluation_id="eval-formal-1",
        evaluation_type="FORMAL_QUALIFICATION",
        consumed_dates=unseen,
        strategy_ids=["STRAT-018"],
        dataset_fingerprint="fp-4",
    )
    assert comp["ok"] is True
    assert comp["holdout_state"] == "EXPOSED"
    reg = guard.load_registry()
    for d in unseen:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"


def test_05_run_cycle_004_exploratory_with_unseen_consumes_dates(tmp_path: Path) -> None:
    """Scenario 5: End-to-end run_cycle_004 with PARTIALLY_VERIFIED provenance & include_unseen_in_exploratory=True consumes holdout."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_report.json"
    orch = ChiefResearchOrchestrator()

    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c4-exp",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
        "symbols": {"RELIANCE": {"valid_session_count": 80}},
    }

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabDatabase"),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine") as mock_eng_cls,
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(80)),
    ):
        mock_eng_cls.return_value.run_date_range.return_value = _dummy_backtest_result(80)
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            include_unseen_in_exploratory=True,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["qualification_allowed"] is False
    assert report["exploratory_research_allowed"] is True
    assert report["exploratory_only"] is True
    assert report["holdout_consumed"] is True
    assert report["holdout_state"] == "EXPOSED"
    assert report["evaluation_type"] == "EXPLORATORY_WITH_UNSEEN"
    assert report["evaluation_status"] == "COMPLETED_EXPLORATORY_WITH_UNSEEN"
    assert len(report["consumed_holdout_dates"]) == 16


def test_06_run_cycle_004_exploratory_train_val_only_preserves_sealed(tmp_path: Path) -> None:
    """Scenario 6: run_cycle_004(include_unseen_in_exploratory=False) runs train+val only and keeps unseen SEALED."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_tv_report.json"
    orch = ChiefResearchOrchestrator()

    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c4-tv",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
        "symbols": {"RELIANCE": {"valid_session_count": 80}},
    }

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabDatabase"),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine") as mock_eng_cls,
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(64)),
    ):
        mock_eng_cls.return_value.run_date_range.return_value = _dummy_backtest_result(64)
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            include_unseen_in_exploratory=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["holdout_consumed"] is False
    assert report["holdout_state"] == "SEALED"
    assert report["evaluation_type"] == "EXPLORATORY_TRAIN_VAL_ONLY"
    assert report["evaluation_status"] == "COMPLETED_EXPLORATORY_TRAIN_VAL_ONLY"
    assert report["consumed_holdout_dates"] == []


def test_07_blocked_all_aborts_and_safely_releases_sealed(tmp_path: Path) -> None:
    """Scenario 7: When both qualification and exploratory are blocked before data access, SEALED dates are released."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_blocked.json"
    orch = ChiefResearchOrchestrator()

    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "FAILED",
        "dataset_fingerprint": "fp-c4-fail",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            allow_exploratory_on_blocked=False,
            include_unseen_in_exploratory=True,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert report["evaluation_status"] == "ABORTED_NO_EVALUATION"
    assert report["holdout_consumed"] is False
    assert report["holdout_state"] == "AVAILABLE"
    guard = SealedHoldoutGuard(registry_path=reg_path)
    reg = guard.load_registry()
    for d in dates[64:]:
        assert reg["records"][d.isoformat()]["state"] == "AVAILABLE"


def test_08_crash_after_data_access_marks_exposed(tmp_path: Path) -> None:
    """Scenario 8: Crash after mark_evaluation_in_progress marks consumed dates EXPOSED."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_crash.json"
    orch = ChiefResearchOrchestrator()

    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-c4-crash",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }

    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabDatabase"),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            orch.run_cycle_004(
                test_report=_FAKE_TEST_REPORT,
                run_live_pytest=False,
                include_unseen_in_exploratory=True,
                registry_path=reg_path,
                output_path=out_path,
            )

    guard = SealedHoldoutGuard(registry_path=reg_path)
    reg = guard.load_registry()
    for d in dates[64:]:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"


def test_09_mark_evaluation_failed_without_data_access_preserves_sealed(tmp_path: Path) -> None:
    """Scenario 9: mark_evaluation_failed with data_accessed=False keeps state SEALED."""
    guard = _make_guard(tmp_path)
    unseen = [date(2026, 4, 10)]
    guard.seal_dates("CYCLE-004", unseen, dataset_fingerprint="fp-x")
    res = guard.mark_evaluation_failed("CYCLE-004", unseen, reason="pre-read error", data_accessed=False)
    assert res["holdout_consumed"] is False
    assert res["exposed_dates"] == []
    assert guard.load_registry()["records"]["2026-04-10"]["state"] == "SEALED"


def test_10_recover_stale_reservations_marks_in_progress_exposed(tmp_path: Path) -> None:
    """Scenario 10: Stale EVALUATION_IN_PROGRESS dates are conservatively recovered as EXPOSED."""
    guard = _make_guard(tmp_path)
    unseen = [date(2026, 4, 15), date(2026, 4, 16)]
    guard.seal_dates("CYCLE-004", unseen, dataset_fingerprint="fp-stale")
    guard.mark_evaluation_in_progress("CYCLE-004", unseen, dataset_fingerprint="fp-stale")

    rec = guard.recover_stale_reservations("CYCLE-004")
    assert sorted(rec["recovered_dates"]) == ["2026-04-15", "2026-04-16"]
    reg = guard.load_registry()
    for d in unseen:
        assert reg["records"][d.isoformat()]["state"] == "EXPOSED"
        assert reg["records"][d.isoformat()]["reservation_status"] == "FAILED_EVALUATION"


def test_11_run_cycle_004_recovers_stale_at_startup(tmp_path: Path) -> None:
    """Scenario 11: run_cycle_004 invokes recover_stale_reservations at startup."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_rec.json"
    guard = SealedHoldoutGuard(registry_path=reg_path)
    dates = _seed_1h_dates(80)
    stale_dates = dates[64:]
    guard.seal_dates("CYCLE-004", stale_dates, dataset_fingerprint="fp-rec")
    guard.mark_evaluation_in_progress("CYCLE-004", stale_dates, dataset_fingerprint="fp-rec")

    orch = ChiefResearchOrchestrator()
    fake_prov = {
        "verified": False,
        "final_provenance_status": "FAILED",
        "dataset_fingerprint": "fp-rec",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }
    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
    ):
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            allow_exploratory_on_blocked=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert len(report["stale_recovery"]["recovered_dates"]) == len(stale_dates)
    assert report["evaluation_status"] == "ABORTED_NO_EVALUATION"


def test_12_mark_in_progress_rejects_exposed_dates(tmp_path: Path) -> None:
    """Scenario 12: mark_evaluation_in_progress returns ok=False for EXPOSED dates."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 1)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-5")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-5")
    guard.complete_evaluation("CYCLE-004", "eval-1", "FORMAL_QUALIFICATION", d, ["S1"], "fp-5")

    res = guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-5")
    assert res["ok"] is False


def test_13_mark_in_progress_rejects_available_dates(tmp_path: Path) -> None:
    """Scenario 13: mark_evaluation_in_progress returns ok=False for unsealed AVAILABLE dates."""
    guard = _make_guard(tmp_path)
    res = guard.mark_evaluation_in_progress("CYCLE-004", [date(2026, 5, 2)], dataset_fingerprint="fp-6")
    assert res["ok"] is False


def test_14_mark_in_progress_rejects_wrong_cycle(tmp_path: Path) -> None:
    """Scenario 14: mark_evaluation_in_progress rejects wrong owning cycle."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 3)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-7")
    res = guard.mark_evaluation_in_progress("CYCLE-005", d, dataset_fingerprint="fp-7")
    assert res["ok"] is False


def test_15_complete_evaluation_rejects_wrong_cycle(tmp_path: Path) -> None:
    """Scenario 15: complete_evaluation blocks wrong owning cycle via ownership_violations."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 4)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-8")
    res = guard.complete_evaluation("CYCLE-005", "eval-bad", "FORMAL_QUALIFICATION", d, ["S1"], "fp-8")
    assert res["ok"] is False
    assert res["ownership_violations"]


def test_16_release_reservation_denied_for_wrong_cycle(tmp_path: Path) -> None:
    """Scenario 16: release_reservation denies release by non-owning cycle."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 5)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-9")
    res = guard.release_reservation("CYCLE-005", d, reason="hijack")
    assert res["ok"] is False
    assert any(item["date"] == "2026-05-05" for item in res["denied"])


def test_17_release_reservation_denied_after_data_accessed(tmp_path: Path) -> None:
    """Scenario 17: release_reservation denies release once data_accessed=True."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 6)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-10")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-10")
    res = guard.release_reservation("CYCLE-004", d, reason="try release")
    assert res["ok"] is False
    assert any(item["date"] == "2026-05-06" for item in res["denied"])


def test_18_fingerprint_mismatch_rejected(tmp_path: Path) -> None:
    """Scenario 18: Fingerprint mismatch in mark_evaluation_in_progress returns ok=False."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 7)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-orig")
    res = guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-tampered")
    assert res["ok"] is False


def test_19_duplicate_dates_in_seal_dates_rejected(tmp_path: Path) -> None:
    """Scenario 19: seal_dates rejects duplicate dates cleanly."""
    guard = _make_guard(tmp_path)
    res = guard.seal_dates("CYCLE-004", [date(2026, 5, 8), date(2026, 5, 8)], dataset_fingerprint="fp-dup")
    assert res["sealed_count"] == 0
    assert res["reservation_status"] == "REJECTED_DUPLICATE_DATES"


def test_20_evaluation_history_records_metadata(tmp_path: Path) -> None:
    """Scenario 20: complete_evaluation appends evaluation_history with full audit metadata."""
    guard = _make_guard(tmp_path)
    d = [date(2026, 5, 9)]
    guard.seal_dates("CYCLE-004", d, dataset_fingerprint="fp-hist")
    guard.mark_evaluation_in_progress("CYCLE-004", d, dataset_fingerprint="fp-hist")
    guard.complete_evaluation(
        "CYCLE-004",
        evaluation_id="eval-hist-1",
        evaluation_type="EXPLORATORY_WITH_UNSEEN",
        consumed_dates=d,
        strategy_ids=["STRAT-018", "STRAT-019"],
        dataset_fingerprint="fp-hist",
        evaluation_ref="ref.json",
    )
    reg = guard.load_registry()
    hist = reg.get("evaluation_history", [])
    assert len(hist) == 1
    assert hist[0]["evaluation_id"] == "eval-hist-1"
    assert hist[0]["evaluation_type"] == "EXPLORATORY_WITH_UNSEEN"
    assert hist[0]["strategy_ids"] == ["STRAT-018", "STRAT-019"]
    assert hist[0]["consumed_dates"] == ["2026-05-09"]


def test_21_corrupted_registry_json_fails_closed(tmp_path: Path) -> None:
    """Scenario 21: Corrupted registry JSON raises an error rather than silently resetting."""
    reg_path = tmp_path / "sealed_holdout_registry.json"
    reg_path.write_text("{not valid json", encoding="utf-8")
    guard = SealedHoldoutGuard(registry_path=reg_path)
    with pytest.raises(json.JSONDecodeError):
        guard.load_registry()


def test_22_cycle_004_report_contains_all_14_required_fields(tmp_path: Path) -> None:
    """Scenario 22: CYCLE-004 report schema includes all 14 Phase 11.3.2 fields."""
    required_keys = {
        "integrity_passed",
        "qualification_allowed",
        "exploratory_research_allowed",
        "exploratory_only",
        "holdout_consumed",
        "holdout_state",
        "evaluation_status",
        "evaluation_type",
        "evaluation_started_at",
        "evaluation_completed_at",
        "consumed_holdout_dates",
        "dataset_fingerprint",
        "strategy_ids_evaluated",
        "failure_reasons",
    }
    reg_path = tmp_path / "sealed_holdout_registry.json"
    out_path = tmp_path / "cycle_004_schema.json"
    orch = ChiefResearchOrchestrator()
    dates = _seed_1h_dates(80)
    fake_prov = {
        "verified": False,
        "final_provenance_status": "PARTIALLY_VERIFIED",
        "dataset_fingerprint": "fp-schema",
        "reconciled_eligible_dates": [d.isoformat() for d in dates],
    }
    with (
        patch("trader.data.loader_1h.get_available_1h_dates", return_value=(dates, [])),
        patch("trader.research.orchestrator.validate_1h_provenance", return_value=(False, fake_prov)),
        patch("trader.research.orchestrator.HistoricalLabDatabase"),
        patch("trader.research.orchestrator.HistoricalLabReplayEngine") as mock_eng_cls,
        patch("trader.research.cycle004.run_cnc_swing_backtest", return_value=_dummy_backtest_result(80)),
    ):
        mock_eng_cls.return_value.run_date_range.return_value = _dummy_backtest_result(80)
        report = orch.run_cycle_004(
            test_report=_FAKE_TEST_REPORT,
            run_live_pytest=False,
            registry_path=reg_path,
            output_path=out_path,
        )

    assert required_keys.issubset(report.keys())


def test_23_dashboard_js_displays_phase11_3_2_fields() -> None:
    """Scenario 23: docs/lab/app.js displays holdout consumption, evaluation type/status, and exposed/untouched dates."""
    app_js = Path("docs/lab/app.js").read_text(encoding="utf-8")
    for token in (
        "holdout_consumed",
        "holdout_state",
        "evaluation_type",
        "evaluation_status",
        "evaluation_completed_at",
        "consumed_holdout_dates",
        "untouched_available_count",
        "Holdout Consumed",
    ):
        assert token in app_js, f"Missing {token} in docs/lab/app.js"


def test_24_no_unsafe_defaults_in_evaluate_seven_gates() -> None:
    """Scenario 24: evaluate_seven_gates defaults remain falsy (None/False) and fail closed."""
    sig = inspect.signature(evaluate_seven_gates)
    assert not sig.parameters["data_provenance_verified"].default
    assert not sig.parameters["independent_holdout_valid"].default
    qual = evaluate_seven_gates(
        strategy_id="S-SAFE",
        strategy_name="Safe Check",
        multi_day_metrics={
            "sessions_evaluated": 100,
            "total_trades": 50,
            "total_return_pct": 5.0,
            "alpha_vs_buy_and_hold_pct": 2.0,
            "max_drawdown_pct": 2.0,
            "sharpe_ratio": 2.0,
            "profit_factor": 2.0,
            "expectancy_inr": 100.0,
        },
        unseen_metrics={"total_return_pct": 2.0, "total_trades": 10},
        conservative_slippage_metrics={"net_return_pct": 4.0},
        stat_validation_verdict="VALIDATED",
        red_team_approved=True,
    )
    assert qual.final_status in ("REJECTED", "BLOCKED_INTEGRITY")
