"""CYCLE-008 — Comprehensive Isolated Test Suite for Historical Data Readiness,
Chronological Walk-Forward Splits, Leakage Controls, Deterministic Metrics,
and Offline End-to-End Strategy Evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts.run_cycle_008_walk_forward_eval import run_cycle_008_eval
from trader.config import LAB_DEFAULT_UNIVERSE
from trader.evaluation.data_readiness import (
    DEFAULT_HOLDOUT_REGISTRY_PATH,
    audit_historical_data_readiness,
)
from trader.evaluation.metrics import (
    TRADING_DAYS_PER_YEAR,
    TradeRecord,
    compute_strategy_metrics,
)
from trader.evaluation.walk_forward import (
    CYCLE008_IMPLEMENTATION_MODULES,
    Cycle008Config,
    Cycle008WalkForwardEvaluator,
    build_chronological_splits,
)
from trader.paper.cycle006b import sha256_file


# ===========================================================================
# 1. Phase 1 — Historical Data Readiness & Provenance Audit
# ===========================================================================


class TestPhase1DataReadiness:
    """Verify Phase 1 audit of committed 1h/15m/5m archives, Stage 1 archives, and holdout registry."""

    def test_1h_coverage_integrity_and_provenance(self) -> None:
        reg_sha_before = sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)
        audit = audit_historical_data_readiness()
        reg_sha_after = sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)

        # Holdout registry must never be mutated by Phase 1 audit
        assert reg_sha_before == reg_sha_after

        h1 = audit["intraday_1h_audit"]
        assert h1["common_trading_days_count"] == 245
        assert h1["total_duplicate_timestamps"] == 0
        assert h1["total_nan_count"] == 0
        assert h1["total_ohlc_invariant_violations"] == 0

        # Hard rule: Yahoo Finance 1h provenance is PARTIALLY_VERIFIED (verified=False)
        assert h1["provenance_status"] == "PARTIALLY_VERIFIED"
        assert h1["verified"] is False
        assert h1["dataset_fingerprint"]

        for sym in LAB_DEFAULT_UNIVERSE:
            sym_meta = h1["per_symbol"][sym]
            assert sym_meta["exists"] is True
            assert sym_meta["sha256"]
            assert sym_meta["row_count"] > 0
            assert sym_meta["trading_days"] >= 245
            assert sym_meta["duplicate_timestamps"] == 0
            assert sym_meta["nan_count"] == 0
            assert sym_meta["ohlc_invariant_violations"] == 0
            assert sym_meta["timezone_valid"] is True

    def test_committed_15m_and_5m_unavailable_never_fabricated(self) -> None:
        audit = audit_historical_data_readiness()

        a15 = audit["intraday_15m_audit"]
        a5 = audit["intraday_5m_audit"]

        assert a15["timeframe"] == "15m"
        assert a15["classification"] == "UNAVAILABLE"
        assert a15["available_symbols"] == []
        assert len(a15["missing_symbols"]) == len(LAB_DEFAULT_UNIVERSE)
        assert a15["fabricated_or_substituted"] is False

        assert a5["timeframe"] == "5m"
        assert a5["classification"] == "UNAVAILABLE"
        assert a5["available_symbols"] == []
        assert len(a5["missing_symbols"]) == len(LAB_DEFAULT_UNIVERSE)
        assert a5["fabricated_or_substituted"] is False

    def test_exact_holdout_partition_196_vs_48_and_1_straddle_zero_overlap(self) -> None:
        audit = audit_historical_data_readiness()
        holdout = audit["holdout_registry_audit"]

        assert holdout["exists"] is True
        assert holdout["sha256"] == sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)
        assert holdout["common_1h_trading_days_count"] == 245
        assert holdout["non_holdout_date_count"] == 196
        assert holdout["non_holdout_date_max"] < "2026-09-09"
        assert holdout["exposed_holdout_date_count"] == 48
        assert holdout["exposed_count"] == 48
        assert holdout["available_count"] == 0
        assert holdout["sealed_count"] == 0
        assert holdout["excluded_straddle_count"] == 1
        assert holdout["excluded_straddle_dates"] == ["2026-11-09"]
        assert holdout["forbidden_for_unseen_claims"] is True

        non_holdout_set = set(holdout["non_holdout_dates"])
        exposed_set = set(holdout["exposed_holdout_dates"])
        straddle_set = set(holdout["excluded_straddle_dates"])
        assert len(non_holdout_set) == 196
        assert len(exposed_set) == 48
        assert len(straddle_set) == 1
        assert non_holdout_set.isdisjoint(exposed_set)
        assert non_holdout_set.isdisjoint(straddle_set)
        assert exposed_set.isdisjoint(straddle_set)
        assert len(non_holdout_set | exposed_set | straddle_set) == 245
        assert max(non_holdout_set) < min(exposed_set)

    def test_stage1_archives_and_registry_immutability(self) -> None:
        reg_before = DEFAULT_HOLDOUT_REGISTRY_PATH.read_bytes()
        audit = audit_historical_data_readiness()
        reg_after = DEFAULT_HOLDOUT_REGISTRY_PATH.read_bytes()
        assert reg_before == reg_after

        s1 = audit["stage1_historical_archives"]
        assert s1["prediction_log"]["exists"] is True
        assert s1["prediction_log"]["observation_date_count"] == 3
        assert len(s1["prediction_log"]["observation_dates"]) == 3

        assert s1["forward_predictions"]["exists"] is True
        assert s1["forward_predictions"]["observation_date_count"] == 3

        assert s1["market_regimes"]["exists"] is True
        assert s1["market_regimes"]["observation_date_count"] == 3

        assert s1["structured_events"]["exists"] is True
        assert s1["structured_events"]["row_count"] > 0
        assert 0.0 <= s1["structured_events"]["collection_timestamp_completeness_pct"] <= 100.0


# ===========================================================================
# 2. Phase 3 — Deterministic Performance & Risk Metrics Calculator
# ===========================================================================


class TestPhase3MetricsCalculator:
    """Verify hand-calculated metrics and edge cases in trader.evaluation.metrics."""

    def test_known_hand_calculated_trade_sequence(self) -> None:
        initial_capital = 100_000.0
        trades = [
            TradeRecord(
                symbol="RELIANCE",
                session_date="2026-01-02",
                side="BUY",
                entry_price=1000.0,
                exit_price=1021.0,
                quantity=50,
                gross_pnl_inr=1050.0,
                brokerage_inr=20.0,
                statutory_charges_inr=15.0,
                slippage_inr=15.0,
                total_friction_inr=50.0,
                net_pnl_inr=1000.0,
                return_pct=2.0,
                mae_pct=-0.5,
                mfe_pct=2.5,
                regime_label="TRENDING_UP_1H_PROXY",
                exit_reason="TARGET",
            ),
            TradeRecord(
                symbol="TCS",
                session_date="2026-01-03",
                side="BUY",
                entry_price=2000.0,
                exit_price=1982.0,
                quantity=25,
                gross_pnl_inr=-450.0,
                brokerage_inr=20.0,
                statutory_charges_inr=15.0,
                slippage_inr=15.0,
                total_friction_inr=50.0,
                net_pnl_inr=-500.0,
                return_pct=-1.0,
                mae_pct=-1.5,
                mfe_pct=0.5,
                regime_label="TRENDING_UP_1H_PROXY",
                exit_reason="STOP_LOSS",
            ),
            TradeRecord(
                symbol="RELIANCE",
                session_date="2026-01-06",
                side="BUY",
                entry_price=1000.0,
                exit_price=1013.0,
                quantity=50,
                gross_pnl_inr=650.0,
                brokerage_inr=20.0,
                statutory_charges_inr=15.0,
                slippage_inr=15.0,
                total_friction_inr=50.0,
                net_pnl_inr=600.0,
                return_pct=1.2,
                mae_pct=-0.4,
                mfe_pct=1.8,
                regime_label="SIDEWAYS_1H_PROXY",
                exit_reason="EOD_SQUARE_OFF",
            ),
            TradeRecord(
                symbol="TCS",
                session_date="2026-01-07",
                side="BUY",
                entry_price=2000.0,
                exit_price=1990.0,
                quantity=25,
                gross_pnl_inr=-250.0,
                brokerage_inr=20.0,
                statutory_charges_inr=15.0,
                slippage_inr=15.0,
                total_friction_inr=50.0,
                net_pnl_inr=-300.0,
                return_pct=-0.6,
                mae_pct=-1.2,
                mfe_pct=0.2,
                regime_label="SIDEWAYS_1H_PROXY",
                exit_reason="STOP_LOSS",
            ),
        ]

        equity_curve = [
            {"date": "2026-01-02", "equity_inr": 101_000.0},
            {"date": "2026-01-03", "equity_inr": 100_500.0},
            {"date": "2026-01-06", "equity_inr": 101_100.0},
            {"date": "2026-01-07", "equity_inr": 100_800.0},
        ]

        directional_outcomes = [
            {
                "symbol": "RELIANCE",
                "session_date": "2026-01-02",
                "predicted_direction": "BULLISH",
                "realized_direction": "BULLISH",
                "regime_label": "TRENDING_UP_1H_PROXY",
            },
            {
                "symbol": "TCS",
                "session_date": "2026-01-03",
                "predicted_direction": "BULLISH",
                "realized_direction": "BEARISH",
                "regime_label": "TRENDING_UP_1H_PROXY",
            },
            {
                "symbol": "RELIANCE",
                "session_date": "2026-01-06",
                "predicted_direction": "BULLISH",
                "realized_direction": "BULLISH",
                "regime_label": "SIDEWAYS_1H_PROXY",
            },
            {
                "symbol": "TCS",
                "session_date": "2026-01-07",
                "predicted_direction": "BEARISH",
                "realized_direction": "BEARISH",
                "regime_label": "SIDEWAYS_1H_PROXY",
            },
        ]

        m = compute_strategy_metrics(
            trades=trades,
            equity_curve=equity_curve,
            directional_outcomes=directional_outcomes,
            initial_capital_inr=initial_capital,
        )

        # Core counts & trade stats
        assert m["total_trades"] == 4
        assert m["winning_trades"] == 2
        assert m["losing_trades"] == 2
        assert m["breakeven_trades"] == 0
        assert m["win_rate"] == pytest.approx(0.5, abs=1e-6)
        assert m["directional_accuracy"] == pytest.approx(0.75, abs=1e-6)
        assert m["avg_win_inr"] == pytest.approx(800.0, abs=1e-4)
        assert m["avg_loss_inr"] == pytest.approx(-400.0, abs=1e-4)
        assert m["net_expectancy_inr"] == pytest.approx(200.0, abs=1e-4)
        assert m["profit_factor"] == pytest.approx(2.0, abs=1e-4)
        assert m["avg_mae_pct"] == pytest.approx(-0.9, abs=1e-4)
        assert m["avg_mfe_pct"] == pytest.approx(1.25, abs=1e-4)

        # Hand-calculated max drawdown: peak=101,000 -> trough=100,500 => dd=500 (0.4950%)
        expected_dd_pct = round((500.0 / 101_000.0) * 100.0, 4)
        assert m["max_drawdown_inr"] == pytest.approx(500.0, abs=1e-4)
        assert m["max_drawdown_pct"] == pytest.approx(expected_dd_pct, abs=1e-4)

        # Hand-calculated Sharpe & Sortino
        daily_rets = [
            (101_000.0 - 100_000.0) / 100_000.0,
            (100_500.0 - 101_000.0) / 101_000.0,
            (101_100.0 - 100_500.0) / 100_500.0,
            (100_800.0 - 101_100.0) / 101_100.0,
        ]
        mean_r = sum(daily_rets) / len(daily_rets)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_rets) / (len(daily_rets) - 1))
        downside_dev = math.sqrt(sum(min(0.0, r) ** 2 for r in daily_rets) / len(daily_rets))
        expected_sharpe = round((mean_r / std_r) * math.sqrt(TRADING_DAYS_PER_YEAR), 4)
        expected_sortino = round((mean_r / downside_dev) * math.sqrt(TRADING_DAYS_PER_YEAR), 4)
        assert m["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-4)
        assert m["sortino_ratio"] == pytest.approx(expected_sortino, abs=1e-4)

        # Cost breakdown: gross - friction == net
        cb = m["cost_breakdown"]
        assert cb["brokerage_inr"] == pytest.approx(80.0, abs=1e-4)
        assert cb["statutory_charges_inr"] == pytest.approx(60.0, abs=1e-4)
        assert cb["slippage_inr"] == pytest.approx(60.0, abs=1e-4)
        assert cb["total_friction_inr"] == pytest.approx(200.0, abs=1e-4)
        assert cb["gross_pnl_inr"] == pytest.approx(1000.0, abs=1e-4)
        assert cb["net_pnl_inr"] == pytest.approx(800.0, abs=1e-4)
        assert cb["gross_pnl_inr"] - cb["total_friction_inr"] == pytest.approx(
            cb["net_pnl_inr"], abs=1e-4
        )
        assert cb["net_return_pct"] == pytest.approx(0.8, abs=1e-4)

        # Per-stock breakdown
        assert set(m["by_stock"].keys()) == {"RELIANCE", "TCS"}
        rel = m["by_stock"]["RELIANCE"]
        assert rel["total_trades"] == 2
        assert rel["win_rate"] == pytest.approx(1.0, abs=1e-6)
        assert rel["profit_factor"] is None  # zero losses
        assert rel["net_pnl_inr"] == pytest.approx(1600.0, abs=1e-4)

        tcs = m["by_stock"]["TCS"]
        assert tcs["total_trades"] == 2
        assert tcs["win_rate"] == pytest.approx(0.0, abs=1e-6)
        assert tcs["profit_factor"] == pytest.approx(0.0, abs=1e-6)
        assert tcs["net_pnl_inr"] == pytest.approx(-800.0, abs=1e-4)

        # Per-regime breakdown
        assert set(m["by_regime"].keys()) == {"TRENDING_UP_1H_PROXY", "SIDEWAYS_1H_PROXY"}
        up = m["by_regime"]["TRENDING_UP_1H_PROXY"]
        assert up["total_trades"] == 2
        assert up["win_rate"] == pytest.approx(0.5, abs=1e-6)
        assert up["profit_factor"] == pytest.approx(2.0, abs=1e-4)
        assert up["net_pnl_inr"] == pytest.approx(500.0, abs=1e-4)

        sw = m["by_regime"]["SIDEWAYS_1H_PROXY"]
        assert sw["total_trades"] == 2
        assert sw["win_rate"] == pytest.approx(0.5, abs=1e-6)
        assert sw["profit_factor"] == pytest.approx(2.0, abs=1e-4)
        assert sw["net_pnl_inr"] == pytest.approx(300.0, abs=1e-4)

    def test_zero_trade_edge_case(self) -> None:
        m = compute_strategy_metrics(
            trades=[],
            equity_curve=[
                {"date": "2026-01-02", "equity_inr": 50_000.0},
                {"date": "2026-01-03", "equity_inr": 50_000.0},
            ],
            directional_outcomes=[],
            initial_capital_inr=50_000.0,
        )

        assert m["total_trades"] == 0
        assert m["profit_factor"] is None
        assert m["win_rate"] is None
        assert m["sharpe_ratio"] is None
        assert m["sortino_ratio"] is None
        assert m["max_drawdown_pct"] == 0.0
        assert m["max_drawdown_inr"] == 0.0
        assert m["avg_mae_pct"] is None
        assert m["avg_mfe_pct"] is None
        assert m["cost_breakdown"]["gross_pnl_inr"] == 0.0
        assert m["cost_breakdown"]["total_friction_inr"] == 0.0
        assert m["cost_breakdown"]["net_pnl_inr"] == 0.0

    def test_non_positive_initial_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_capital_inr"):
            compute_strategy_metrics([], [], [], initial_capital_inr=0.0)


# ===========================================================================
# 3. Phase 2 — Chronological Splits & Leakage Controls
# ===========================================================================


class TestPhase2ChronologicalSplits:
    """Verify build_chronological_splits ordering, disjointness, and error handling."""

    def test_chronological_splits_monotonic_disjoint_and_zero_holdout(self) -> None:
        audit = audit_historical_data_readiness()
        non_holdout_dates = audit["holdout_registry_audit"]["non_holdout_dates"]
        exposed_set = set(audit["holdout_registry_audit"]["exposed_holdout_dates"])

        # Deliberately pass reversed order to prove build_chronological_splits enforces sorting
        splits = build_chronological_splits(list(reversed(non_holdout_dates)))
        train = splits["train"]
        val = splits["validation"]
        test = splits["dev_test_oos"]

        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        assert train == sorted(train)
        assert val == sorted(val)
        assert test == sorted(test)

        # Strictly monotonic across boundaries
        assert max(train) < min(val)
        assert max(val) < min(test)

        # Disjoint and complete union
        train_s, val_s, test_s = set(train), set(val), set(test)
        assert train_s.isdisjoint(val_s)
        assert train_s.isdisjoint(test_s)
        assert val_s.isdisjoint(test_s)
        assert (train_s | val_s | test_s) == set(non_holdout_dates)
        assert len(train) + len(val) + len(test) == 196

        # Zero exposed holdout or straddle dates anywhere in the walk-forward splits
        assert (train_s | val_s | test_s).isdisjoint(exposed_set)
        assert max(test) < "2026-09-09"
        assert "2026-11-09" not in (train_s | val_s | test_s)

    @pytest.mark.parametrize(
        "bad_dates",
        [
            [],
            ["2026-01-02"],
            ["2026-01-02", "2026-01-03"],
            ["2026-01-02", "2026-01-02", "2026-01-03"],  # only 2 unique dates
        ],
    )
    def test_insufficient_dates_raises_value_error(self, bad_dates: list[str]) -> None:
        with pytest.raises(ValueError):
            build_chronological_splits(bad_dates)

    @pytest.mark.parametrize(
        ("tr", "vr", "ter"),
        [
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (-0.1, 0.6, 0.5),
            (0.6, 0.3, 0.3),  # sums to 1.2 != 1.0
            (0.4, 0.2, 0.2),  # sums to 0.8 != 1.0
        ],
    )
    def test_invalid_ratios_raise_value_error(self, tr: float, vr: float, ter: float) -> None:
        sample_dates = [f"2026-01-{d:02d}" for d in range(1, 21)]
        with pytest.raises(ValueError):
            Cycle008Config(train_ratio=tr, val_ratio=vr, test_ratio=ter)
        with pytest.raises(ValueError):
            build_chronological_splits(
                sample_dates,
                train_ratio=tr,
                val_ratio=vr,
                test_ratio=ter,
            )

    def test_missing_holdout_registry_fails_closed_with_file_not_found_error(
        self, tmp_path: Path
    ) -> None:
        missing_registry = tmp_path / "does_not_exist_registry.json"
        evaluator = Cycle008WalkForwardEvaluator(
            Cycle008Config(holdout_registry_path=missing_registry)
        )
        with pytest.raises(FileNotFoundError):
            evaluator.run()


# ===========================================================================
# 4. End-to-End CYCLE-008 Evaluation (strictly into tmp_path)
# ===========================================================================


@pytest.fixture(scope="module")
def cycle008_e2e_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run run_cycle_008_eval once into an isolated tmp_path for end-to-end assertions."""
    out_dir = tmp_path_factory.mktemp("cycle008_e2e")
    json_out = out_dir / "cycle_008_walk_forward_eval_report.json"
    md_out = out_dir / "CYCLE_008_WALK_FORWARD_EVAL_REPORT.md"

    reg_sha_before = sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)
    report = run_cycle_008_eval(json_out=json_out, md_out=md_out)
    reg_sha_after = sha256_file(DEFAULT_HOLDOUT_REGISTRY_PATH)

    return {
        "report": report,
        "json_out": json_out,
        "md_out": md_out,
        "reg_sha_before": reg_sha_before,
        "reg_sha_after": reg_sha_after,
    }


class TestCycle008EndToEnd:
    """Verify full offline CYCLE-008 walk-forward run, governance flags, ledgers, and determinism."""

    def test_governance_safety_and_isolation_flags(self, cycle008_e2e_run: dict[str, Any]) -> None:
        report = cycle008_e2e_run["report"]
        json_out: Path = cycle008_e2e_run["json_out"]
        md_out: Path = cycle008_e2e_run["md_out"]

        assert json_out.exists()
        assert md_out.exists()
        loaded = json.loads(json_out.read_text(encoding="utf-8"))
        assert loaded["output_report_sha256"] == report["output_report_sha256"]

        assert report["cycle_id"] == "CYCLE-008"
        assert report["run_status"] == "COMPLETED"
        assert report["overall_classification"] == "INSUFFICIENT_EVIDENCE"
        assert report["confidence_calibration_status"] == "UNCALIBRATED"
        assert report["qualification_allowed"] is False
        assert report["unseen_evaluation_status"] == "NOT_EVALUATED"
        assert report["live_trading_enabled"] is False
        assert report["real_money_trading"] is False
        assert report["external_api_calls"] == 0
        assert report["broker_connections"] == 0
        assert report["paid_api_cost"] == 0.0
        assert report["synthetic_candles_used"] == 0

    def test_data_leakage_audit_passes_with_zero_violations(
        self, cycle008_e2e_run: dict[str, Any]
    ) -> None:
        leak = cycle008_e2e_run["report"]["data_leakage_audit"]
        assert leak["passed"] is True
        assert leak["exposed_holdout_dates_in_walk_forward"] == 0
        assert leak["lookahead_bar_violations"] == 0
        assert leak["split_overlap_count"] == 0
        assert leak["parameter_tuning_on_dev_test_oos"] == 0
        assert leak["synthetic_candles_used"] == 0
        assert leak["fabricated_timeframes"] == 0
        assert leak["evaluated_symbol_sessions_checked"] > 0

    def test_all_five_strategies_present_tagged_and_ledgers_valid(
        self, cycle008_e2e_run: dict[str, Any]
    ) -> None:
        strats = cycle008_e2e_run["report"]["strategies"]
        expected_ids = {
            "independent_candle_timing_agent",
            "simple_technical_baseline",
            "no_trade_baseline",
            "central_directional_agent_archived_diagnostic",
            "combined_cycle007_pipeline_archived_diagnostic",
        }
        assert set(strats.keys()) == expected_ids

        # Non-holdout walk-forward strategies: valid SHA-256 chains & zero open positions at end
        for sid in (
            "independent_candle_timing_agent",
            "simple_technical_baseline",
            "no_trade_baseline",
        ):
            for split_key in ("dev_test_oos", "full_non_holdout"):
                block = strats[sid][split_key]
                assert block["ledger_reconciliation"]["sha256_chain_valid"] is True
                assert block["ledger_reconciliation"]["open_positions_at_end"] == 0

        # No-trade baseline invariants
        nt_oos = strats["no_trade_baseline"]["dev_test_oos"]["metrics"]
        assert nt_oos["total_trades"] == 0
        assert nt_oos["profit_factor"] is None
        assert nt_oos["cost_breakdown"]["net_pnl_inr"] == 0.0
        assert nt_oos["cost_breakdown"]["total_friction_inr"] == 0.0
        assert nt_oos["max_drawdown_pct"] == 0.0

        # Archived historical diagnostics: explicit scope & verdict + valid SHA-256 chains
        s4 = strats["central_directional_agent_archived_diagnostic"]
        assert s4["evaluation_scope"] == "EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY"
        assert s4["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert s4["ledger_reconciliation"]["sha256_chain_valid"] is True

        s5 = strats["combined_cycle007_pipeline_archived_diagnostic"]
        assert s5["evaluation_scope"] == "EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY"
        assert s5["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert s5["ledger_reconciliation"]["all_session_sha256_chains_valid"] is True

    def test_execution_provenance_and_holdout_registry_unchanged(
        self, cycle008_e2e_run: dict[str, Any]
    ) -> None:
        report = cycle008_e2e_run["report"]
        exec_prov = report["execution_provenance"]

        assert exec_prov["implementation_commit_sha"]
        assert isinstance(exec_prov["working_tree_clean_at_execution"], bool)
        assert isinstance(exec_prov["dirty_paths_at_execution"], list)
        assert exec_prov["report_commit_distinction"]
        for rel_mod in CYCLE008_IMPLEMENTATION_MODULES:
            assert rel_mod in exec_prov["implementation_file_hashes"]
            assert len(exec_prov["implementation_file_hashes"][rel_mod]) == 64

        reg_int = report["holdout_registry_integrity"]
        assert reg_int["unchanged"] is True
        assert reg_int["sha256_before"] == cycle008_e2e_run["reg_sha_before"]
        assert reg_int["sha256_after"] == cycle008_e2e_run["reg_sha_after"]
        assert cycle008_e2e_run["reg_sha_before"] == cycle008_e2e_run["reg_sha_after"]

        holdout = report["phase1_data_readiness"]["holdout_registry_audit"]
        assert holdout["exposed_count"] == 48
        assert holdout["available_count"] == 0
        assert holdout["sealed_count"] == 0

    def test_deterministic_replay_identical_dev_test_oos(
        self, cycle008_e2e_run: dict[str, Any]
    ) -> None:
        report_a = cycle008_e2e_run["report"]
        non_holdout_dates = report_a["phase1_data_readiness"]["holdout_registry_audit"][
            "non_holdout_dates"
        ]
        splits = build_chronological_splits(non_holdout_dates)
        dev_test_dates = splits["dev_test_oos"]

        from trader.evaluation.walk_forward import _load_validated_1h_frames

        evaluator_b = Cycle008WalkForwardEvaluator()
        frames = _load_validated_1h_frames(
            evaluator_b.config.symbols, evaluator_b.config.cache_dir
        )
        leakage_b = {"evaluated_symbol_sessions_checked": 0, "lookahead_bar_violations": 0}

        replay_runs = {
            "independent_candle_timing_agent": evaluator_b._simulate_strategy_on_dates(
                dates_iso=dev_test_dates,
                frames=frames,
                signal_fn=evaluator_b._make_independent_candle_timing_signal_fn(),
                leakage_tracker=leakage_b,
                enforce_timing_gap_rules=True,
            ),
            "simple_technical_baseline": evaluator_b._simulate_strategy_on_dates(
                dates_iso=dev_test_dates,
                frames=frames,
                signal_fn=evaluator_b._make_simple_technical_baseline_signal_fn(),
                leakage_tracker=leakage_b,
                enforce_timing_gap_rules=False,
            ),
            "no_trade_baseline": evaluator_b._simulate_strategy_on_dates(
                dates_iso=dev_test_dates,
                frames=frames,
                signal_fn=evaluator_b._make_no_trade_baseline_signal_fn(),
                leakage_tracker=leakage_b,
                enforce_timing_gap_rules=False,
            ),
        }

        assert leakage_b["lookahead_bar_violations"] == 0
        for sid, replay_block in replay_runs.items():
            m_a = report_a["strategies"][sid]["dev_test_oos"]["metrics"]
            m_b = replay_block["metrics"]
            assert m_a["total_trades"] == m_b["total_trades"]
            assert m_a["cost_breakdown"]["net_pnl_inr"] == m_b["cost_breakdown"]["net_pnl_inr"]
            assert (
                m_a["cost_breakdown"]["total_friction_inr"]
                == m_b["cost_breakdown"]["total_friction_inr"]
            )
            assert m_a["win_rate"] == m_b["win_rate"]
            assert m_a["profit_factor"] == m_b["profit_factor"]
            assert m_a["max_drawdown_pct"] == m_b["max_drawdown_pct"]
