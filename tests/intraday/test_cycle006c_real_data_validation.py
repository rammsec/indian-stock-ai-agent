"""CYCLE-006C — Isolated Tests for Real-Data Snapshot Reconstruction, Provenance & Paper Execution."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_cycle_006c_real_data_validation import run_cycle_006c_validation
from trader.config import IST_TZ, LAB_DEFAULT_UNIVERSE
from trader.paper.cycle006b import (
    DEFAULT_CACHE_DIR,
    Cycle006BConfig,
    Cycle006BPaperTradingRunner,
    sha256_file,
)
from trader.paper.cycle006c_snapshot import (
    DEFAULT_FORWARD_PREDICTIONS_PATH,
    DEFAULT_MARKET_REGIMES_PATH,
    DEFAULT_PREDICTION_LOG_PATH,
    DEFAULT_STRUCTURED_EVENTS_PATH,
    classify_timeframe_provenance,
    reconstruct_as_of_snapshot,
)


def test_reconstruct_genuine_2026_11_19_snapshot_from_committed_archives(tmp_path: Path) -> None:
    """Reconstructing 2026-11-19 @ 09:15 IST uses only genuine pre-cutoff committed archives."""
    before_hashes = {
        p: sha256_file(p)
        for p in (
            DEFAULT_PREDICTION_LOG_PATH,
            DEFAULT_MARKET_REGIMES_PATH,
            DEFAULT_FORWARD_PREDICTIONS_PATH,
            DEFAULT_STRUCTURED_EVENTS_PATH,
        )
    }

    manifest = reconstruct_as_of_snapshot(
        session_date=date(2026, 11, 19),
        output_dir=tmp_path / "snap",
    )

    assert manifest["snapshot_status"] == "RECONSTRUCTED_FROM_ARCHIVES"
    assert manifest["provenance_classification"] == "ARCHIVED_PRE_CUTOFF_RECONSTRUCTED"
    assert manifest["target_observation_date"] == "2026-11-18"
    assert manifest["as_of_cutoff_ist"] == "2026-11-19T09:15:00+05:30"

    cutoff = datetime(2026, 11, 19, 9, 15, tzinfo=IST_TZ)
    for ts_key in (
        "prediction_generated_at_ist",
        "regime_availability_ist",
        "intelligence_generated_at_ist",
        "max_verified_event_collection_ist",
    ):
        ts_val = datetime.fromisoformat(manifest["timestamps"][ts_key])
        assert ts_val <= cutoff

    ev_audit = manifest["structured_events_audit"]
    assert ev_audit["included_verified_events"] == 81
    assert ev_audit["excluded_reasons"]["missing_collection_timestamp"] == 156
    assert ev_audit["excluded_reasons"]["published_after_cutoff"] == 4
    assert ev_audit["excluded_reasons"]["collected_after_cutoff"] == 39
    for sym in LAB_DEFAULT_UNIVERSE:
        assert ev_audit["included_by_symbol"][sym] > 0

    # Source archives must remain completely untouched
    for p, h in before_hashes.items():
        assert sha256_file(p) == h


def test_reconstruct_snapshot_fails_closed_on_missing_or_future_data(tmp_path: Path) -> None:
    """Snapshot reconstruction rejects missing archives, pre-history dates, and post-cutoff events."""
    with pytest.raises(FileNotFoundError):
        reconstruct_as_of_snapshot(
            session_date=date(2026, 11, 19),
            output_dir=tmp_path / "missing",
            prediction_log_path=tmp_path / "does_not_exist.parquet",
        )

    with pytest.raises(ValueError, match="HISTORICAL_PREDICTIONS_NOT_RECONSTRUCTABLE"):
        reconstruct_as_of_snapshot(
            session_date=date(2020, 1, 2),
            output_dir=tmp_path / "too_early",
        )

    # Events published before cutoff but collected AFTER cutoff (or missing collection_timestamp)
    # must be excluded; if a symbol has zero valid events, reconstruction must fail closed.
    se_bad = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "publication_timestamp": "2026-11-18T12:00:00+05:30",
                "collection_timestamp": "2026-11-19T09:16:00+05:30",  # post-cutoff collection!
            },
            {
                "ticker": "RELIANCE",
                "publication_timestamp": "2026-11-18T10:00:00+05:30",
                "collection_timestamp": None,  # missing collection timestamp!
            },
        ]
    )
    bad_se_path = tmp_path / "bad_events.parquet"
    se_bad.to_parquet(bad_se_path, index=False)

    with pytest.raises(ValueError, match="HISTORICAL_INTELLIGENCE_NOT_RECONSTRUCTABLE"):
        reconstruct_as_of_snapshot(
            session_date=date(2026, 11, 19),
            output_dir=tmp_path / "bad_ev_out",
            structured_events_path=bad_se_path,
        )


def test_reconstruct_snapshot_rejects_naive_timestamps_and_regime_mismatch(tmp_path: Path) -> None:
    """Timezone-naive timestamps and regime mismatches must fail closed."""
    pl_df = pd.read_parquet(DEFAULT_PREDICTION_LOG_PATH).copy()
    pl_df["generated_at"] = "2026-11-18T21:04:26"  # naive timestamp
    naive_pl = tmp_path / "naive_pl.parquet"
    pl_df.to_parquet(naive_pl, index=False)

    with pytest.raises(ValueError, match="Timezone-naive"):
        reconstruct_as_of_snapshot(
            session_date=date(2026, 11, 19),
            output_dir=tmp_path / "naive_out",
            prediction_log_path=naive_pl,
        )

    mr_df = pd.read_parquet(DEFAULT_MARKET_REGIMES_PATH).copy()
    mr_df["regime"] = "Strong Bear"  # conflicts with prediction_log Weak Bull
    bad_mr = tmp_path / "bad_mr.parquet"
    mr_df.to_parquet(bad_mr, index=False)

    with pytest.raises(ValueError, match="Regime mismatch"):
        reconstruct_as_of_snapshot(
            session_date=date(2026, 11, 19),
            output_dir=tmp_path / "mismatch_out",
            market_regimes_path=bad_mr,
        )


def test_real_data_cycle006b_execution_completes_deterministically_with_zero_trades(
    tmp_path: Path,
) -> None:
    """Running CYCLE-006B on genuine 1h archives + reconstructed 2026-11-19 snapshot completes cleanly."""
    snap = reconstruct_as_of_snapshot(
        session_date=date(2026, 11, 19),
        output_dir=tmp_path / "snap",
    )
    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 19),
        symbols=LAB_DEFAULT_UNIVERSE,
        initial_capital_inr=50_000.0,
        cache_dir=DEFAULT_CACHE_DIR,
        predictions_path=Path(snap["output_files"]["predictions_path"]),
        intelligence_path=Path(snap["output_files"]["intelligence_path"]),
        regime_path=Path(snap["output_files"]["regime_path"]),
    )

    rep1 = Cycle006BPaperTradingRunner(cfg).run()
    rep2 = Cycle006BPaperTradingRunner(cfg).run()

    for rep in (rep1, rep2):
        assert rep["run_status"] == "COMPLETED"
        assert rep["order_count"] == 0
        assert rep["filled_order_count"] == 0
        assert len(rep["risk_decisions"]) == len(LAB_DEFAULT_UNIVERSE)
        assert rep["portfolio_results"]["initial_capital_inr"] == 50_000.0
        assert rep["portfolio_results"]["ending_cash_inr"] == 50_000.0
        assert rep["portfolio_results"]["ending_equity_inr"] == 50_000.0
        assert rep["portfolio_results"]["net_pnl_inr"] == 0.0
        assert rep["cost_accounting"]["total_friction_inr"] == 0.0
        assert rep["ledger_reconciliation"]["sha256_chain_valid"] is True
        assert rep["ledger_reconciliation"]["cash_plus_invested_matches_equity"] is True
        assert rep["synthetic_candles_used"] == 0
        assert rep["external_api_calls"] == 0
        assert rep["broker_connections"] == 0
        assert rep["paid_api_cost"] == 0.0
        assert rep["qualification_allowed"] is False
        assert rep["unseen_evaluation_status"] == "NOT_EVALUATED"
        assert rep["real_money_trading"] is False

    assert rep1["portfolio_results"] == rep2["portfolio_results"]
    assert rep1["risk_decisions"] == rep2["risk_decisions"]
    assert rep1["order_count"] == rep2["order_count"]


def test_timeframe_provenance_classification_committed_clean_and_corrupt(tmp_path: Path) -> None:
    """1h/5m/15m provenance classification distinguishes committed UNAVAILABLE, PARTIALLY_VERIFIED, and corrupt UNVERIFIED."""
    committed = classify_timeframe_provenance(
        session_date=date(2026, 11, 19),
        symbols=LAB_DEFAULT_UNIVERSE,
        cache_dir_1h=DEFAULT_CACHE_DIR,
        cache_dir_5m=None,
    )
    assert committed["1h"]["classification"] == "PARTIALLY_VERIFIED"
    assert committed["5m"]["classification"] == "UNAVAILABLE"
    assert committed["5m"]["committed_in_repo"] is False
    assert committed["15m"]["classification"] == "UNAVAILABLE"

    # Build a clean 5m fixture (6 bars -> 2 completed 15m bars) for 2 symbols
    clean_dir = tmp_path / "clean_5m"
    clean_dir.mkdir()
    base_ts = datetime(2026, 11, 19, 9, 15, tzinfo=IST_TZ)
    for sym in ("RELIANCE", "TCS"):
        rows = [
            {
                "timestamp": base_ts + timedelta(minutes=5 * i),
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 1000,
            }
            for i in range(6)
        ]
        pd.DataFrame(rows).to_parquet(clean_dir / f"{sym}_NSE_5m.parquet", index=False)

    clean_prov = classify_timeframe_provenance(
        session_date=date(2026, 11, 19),
        symbols=("RELIANCE", "TCS"),
        cache_dir_1h=DEFAULT_CACHE_DIR,
        cache_dir_5m=clean_dir,
    )
    assert clean_prov["5m"]["classification"] == "PARTIALLY_VERIFIED"
    assert clean_prov["15m"]["classification"] == "PARTIALLY_VERIFIED"
    assert clean_prov["15m"]["symbols"]["RELIANCE"]["aggregated_completed_bars"] == 2

    # Corrupt OHLC (high < low) must fail closed to UNVERIFIED
    corrupt_dir = tmp_path / "corrupt_5m"
    corrupt_dir.mkdir()
    for sym in ("RELIANCE", "TCS"):
        rows = [
            {
                "timestamp": base_ts,
                "open": 100.0,
                "high": 95.0,  # invalid: high < low
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            }
        ]
        pd.DataFrame(rows).to_parquet(corrupt_dir / f"{sym}_NSE_5m.parquet", index=False)

    corrupt_prov = classify_timeframe_provenance(
        session_date=date(2026, 11, 19),
        symbols=("RELIANCE", "TCS"),
        cache_dir_1h=DEFAULT_CACHE_DIR,
        cache_dir_5m=corrupt_dir,
    )
    assert corrupt_prov["5m"]["classification"] == "UNVERIFIED"
    assert corrupt_prov["15m"]["classification"] == "UNVERIFIED"


def test_end_to_end_cycle006c_validation_runner_and_holdout_immutability(tmp_path: Path) -> None:
    """End-to-end CYCLE-006C runner produces valid reports without touching sealed_holdout_registry.json."""
    reg_path = Path("data/lab/sealed_holdout_registry.json")
    reg_hash_before = sha256_file(reg_path)

    json_out = tmp_path / "cycle006c.json"
    md_out = tmp_path / "cycle006c.md"
    report = run_cycle_006c_validation(
        session_date=date(2026, 11, 19),
        snapshot_dir=tmp_path / "snap",
        json_out=json_out,
        md_out=md_out,
        iso_5m_cache_dir=None,
    )

    assert report["primary_classification"] == "PASS_INFRASTRUCTURE_WITH_ZERO_TRADES"
    assert report["governance"]["qualification_allowed"] is False
    assert report["governance"]["unseen_evaluation_status"] == "NOT_EVALUATED"
    assert report["governance"]["live_trading_enabled"] is False
    assert report["governance"]["synthetic_candles_used"] == 0
    assert report["governance"]["external_api_calls"] == 0
    assert report["governance"]["cycle_007_implemented"] is False
    assert json_out.exists()
    assert md_out.exists()
    assert "PASS_INFRASTRUCTURE_WITH_ZERO_TRADES" in md_out.read_text(encoding="utf-8")

    # Verify holdout registry is untouched
    assert sha256_file(reg_path) == reg_hash_before
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    statuses = [d["state"] for d in reg.get("records", {}).values()]
    assert statuses.count("EXPOSED") == 48
    assert statuses.count("AVAILABLE") == 0
    assert statuses.count("SEALED") == 0
