"""Isolated unit and integration tests for CYCLE-006B Paper-Trading Infrastructure."""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.run_cycle_006b_paper_dry_run import main as cli_main
from trader.config import IST_TZ
from trader.paper.cycle006b import (
    Cycle006BConfig,
    Cycle006BPaperTradingRunner,
    ExecutionInstrumentation,
    ExternalNetworkBlockedError,
    OfflineExecutionGuard,
    sha256_file,
)
from trader.risk.engine import DeterministicRiskEngine, RiskEvaluationResult


def _build_1h_fixture_dir(
    tmp_path: Path,
    symbols: tuple[str, ...] = ("RELIANCE", "TCS"),
    session_date: date = date(2026, 11, 20),
    num_sessions: int = 65,
) -> Path:
    """Create deterministic, valid 1h parquet fixtures in tmp_path (zero repo writes)."""
    cache_dir = tmp_path / "intraday"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate num_sessions trading weekdays ending on session_date
    days: list[date] = []
    cur = session_date
    while len(days) < num_sessions:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    days.reverse()

    hours = [9, 10, 11, 12, 13, 14, 15]
    for s_idx, sym in enumerate(symbols):
        base = 1200.0 + s_idx * 400.0
        rows: list[dict[str, Any]] = []
        for d_idx, d in enumerate(days):
            for h_idx, hr in enumerate(hours):
                ts = datetime(d.year, d.month, d.day, hr, 15, tzinfo=IST_TZ)
                o = round(base + d_idx * 1.5 + h_idx * 2.0, 2)
                c = round(o + 3.0, 2)
                h = round(max(o, c) + 2.0, 2)
                l = round(min(o, c) - 1.5, 2)
                rows.append({
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": 50000 + h_idx * 1000,
                })
        pd.DataFrame(rows).to_parquet(cache_dir / f"{sym}_1h.parquet", index=False)
    return cache_dir


def _build_aligned_inputs(
    tmp_path: Path,
    symbols: tuple[str, ...] = ("RELIANCE", "TCS"),
    session_date: date = date(2026, 11, 20),
    pred_ts: str = "2026-11-19T18:00:00+05:30",
    intel_ts: str = "2026-11-19T19:00:00+05:30",
    regime_obs: str = "2026-11-19",
    regime_label: str = "Weak Bull",
    prob_up: float = 0.68,
    exp_ret: float = 0.85,
    news_sent: float = 0.25,
    model_version: str = "5.0.0-pilot",
) -> tuple[Path, Path, Path]:
    pred_path = tmp_path / "latest_predictions.json"
    intel_path = tmp_path / "phase5_intelligence.json"
    regime_path = tmp_path / "regime.json"

    pred_obs = pred_ts[:10]
    intel_obs = intel_ts[:10]

    pred_payload = {
        "generated_at": pred_ts,
        "market_regime": regime_label,
        "predictions": [
            {
                "symbol": sym,
                "observation_date": pred_obs,
                "generated_at": pred_ts,
                "model_version": "3.0.0",
                "prob_next_day_up": prob_up,
                "expected_next_day_return_pct": exp_ret,
            }
            for sym in symbols
        ],
    }
    intel_payload = {
        "generated_at": intel_ts,
        "model_version": model_version,
        "stocks": {
            sym: {
                "symbol": sym,
                "observation_date": intel_obs,
                "news_sentiment_score": news_sent,
            }
            for sym in symbols
        },
    }
    regime_payload = {
        "observation_date": regime_obs,
        "regime": regime_label,
    }

    pred_path.write_text(json.dumps(pred_payload, indent=2), encoding="utf-8")
    intel_path.write_text(json.dumps(intel_payload, indent=2), encoding="utf-8")
    regime_path.write_text(json.dumps(regime_payload, indent=2), encoding="utf-8")
    return pred_path, intel_path, regime_path


def _make_valid_config(tmp_path: Path, symbols: tuple[str, ...] = ("RELIANCE", "TCS")) -> Cycle006BConfig:
    cache_dir = _build_1h_fixture_dir(tmp_path, symbols=symbols)
    pred_p, intel_p, reg_p = _build_aligned_inputs(tmp_path, symbols=symbols)
    return Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=symbols,
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )


# ---------------------------------------------------------------------------
# 1–11: Input Validation & Provenance Fail-Closed Tests
# ---------------------------------------------------------------------------

def test_01_missing_market_data_fails_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)
    (cfg.cache_dir / "TCS_1h.parquet").unlink()
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert rep["order_count"] == 0
    assert rep["rejection_reasons"]


def test_02_missing_predictions_fails_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)
    cfg.predictions_path.unlink()
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert rep["order_count"] == 0


def test_03_missing_intelligence_fails_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)
    cfg.intelligence_path.unlink()
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert rep["order_count"] == 0


def test_04_missing_regime_fails_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)
    cfg.regime_path.unlink()
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert rep["order_count"] == 0


def test_05_malformed_json_fails_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)
    cfg.predictions_path.write_text("{bad json", encoding="utf-8")
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert any("Malformed JSON" in r for r in rep["rejection_reasons"])


def test_06_stale_inputs_fail_closed(tmp_path: Path) -> None:
    cache_dir = _build_1h_fixture_dir(tmp_path)
    pred_p, intel_p, reg_p = _build_aligned_inputs(
        tmp_path,
        pred_ts="2026-11-10T18:00:00+05:30",
        intel_ts="2026-11-10T19:00:00+05:30",
        regime_obs="2026-11-10",
    )
    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=("RELIANCE", "TCS"),
        max_staleness_hours=48.0,
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert any("Stale input" in r for r in rep["rejection_reasons"])


def test_07_future_dated_inputs_fail_closed(tmp_path: Path) -> None:
    cache_dir = _build_1h_fixture_dir(tmp_path)
    pred_p, intel_p, reg_p = _build_aligned_inputs(
        tmp_path,
        pred_ts="2026-11-20T17:30:00+05:30",  # After 09:15 IST as_of
        intel_ts="2026-11-19T18:00:00+05:30",
        regime_obs="2026-11-19",
    )
    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=("RELIANCE", "TCS"),
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert any("Future-dated" in r for r in rep["rejection_reasons"])


def test_08_conflicting_timestamps_fail_closed(tmp_path: Path) -> None:
    cache_dir = _build_1h_fixture_dir(tmp_path)
    pred_p, intel_p, reg_p = _build_aligned_inputs(
        tmp_path,
        pred_ts="2026-11-17T18:00:00+05:30",
        intel_ts="2026-11-20T08:00:00+05:30",  # 62h spread > 24h max_input_spread_hours
        regime_obs="2026-11-19",
    )
    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=("RELIANCE", "TCS"),
        max_staleness_hours=96.0,
        max_input_spread_hours=24.0,
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert any("Conflicting input timestamps" in r for r in rep["rejection_reasons"])


def test_09_mismatched_instruments_fail_closed(tmp_path: Path) -> None:
    cache_dir = _build_1h_fixture_dir(tmp_path, symbols=("RELIANCE", "TCS"))
    pred_p, intel_p, reg_p = _build_aligned_inputs(tmp_path, symbols=("RELIANCE", "INFY"))
    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=("RELIANCE", "TCS"),
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert any("Mismatched instruments" in r for r in rep["rejection_reasons"])


def test_10_invalid_and_incomplete_market_data_fails_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)
    pq_path = cfg.cache_dir / "RELIANCE_1h.parquet"
    df = pd.read_parquet(pq_path)
    # Inject NaN in last row close
    df.loc[df.index[-1], "close"] = float("nan")
    df.to_parquet(pq_path, index=False)

    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"
    assert rep["order_count"] == 0


def test_11_partially_verified_provenance_preserved_never_verified(tmp_path: Path) -> None:
    """Real committed 1h archive is PARTIALLY_VERIFIED and must never be reported as VERIFIED."""
    real_cache = Path("data/intraday")
    pred_p, intel_p, reg_p = _build_aligned_inputs(
        tmp_path,
        symbols=("RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI"),
        session_date=date(2026, 11, 20),
    )
    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=("RELIANCE", "TCS", "BIRLACABLE", "ITC", "TATAELXSI"),
        cache_dir=real_cache,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "COMPLETED"
    assert rep["market_data_provenance"]["provenance_status"] == "PARTIALLY_VERIFIED"
    assert rep["market_data_provenance"]["provenance_verified"] is False
    assert rep["qualification_allowed"] is False


# ---------------------------------------------------------------------------
# 12–16: Risk Engine Mandatory Gate Tests
# ---------------------------------------------------------------------------

class _RejectAllRiskEngine(DeterministicRiskEngine):
    def evaluate_order(self, *args: Any, **kwargs: Any) -> RiskEvaluationResult:
        return RiskEvaluationResult(
            approved=False,
            status="REJECTED_TEST_LIMIT",
            approved_quantity=0,
            requested_quantity=int(kwargs.get("requested_quantity", 1)),
            rejection_reasons=["Forced test risk rejection"],
            max_risk_amount_inr=250.0,
            estimated_trade_risk_inr=500.0,
        )


class _NoneRiskEngine(DeterministicRiskEngine):
    def evaluate_order(self, *args: Any, **kwargs: Any) -> Any:
        return None


def test_12_to_16_risk_engine_rejection_and_missing_decision_fail_closed(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)

    # Rejection creates 0 orders, 0 positions, and records reasons
    runner_rej = Cycle006BPaperTradingRunner(cfg, risk_engine=_RejectAllRiskEngine())
    rep_rej = runner_rej.run()
    assert rep_rej["run_status"] == "COMPLETED"
    assert rep_rej["order_count"] == 0
    assert rep_rej["open_positions_count"] == 0
    assert rep_rej["closed_trades_count"] == 0
    assert len(runner_rej.portfolio.orders) == 0
    assert len(runner_rej.portfolio.positions) == 0
    assert any("Forced test risk rejection" in r for r in rep_rej["rejection_reasons"])
    assert all(d["stage"] == "RISK_ENGINE" and d["approved"] is False for d in rep_rej["risk_decisions"])

    # Missing risk decision (None) fails closed
    runner_none = Cycle006BPaperTradingRunner(cfg, risk_engine=_NoneRiskEngine())
    rep_none = runner_none.run()
    assert rep_none["order_count"] == 0
    assert rep_none["open_positions_count"] == 0
    assert any("Missing risk decision" in r for r in rep_none["rejection_reasons"])
    assert all(d["status"] == "FAIL_CLOSED_MISSING_DECISION" for d in rep_none["risk_decisions"])


# ---------------------------------------------------------------------------
# 17–24: Portfolio, Cost Accounting, SHA-256 Ledger & Determinism Tests
# ---------------------------------------------------------------------------

def test_17_to_24_portfolio_costs_ledger_tamper_and_determinism(tmp_path: Path) -> None:
    cfg = _make_valid_config(tmp_path)

    runner1 = Cycle006BPaperTradingRunner(cfg)
    rep1 = runner1.run()
    assert rep1["run_status"] == "COMPLETED"
    assert rep1["filled_order_count"] >= 2  # At least 1 BUY + 1 EOD square-off SELL
    assert rep1["open_positions_count"] == 0  # Squared off at session close
    assert rep1["closed_trades_count"] >= 1

    # Costs, slippage, and gross vs net distinction
    costs = rep1["cost_accounting"]
    assert costs["brokerage_inr"] > 0.0
    assert costs["statutory_charges_inr"] > 0.0
    assert costs["slippage_inr"] > 0.0
    assert costs["total_friction_inr"] > 0.0
    res1 = rep1["portfolio_results"]
    assert res1["gross_pnl_inr"] != res1["net_pnl_inr"]
    assert res1["gross_pnl_inr"] > res1["net_pnl_inr"]

    # Ledger reconciliation & SHA-256 chain validation
    rec = rep1["ledger_reconciliation"]
    assert rec["sha256_chain_valid"] is True
    assert rec["cash_plus_invested_matches_equity"] is True
    assert rec["ledger_event_count"] > 0

    # Tampered ledger is detected
    runner1.portfolio.event_ledger[0].details["tampered"] = True
    tampered_ok, _ = runner1.portfolio.verify_ledger_integrity()
    assert tampered_ok is False

    # Determinism across repeated runs with identical inputs
    runner2 = Cycle006BPaperTradingRunner(cfg)
    rep2 = runner2.run()
    assert rep1["portfolio_results"] == rep2["portfolio_results"]
    assert rep1["cost_accounting"] == rep2["cost_accounting"]
    assert rep1["order_count"] == rep2["order_count"]
    assert rep1["filled_order_count"] == rep2["filled_order_count"]


# ---------------------------------------------------------------------------
# 25–34: Safety, Offline Instrumentation, Immutability & CLI Tests
# ---------------------------------------------------------------------------

def test_25_to_34_safety_instrumentation_immutability_and_cli(tmp_path: Path) -> None:
    # Verify OfflineExecutionGuard actually blocks & counts attempted network calls
    inst = ExecutionInstrumentation()
    with OfflineExecutionGuard(inst):
        with pytest.raises(ExternalNetworkBlockedError):
            socket.create_connection(("example.com", 443), timeout=1)
    assert inst.network_guard_active is True
    assert inst.external_api_calls == 1
    assert len(inst.blocked_network_attempts) == 1

    # Snapshot protected repo files before running CLI
    protected_paths = [
        Path("data/lab/sealed_holdout_registry.json"),
        Path("data/lab/cycle_005_report.json"),
        Path("data/lab/cycle_005_report_v2.json"),
        Path("data/intraday/RELIANCE_1h.parquet"),
    ]
    hashes_before = {p: sha256_file(p) for p in protected_paths}

    cfg = _make_valid_config(tmp_path)
    out_file = tmp_path / "out" / "cycle_006b_test_report.json"
    rc = cli_main([
        "--session-date", "2026-11-20",
        "--symbols", "RELIANCE,TCS",
        "--cache-dir", str(cfg.cache_dir),
        "--predictions-path", str(cfg.predictions_path),
        "--intelligence-path", str(cfg.intelligence_path),
        "--regime-path", str(cfg.regime_path),
        "--output-path", str(out_file),
    ])
    assert rc == 0
    assert out_file.exists()

    rep = json.loads(out_file.read_text(encoding="utf-8"))
    assert rep["cycle_id"] == "CYCLE-006B"
    assert rep["run_mode"] == "HISTORICAL_PAPER_TRADING_DRY_RUN"
    assert rep["qualification_allowed"] is False
    assert rep["unseen_evaluation_status"] == "NOT_EVALUATED"
    assert rep["unseen_return_pct"] is None
    assert rep["real_money_trading"] is False
    assert rep["synthetic_candles_used"] == 0
    assert rep["broker_connections"] == 0
    assert rep["external_api_calls"] == 0
    assert rep["paid_api_cost"] == 0.0
    assert rep["instrumentation"]["network_guard_active"] is True
    assert rep["output_report_sha256"]

    # Verify output_report_sha256 matches canonical payload
    recorded_hash = rep.pop("output_report_sha256")
    recomputed = hashlib.sha256(
        json.dumps(rep, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded_hash == recomputed

    # Verify protected repo files are byte-for-byte unchanged
    for p, h_before in hashes_before.items():
        assert sha256_file(p) == h_before


# ---------------------------------------------------------------------------
# 35–41: Guardian Audit Regression Tests (Auto-Exit, Rejected Costs, Sockets,
#        NaN/Inf/Date Validation, Canonical Determinism & CLI Error Handling)
# ---------------------------------------------------------------------------

def test_35_intraday_stop_loss_auto_exit_recorded_and_reconciled(tmp_path: Path) -> None:
    """Auto-executed SL/target exits from update_market_price must be recorded and cost-accounted."""
    cfg = _make_valid_config(tmp_path, symbols=("RELIANCE",))
    pq_path = cfg.cache_dir / "RELIANCE_1h.parquet"
    df = pd.read_parquet(pq_path)

    # Trigger stop-loss on bar 2 of session_date (drop close by > 3% below bar 0 open)
    ts_ist = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST_TZ)
    day_mask = ts_ist.dt.date == cfg.session_date
    day_indices = df.index[day_mask].tolist()
    entry_open = float(df.loc[day_indices[0], "open"])
    sl_crash_px = round(entry_open * 0.95, 2)
    df.loc[day_indices[2], "open"] = sl_crash_px
    df.loc[day_indices[2], "high"] = round(sl_crash_px + 1.0, 2)
    df.loc[day_indices[2], "low"] = round(sl_crash_px - 1.0, 2)
    df.loc[day_indices[2], "close"] = sl_crash_px
    df.to_parquet(pq_path, index=False)

    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "COMPLETED"
    assert rep["open_positions_count"] == 0
    assert rep["closed_trades_count"] == 1
    assert rep["order_count"] == 2
    assert rep["filled_order_count"] == 2
    sides = [o["side"] for o in rep["orders"]]
    assert sides == ["BUY", "SELL"]
    # Exit happened at bar 2 (stop-loss), not final bar
    assert rep["orders"][1]["timestamp_ist"] >= rep["orders"][0]["timestamp_ist"]
    # Gross PnL - total friction == Net PnL within rounding
    res = rep["portfolio_results"]
    costs = rep["cost_accounting"]
    assert abs((res["gross_pnl_inr"] - costs["total_friction_inr"]) - res["net_pnl_inr"]) <= 0.02


def test_36_rejected_portfolio_order_excluded_from_costs_and_gross_pnl(tmp_path: Path) -> None:
    """Orders rejected by portfolio.submit_order (insufficient cash after friction) must not inflate costs or gross PnL."""
    cache_dir = _build_1h_fixture_dir(tmp_path, symbols=("RELIANCE",))
    pred_p, intel_p, reg_p = _build_aligned_inputs(tmp_path, symbols=("RELIANCE",))

    # Set initial_capital_inr to exact 1-share price so risk engine approves 1 share
    # (cash // entry_price == 1) but portfolio rejects due to price + friction > cash.
    df = pd.read_parquet(cache_dir / "RELIANCE_1h.parquet")
    ts_ist = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST_TZ)
    entry_open = float(df.loc[ts_ist.dt.date == date(2026, 11, 20), "open"].iloc[0])

    cfg = Cycle006BConfig(
        session_date=date(2026, 11, 20),
        symbols=("RELIANCE",),
        initial_capital_inr=entry_open,
        max_risk_per_trade_pct=1.0,
        max_position_weight_pct=1.0,
        max_portfolio_exposure_pct=1.0,
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "COMPLETED"
    assert rep["order_count"] == 1
    assert rep["filled_order_count"] == 0
    assert rep["orders"][0]["status"] == "REJECTED"
    assert rep["cost_accounting"]["total_friction_inr"] == 0.0
    assert rep["cost_accounting"]["brokerage_inr"] == 0.0
    assert rep["portfolio_results"]["gross_pnl_inr"] == 0.0
    assert rep["portfolio_results"]["net_pnl_inr"] == 0.0
    assert any("Insufficient cash" in r for r in rep["rejection_reasons"])


def test_37_offline_guard_blocks_raw_socket_connect_and_connect_ex() -> None:
    """OfflineExecutionGuard must block direct socket.socket.connect and connect_ex calls."""
    inst = ExecutionInstrumentation()
    with OfflineExecutionGuard(inst):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ExternalNetworkBlockedError):
                s.connect(("93.184.216.34", 443))
            with pytest.raises(ExternalNetworkBlockedError):
                s.connect_ex(("93.184.216.34", 443))
        finally:
            s.close()
    assert inst.external_api_calls == 2
    assert len(inst.blocked_network_attempts) == 2


def test_38_nan_inf_non_numeric_and_garbage_date_fail_closed(tmp_path: Path) -> None:
    """Non-numeric, NaN/Inf, or malformed observation_date inputs must fail closed cleanly."""
    # Non-numeric prob_next_day_up
    cfg = _make_valid_config(tmp_path / "case_non_num")
    payload = json.loads(cfg.predictions_path.read_text(encoding="utf-8"))
    payload["predictions"][0]["prob_next_day_up"] = "not_a_float"
    cfg.predictions_path.write_text(json.dumps(payload), encoding="utf-8")
    rep = Cycle006BPaperTradingRunner(cfg).run()
    assert rep["run_status"] == "BLOCKED_INPUT_VALIDATION"

    # Non-standard JSON NaN constant
    cfg2 = _make_valid_config(tmp_path / "case_nan_json")
    raw_txt = cfg2.predictions_path.read_text(encoding="utf-8").replace("0.68", "NaN")
    cfg2.predictions_path.write_text(raw_txt, encoding="utf-8")
    rep2 = Cycle006BPaperTradingRunner(cfg2).run()
    assert rep2["run_status"] == "BLOCKED_INPUT_VALIDATION"

    # Trailing garbage in observation_date
    cfg3 = _make_valid_config(tmp_path / "case_bad_obs")
    reg_payload = json.loads(cfg3.regime_path.read_text(encoding="utf-8"))
    reg_payload["observation_date"] = "2026-11-19GARBAGE"
    cfg3.regime_path.write_text(json.dumps(reg_payload), encoding="utf-8")
    rep3 = Cycle006BPaperTradingRunner(cfg3).run()
    assert rep3["run_status"] == "BLOCKED_INPUT_VALIDATION"

    # Inf in parquet OHLCV
    cfg4 = _make_valid_config(tmp_path / "case_inf_pq")
    pq_path = cfg4.cache_dir / "RELIANCE_1h.parquet"
    df = pd.read_parquet(pq_path)
    df.loc[df.index[-1], "high"] = float("inf")
    df.to_parquet(pq_path, index=False)
    rep4 = Cycle006BPaperTradingRunner(cfg4).run()
    assert rep4["run_status"] == "BLOCKED_INPUT_VALIDATION"


def test_39_canonical_payload_determinism_excluding_wall_clock(tmp_path: Path) -> None:
    """Two runs with identical inputs must produce byte-for-byte identical canonical payloads (excluding wall-clock fields)."""
    cfg = _make_valid_config(tmp_path)
    rep1 = Cycle006BPaperTradingRunner(cfg).run()
    rep2 = Cycle006BPaperTradingRunner(cfg).run()

    for key in ("started_at_ist", "completed_at_ist", "output_report_sha256"):
        rep1.pop(key, None)
        rep2.pop(key, None)

    c1 = json.dumps(rep1, sort_keys=True, separators=(",", ":"))
    c2 = json.dumps(rep2, sort_keys=True, separators=(",", ":"))
    assert c1 == c2


def test_40_cli_invalid_date_and_bad_config_fail_closed_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI must return exit code 2 and emit a clean stderr message on invalid date or bad args."""
    rc_date = cli_main(["--session-date", "not-a-date"])
    assert rc_date == 2
    assert "configuration error" in capsys.readouterr().err.lower()

    rc_cap = cli_main(["--session-date", "2026-11-20", "--capital", "-500"])
    assert rc_cap == 2
    assert "configuration error" in capsys.readouterr().err.lower()

