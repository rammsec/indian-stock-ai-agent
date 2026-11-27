"""REMED-02 Deferred Hardening — Genuine Behavioral & Regression Test Suite.

Covers all 6 REMED-02 items:
1. PY-04: MandatoryExecutionGate non-bypass (AST + dynamic stack spy + fail-closed missing Human permission)
   across both Cycle006BPaperTradingRunner and Cycle007Pipeline.
2. R5-D: Honest unclipped short-gap losses (>100% adverse move), negative cash, margin_call_triggered,
   MARGIN_CALL_DEFICIT ledger event, intact SHA-256 chain, and RiskEngine HALTED + EXIT_ONLY (RISK_EXIT).
3. D6-09: Fail-closed timezone-naive and missing/NaT timestamp rejection in normalize_1m_dataframe
   and loader_1h, with clean UTC/IST tz-aware conversion to IST.
4. D6-11: build_trader_dataset(HISTORICAL_REAL) never calls generate_synthetic_5m_session and fails
   closed (status="UNAVAILABLE", synthetic_substitution=False); SYNTHETIC_TEST tags source_mode.
5. D6-03: validate_1h_provenance() defaults enforce_provider_authority_cap=True (caps Yahoo 1h to
   PARTIALLY_VERIFIED, qualification_allowed=False, provider_authority_cap_opted_out=False);
   explicit opt-out sets provider_authority_cap_opted_out=True.
6. DEF-7.9: .gitignore ignores data/intraday_1m/*_report.json while NOT ignoring provenance_1m.json.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts import generate_trader_data
from trader.authority.instructions import PermissionType
from trader.config import IST_TZ, is_nse_trading_day
from trader.data import loader_1h
from trader.data.loader import DataUnavailableError
from trader.data.loader_1m import normalize_1m_dataframe
from trader.data.provenance_1h import validate_1h_provenance
from trader.data.schema import DataSourceMode
from trader.paper.cycle006b import Cycle006BConfig, Cycle006BPaperTradingRunner
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from trader.risk.engine import DeterministicRiskEngine, RiskState
from trader.timing.cycle007_pipeline import Cycle007Config, Cycle007Pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers for isolated tmp_path fixtures (zero repository writes)
# ---------------------------------------------------------------------------

def _trading_days_ending(end_date: date, count: int) -> list[date]:
    days: list[date] = []
    cur = end_date
    while len(days) < count:
        if is_nse_trading_day(cur):
            days.append(cur)
        cur -= timedelta(days=1)
    days.reverse()
    return days


def _write_1h_fixture(
    cache_dir: Path,
    symbols: tuple[str, ...],
    session_date: date,
    num_sessions: int = 65,
    *,
    bullish_breakout_on_last_day: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = _trading_days_ending(session_date, num_sessions)
    hours = [9, 10, 11, 12, 13, 14, 15]

    for s_idx, sym in enumerate(symbols):
        base = 1200.0 + s_idx * 350.0
        rows: list[dict[str, Any]] = []
        for d_idx, d in enumerate(days):
            for h_idx, hr in enumerate(hours):
                ts = datetime(d.year, d.month, d.day, hr, 15, tzinfo=IST_TZ)
                if bullish_breakout_on_last_day and d == session_date:
                    # Clean upward bullish session for Cycle007 ENTER_NOW alignment
                    o = round(base + 120.0 + h_idx * 4.0, 2)
                    c = round(o + 5.0, 2)
                    h = round(c + 2.0, 2)
                    l = round(o - 1.0, 2)
                    vol = 250_000.0 + h_idx * 15_000.0
                elif bullish_breakout_on_last_day:
                    # Steady prior uptrend so Stage-2 timing engine confirms trend & breakout
                    step = d_idx * 1.5 + h_idx * 0.4
                    o = round(base + step, 2)
                    c = round(o + 1.2, 2)
                    h = round(c + 0.8, 2)
                    l = round(o - 0.5, 2)
                    vol = 120_000.0 + h_idx * 5_000.0
                else:
                    drift = d_idx * 0.5 + h_idx * 1.25
                    o = round(base + drift, 2)
                    c = round(o + 1.5, 2)
                    h = round(max(o, c) + 1.0, 2)
                    l = round(min(o, c) - 0.8, 2)
                    vol = 100_000.0 + h_idx * 10_000.0
                rows.append(
                    {
                        "timestamp": ts,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": vol,
                    }
                )
        pd.DataFrame(rows).to_parquet(cache_dir / f"{sym}_1h.parquet", index=False)


def _write_stage1_inputs(
    tmp_path: Path,
    symbols: tuple[str, ...],
    session_date: date,
) -> tuple[Path, Path, Path]:
    obs_date = (session_date - timedelta(days=1)).isoformat()
    gen_at = f"{obs_date}T21:00:00+05:30"

    preds = {
        "generated_at": gen_at,
        "market_regime": "Bull",
        "predictions": [
            {
                "symbol": sym,
                "observation_date": obs_date,
                "generated_at": gen_at,
                "model_version": "3.0.0",
                "prob_next_day_up": 0.72,
                "expected_next_day_return_pct": 1.65,
            }
            for sym in symbols
        ],
    }
    intel = {
        "model_version": "5.0.0-pilot",
        "generated_at": gen_at,
        "stocks": {
            sym: {
                "symbol": sym,
                "observation_date": obs_date,
                "news_sentiment_score": 0.35,
            }
            for sym in symbols
        },
    }
    regime = {
        "generated_at": gen_at,
        "observation_date": obs_date,
        "regime": "Bull",
        "confidence": 0.85,
    }

    pred_p = tmp_path / "latest_predictions.json"
    intel_p = tmp_path / "phase5_intelligence.json"
    reg_p = tmp_path / "regime.json"
    pred_p.write_text(json.dumps(preds), encoding="utf-8")
    intel_p.write_text(json.dumps(intel), encoding="utf-8")
    reg_p.write_text(json.dumps(regime), encoding="utf-8")
    return pred_p, intel_p, reg_p


# ---------------------------------------------------------------------------
# 1. PY-04: MandatoryExecutionGate Non-Bypass & Fail-Closed Human Authority
# ---------------------------------------------------------------------------

def test_py04_ast_no_direct_submit_order_in_cycle006b_and_cycle007() -> None:
    """AST static check: neither Cycle006B nor Cycle007 calls portfolio.submit_order directly."""
    targets = [
        REPO_ROOT / "trader/paper/cycle006b.py",
        REPO_ROOT / "trader/timing/cycle007_pipeline.py",
    ]
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        direct_submit_calls: list[int] = []
        gate_calls: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "submit_order":
                    direct_submit_calls.append(node.lineno)
                elif node.func.attr == "execute_with_gate_protection":
                    gate_calls.append(node.lineno)
        assert direct_submit_calls == [], (
            f"Direct submit_order bypass found in {path.name} at lines {direct_submit_calls}"
        )
        assert len(gate_calls) >= 3, (
            f"Expected >= 3 execute_with_gate_protection call sites in {path.name}, found {len(gate_calls)}"
        )


def test_py04_cycle006b_fail_closed_missing_human_permission_and_dynamic_gate_spy(
    tmp_path: Path,
) -> None:
    """Cycle006B fails closed without Human permission and routes 100% of orders through the gate."""
    session_date = date(2026, 11, 20)
    symbols = ("RELIANCE",)
    cache_dir = tmp_path / "intraday_006b"
    _write_1h_fixture(cache_dir, symbols, session_date, num_sessions=65)
    pred_p, intel_p, reg_p = _write_stage1_inputs(tmp_path, symbols, session_date)

    cfg = Cycle006BConfig(
        session_date=session_date,
        symbols=symbols,
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )

    # 1. Fail-closed when no Human permission is supplied
    rep_no_auth = Cycle006BPaperTradingRunner(cfg).run()
    assert rep_no_auth["run_status"] == "COMPLETED"
    assert rep_no_auth["order_count"] == 0
    assert rep_no_auth["filled_order_count"] == 0
    assert rep_no_auth["open_positions_count"] == 0
    assert any("HUMAN_INSTRUCTION_MISSING" in r for r in rep_no_auth["rejection_reasons"])

    # 2. Dynamic inspect.stack() spy when explicit ALLOW_LONG is supplied
    runner = Cycle006BPaperTradingRunner(cfg, human_permission=PermissionType.ALLOW_LONG)
    orig_submit = runner.portfolio.submit_order
    observed_calls: list[bool] = []

    def _spy_submit(*args: Any, **kwargs: Any) -> Any:
        stack_fns = [frame.function for frame in inspect.stack()]
        observed_calls.append("execute_with_gate_protection" in stack_fns)
        return orig_submit(*args, **kwargs)

    runner.portfolio.submit_order = _spy_submit  # type: ignore[method-assign]
    rep_auth = runner.run()

    assert rep_auth["run_status"] == "COMPLETED"
    assert rep_auth["filled_order_count"] >= 2
    assert len(observed_calls) >= 2
    assert all(observed_calls), "At least one Cycle006B order bypassed MandatoryExecutionGate"


def test_py04_cycle007_fail_closed_missing_human_permission_and_dynamic_gate_spy(
    tmp_path: Path,
) -> None:
    """Cycle007 fails closed without Human permission and routes 100% of orders through the gate."""
    session_date = date(2026, 11, 20)
    symbols = ("RELIANCE",)
    cache_dir = tmp_path / "intraday_007"
    _write_1h_fixture(cache_dir, symbols, session_date, num_sessions=65)

    # Craft a clean bullish engulfing breakout on the final two pre-cutoff bars
    # so Stage-2 emits ENTER_NOW and execution reaches MandatoryExecutionGate.
    pq_path = cache_dir / "RELIANCE_1h.parquet"
    df = pd.read_parquet(pq_path).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST_TZ)
    pre_mask = df["timestamp"].dt.date < session_date
    day_mask = df["timestamp"].dt.date == session_date

    tail_24 = df.index[pre_mask].tolist()[-24:]
    px = 1400.0
    for idx_pos, row_idx in enumerate(tail_24[:-2]):
        o = px
        c = px + 1.4
        df.loc[row_idx, ["open", "high", "low", "close", "volume"]] = [
            o, c + 0.5, o - 0.4, c, 120_000.0 + (idx_pos % 3) * 5_000.0
        ]
        px = c
    i_prev, i_curr = tail_24[-2], tail_24[-1]
    prev_o, prev_c = px + 0.8, px - 1.2
    df.loc[i_prev, ["open", "high", "low", "close", "volume"]] = [
        prev_o, prev_o + 0.4, prev_c - 0.4, prev_c, 115_000.0
    ]
    curr_o, curr_c = prev_c - 0.2, prev_o + 4.0
    df.loc[i_curr, ["open", "high", "low", "close", "volume"]] = [
        curr_o, curr_c + 0.3, curr_o - 0.2, curr_c, 220_000.0
    ]
    s_px = curr_c + 0.5
    for step, row_idx in enumerate(df.index[day_mask].tolist()):
        o, c = s_px, s_px + 2.0
        df.loc[row_idx, ["open", "high", "low", "close", "volume"]] = [
            o, c + 0.8, o - 0.5, c, 150_000.0 + step * 5_000.0
        ]
        s_px = c
    df.to_parquet(pq_path, index=False)

    pred_p, intel_p, reg_p = _write_stage1_inputs(tmp_path, symbols, session_date)

    cfg = Cycle007Config(
        session_date=session_date,
        symbols=symbols,
        cache_dir=cache_dir,
        predictions_path=pred_p,
        intelligence_path=intel_p,
        regime_path=reg_p,
    )

    # 1. Fail-closed when no Human permission is supplied
    rep_no_auth = Cycle007Pipeline(cfg).run()
    assert rep_no_auth["run_status"] == "COMPLETED"
    assert rep_no_auth["stage2_timing_assessments"]["RELIANCE"]["decision"] == "ENTER_NOW"
    assert rep_no_auth["order_count"] == 0
    assert rep_no_auth["filled_order_count"] == 0
    assert rep_no_auth["open_positions_count"] == 0
    assert any("HUMAN_INSTRUCTION_MISSING" in r for r in rep_no_auth["rejection_reasons"])

    # 2. Dynamic inspect.stack() spy when explicit ALLOW_LONG is supplied
    pipeline = Cycle007Pipeline(cfg, human_permission=PermissionType.ALLOW_LONG)
    orig_submit = pipeline.portfolio.submit_order
    observed_calls: list[bool] = []

    def _spy_submit(*args: Any, **kwargs: Any) -> Any:
        stack_fns = [frame.function for frame in inspect.stack()]
        observed_calls.append("execute_with_gate_protection" in stack_fns)
        return orig_submit(*args, **kwargs)

    pipeline.portfolio.submit_order = _spy_submit  # type: ignore[method-assign]
    rep_auth = pipeline.run()

    assert rep_auth["run_status"] == "COMPLETED"
    assert rep_auth["stage2_timing_assessments"]["RELIANCE"]["decision"] == "ENTER_NOW"
    assert rep_auth["filled_order_count"] >= 2
    assert len(observed_calls) >= 2
    assert all(observed_calls), "At least one Cycle007 order bypassed MandatoryExecutionGate"


# ---------------------------------------------------------------------------
# 2. R5-D: Unclipped >100% Adverse Short-Gap Loss & Margin-Call Deficit Halt
# ---------------------------------------------------------------------------

def test_r5d_short_gap_over_100pct_unclipped_negative_cash_margin_call_and_risk_halt() -> None:
    """Extreme >100% adverse short-gap cover produces exact negative cash, margin call, and HALTED state."""
    portfolio = IntradayPaperPortfolio(initial_capital=100_000.0)
    risk_engine = DeterministicRiskEngine(
        max_risk_per_trade_pct=0.02,
        max_position_weight_pct=0.80,
        max_portfolio_exposure_pct=0.95,
        max_daily_loss_pct=0.05,
    )

    t0 = datetime(2026, 11, 20, 9, 30, tzinfo=IST_TZ)
    t1 = datetime(2026, 11, 20, 9, 35, tzinfo=IST_TZ)
    t2 = datetime(2026, 11, 20, 10, 30, tzinfo=IST_TZ)
    t3 = datetime(2026, 11, 20, 10, 35, tzinfo=IST_TZ)

    # Open a small long position in TCS so we can verify RISK_EXIT stays permitted after halt
    long_order = portfolio.submit_order(
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=5,
        price=1000.0,
        timestamp=t0,
        position_effect="OPEN_LONG",
    )
    assert long_order.status == OrderStatus.FILLED

    # Open short position in RELIANCE: 50 shares @ Rs. 1,000 (Rs. 50,000 margin locked)
    short_entry = portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.SELL,
        quantity=50,
        price=1000.0,
        timestamp=t1,
        position_effect="OPEN_SHORT",
    )
    assert short_entry.status == OrderStatus.FILLED
    cash_before_cover = portfolio.cash
    assert cash_before_cover > 0.0

    # Extreme >100% adverse gap: cover @ Rs. 3,500 -> gross_pnl = (1000 - 3500) * 50 = -125,000
    cover_order = portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=50,
        price=3500.0,
        timestamp=t2,
        position_effect="COVER_SHORT",
    )
    assert cover_order.status == OrderStatus.FILLED

    # Verify exact unclipped negative cash & realized PnL
    expected_gross_pnl = (1000.0 - 3500.0) * 50
    expected_cash = cash_before_cover + (1000.0 * 50) + expected_gross_pnl - cover_order.cost_breakdown.total_friction_inr
    assert portfolio.cash < 0.0
    assert portfolio.cash == pytest.approx(expected_cash, abs=1e-6)

    closed = portfolio.closed_trades[-1]
    assert closed["symbol"] == "RELIANCE"
    assert closed["side"] == "SHORT"
    assert closed["gross_pnl_inr"] == pytest.approx(expected_gross_pnl, abs=0.01)
    assert closed["net_pnl_inr"] < expected_gross_pnl  # includes friction, never clipped

    # Margin call state, events, and SHA-256 ledger integrity
    assert portfolio.margin_call_triggered is True
    assert len(portfolio.margin_call_events) == 1
    mc_evt = portfolio.margin_call_events[0]
    assert mc_evt["symbol"] == "RELIANCE"
    assert mc_evt["cash_after_inr"] == pytest.approx(portfolio.cash, abs=1e-6)
    assert mc_evt["cash_deficit_inr"] == pytest.approx(abs(portfolio.cash), abs=1e-6)

    mc_ledger_events = [e for e in portfolio.event_ledger if e.event_type == "MARGIN_CALL_DEFICIT"]
    assert len(mc_ledger_events) == 1

    ledger_ok, ledger_msg = portfolio.verify_ledger_integrity()
    assert ledger_ok is True, f"Ledger integrity failed: {ledger_msg}"

    # RiskEngine transitions to HALTED, blocks new entries with MARGIN_CALL_DEFICIT, permits RISK_EXIT
    breached = risk_engine.check_daily_loss_limit(portfolio)
    assert breached is True
    assert risk_engine.state == RiskState.HALTED

    new_entry_res = risk_engine.evaluate_order(
        symbol="INFY",
        side="BUY",
        requested_quantity=5,
        entry_price=1500.0,
        stop_loss_price=1450.0,
        portfolio=portfolio,
        current_time=t3,
    )
    assert new_entry_res.approved is False
    assert new_entry_res.approved_quantity == 0
    assert any("MARGIN_CALL_DEFICIT" in r for r in new_entry_res.rejection_reasons)

    risk_exit_res = risk_engine.evaluate_order(
        symbol="TCS",
        side="SELL",
        requested_quantity=5,
        entry_price=1005.0,
        stop_loss_price=None,
        portfolio=portfolio,
        current_time=t3,
        action_type="RISK_EXIT",
    )
    assert risk_exit_res.approved is True
    assert risk_exit_res.approved_quantity == 5


# ---------------------------------------------------------------------------
# 3. D6-09: Fail-Closed Timestamp Parsing (1m & 1h)
# ---------------------------------------------------------------------------

def test_d6_09_normalize_1m_rejects_naive_and_nat_and_converts_tz_aware() -> None:
    """normalize_1m_dataframe rejects NaT and naive timestamps while converting UTC & IST cleanly."""
    # Naive datetime64 series -> ValueError
    df_naive_dt = pd.DataFrame(
        {
            "timestamp_ist": pd.to_datetime(["2026-11-20 09:15:00", "2026-11-20 09:16:00"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    )
    with pytest.raises(ValueError, match="AMBIGUOUS_NAIVE_TIMESTAMP"):
        normalize_1m_dataframe(df_naive_dt, "RELIANCE")

    # Naive ISO string series -> ValueError
    df_naive_str = df_naive_dt.copy()
    df_naive_str["timestamp_ist"] = ["2026-11-20T09:15:00", "2026-11-20T09:16:00"]
    with pytest.raises(ValueError, match="AMBIGUOUS_NAIVE_TIMESTAMP"):
        normalize_1m_dataframe(df_naive_str, "RELIANCE")

    # Missing / NaT timestamp -> ValueError
    df_nat = df_naive_dt.copy()
    df_nat["timestamp_ist"] = [
        datetime(2026, 11, 20, 9, 15, tzinfo=IST_TZ),
        pd.NaT,
    ]
    with pytest.raises(ValueError, match="Missing/NaT"):
        normalize_1m_dataframe(df_nat, "RELIANCE")

    # Tz-aware UTC -> converted cleanly to IST (04:00 UTC == 09:30 IST)
    df_utc = pd.DataFrame(
        {
            "timestamp_ist": [
                datetime(2026, 11, 20, 4, 0, tzinfo=timezone.utc),
                datetime(2026, 11, 20, 4, 1, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    )
    norm_utc = normalize_1m_dataframe(df_utc, "RELIANCE")
    assert len(norm_utc) == 2
    assert str(norm_utc["timestamp_ist"].dt.tz) == "Asia/Kolkata"
    assert norm_utc["timestamp_ist"].iloc[0].hour == 9
    assert norm_utc["timestamp_ist"].iloc[0].minute == 30

    # Tz-aware IST -> preserved cleanly in IST
    df_ist = df_utc.copy()
    df_ist["timestamp_ist"] = [
        datetime(2026, 11, 20, 9, 30, tzinfo=IST_TZ),
        datetime(2026, 11, 20, 9, 31, tzinfo=IST_TZ),
    ]
    norm_ist = normalize_1m_dataframe(df_ist, "RELIANCE")
    assert len(norm_ist) == 2
    assert str(norm_ist["timestamp_ist"].dt.tz) == "Asia/Kolkata"
    assert norm_ist["timestamp_ist"].iloc[0] == norm_utc["timestamp_ist"].iloc[0]


def test_d6_09_loader_1h_rejects_naive_and_nat_and_converts_tz_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """loader_1h rejects naive and NaT timestamps in parquet files and converts UTC/IST cleanly."""
    monkeypatch.setattr(loader_1h, "CACHE_DIR", tmp_path)
    pq_path = tmp_path / "RELIANCE_1h.parquet"

    # 1. Naive timestamps -> ValueError
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-11-20 09:15:00", "2026-11-20 10:15:00"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    ).to_parquet(pq_path, index=False)

    with pytest.raises(ValueError, match="AMBIGUOUS_NAIVE_TIMESTAMP"):
        loader_1h.get_available_1h_dates(symbols=("RELIANCE",), min_bars_per_stock=1)
    with pytest.raises(ValueError, match="AMBIGUOUS_NAIVE_TIMESTAMP"):
        loader_1h.load_1h_session_with_warmup("RELIANCE", date(2026, 11, 20))

    # 2. NaT timestamps -> ValueError
    pd.DataFrame(
        {
            "timestamp": [datetime(2026, 11, 20, 9, 15, tzinfo=IST_TZ), pd.NaT],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    ).to_parquet(pq_path, index=False)

    with pytest.raises(ValueError, match="Missing/NaT"):
        loader_1h.get_available_1h_dates(symbols=("RELIANCE",), min_bars_per_stock=1)
    with pytest.raises(ValueError, match="Missing/NaT"):
        loader_1h.load_1h_session_with_warmup("RELIANCE", date(2026, 11, 20))

    # 3. Tz-aware UTC timestamps -> converted cleanly to IST
    pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 11, 20, 3, 45, tzinfo=timezone.utc),  # 09:15 IST
                datetime(2026, 11, 20, 4, 45, tzinfo=timezone.utc),  # 10:15 IST
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    ).to_parquet(pq_path, index=False)

    avail, _excluded = loader_1h.get_available_1h_dates(symbols=("RELIANCE",), min_bars_per_stock=2)
    assert avail == [date(2026, 11, 20)]
    warmup, session, _prov = loader_1h.load_1h_session_with_warmup("RELIANCE", date(2026, 11, 20))
    assert warmup == []
    assert len(session) == 2
    assert session[0].timestamp.tzinfo is not None
    assert session[0].timestamp.astimezone(IST_TZ).hour == 9
    assert session[0].timestamp.astimezone(IST_TZ).minute == 15


# ---------------------------------------------------------------------------
# 4. D6-11: No Silent Synthetic Substitution in HISTORICAL_REAL
# ---------------------------------------------------------------------------

def test_d6_11_historical_real_never_substitutes_synthetic_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """build_trader_dataset(HISTORICAL_REAL) never calls generate_synthetic_5m_session and fails closed."""
    monkeypatch.setattr(generate_trader_data, "OUTPUT_DIR", tmp_path)
    # Pre-generate fixture candles strictly for the watchlist scanner step before monkeypatching
    watchlist_fixtures: dict[str, tuple[Any, Any]] = {}
    for sym, bp in [("RELIANCE", 2520.0), ("TCS", 3540.0), ("HDFCBANK", 1625.0), ("INFY", 1820.0), ("ITC", 435.0)]:
        watchlist_fixtures[sym] = generate_trader_data.generate_synthetic_5m_session(
            symbol=sym, base_price=bp, daily_trend_pct=0.8, return_provenance=True
        )

    def _forbid_synthetic(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generate_synthetic_5m_session must NEVER be called in HISTORICAL_REAL mode")

    def _mock_load_5m(*args: Any, **kwargs: Any) -> Any:
        # Step 2 (watchlist scanner) passes exchange="NSE"; Step 5 (backtest comparison) omits exchange
        if "exchange" in kwargs:
            sym = kwargs["symbol"]
            return watchlist_fixtures[sym]
        raise DataUnavailableError("Real 5m backtest archive unavailable for RELIANCE")

    monkeypatch.setattr(generate_trader_data, "generate_synthetic_5m_session", _forbid_synthetic)
    monkeypatch.setattr(generate_trader_data, "load_historical_5m_candles", _mock_load_5m)

    payload_real = generate_trader_data.build_trader_dataset(DataSourceMode.HISTORICAL_REAL)
    perf_real = payload_real["strategy_performance"]
    assert len(perf_real) == 4
    for row in perf_real:
        assert row["status"] == "UNAVAILABLE"
        assert row["synthetic_substitution"] is False
        assert row["source_mode"] == DataSourceMode.HISTORICAL_REAL.value
        assert row["total_trades"] == 0
        assert row["win_rate_pct"] is None
        assert row["profit_factor"] is None


def test_d6_11_synthetic_test_mode_explicitly_tags_source_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """build_trader_dataset(SYNTHETIC_TEST) runs backtests and tags source_mode='SYNTHETIC_TEST'."""
    monkeypatch.setattr(generate_trader_data, "OUTPUT_DIR", tmp_path)
    payload_syn = generate_trader_data.build_trader_dataset(DataSourceMode.SYNTHETIC_TEST)
    perf_syn = payload_syn["strategy_performance"]
    assert len(perf_syn) == 4
    for row in perf_syn:
        assert row["status"] == "AVAILABLE"
        assert row["source_mode"] == DataSourceMode.SYNTHETIC_TEST.value
        assert row["synthetic_substitution"] is True


# ---------------------------------------------------------------------------
# 5. D6-03: Provider Authority Cap Default & Explicit Opt-Out Auditability
# ---------------------------------------------------------------------------

def test_d6_03_validate_1h_provenance_defaults_provider_cap_and_explicit_opt_out(
    tmp_path: Path,
) -> None:
    """validate_1h_provenance defaults to enforce_provider_authority_cap=True and records opt-out."""
    _write_1h_fixture(tmp_path, ("RELIANCE",), date(2026, 11, 20), num_sessions=60)

    # Default call: Yahoo 1h capped at PARTIALLY_VERIFIED, qualification_allowed=False
    ok_def, rep_def = validate_1h_provenance(
        ["RELIANCE"],
        cache_dir=tmp_path,
        min_sessions_required=55,
    )
    assert ok_def is False
    assert rep_def["final_provenance_status"] == "PARTIALLY_VERIFIED"
    assert rep_def["verified"] is False
    assert rep_def["qualification_allowed"] is False
    assert rep_def["enforce_provider_authority_cap"] is True
    assert rep_def["provider_authority_cap_opted_out"] is False

    # Explicit opt-out: allows VERIFIED but explicitly flags provider_authority_cap_opted_out=True
    ok_opt, rep_opt = validate_1h_provenance(
        ["RELIANCE"],
        cache_dir=tmp_path,
        min_sessions_required=55,
        enforce_provider_authority_cap=False,
    )
    assert ok_opt is True
    assert rep_opt["final_provenance_status"] == "VERIFIED"
    assert rep_opt["verified"] is True
    assert rep_opt["qualification_allowed"] is True
    assert rep_opt["enforce_provider_authority_cap"] is False
    assert rep_opt["provider_authority_cap_opted_out"] is True


# ---------------------------------------------------------------------------
# 6. DEF-7.9: .gitignore Hygiene for data/intraday_1m/*_report.json
# ---------------------------------------------------------------------------

def test_def_7_9_gitignore_ignores_1m_reports_but_not_provenance_1m() -> None:
    """Verify .gitignore ignores data/intraday_1m/*_report.json without ignoring provenance_1m.json."""
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/intraday_1m/*_report.json" in gitignore_text.splitlines()

    ignored = subprocess.run(
        ["git", "check-ignore", "data/intraday_1m/sample_report.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0, f"Expected sample_report.json to be ignored, stderr={ignored.stderr!r}"

    not_ignored = subprocess.run(
        ["git", "check-ignore", "data/intraday_1m/provenance_1m.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert not_ignored.returncode == 1, (
        f"Expected provenance_1m.json NOT to be ignored, stdout={not_ignored.stdout!r}"
    )
