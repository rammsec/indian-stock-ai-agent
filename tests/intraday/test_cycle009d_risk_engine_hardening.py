"""CYCLE-009D: Deterministic Risk Engine Hardening Tests."""

from __future__ import annotations

import ast
import threading
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

from trader.config import IST_TZ
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine, RiskState

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_portfolio_with_position(symbol: str = "RELIANCE", qty: int = 20, entry: float = 2500.0) -> IntradayPaperPortfolio:
    p = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    ts = datetime(2026, 11, 20, 9, 30, tzinfo=IST_TZ)
    p.submit_order(symbol=symbol, side="BUY", quantity=qty, price=entry, timestamp=ts)
    return p


def test_cycle009d_state_machine_and_1505_reducing_allows_sell_blocks_buy() -> None:
    p = _make_portfolio_with_position("RELIANCE", qty=20, entry=2500.0)
    engine = DeterministicRiskEngine()
    assert engine.state == RiskState.ACTIVE

    ts_1505 = datetime(2026, 11, 20, 15, 5, tzinfo=IST_TZ)
    buy_res = engine.evaluate_order("TCS", "BUY", 10, 3500.0, 3450.0, p, ts_1505)
    assert not buy_res.approved
    assert engine.state == RiskState.REDUCING

    # SELL exit at 15:05 and 15:15 must succeed up to held qty (20)
    sell_res = engine.evaluate_order("RELIANCE", "SELL", 50, 2510.0, None, p, ts_1505)
    assert sell_res.approved
    assert sell_res.status == "SIZED_DOWN"
    assert sell_res.approved_quantity == 20

    ts_1516 = datetime(2026, 11, 20, 15, 16, tzinfo=IST_TZ)
    sell_1516 = engine.evaluate_order("RELIANCE", "SELL", 20, 2510.0, None, p, ts_1516)
    assert sell_1516.approved
    assert sell_1516.approved_quantity == 20

    # Reset on new session date restores ACTIVE
    engine.reset_daily_session(date(2026, 11, 23))
    assert engine.state == RiskState.ACTIVE
    assert engine.is_daily_halted is False


def test_cycle009d_halted_allows_force_reduce_only_and_preserves_is_daily_halted_setter() -> None:
    p = _make_portfolio_with_position("TCS", qty=15, entry=3000.0)
    engine = DeterministicRiskEngine()
    engine.is_daily_halted = True
    assert engine.state == RiskState.HALTED
    assert engine.is_daily_halted is True

    ts = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)
    assert not engine.evaluate_order("TCS", "SELL", 10, 3000.0, None, p, ts).approved
    forced = engine.evaluate_order("TCS", "SELL", 10, 3000.0, None, p, ts, force_reduce_only=True)
    assert forced.approved
    assert forced.approved_quantity == 10

    engine.is_daily_halted = False
    assert engine.state == RiskState.ACTIVE


def test_cycle009d_thread_safe_cash_reservation_ledger_prevents_double_spend() -> None:
    p = IntradayPaperPortfolio(initial_capital=100_000.0)
    engine = DeterministicRiskEngine(
        max_risk_per_trade_pct=0.5,
        max_position_weight_pct=1.0,
        max_portfolio_exposure_pct=1.0,
    )
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    engine.cash_ledger.reserve("ord_1", "RELIANCE", 90_000.0)

    # Concurrent reservations & release checks
    def _worker(idx: int) -> None:
        engine.cash_ledger.reserve(f"w_{idx}", "INFY", 1_000.0)
        engine.cash_ledger.release(f"w_{idx}")

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Only 10,000 INR remains unreserved -> at price 2,000, max 5 shares
    res = engine.evaluate_order("INFY", "BUY", 20, 2000.0, 1900.0, p, ts)
    assert res.approved
    assert res.approved_quantity == 5
    engine.cash_ledger.release("ord_1")
    assert engine.cash_ledger.total_reserved_inr == 0.0


def test_cycle009d_intraday_mtm_peak_to_trough_drawdown_halts() -> None:
    p = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    engine = DeterministicRiskEngine(max_daily_loss_pct=0.10, max_intraday_mtm_drawdown_pct=0.03)
    ts1 = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Simulate peak equity at 1,100,000 via position MTM
    p.submit_order(symbol="RELIANCE", side="BUY", quantity=100, price=2000.0, timestamp=ts1)
    p.update_market_price("RELIANCE", 3000.0, ts1)  # total_equity = 1,100,000
    assert not engine.check_daily_loss_limit(p)

    # Drop from peak 1,100,000 to 1,050,000 (~4.5% MTM DD >= 3.0% threshold, even though day PnL is +5%)
    p.update_market_price("RELIANCE", 2500.0, ts1)
    assert engine.check_daily_loss_limit(p) is True
    assert engine.state == RiskState.HALTED


def test_cycle009d_order_rate_throttler_and_naive_datetime_rejection() -> None:
    p = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    engine = DeterministicRiskEngine(max_orders_per_minute=3)

    # Naive datetime must fail closed
    naive_res = engine.evaluate_order("TCS", "BUY", 5, 3000.0, 2950.0, p, datetime(2026, 11, 20, 10, 0))
    assert not naive_res.approved
    assert "Naive datetime" in naive_res.rejection_reasons[0]

    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    for i in range(3):
        r = engine.evaluate_order("TCS", "BUY", 1, 3000.0, 2950.0, p, ts + timedelta(seconds=i * 5))
        assert r.approved
    throttled = engine.evaluate_order("TCS", "BUY", 1, 3000.0, 2950.0, p, ts + timedelta(seconds=20))
    assert not throttled.approved
    assert "throttle" in throttled.rejection_reasons[0].lower()


def test_cycle009d_sha256_audit_chain_catches_4_tamper_vectors() -> None:
    p = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    engine = DeterministicRiskEngine()
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    engine.evaluate_order("TCS", "BUY", 5, 3000.0, 2950.0, p, ts)
    engine.evaluate_order("INFY", "BUY", 5, 1500.0, 1450.0, p, ts + timedelta(minutes=1))
    assert engine.verify_audit_chain() is True

    log = list(engine.audit_log)
    # 1. Tamper quantity
    t1 = [replace(log[0], approved_quantity=999), log[1]]
    assert engine.verify_audit_chain(t1) is False
    # 2. Tamper prev_hash linkage
    t2 = [log[0], replace(log[1], prev_hash="f" * 64)]
    assert engine.verify_audit_chain(t2) is False
    # 3. Tamper event_hash
    t3 = [replace(log[0], event_hash="a" * 64), log[1]]
    assert engine.verify_audit_chain(t3) is False
    # 4. Drop middle/first event (seq mismatch)
    t4 = [log[1]]
    assert engine.verify_audit_chain(t4) is False


def test_cycle009d_risk_engine_zero_imports_from_intelligence_agents_timing() -> None:
    src = (REPO_ROOT / "trader" / "risk" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = ("intelligence", "agents", "timing")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(f in alias.name.split(".") for f in forbidden), alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not any(f in node.module.split(".") for f in forbidden), node.module
