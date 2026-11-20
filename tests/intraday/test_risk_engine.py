"""Tests for deterministic risk engine, position sizing, and circuit breakers."""

from datetime import datetime, time
import pytest

from trader.config import IST_TZ
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide
from trader.risk.engine import DeterministicRiskEngine


def test_fractional_risk_position_sizing():
    """Verify that order quantity is properly calculated based on stop-loss distance."""
    risk_engine = DeterministicRiskEngine(max_risk_per_trade_pct=0.01)  # 1% risk
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)    # Equity: 10 Lakhs -> Max risk = ₹10,000

    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)
    # Entry 1000, Stop 980 -> Risk per share = ₹20
    # Expected max shares by risk = 10,000 / 20 = 500 shares
    res = risk_engine.evaluate_order(
        symbol="RELIANCE",
        side="BUY",
        requested_quantity=1000,  # Requesting more than allowed
        entry_price=1000.0,
        stop_loss_price=980.0,
        portfolio=portfolio,
        current_time=ts,
    )

    assert res.approved is True
    assert res.status == "SIZED_DOWN"
    # Position cap is 15% of 10 Lakhs = ₹1,50,000 / 1000 = 150 shares
    # Thus position ceiling (150 shares) caps the risk shares (500 shares)
    assert res.approved_quantity == 150
    assert res.estimated_trade_risk_inr == 150 * 20.0  # ₹3,000 <= ₹10,000


def test_missing_stop_loss_rejection():
    """Assert buy orders with missing or invalid stop-loss are strictly rejected."""
    risk_engine = DeterministicRiskEngine()
    portfolio = IntradayPaperPortfolio()
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Missing stop loss
    res = risk_engine.evaluate_order(
        symbol="TCS",
        side="BUY",
        requested_quantity=10,
        entry_price=3500.0,
        stop_loss_price=None,
        portfolio=portfolio,
        current_time=ts,
    )
    assert res.approved is False
    assert res.status == "REJECTED"
    assert "Invalid stop-loss" in res.rejection_reasons[0]

    # Stop loss >= entry price
    res2 = risk_engine.evaluate_order(
        symbol="TCS",
        side="BUY",
        requested_quantity=10,
        entry_price=3500.0,
        stop_loss_price=3550.0,
        portfolio=portfolio,
        current_time=ts,
    )
    assert res2.approved is False
    assert res2.status == "REJECTED"


def test_daily_drawdown_circuit_breaker():
    """Assert trading halts when daily loss reaches 2.5%."""
    risk_engine = DeterministicRiskEngine(max_daily_loss_pct=0.025)
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)

    # Simulate losing trades reducing equity by ₹30,000 (3.0% loss > 2.5% threshold)
    portfolio.cash = 970_000.0
    ts = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)

    res = risk_engine.evaluate_order(
        symbol="INFY",
        side="BUY",
        requested_quantity=20,
        entry_price=1800.0,
        stop_loss_price=1750.0,
        portfolio=portfolio,
        current_time=ts,
    )
    assert res.approved is False
    assert res.status == "REJECTED"
    assert "Daily loss limit breached" in res.rejection_reasons[0]
    assert risk_engine.is_daily_halted is True


def test_time_of_day_entry_cutoff():
    """Assert entry orders past 14:45 IST are rejected."""
    risk_engine = DeterministicRiskEngine()
    portfolio = IntradayPaperPortfolio()

    # 14:46 IST (past 14:45 cutoff)
    ts_late = datetime(2026, 11, 20, 14, 46, tzinfo=IST_TZ)

    res = risk_engine.evaluate_order(
        symbol="HDFCBANK",
        side="BUY",
        requested_quantity=10,
        entry_price=1600.0,
        stop_loss_price=1580.0,
        portfolio=portfolio,
        current_time=ts_late,
    )
    assert res.approved is False
    assert res.status == "REJECTED"
    assert "Past 14:45 IST intraday entry cutoff" in res.rejection_reasons[0]
