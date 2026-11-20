"""Tests for Indian statutory cost calculation and double-entry paper portfolio ledger."""

from datetime import datetime
import pytest

from trader.config import IST_TZ
from trader.portfolio.cost_calculator import calculate_intraday_costs
from trader.portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus


def test_statutory_cost_calculation():
    """Verify exact Indian statutory friction breakdown for MIS intraday trades."""
    # Test Buy Order of ₹1,00,000 turnover
    buy_cost = calculate_intraday_costs(price=1000.0, quantity=100, side="BUY", exchange="NSE")
    assert buy_cost.turnover_inr == 100_000.0
    # Brokerage: 0.03% = ₹30, capped at ₹20
    assert buy_cost.brokerage_inr == 20.0
    # STT: ₹0 on Buy intraday
    assert buy_cost.stt_inr == 0.0
    # Stamp duty: 0.003% on ₹1,00,000 = ₹3.0
    assert buy_cost.stamp_duty_inr == 3.0
    # NSE charge: 0.00297% = ₹2.97
    assert buy_cost.exchange_charges_inr == 2.97
    # GST: 18% of (20 + 2.97 + 0.1) = ~₹4.15
    assert abs(buy_cost.gst_inr - 4.1526) < 0.01

    # Test Sell Order of ₹1,00,000 turnover
    sell_cost = calculate_intraday_costs(price=1000.0, quantity=100, side="SELL", exchange="NSE")
    # STT: 0.025% on Sell = ₹25.0
    assert sell_cost.stt_inr == 25.0
    # Stamp duty: ₹0 on Sell
    assert sell_cost.stamp_duty_inr == 0.0


def test_ledger_double_entry_and_hash_integrity():
    """Assert conservation of capital and unbroken SHA-256 event chaining."""
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # 1. Execute Buy
    buy_ord = portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=50,
        price=2500.0,
        exchange="NSE",
        stop_loss=2450.0,
        target=2600.0,
        timestamp=ts,
    )
    assert buy_ord.status == OrderStatus.FILLED
    assert "RELIANCE" in portfolio.positions
    assert portfolio.positions["RELIANCE"].quantity == 50

    # Verify accounting invariant after buy
    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg
    assert abs(portfolio.cash + portfolio.invested_capital - portfolio.total_equity) < 1e-4

    # 2. Update price to 2550 (unrealized profit)
    ts_update = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)
    portfolio.update_market_price("RELIANCE", 2550.0, timestamp=ts_update)
    assert portfolio.unrealized_pnl == 50 * (2550.0 - 2500.0)

    # 3. Target hit at 2600 (automatic exit)
    ts_target = datetime(2026, 11, 20, 12, 0, tzinfo=IST_TZ)
    executed = portfolio.update_market_price("RELIANCE", 2605.0, timestamp=ts_target)
    assert len(executed) == 1
    assert "RELIANCE" not in portfolio.positions  # Fully exited
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0]["net_pnl_inr"] > 0

    # Verify unbroken SHA-256 hash chain and capital conservation
    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg
    assert portfolio.cash == portfolio.total_equity  # All in cash after exit


def test_mandatory_square_off():
    """Assert all open positions are force-exited upon square-off call."""
    portfolio = IntradayPaperPortfolio(initial_capital=500_000.0)
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    portfolio.submit_order("TCS", OrderSide.BUY, quantity=10, price=3500.0, timestamp=ts)
    portfolio.submit_order("INFY", OrderSide.BUY, quantity=20, price=1800.0, timestamp=ts)
    assert len(portfolio.positions) == 2

    # Force square-off at 15:15 IST
    ts_sq = datetime(2026, 11, 20, 15, 15, tzinfo=IST_TZ)
    orders = portfolio.square_off_all_positions(
        current_prices={"TCS": 3520.0, "INFY": 1790.0},
        timestamp=ts_sq,
    )
    assert len(orders) == 2
    assert len(portfolio.positions) == 0
    assert len(portfolio.closed_trades) == 2

    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg
