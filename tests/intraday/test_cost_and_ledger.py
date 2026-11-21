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


def test_partial_fill_accounting():
    """Verify partial fill handling, order status, cash deductions, and position state."""
    portfolio = IntradayPaperPortfolio(initial_capital=500_000.0)
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Request 100 shares, but partial fill is 40 shares
    ord_buy = portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=100,
        price=2500.0,
        timestamp=ts,
        partial_fill_quantity=40,
    )

    assert ord_buy.status == OrderStatus.PARTIALLY_FILLED
    assert ord_buy.filled_quantity == 40
    assert ord_buy.remaining_quantity == 60
    assert portfolio.positions["RELIANCE"].quantity == 40

    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg

    # Partial exit of 25 shares out of 40
    ts_exit = datetime(2026, 11, 20, 11, 0, tzinfo=IST_TZ)
    ord_sell = portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.SELL,
        quantity=40,
        price=2550.0,
        timestamp=ts_exit,
        partial_fill_quantity=25,
    )

    assert ord_sell.status == OrderStatus.PARTIALLY_FILLED
    assert ord_sell.filled_quantity == 25
    assert ord_sell.remaining_quantity == 15
    assert portfolio.positions["RELIANCE"].quantity == 15  # 40 - 25
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0]["quantity"] == 25

    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg


def test_multi_exchange_concurrent_positions():
    """Verify that identical stock symbols held on different exchanges (NSE and BSE) coexist without collision."""
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Buy RELIANCE on NSE
    portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=50,
        price=2500.0,
        exchange="NSE",
        timestamp=ts,
    )

    # Buy RELIANCE on BSE
    portfolio.submit_order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=30,
        price=2501.50,
        exchange="BSE",
        timestamp=ts,
    )

    assert len(portfolio.positions) == 2
    assert "NSE:RELIANCE" in portfolio.positions
    assert "BSE:RELIANCE" in portfolio.positions

    pos_nse = portfolio.positions["NSE:RELIANCE"]
    pos_bse = portfolio.positions["BSE:RELIANCE"]
    assert pos_nse.quantity == 50
    assert pos_nse.entry_price == 2500.0
    assert pos_bse.quantity == 30
    assert pos_bse.entry_price == 2501.50

    # Square-off closes both accurately
    ts_sq = datetime(2026, 11, 20, 15, 15, tzinfo=IST_TZ)
    orders = portfolio.square_off_all_positions(
        current_prices={"RELIANCE": 2520.0},
        timestamp=ts_sq,
    )
    assert len(orders) == 2
    assert len(portfolio.positions) == 0
    assert len(portfolio.closed_trades) == 2

    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg


def test_configurable_cost_parameters():
    """Verify that CostConfig accurately modifies statutory rates and caps."""
    from trader.config import CostConfig

    # Standard default cost config
    cost_default = calculate_intraday_costs(price=1000.0, quantity=100, side="BUY", exchange="NSE")
    assert cost_default.brokerage_inr == 20.0

    # Custom discounted broker cost config (0.01% with max ₹10)
    custom_config = CostConfig(
        brokerage_pct=0.0001,
        max_brokerage_per_order_inr=10.0,
    )
    cost_custom = calculate_intraday_costs(
        price=1000.0, quantity=100, side="BUY", exchange="NSE", cost_config=custom_config
    )
    # 0.01% of 100,000 = 10.0
    assert cost_custom.brokerage_inr == 10.0
    assert cost_custom.total_friction_inr < cost_default.total_friction_inr


def test_daily_session_reset_and_ledger():
    """Verify daily portfolio equity reset and ledger recording."""
    portfolio = IntradayPaperPortfolio(initial_capital=500_000.0)
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Day 1: Make a profitable trade
    portfolio.submit_order("TCS", OrderSide.BUY, quantity=10, price=3500.0, timestamp=ts)
    ts_sell = datetime(2026, 11, 20, 14, 0, tzinfo=IST_TZ)
    portfolio.submit_order("TCS", OrderSide.SELL, quantity=10, price=3600.0, timestamp=ts_sell)

    day1_equity = portfolio.total_equity
    assert day1_equity > 500_000.0
    assert portfolio.day_start_equity == 500_000.0

    # Day 2: Start new daily session
    evt = portfolio.reset_daily_session(day_date="2026-11-21")
    assert evt.event_type == "DAILY_SESSION_OPENED"
    assert portfolio.day_start_equity == day1_equity

    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg


def test_split_exit_paisa_reconciliation():
    """Verify exact down-to-the-paisa capital conservation across multiple partial exits."""
    portfolio = IntradayPaperPortfolio(initial_capital=500_000.0)
    ts = datetime(2026, 11, 20, 10, 0, tzinfo=IST_TZ)

    # Buy 100 shares of SBIN at ₹800
    portfolio.submit_order("SBIN", OrderSide.BUY, quantity=100, price=800.0, timestamp=ts)

    # Exit 30 shares at ₹810
    portfolio.submit_order("SBIN", OrderSide.SELL, quantity=30, price=810.0, timestamp=ts)
    # Exit 35 shares at ₹815
    portfolio.submit_order("SBIN", OrderSide.SELL, quantity=35, price=815.0, timestamp=ts)
    # Exit remaining 35 shares at ₹805
    portfolio.submit_order("SBIN", OrderSide.SELL, quantity=35, price=805.0, timestamp=ts)

    assert len(portfolio.positions) == 0
    assert len(portfolio.closed_trades) == 3

    # All cash, no positions remaining
    valid, msg = portfolio.verify_ledger_integrity()
    assert valid, msg

    # Net PnL matches exactly the sum of closed trades
    total_net_pnl = sum(t["net_pnl_inr"] for t in portfolio.closed_trades)
    assert abs(portfolio.cash - (500_000.0 + total_net_pnl)) < 0.05

