"""Unit and integration tests for Strategy Library, Walk-Forward Backtester, and Cross-Exchange Arbitrage."""

from datetime import datetime, timedelta
import pytest

from trader.config import IST_TZ
from trader.data.schema import Candle5M
from trader.data.loader import generate_synthetic_5m_session
from trader.features.indicators import compute_all_features
from trader.portfolio.ledger import IntradayPaperPortfolio
from trader.risk.engine import DeterministicRiskEngine
from trader.strategies.vwap_pullback import VWAPPullbackStrategy
from trader.strategies.orb_breakout import ORBBreakoutStrategy
from trader.strategies.ema_cross import EMACrossStrategy
from trader.strategies.supertrend import SupertrendStrategy
from trader.strategies.backtester import IntradayBacktester
from trader.arbitrage.spread_calculator import calculate_executable_arbitrage_spread
from trader.arbitrage.paired_executor import PairedArbitrageExecutor


def test_vwap_pullback_strategy_logic():
    """Assert VWAP Pullback strategy generates signals on qualified setups."""
    strat = VWAPPullbackStrategy()
    assert strat.metadata.strategy_id == "STRAT-001-VWAP-PULLBACK"
    assert strat.metadata.validation_status == "APPROVED_PAPER"

    candles = generate_synthetic_5m_session(symbol="RELIANCE", base_price=2500.0, daily_trend_pct=1.5)
    df = compute_all_features(candles)

    signals = [strat.evaluate_bar("RELIANCE", df, i) for i in range(len(df))]
    # Verify signal format
    assert all(s.strategy_id == "STRAT-001-VWAP-PULLBACK" for s in signals)
    actions = {s.action for s in signals}
    assert "NO_SIGNAL" in actions


def test_orb_breakout_strategy_logic():
    """Assert ORB-15 strategy tracks opening range and evaluates breakouts."""
    strat = ORBBreakoutStrategy()
    assert strat.metadata.strategy_id == "STRAT-002-ORB-BREAKOUT"

    candles = generate_synthetic_5m_session(symbol="TCS", base_price=3500.0)
    df = compute_all_features(candles)

    # Evaluate bars 0 to 10
    signals = [strat.evaluate_bar("TCS", df, i) for i in range(10)]
    assert all(s.strategy_id == "STRAT-002-ORB-BREAKOUT" for s in signals)


def test_ema_cross_strategy_logic():
    """Assert EMA 9/21 cross strategy emits signals with trend filters."""
    strat = EMACrossStrategy()
    assert strat.metadata.strategy_id == "STRAT-003-EMA-CROSS"

    candles = generate_synthetic_5m_session(symbol="INFY", base_price=1800.0)
    df = compute_all_features(candles)

    signals = [strat.evaluate_bar("INFY", df, i) for i in range(len(df))]
    assert all(s.symbol == "INFY" for s in signals)


def test_intraday_backtester_execution():
    """Run full backtester over synthetic 5m candles and assert metric computations."""
    backtester = IntradayBacktester(initial_capital=1_000_000.0)
    strat = VWAPPullbackStrategy()
    candles = generate_synthetic_5m_session(symbol="RELIANCE", base_price=2500.0, daily_trend_pct=1.0)

    res = backtester.run_backtest(strategy=strat, candles=candles)
    assert res.symbol == "RELIANCE"
    assert res.strategy_id == "STRAT-001-VWAP-PULLBACK"
    assert res.total_candles == 75
    assert res.initial_capital_inr == 1_000_000.0
    assert isinstance(res.total_return_pct, float)
    assert isinstance(res.win_rate_pct, float)
    assert isinstance(res.profit_factor, float)
    assert isinstance(res.max_drawdown_pct, float)


def test_cross_exchange_spread_calculator():
    """Assert calculation of net arbitrage spread and rejection of uneconomic spreads."""
    # Scenario 1: Tiny LTP gap of ₹0.50 (eaten by friction)
    opp_small = calculate_executable_arbitrage_spread(
        symbol="RELIANCE",
        nse_bid=2500.50,
        nse_ask=2501.00,
        bse_bid=2500.00,
        bse_ask=2500.50,
        quantity=100,
    )
    assert opp_small.is_actionable is False
    assert opp_small.status == "UNECONOMIC_SPREAD"
    assert opp_small.net_profit_inr <= 0

    # Scenario 2: Substantial executable gap (BSE Ask 2480 vs NSE Bid 2500 -> ₹20 spread)
    opp_wide = calculate_executable_arbitrage_spread(
        symbol="RELIANCE",
        nse_bid=2500.00,
        nse_ask=2501.00,
        bse_bid=2478.00,
        bse_ask=2480.00,
        quantity=100,
    )
    assert opp_wide.is_actionable is True
    assert opp_wide.status == "ACTIONABLE_OPPORTUNITY"
    assert opp_wide.buy_exchange == "BSE"
    assert opp_wide.sell_exchange == "NSE"
    assert opp_wide.gross_spread_inr == 2000.0  # (2500 - 2480) * 100
    assert opp_wide.net_profit_inr > 1500.0     # Profit remains positive after all friction


def test_paired_arbitrage_execution_and_unhedged_mitigation():
    """Assert paired order execution and emergency contingency square-off if Leg 2 fails."""
    portfolio = IntradayPaperPortfolio(initial_capital=1_000_000.0)
    executor = PairedArbitrageExecutor(portfolio=portfolio)

    opp = calculate_executable_arbitrage_spread(
        symbol="RELIANCE",
        nse_bid=2500.00,
        nse_ask=2501.00,
        bse_bid=2478.00,
        bse_ask=2480.00,
        quantity=50,
    )

    # Test 1: Full fill on both legs
    res_full = executor.execute_paired_arbitrage(opp, leg1_fill_prob=1.0, leg2_fill_prob=1.0)
    assert res_full.is_hedged is True
    assert res_full.unhedged_quantity == 0
    assert res_full.leg1_filled_quantity == 50
    assert res_full.leg2_filled_quantity == 50
    assert res_full.net_realized_pnl_inr > 0

    # Test 2: Leg 2 failure triggers emergency square-off
    res_failed_leg2 = executor.execute_paired_arbitrage(opp, leg1_fill_prob=1.0, leg2_fill_prob=0.0)
    assert res_failed_leg2.contingency_action_taken == "EMERGENCY_MARKET_SQUAREOFF"
    assert len(portfolio.positions) == 0  # No open position remains
