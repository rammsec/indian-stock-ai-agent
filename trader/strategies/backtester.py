"""Deterministic Walk-Forward Intraday Backtester with Indian Friction Modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any
import numpy as np
import pandas as pd

from ..config import IST_TZ, INTRADAY_MANDATORY_SQUAREOFF_TIME
from ..data.schema import Candle5M
from ..features.indicators import compute_all_features
from ..portfolio.ledger import IntradayPaperPortfolio, OrderSide
from ..risk.engine import DeterministicRiskEngine
from .base import BaseStrategy, StrategySignal


@dataclass
class BacktestResult:
    """Comprehensive performance report of a backtest run."""

    strategy_id: str
    symbol: str
    total_candles: int
    initial_capital_inr: float
    ending_capital_inr: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_friction_paid_inr: float
    net_expectancy_per_trade_inr: float
    trades: list[dict[str, Any]] = field(default_factory=list)


class IntradayBacktester:
    """Rigorous, deterministic backtester with statutory Indian costs, slippage, and no lookahead."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_engine: DeterministicRiskEngine | None = None,
        execution_mode: Literal["NEXT_BAR_OPEN", "SAME_BAR_CLOSE"] = "NEXT_BAR_OPEN",
        slippage_bps: float = 5.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.risk_engine = risk_engine or DeterministicRiskEngine()
        self.execution_mode = execution_mode
        self.slippage_bps = slippage_bps

    def run_backtest(
        self,
        strategy: BaseStrategy,
        candles: list[Candle5M],
    ) -> BacktestResult:
        """Run backtest candle-by-candle over historical 5m bars."""
        if not candles:
            raise ValueError("No candles provided for backtesting")

        symbol = candles[0].symbol
        portfolio = IntradayPaperPortfolio(initial_capital=self.initial_capital)
        features_df = compute_all_features(candles)

        daily_returns: list[float] = []
        prior_equity = self.initial_capital
        pending_signal = None  # Tuple of (action, signal)

        for i in range(1, len(features_df)):
            curr_row = features_df.iloc[i]
            ts = curr_row["timestamp"].to_pydatetime() if hasattr(curr_row["timestamp"], "to_pydatetime") else curr_row["timestamp"]
            open_price = float(curr_row["open"])
            high_price = float(curr_row["high"])
            low_price = float(curr_row["low"])
            close_price = float(curr_row["close"])
            current_time = ts.astimezone(IST_TZ).time()

            # 1. Execute any pending signal from previous bar at NEXT BAR OPEN (with slippage)
            if pending_signal is not None:
                p_action, p_sig = pending_signal
                pending_signal = None

                if p_action == "ENTER_LONG" and symbol not in portfolio.positions:
                    slippage_mult = 1.0 + (self.slippage_bps / 10000.0)
                    exec_price = round(open_price * slippage_mult, 2)
                    risk_res = self.risk_engine.evaluate_order(
                        symbol=symbol,
                        side="BUY",
                        requested_quantity=100,
                        entry_price=exec_price,
                        stop_loss_price=p_sig.stop_loss,
                        portfolio=portfolio,
                        current_time=ts,
                    )
                    if risk_res.approved and risk_res.approved_quantity > 0:
                        portfolio.submit_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=risk_res.approved_quantity,
                            price=exec_price,
                            exchange=candles[0].exchange,
                            stop_loss=p_sig.stop_loss,
                            target=p_sig.target,
                            timestamp=ts,
                        )
                elif p_action == "EXIT" and symbol in portfolio.positions:
                    slippage_mult = 1.0 - (self.slippage_bps / 10000.0)
                    exec_price = round(open_price * slippage_mult, 2)
                    pos = portfolio.positions[symbol]
                    portfolio.submit_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        price=exec_price,
                        exchange=candles[0].exchange,
                        timestamp=ts,
                    )

            # 2. Check in-bar price progression for Stop-Loss & Target breaches (Low then High)
            if symbol in portfolio.positions:
                # Test low price for stop loss
                portfolio.update_market_price(symbol, low_price, timestamp=ts)
                # If still open, test high price for target
                if symbol in portfolio.positions:
                    portfolio.update_market_price(symbol, high_price, timestamp=ts)
                # Update mark-to-market at bar close
                if symbol in portfolio.positions:
                    portfolio.update_market_price(symbol, close_price, timestamp=ts)

            # 3. Check 15:15 IST Mandatory Square-Off
            if current_time >= INTRADAY_MANDATORY_SQUAREOFF_TIME and symbol in portfolio.positions:
                portfolio.square_off_all_positions(current_prices={symbol: close_price}, timestamp=ts)
                pending_signal = None
                continue

            # 4. Strategy evaluation on closed candle i
            signal = strategy.evaluate_bar(symbol, features_df, current_bar_idx=i)

            if self.execution_mode == "NEXT_BAR_OPEN":
                if signal.action == "ENTER_LONG" and symbol not in portfolio.positions:
                    pending_signal = ("ENTER_LONG", signal)
                elif signal.action == "EXIT" and symbol in portfolio.positions:
                    pending_signal = ("EXIT", signal)
            else:
                # SAME_BAR_CLOSE mode
                if signal.action == "ENTER_LONG" and symbol not in portfolio.positions:
                    risk_res = self.risk_engine.evaluate_order(
                        symbol=symbol,
                        side="BUY",
                        requested_quantity=100,
                        entry_price=close_price,
                        stop_loss_price=signal.stop_loss,
                        portfolio=portfolio,
                        current_time=ts,
                    )
                    if risk_res.approved and risk_res.approved_quantity > 0:
                        portfolio.submit_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=risk_res.approved_quantity,
                            price=close_price,
                            exchange=candles[0].exchange,
                            stop_loss=signal.stop_loss,
                            target=signal.target,
                            timestamp=ts,
                        )
                elif signal.action == "EXIT" and symbol in portfolio.positions:
                    pos = portfolio.positions[symbol]
                    portfolio.submit_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        price=close_price,
                        exchange=candles[0].exchange,
                        timestamp=ts,
                    )

        # Force square-off remaining open positions at final bar
        if symbol in portfolio.positions:
            portfolio.square_off_all_positions(
                current_prices={symbol: float(features_df.iloc[-1]["close"])},
                timestamp=features_df.iloc[-1]["timestamp"],
            )

        # Performance Metrics
        summary = portfolio.get_portfolio_summary()
        trades = portfolio.closed_trades
        total_trades = len(trades)
        wins = [t for t in trades if t["net_pnl_inr"] > 0]
        losses = [t for t in trades if t["net_pnl_inr"] <= 0]
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t["net_pnl_inr"] for t in wins)
        gross_loss = abs(sum(t["net_pnl_inr"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

        trade_pnls = [t["net_pnl_inr"] for t in trades]
        net_expectancy = (sum(trade_pnls) / total_trades) if total_trades > 0 else 0.0
        total_friction = sum(t.get("total_friction_inr", 0.0) for t in trades)

        # Sharpe and Sortino ratios
        if len(trade_pnls) > 1:
            pnl_std = float(np.std(trade_pnls))
            downside_std = float(np.std([p for p in trade_pnls if p < 0])) if losses else 1e-4
            mean_pnl = float(np.mean(trade_pnls))
            sharpe = (mean_pnl / pnl_std * math.sqrt(252)) if pnl_std > 0 else 0.0
            sortino = (mean_pnl / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        return BacktestResult(
            strategy_id=strategy.metadata.strategy_id,
            symbol=symbol,
            total_candles=len(candles),
            initial_capital_inr=self.initial_capital,
            ending_capital_inr=summary["total_equity_inr"],
            total_return_pct=summary["total_return_pct"],
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=summary["current_drawdown_pct"],
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            total_friction_paid_inr=round(total_friction, 2),
            net_expectancy_per_trade_inr=round(net_expectancy, 2),
            trades=trades,
        )
