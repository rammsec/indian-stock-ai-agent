"""Paired Order Execution Simulator for Cross-Exchange Arbitrage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..config import IST_TZ
from ..portfolio.ledger import IntradayPaperPortfolio, OrderSide, OrderStatus
from .spread_calculator import CrossExchangeOpportunity


@dataclass
class PairedExecutionResult:
    """Outcome of a simulated paired cross-exchange execution."""

    opportunity_id: str
    symbol: str
    isin: str
    leg1_status: OrderStatus
    leg2_status: OrderStatus
    leg1_filled_quantity: int
    leg2_filled_quantity: int
    is_hedged: bool
    unhedged_quantity: int
    net_realized_pnl_inr: float
    contingency_action_taken: str | None = None
    execution_notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.execution_notes is None:
            self.execution_notes = []


class PairedArbitrageExecutor:
    """Simulates simultaneous paired orders and handles partial fill risks."""

    def __init__(self, portfolio: IntradayPaperPortfolio) -> None:
        self.portfolio = portfolio

    def execute_paired_arbitrage(
        self,
        opp: CrossExchangeOpportunity,
        leg1_fill_prob: float = 1.0,
        leg2_fill_prob: float = 1.0,
        timestamp: datetime | None = None,
    ) -> PairedExecutionResult:
        """Simulate paired execution of Buy leg and Sell leg with contingency unhedged leg handling."""
        if timestamp is None:
            timestamp = datetime.now(tz=IST_TZ)

        opp_id = f"arb_{opp.symbol}_{int(timestamp.timestamp())}"
        notes: list[str] = []

        if not opp.is_actionable:
            return PairedExecutionResult(
                opportunity_id=opp_id,
                symbol=opp.symbol,
                isin=opp.isin,
                leg1_status=OrderStatus.REJECTED,
                leg2_status=OrderStatus.REJECTED,
                leg1_filled_quantity=0,
                leg2_filled_quantity=0,
                is_hedged=True,
                unhedged_quantity=0,
                net_realized_pnl_inr=0.0,
                contingency_action_taken="ABORTED_NOT_ACTIONABLE",
                execution_notes=["Opportunity rejected: uneconomic net spread or negative yield"],
            )

        # 1. Execute Leg 1 (Buy)
        leg1_qty = opp.quantity if leg1_fill_prob >= 0.5 else int(opp.quantity * leg1_fill_prob)
        if leg1_qty <= 0:
            return PairedExecutionResult(
                opportunity_id=opp_id,
                symbol=opp.symbol,
                isin=opp.isin,
                leg1_status=OrderStatus.REJECTED,
                leg2_status=OrderStatus.CANCELLED,
                leg1_filled_quantity=0,
                leg2_filled_quantity=0,
                is_hedged=True,
                unhedged_quantity=0,
                net_realized_pnl_inr=0.0,
                contingency_action_taken="LEG1_FILL_FAILED",
                execution_notes=["Leg 1 Buy order was not filled; Leg 2 aborted"],
            )

        buy_order = self.portfolio.submit_order(
            symbol=opp.symbol,
            side=OrderSide.BUY,
            quantity=leg1_qty,
            price=opp.buy_ask_price,
            exchange=opp.buy_exchange,
            timestamp=timestamp,
        )

        if buy_order.status != OrderStatus.FILLED:
            return PairedExecutionResult(
                opportunity_id=opp_id,
                symbol=opp.symbol,
                isin=opp.isin,
                leg1_status=buy_order.status,
                leg2_status=OrderStatus.CANCELLED,
                leg1_filled_quantity=0,
                leg2_filled_quantity=0,
                is_hedged=True,
                unhedged_quantity=0,
                net_realized_pnl_inr=0.0,
                contingency_action_taken="LEG1_PORTFOLIO_REJECTION",
                execution_notes=[f"Leg 1 rejected: {buy_order.rejection_reason}"],
            )

        notes.append(f"Leg 1 filled: Bought {leg1_qty} {opp.symbol} on {opp.buy_exchange} @ ₹{opp.buy_ask_price:.2f}")

        # 2. Execute Leg 2 (Sell)
        leg2_qty = leg1_qty if leg2_fill_prob >= 0.95 else int(leg1_qty * leg2_fill_prob)

        if leg2_qty > 0:
            sell_order = self.portfolio.submit_order(
                symbol=opp.symbol,
                side=OrderSide.SELL,
                quantity=leg2_qty,
                price=opp.sell_bid_price,
                exchange=opp.sell_exchange,
                timestamp=timestamp,
            )
            notes.append(f"Leg 2 filled: Sold {leg2_qty} {opp.symbol} on {opp.sell_exchange} @ ₹{opp.sell_bid_price:.2f}")
        else:
            sell_order = None
            notes.append(f"Leg 2 FAILED to fill on {opp.sell_exchange}! Unhedged risk detected!")

        unhedged_qty = leg1_qty - (leg2_qty if sell_order and sell_order.status == OrderStatus.FILLED else 0)
        contingency = None

        # 3. Contingency Unhedged Leg Mitigation
        if unhedged_qty > 0:
            contingency = "EMERGENCY_MARKET_SQUAREOFF"
            notes.append(
                f"Contingency triggered: Immediately squaring off {unhedged_qty} unhedged shares at current market bid"
            )
            # Emergency exit on the same exchange to eliminate naked overnight risk
            self.portfolio.submit_order(
                symbol=opp.symbol,
                side=OrderSide.SELL,
                quantity=unhedged_qty,
                price=opp.buy_ask_price * 0.998,  # Model small emergency exit slippage
                exchange=opp.buy_exchange,
                timestamp=timestamp,
            )

        # Realized P&L from latest trade(s)
        pnl = sum(t["net_pnl_inr"] for t in self.portfolio.closed_trades[-2:]) if self.portfolio.closed_trades else 0.0

        return PairedExecutionResult(
            opportunity_id=opp_id,
            symbol=opp.symbol,
            isin=opp.isin,
            leg1_status=OrderStatus.FILLED,
            leg2_status=OrderStatus.FILLED if unhedged_qty == 0 else OrderStatus.REJECTED,
            leg1_filled_quantity=leg1_qty,
            leg2_filled_quantity=leg2_qty,
            is_hedged=(unhedged_qty == 0),
            unhedged_quantity=unhedged_qty,
            net_realized_pnl_inr=round(pnl, 2),
            contingency_action_taken=contingency,
            execution_notes=notes,
        )
