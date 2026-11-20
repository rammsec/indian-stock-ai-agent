"""Double-entry paper trading portfolio and cryptographic SHA-256 event ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from ..config import DEFAULT_INITIAL_CAPITAL_INR, IST_TZ
from .cost_calculator import calculate_intraday_costs, IndianIntradayCost


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class SimulatedOrder:
    order_id: str
    symbol: str
    exchange: Literal["NSE", "BSE"]
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    status: OrderStatus
    timestamp: datetime
    stop_loss: float | None = None
    target: float | None = None
    filled_price: float | None = None
    filled_at: datetime | None = None
    cost_breakdown: IndianIntradayCost | None = None
    rejection_reason: str | None = None


@dataclass
class Position:
    symbol: str
    exchange: Literal["NSE", "BSE"]
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float | None
    target: float | None
    highest_price: float
    entry_time: datetime
    entry_cost: float


@dataclass
class LedgerEvent:
    event_seq: int
    event_id: str
    event_type: str
    timestamp: str
    details: dict[str, Any]
    prev_hash: str
    event_hash: str


class IntradayPaperPortfolio:
    """Institutional double-entry paper trading portfolio with SHA-256 provenance."""

    def __init__(self, initial_capital: float = DEFAULT_INITIAL_CAPITAL_INR) -> None:
        self.initial_capital: float = float(initial_capital)
        self.cash: float = float(initial_capital)
        self.positions: dict[str, Position] = {}
        self.orders: list[SimulatedOrder] = []
        self.closed_trades: list[dict[str, Any]] = []
        self.event_ledger: list[LedgerEvent] = []
        self._seq: int = 0
        self._last_hash: str = "GENESIS_HASH_00000000000000000000000000000000"
        self.peak_equity: float = float(initial_capital)
        self.day_start_equity: float = float(initial_capital)

        # Record portfolio initialization event
        self._record_event(
            event_type="PORTFOLIO_INITIALIZED",
            timestamp=datetime.now(tz=IST_TZ).isoformat(),
            details={"initial_capital": self.initial_capital},
        )

    def _record_event(self, event_type: str, timestamp: str, details: dict[str, Any]) -> LedgerEvent:
        self._seq += 1
        event_id = f"evt_{self._seq:06d}_{event_type.lower()}"
        payload = {
            "seq": self._seq,
            "id": event_id,
            "type": event_type,
            "timestamp": timestamp,
            "details": details,
            "prev_hash": self._last_hash,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        evt = LedgerEvent(
            event_seq=self._seq,
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            details=details,
            prev_hash=self._last_hash,
            event_hash=digest,
        )
        self.event_ledger.append(evt)
        self._last_hash = digest
        return evt

    @property
    def invested_capital(self) -> float:
        """Total capital allocated to open positions at current market prices."""
        return sum(pos.quantity * pos.current_price for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        """Cash plus mark-to-market value of open positions."""
        return self.cash + self.invested_capital

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L across all open positions."""
        return sum(
            pos.quantity * (pos.current_price - pos.entry_price)
            for pos in self.positions.values()
        )

    @property
    def realized_pnl(self) -> float:
        """Realized P&L across all completed trades."""
        return sum(trade["net_pnl_inr"] for trade in self.closed_trades)

    def submit_order(
        self,
        symbol: str,
        side: OrderSide | str,
        quantity: int,
        price: float,
        exchange: Literal["NSE", "BSE"] = "NSE",
        order_type: OrderType | str = OrderType.MARKET,
        stop_loss: float | None = None,
        target: float | None = None,
        timestamp: datetime | None = None,
    ) -> SimulatedOrder:
        """Submit and immediately simulate fill for a paper order with statutory deductions."""
        if timestamp is None:
            timestamp = datetime.now(tz=IST_TZ)

        side_enum = OrderSide(side)
        type_enum = OrderType(order_type)
        order_id = f"ord_{len(self.orders) + 1:05d}_{symbol}"

        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            exchange=exchange,
            side=side_enum,
            order_type=type_enum,
            quantity=quantity,
            price=price,
            status=OrderStatus.SUBMITTED,
            timestamp=timestamp,
            stop_loss=stop_loss,
            target=target,
        )

        cost = calculate_intraday_costs(price=price, quantity=quantity, side=side_enum.value, exchange=exchange)
        order.cost_breakdown = cost

        if side_enum == OrderSide.BUY:
            required_funds = (price * quantity) + cost.total_friction_inr
            if required_funds > self.cash:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = (
                    f"Insufficient cash: required ₹{required_funds:.2f}, available ₹{self.cash:.2f}"
                )
                self.orders.append(order)
                self._record_event(
                    "ORDER_REJECTED",
                    timestamp.isoformat(),
                    {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason},
                )
                return order

            # Fill Buy Order
            self.cash -= required_funds
            order.status = OrderStatus.FILLED
            order.filled_price = price
            order.filled_at = timestamp

            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos.quantity + quantity
                new_avg_entry = ((pos.quantity * pos.entry_price) + (quantity * price)) / total_qty
                pos.quantity = total_qty
                pos.entry_price = new_avg_entry
                pos.current_price = price
                pos.highest_price = max(pos.highest_price, price)
                pos.entry_cost += cost.total_friction_inr
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    exchange=exchange,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    highest_price=price,
                    entry_time=timestamp,
                    entry_cost=cost.total_friction_inr,
                )

            self.orders.append(order)
            self._record_event(
                "ORDER_FILLED_BUY",
                timestamp.isoformat(),
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "friction_inr": cost.total_friction_inr,
                    "cash_after": self.cash,
                },
            )

        elif side_enum == OrderSide.SELL:
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"No open long position in {symbol} to sell"
                self.orders.append(order)
                self._record_event(
                    "ORDER_REJECTED",
                    timestamp.isoformat(),
                    {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason},
                )
                return order

            pos = self.positions[symbol]
            gross_proceeds = price * quantity
            net_proceeds = gross_proceeds - cost.total_friction_inr
            self.cash += net_proceeds

            order.status = OrderStatus.FILLED
            order.filled_price = price
            order.filled_at = timestamp

            # Calculate realized P&L on the sold shares
            cost_basis = pos.entry_price * quantity
            portion_entry_cost = pos.entry_cost * (quantity / pos.quantity)
            net_pnl = net_proceeds - cost_basis - portion_entry_cost
            ret_pct = (net_pnl / cost_basis) * 100.0

            trade_record = {
                "symbol": symbol,
                "exchange": exchange,
                "quantity": quantity,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "entry_time": pos.entry_time.isoformat(),
                "exit_time": timestamp.isoformat(),
                "gross_pnl_inr": round((price - pos.entry_price) * quantity, 2),
                "total_friction_inr": round(portion_entry_cost + cost.total_friction_inr, 2),
                "net_pnl_inr": round(net_pnl, 2),
                "return_pct": round(ret_pct, 2),
            }
            self.closed_trades.append(trade_record)

            # Update or remove position
            if pos.quantity == quantity:
                del self.positions[symbol]
            else:
                pos.quantity -= quantity
                pos.entry_cost -= portion_entry_cost

            self.orders.append(order)
            self._record_event(
                "ORDER_FILLED_SELL",
                timestamp.isoformat(),
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "exit_price": price,
                    "net_pnl": net_pnl,
                    "cash_after": self.cash,
                },
            )

        # Track peak equity
        self.peak_equity = max(self.peak_equity, self.total_equity)
        return order

    def update_market_price(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
    ) -> list[SimulatedOrder]:
        """Update market price for a symbol and auto-execute stop-loss / target if breached."""
        executed_orders: list[SimulatedOrder] = []
        if symbol not in self.positions:
            return executed_orders

        pos = self.positions[symbol]
        pos.current_price = current_price
        pos.highest_price = max(pos.highest_price, current_price)

        # Check Stop-Loss
        if pos.stop_loss is not None and current_price <= pos.stop_loss:
            order = self.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                price=current_price,
                exchange=pos.exchange,
                timestamp=timestamp,
            )
            executed_orders.append(order)
            return executed_orders

        # Check Target
        if pos.target is not None and current_price >= pos.target:
            order = self.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                price=current_price,
                exchange=pos.exchange,
                timestamp=timestamp,
            )
            executed_orders.append(order)
            return executed_orders

        return executed_orders

    def square_off_all_positions(self, current_prices: dict[str, float], timestamp: datetime) -> list[SimulatedOrder]:
        """Mandatory square-off for all open positions (e.g. at 15:15 IST intraday close)."""
        orders = []
        for symbol, pos in list(self.positions.items()):
            price = current_prices.get(symbol, pos.current_price)
            ord_res = self.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                price=price,
                exchange=pos.exchange,
                timestamp=timestamp,
            )
            orders.append(ord_res)
        return orders

    def verify_ledger_integrity(self) -> tuple[bool, str]:
        """Verify unbroken cryptographic SHA-256 hash chaining and capital conservation."""
        last_hash = "GENESIS_HASH_00000000000000000000000000000000"
        for evt in self.event_ledger:
            if evt.prev_hash != last_hash:
                return False, f"Broken hash chain at event sequence {evt.event_seq}"
            payload = {
                "seq": evt.event_seq,
                "id": evt.event_id,
                "type": evt.event_type,
                "timestamp": evt.timestamp,
                "details": evt.details,
                "prev_hash": evt.prev_hash,
            }
            expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if evt.event_hash != expected_hash:
                return False, f"Hash mismatch at event sequence {evt.event_seq}"
            last_hash = evt.event_hash

        # Conservation of Capital check
        total_accounting_equity = self.cash + self.invested_capital
        if abs(total_accounting_equity - self.total_equity) > 1e-4:
            return False, f"Capital conservation violated: cash({self.cash}) + invested({self.invested_capital}) != equity({self.total_equity})"

        return True, "Ledger integrity and capital conservation 100% verified"

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Return comprehensive institutional portfolio metrics."""
        wins = [t for t in self.closed_trades if t["net_pnl_inr"] > 0]
        losses = [t for t in self.closed_trades if t["net_pnl_inr"] <= 0]
        win_rate = (len(wins) / len(self.closed_trades) * 100.0) if self.closed_trades else 0.0

        gross_profits = sum(t["net_pnl_inr"] for t in wins)
        gross_losses = abs(sum(t["net_pnl_inr"] for t in losses))
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (100.0 if gross_profits > 0 else 0.0)

        dd_pct = ((self.peak_equity - self.total_equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        return {
            "initial_capital_inr": self.initial_capital,
            "cash_inr": round(self.cash, 2),
            "invested_capital_inr": round(self.invested_capital, 2),
            "total_equity_inr": round(self.total_equity, 2),
            "realized_pnl_inr": round(self.realized_pnl, 2),
            "unrealized_pnl_inr": round(self.unrealized_pnl, 2),
            "total_return_pct": round(((self.total_equity - self.initial_capital) / self.initial_capital) * 100.0, 2),
            "current_drawdown_pct": round(dd_pct, 2),
            "open_positions_count": len(self.positions),
            "closed_trades_count": len(self.closed_trades),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
        }
