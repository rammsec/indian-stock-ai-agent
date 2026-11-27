"""Double-entry paper trading portfolio and cryptographic SHA-256 event ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from ..config import DEFAULT_INITIAL_CAPITAL_INR, IST_TZ, CostConfig
from .cost_calculator import calculate_intraday_costs, IndianIntradayCost


NSE_TICK_SIZE_INR: float = 0.05


def round_to_nse_tick(price: float, tick_size: float = NSE_TICK_SIZE_INR) -> float:
    """Round an INR equity price to the nearest valid NSE tick increment (default ₹0.05)."""
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive; got {tick_size}")
    return round(round(float(price) / tick_size) * tick_size, 2)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


PositionEffectLiteral = Literal["AUTO", "OPEN_LONG", "EXIT_LONG", "OPEN_SHORT", "COVER_SHORT"]


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
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
    filled_quantity: int = 0
    remaining_quantity: int = 0
    stop_loss: float | None = None
    target: float | None = None
    filled_price: float | None = None
    filled_at: datetime | None = None
    cost_breakdown: IndianIntradayCost | None = None
    rejection_reason: str | None = None
    position_effect: str = "AUTO"


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
    side: Literal["LONG", "SHORT"] = "LONG"
    lowest_price: float | None = None

    def __post_init__(self) -> None:
        if self.lowest_price is None:
            self.lowest_price = float(self.entry_price)


class PositionDict(dict):
    """Dictionary supporting position keying by 'EXCHANGE:SYMBOL' or bare 'SYMBOL'."""

    def __getitem__(self, key: str) -> Position:
        if super().__contains__(key):
            return super().__getitem__(key)
        for k in self:
            if k.endswith(f":{key}") or k == key:
                return super().__getitem__(k)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            for k in self:
                if k.endswith(f":{key}") or k == key:
                    return True
        return False

    def __delitem__(self, key: str) -> None:
        if super().__contains__(key):
            super().__delitem__(key)
            return
        for k in list(self.keys()):
            if k.endswith(f":{key}") or k == key:
                super().__delitem__(k)
                return
        raise KeyError(key)

    def get(self, key: object, default: Any = None) -> Any:
        try:
            return self[str(key)]
        except KeyError:
            return default

    def get_position(self, symbol: str, exchange: str | None = None) -> Position | None:
        """Lookup position with optional exchange qualification."""
        if exchange:
            k = f"{exchange.upper()}:{symbol.upper()}"
            return super().get(k, None)
        try:
            return self[symbol]
        except KeyError:
            return None


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

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL_INR,
        cost_config: CostConfig | None = None,
    ) -> None:
        self.initial_capital: float = float(initial_capital)
        self.cost_config: CostConfig | None = cost_config
        self.cash: float = float(initial_capital)
        self.positions: PositionDict = PositionDict()
        self.orders: list[SimulatedOrder] = []
        self.closed_trades: list[dict[str, Any]] = []
        self.event_ledger: list[LedgerEvent] = []
        self._seq: int = 0
        self._last_hash: str = "GENESIS_HASH_00000000000000000000000000000000"
        self.peak_equity: float = float(initial_capital)
        self.day_start_equity: float = float(initial_capital)
        self.margin_call_triggered: bool = False
        self.margin_call_events: list[dict[str, Any]] = []

        # Record portfolio initialization event
        self._record_event(
            event_type="PORTFOLIO_INITIALIZED",
            timestamp=datetime.now(tz=IST_TZ).isoformat(),
            details={"initial_capital": self.initial_capital},
        )

    def reset_daily_session(self, day_date: datetime.date | str | None = None) -> LedgerEvent:
        """Reset day_start_equity at the start of a new trading session for accurate daily drawdown calculation."""
        now_iso = datetime.now(tz=IST_TZ).isoformat()
        self.day_start_equity = self.total_equity
        return self._record_event(
            event_type="DAILY_SESSION_OPENED",
            timestamp=now_iso,
            details={
                "session_date": str(day_date) if day_date else now_iso[:10],
                "day_start_equity": self.day_start_equity,
                "cash": self.cash,
                "open_positions": len(self.positions),
            },
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
        """Total mark-to-market equity value of open long and short positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.side == "SHORT":
                # 100% notional margin locked at entry + unrealized short P&L
                total += (pos.quantity * pos.entry_price) + (pos.quantity * (pos.entry_price - pos.current_price))
            else:
                total += pos.quantity * pos.current_price
        return total

    @property
    def total_equity(self) -> float:
        """Cash plus mark-to-market equity value of open positions."""
        return self.cash + self.invested_capital

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L across all open long and short positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.side == "SHORT":
                total += pos.quantity * (pos.entry_price - pos.current_price)
            else:
                total += pos.quantity * (pos.current_price - pos.entry_price)
        return total

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
        partial_fill_quantity: int | None = None,
        position_effect: PositionEffectLiteral = "AUTO",
        round_ticks: bool = False,
    ) -> SimulatedOrder:
        """Submit and immediately simulate fill for a paper order with statutory deductions.

        Explicitly distinguishes OPEN_LONG, EXIT_LONG, OPEN_SHORT, and COVER_SHORT so that
        exiting a long position can never accidentally open a short (and vice versa).
        """
        if timestamp is None:
            timestamp = datetime.now(tz=IST_TZ)

        side_enum = OrderSide(side)
        type_enum = OrderType(order_type)
        order_id = f"ord_{len(self.orders) + 1:05d}_{symbol}"

        if round_ticks:
            price = round_to_nse_tick(price)
            if stop_loss is not None:
                stop_loss = round_to_nse_tick(stop_loss)
            if target is not None:
                target = round_to_nse_tick(target)

        # Resolve effective position action while preserving backward-compatible AUTO behavior
        effect = position_effect
        if effect == "AUTO":
            effect = "OPEN_LONG" if side_enum == OrderSide.BUY else "EXIT_LONG"

        exec_quantity = quantity
        if partial_fill_quantity is not None and 0 < partial_fill_quantity < quantity:
            exec_quantity = partial_fill_quantity
            status_on_fill = OrderStatus.PARTIALLY_FILLED
        else:
            status_on_fill = OrderStatus.FILLED

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
            filled_quantity=0,
            remaining_quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            position_effect=effect,
        )

        # Fail closed on non-positive quantity/price or unknown position_effect
        if quantity <= 0 or price <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"Invalid order parameters: quantity={quantity}, price={price} must be positive"
            self.orders.append(order)
            self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
            return order
        if effect not in ("OPEN_LONG", "EXIT_LONG", "OPEN_SHORT", "COVER_SHORT"):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"Unrecognized position_effect: {effect!r}"
            self.orders.append(order)
            self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
            return order

        # Validate side vs position_effect consistency (fail closed)
        if effect in ("OPEN_LONG", "COVER_SHORT") and side_enum != OrderSide.BUY:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"Position effect {effect} requires side=BUY, got {side_enum.value}"
            self.orders.append(order)
            self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
            return order
        if effect in ("EXIT_LONG", "OPEN_SHORT") and side_enum != OrderSide.SELL:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"Position effect {effect} requires side=SELL, got {side_enum.value}"
            self.orders.append(order)
            self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
            return order

        cost = calculate_intraday_costs(
            price=price,
            quantity=exec_quantity,
            side=side_enum.value,
            exchange=exchange,
            cost_config=self.cost_config,
        )
        order.cost_breakdown = cost

        pos_key = f"{exchange}:{symbol}"
        matched_pos_key = pos_key if pos_key in self.positions else (symbol if symbol in self.positions else None)

        if effect == "OPEN_LONG":
            if matched_pos_key and self.positions[matched_pos_key].side == "SHORT":
                order.status = OrderStatus.REJECTED
                order.rejection_reason = (
                    f"Cannot OPEN_LONG in {symbol} while an open SHORT position exists; use COVER_SHORT first"
                )
                self.orders.append(order)
                self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
                return order

            required_funds = (price * exec_quantity) + cost.total_friction_inr
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

            # Fill / Partially Fill Buy Order (OPEN_LONG)
            self.cash -= required_funds
            order.status = status_on_fill
            order.filled_price = price
            order.filled_at = timestamp
            order.filled_quantity = exec_quantity
            order.remaining_quantity = quantity - exec_quantity

            # Update or create LONG position
            if pos_key in self.positions:
                pos = self.positions[pos_key]
                total_qty = pos.quantity + exec_quantity
                new_avg_entry = ((pos.quantity * pos.entry_price) + (exec_quantity * price)) / total_qty
                pos.quantity = total_qty
                pos.entry_price = new_avg_entry
                pos.current_price = price
                pos.highest_price = max(pos.highest_price, price)
                pos.lowest_price = min(pos.lowest_price if pos.lowest_price is not None else price, price)
                pos.entry_cost += cost.total_friction_inr
                if stop_loss is not None:
                    pos.stop_loss = stop_loss
                if target is not None:
                    pos.target = target
            else:
                self.positions[pos_key] = Position(
                    symbol=symbol,
                    exchange=exchange,
                    quantity=exec_quantity,
                    entry_price=price,
                    current_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    highest_price=price,
                    entry_time=timestamp,
                    entry_cost=cost.total_friction_inr,
                    side="LONG",
                    lowest_price=price,
                )

            self.orders.append(order)
            self._record_event(
                "ORDER_FILLED_BUY" if order.status == OrderStatus.FILLED else "ORDER_PARTIALLY_FILLED_BUY",
                timestamp.isoformat(),
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "exchange": exchange,
                    "position_effect": effect,
                    "requested_quantity": quantity,
                    "filled_quantity": exec_quantity,
                    "price": price,
                    "friction_inr": cost.total_friction_inr,
                    "cash_after": self.cash,
                },
            )

        elif effect == "EXIT_LONG":
            if (
                not matched_pos_key
                or self.positions[matched_pos_key].side != "LONG"
                or self.positions[matched_pos_key].quantity < exec_quantity
            ):
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"No open long position in {symbol} ({exchange}) to sell {exec_quantity} shares"
                self.orders.append(order)
                self._record_event(
                    "ORDER_REJECTED",
                    timestamp.isoformat(),
                    {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason},
                )
                return order

            pos = self.positions[matched_pos_key]
            gross_proceeds = price * exec_quantity
            net_proceeds = gross_proceeds - cost.total_friction_inr
            self.cash += net_proceeds

            order.status = status_on_fill
            order.filled_price = price
            order.filled_at = timestamp
            order.filled_quantity = exec_quantity
            order.remaining_quantity = quantity - exec_quantity

            # Calculate realized P&L on the sold long shares
            cost_basis = pos.entry_price * exec_quantity
            portion_entry_cost = pos.entry_cost * (exec_quantity / pos.quantity)
            net_pnl = net_proceeds - cost_basis - portion_entry_cost
            ret_pct = (net_pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0

            trade_record = {
                "symbol": symbol,
                "exchange": exchange,
                "side": "LONG",
                "quantity": exec_quantity,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "entry_time": pos.entry_time.isoformat(),
                "exit_time": timestamp.isoformat(),
                "gross_pnl_inr": round((price - pos.entry_price) * exec_quantity, 2),
                "total_friction_inr": round(portion_entry_cost + cost.total_friction_inr, 2),
                "net_pnl_inr": round(net_pnl, 2),
                "return_pct": round(ret_pct, 2),
            }
            self.closed_trades.append(trade_record)

            # Update or remove position
            if pos.quantity == exec_quantity:
                del self.positions[matched_pos_key]
            else:
                pos.quantity -= exec_quantity
                pos.entry_cost -= portion_entry_cost

            self.orders.append(order)
            self._record_event(
                "ORDER_FILLED_SELL" if order.status == OrderStatus.FILLED else "ORDER_PARTIALLY_FILLED_SELL",
                timestamp.isoformat(),
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "exchange": exchange,
                    "position_effect": effect,
                    "requested_quantity": quantity,
                    "filled_quantity": exec_quantity,
                    "exit_price": price,
                    "net_pnl": net_pnl,
                    "cash_after": self.cash,
                },
            )

        elif effect == "OPEN_SHORT":
            if matched_pos_key and self.positions[matched_pos_key].side == "LONG":
                order.status = OrderStatus.REJECTED
                order.rejection_reason = (
                    f"Cannot OPEN_SHORT in {symbol} while an open LONG position exists; use EXIT_LONG first"
                )
                self.orders.append(order)
                self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
                return order

            required_margin = (price * exec_quantity) + cost.total_friction_inr
            if required_margin > self.cash:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = (
                    f"Insufficient cash margin for short: required ₹{required_margin:.2f}, available ₹{self.cash:.2f}"
                )
                self.orders.append(order)
                self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
                return order

            self.cash -= required_margin
            order.status = status_on_fill
            order.filled_price = price
            order.filled_at = timestamp
            order.filled_quantity = exec_quantity
            order.remaining_quantity = quantity - exec_quantity

            if pos_key in self.positions:
                pos = self.positions[pos_key]
                total_qty = pos.quantity + exec_quantity
                new_avg_entry = ((pos.quantity * pos.entry_price) + (exec_quantity * price)) / total_qty
                pos.quantity = total_qty
                pos.entry_price = new_avg_entry
                pos.current_price = price
                pos.highest_price = max(pos.highest_price, price)
                pos.lowest_price = min(pos.lowest_price if pos.lowest_price is not None else price, price)
                pos.entry_cost += cost.total_friction_inr
                if stop_loss is not None:
                    pos.stop_loss = stop_loss
                if target is not None:
                    pos.target = target
            else:
                self.positions[pos_key] = Position(
                    symbol=symbol,
                    exchange=exchange,
                    quantity=exec_quantity,
                    entry_price=price,
                    current_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    highest_price=price,
                    entry_time=timestamp,
                    entry_cost=cost.total_friction_inr,
                    side="SHORT",
                    lowest_price=price,
                )

            self.orders.append(order)
            self._record_event(
                "ORDER_FILLED_OPEN_SHORT" if order.status == OrderStatus.FILLED else "ORDER_PARTIALLY_FILLED_OPEN_SHORT",
                timestamp.isoformat(),
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "exchange": exchange,
                    "position_effect": effect,
                    "requested_quantity": quantity,
                    "filled_quantity": exec_quantity,
                    "price": price,
                    "friction_inr": cost.total_friction_inr,
                    "cash_after": self.cash,
                },
            )

        elif effect == "COVER_SHORT":
            if (
                not matched_pos_key
                or self.positions[matched_pos_key].side != "SHORT"
                or self.positions[matched_pos_key].quantity < exec_quantity
            ):
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"No open short position in {symbol} ({exchange}) to cover {exec_quantity} shares"
                self.orders.append(order)
                self._record_event("ORDER_REJECTED", timestamp.isoformat(), {"order_id": order_id, "symbol": symbol, "reason": order.rejection_reason})
                return order

            pos = self.positions[matched_pos_key]
            margin_basis = pos.entry_price * exec_quantity
            gross_pnl = (pos.entry_price - price) * exec_quantity
            portion_entry_cost = pos.entry_cost * (exec_quantity / pos.quantity)
            net_pnl = gross_pnl - portion_entry_cost - cost.total_friction_inr
            # Release locked margin + realized gross P&L - exit friction
            self.cash += margin_basis + gross_pnl - cost.total_friction_inr

            order.status = status_on_fill
            order.filled_price = price
            order.filled_at = timestamp
            order.filled_quantity = exec_quantity
            order.remaining_quantity = quantity - exec_quantity

            ret_pct = (net_pnl / margin_basis) * 100.0 if margin_basis > 0 else 0.0
            trade_record = {
                "symbol": symbol,
                "exchange": exchange,
                "side": "SHORT",
                "quantity": exec_quantity,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "entry_time": pos.entry_time.isoformat(),
                "exit_time": timestamp.isoformat(),
                "gross_pnl_inr": round(gross_pnl, 2),
                "total_friction_inr": round(portion_entry_cost + cost.total_friction_inr, 2),
                "net_pnl_inr": round(net_pnl, 2),
                "return_pct": round(ret_pct, 2),
            }
            self.closed_trades.append(trade_record)

            if pos.quantity == exec_quantity:
                del self.positions[matched_pos_key]
            else:
                pos.quantity -= exec_quantity
                pos.entry_cost -= portion_entry_cost

            self.orders.append(order)
            self._record_event(
                "ORDER_FILLED_COVER_SHORT" if order.status == OrderStatus.FILLED else "ORDER_PARTIALLY_FILLED_COVER_SHORT",
                timestamp.isoformat(),
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "exchange": exchange,
                    "position_effect": effect,
                    "requested_quantity": quantity,
                    "filled_quantity": exec_quantity,
                    "exit_price": price,
                    "net_pnl": net_pnl,
                    "cash_after": self.cash,
                },
            )

            # R5-D: Honest short-gap losses — NEVER clip negative P&L or floor cash to 0.
            # If adverse short cover drives cash negative, trigger margin-call deficit state.
            if self.cash < 0.0:
                self.margin_call_triggered = True
                mc_event: dict[str, Any] = {
                    "symbol": symbol,
                    "order_id": order_id,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "cash_after_inr": self.cash,
                    "cash_deficit_inr": abs(self.cash),
                    "total_equity_inr": self.total_equity,
                }
                self.margin_call_events.append(mc_event)
                self._record_event(
                    "MARGIN_CALL_DEFICIT",
                    timestamp.isoformat(),
                    mc_event,
                )

        # Track peak equity
        self.peak_equity = max(self.peak_equity, self.total_equity)
        return order

    def update_market_price(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        exchange: Literal["NSE", "BSE"] | None = None,
    ) -> list[SimulatedOrder]:
        """Update market price for a symbol and auto-execute stop-loss / target if breached."""
        executed_orders: list[SimulatedOrder] = []
        pos = self.positions.get_position(symbol, exchange) if hasattr(self.positions, "get_position") else self.positions.get(symbol)
        if pos is None:
            return executed_orders

        pos.current_price = current_price
        pos.highest_price = max(pos.highest_price, current_price)
        pos.lowest_price = min(pos.lowest_price if pos.lowest_price is not None else current_price, current_price)

        if pos.side == "SHORT":
            # Short stop-loss triggers when price rises >= stop_loss
            if pos.stop_loss is not None and current_price >= pos.stop_loss:
                order = self.submit_order(
                    symbol=pos.symbol,
                    side=OrderSide.BUY,
                    quantity=pos.quantity,
                    price=current_price,
                    exchange=pos.exchange,
                    timestamp=timestamp,
                    position_effect="COVER_SHORT",
                )
                executed_orders.append(order)
                return executed_orders
            # Short target triggers when price drops <= target
            if pos.target is not None and current_price <= pos.target:
                order = self.submit_order(
                    symbol=pos.symbol,
                    side=OrderSide.BUY,
                    quantity=pos.quantity,
                    price=current_price,
                    exchange=pos.exchange,
                    timestamp=timestamp,
                    position_effect="COVER_SHORT",
                )
                executed_orders.append(order)
                return executed_orders
            return executed_orders

        # LONG position checks
        if pos.stop_loss is not None and current_price <= pos.stop_loss:
            order = self.submit_order(
                symbol=pos.symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                price=current_price,
                exchange=pos.exchange,
                timestamp=timestamp,
                position_effect="EXIT_LONG",
            )
            executed_orders.append(order)
            return executed_orders

        if pos.target is not None and current_price >= pos.target:
            order = self.submit_order(
                symbol=pos.symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                price=current_price,
                exchange=pos.exchange,
                timestamp=timestamp,
                position_effect="EXIT_LONG",
            )
            executed_orders.append(order)
            return executed_orders

        return executed_orders

    def update_candle_bar(
        self,
        symbol: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        timestamp: datetime,
        exchange: Literal["NSE", "BSE"] | None = None,
        round_ticks: bool = True,
    ) -> list[SimulatedOrder]:
        """Evaluate intrabar OHLC stop-loss and target triggers using conservative stop-first precedence.

        1. Gap-open breach against stop-loss fills at open_price.
        2. Gap-open breach against target fills at open_price.
        3. Intrabar stop-loss breach fills at stop_loss (takes precedence if both SL and TP occur in the same bar).
        4. Intrabar target breach fills at target.
        5. Otherwise marks position to close_price and updates highest/lowest excursion.
        """
        executed_orders: list[SimulatedOrder] = []
        pos = self.positions.get_position(symbol, exchange) if hasattr(self.positions, "get_position") else self.positions.get(symbol)
        if pos is None:
            return executed_orders

        pos.highest_price = max(pos.highest_price, high_price)
        pos.lowest_price = min(pos.lowest_price if pos.lowest_price is not None else low_price, low_price)

        def _px(v: float) -> float:
            return round_to_nse_tick(v) if round_ticks else float(v)

        if pos.side == "SHORT":
            exit_px: float | None = None
            if pos.stop_loss is not None and open_price >= pos.stop_loss:
                exit_px = _px(open_price)
            elif pos.target is not None and open_price <= pos.target:
                exit_px = _px(open_price)
            elif pos.stop_loss is not None and high_price >= pos.stop_loss:
                exit_px = _px(pos.stop_loss)
            elif pos.target is not None and low_price <= pos.target:
                exit_px = _px(pos.target)

            if exit_px is not None:
                order = self.submit_order(
                    symbol=pos.symbol,
                    side=OrderSide.BUY,
                    quantity=pos.quantity,
                    price=exit_px,
                    exchange=pos.exchange,
                    timestamp=timestamp,
                    position_effect="COVER_SHORT",
                )
                executed_orders.append(order)
                return executed_orders
        else:
            exit_px = None
            if pos.stop_loss is not None and open_price <= pos.stop_loss:
                exit_px = _px(open_price)
            elif pos.target is not None and open_price >= pos.target:
                exit_px = _px(open_price)
            elif pos.stop_loss is not None and low_price <= pos.stop_loss:
                exit_px = _px(pos.stop_loss)
            elif pos.target is not None and high_price >= pos.target:
                exit_px = _px(pos.target)

            if exit_px is not None:
                order = self.submit_order(
                    symbol=pos.symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    price=exit_px,
                    exchange=pos.exchange,
                    timestamp=timestamp,
                    position_effect="EXIT_LONG",
                )
                executed_orders.append(order)
                return executed_orders

        pos.current_price = close_price
        self.peak_equity = max(self.peak_equity, self.total_equity)
        return executed_orders

    def square_off_all_positions(self, current_prices: dict[str, float], timestamp: datetime) -> list[SimulatedOrder]:
        """Mandatory square-off for all open long and short positions (e.g. at 15:15 IST intraday close)."""
        orders = []
        for key, pos in list(self.positions.items()):
            price = current_prices.get(pos.symbol, current_prices.get(key, pos.current_price))
            if pos.side == "SHORT":
                ord_res = self.submit_order(
                    symbol=pos.symbol,
                    side=OrderSide.BUY,
                    quantity=pos.quantity,
                    price=price,
                    exchange=pos.exchange,
                    timestamp=timestamp,
                    position_effect="COVER_SHORT",
                )
            else:
                ord_res = self.submit_order(
                    symbol=pos.symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    price=price,
                    exchange=pos.exchange,
                    timestamp=timestamp,
                    position_effect="EXIT_LONG",
                )
            orders.append(ord_res)
        return orders

    def verify_ledger_integrity(self) -> tuple[bool, str]:
        """Verify unbroken cryptographic SHA-256 hash chaining and true double-entry capital conservation."""
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

        # True Double-Entry Conservation of Capital check:
        # initial_capital + realized_pnl + unrealized_pnl - open_entry_costs == cash + invested_capital
        open_entry_costs = sum(pos.entry_cost for pos in self.positions.values())
        expected_equity = self.initial_capital + self.realized_pnl + self.unrealized_pnl - open_entry_costs
        tolerance = 0.02 * len(self.closed_trades) + 1e-3
        if abs(expected_equity - self.total_equity) > tolerance:
            return False, (
                f"Capital conservation violated: expected_equity({expected_equity:.4f}) "
                f"!= total_equity({self.total_equity:.4f})"
            )

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
