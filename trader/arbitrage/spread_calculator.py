"""NSE & BSE Cross-Exchange Executable Spread & Statutory Friction Calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..data.instruments import get_instrument_by_isin, get_instrument_by_symbol
from ..portfolio.cost_calculator import calculate_intraday_costs


@dataclass(frozen=True)
class CrossExchangeOpportunity:
    """Detailed net arbitrage opportunity evaluation."""

    symbol: str
    isin: str
    buy_exchange: Literal["NSE", "BSE"]
    sell_exchange: Literal["NSE", "BSE"]
    buy_ask_price: float
    sell_bid_price: float
    quantity: int
    gross_spread_inr: float
    gross_spread_pct: float
    total_friction_inr: float
    net_profit_inr: float
    net_return_pct: float
    is_actionable: bool
    status: Literal["ACTIONABLE_OPPORTUNITY", "UNECONOMIC_SPREAD", "INSUFFICIENT_LIQUIDITY", "STALE_QUOTE"]
    notes: list[str]


def calculate_executable_arbitrage_spread(
    symbol: str,
    nse_bid: float,
    nse_ask: float,
    bse_bid: float,
    bse_ask: float,
    quantity: int = 100,
    min_net_profit_margin_pct: float = 0.15,
    quote_timestamp: datetime | None = None,
    current_time: datetime | None = None,
    max_quote_age_seconds: float = 15.0,
    available_buy_depth_qty: int | None = None,
    available_sell_depth_qty: int | None = None,
) -> CrossExchangeOpportunity:
    """Evaluate whether price differences between NSE and BSE yield a genuine net arbitrage.

    Crucial: Uses executable prices (Buy at Ask, Sell at Bid), NOT closing LTPs.
    Deducts all Indian statutory taxes (STT, Stamp Duty, Brokerage, Turnover fee, GST, slippage) on BOTH legs.
    Validates quote freshness (max 15s) and order book depth availability.
    """
    inst = get_instrument_by_symbol(symbol)
    isin = inst.isin if inst else "UNKNOWN_ISIN"

    # 1. Validate Quote Freshness
    if quote_timestamp is not None:
        curr_t = current_time or datetime.now(tz=quote_timestamp.tzinfo)
        quote_age = abs((curr_t - quote_timestamp).total_seconds())
        if quote_age > max_quote_age_seconds:
            return CrossExchangeOpportunity(
                symbol=symbol,
                isin=isin,
                buy_exchange="NSE",
                sell_exchange="BSE",
                buy_ask_price=nse_ask,
                sell_bid_price=bse_bid,
                quantity=quantity,
                gross_spread_inr=0.0,
                gross_spread_pct=0.0,
                total_friction_inr=0.0,
                net_profit_inr=0.0,
                net_return_pct=0.0,
                is_actionable=False,
                status="STALE_QUOTE",
                notes=[f"Quote age ({quote_age:.1f}s) exceeds maximum allowed age of {max_quote_age_seconds}s; arbitrage opportunity expired."],
            )

    # 2. Validate Order Book Depth
    if available_buy_depth_qty is not None and available_buy_depth_qty < quantity:
        return CrossExchangeOpportunity(
            symbol=symbol,
            isin=isin,
            buy_exchange="NSE",
            sell_exchange="BSE",
            buy_ask_price=nse_ask,
            sell_bid_price=bse_bid,
            quantity=quantity,
            gross_spread_inr=0.0,
            gross_spread_pct=0.0,
            total_friction_inr=0.0,
            net_profit_inr=0.0,
            net_return_pct=0.0,
            is_actionable=False,
            status="INSUFFICIENT_LIQUIDITY",
            notes=[f"Available buy book depth ({available_buy_depth_qty}) is less than required execution quantity ({quantity})."],
        )

    if available_sell_depth_qty is not None and available_sell_depth_qty < quantity:
        return CrossExchangeOpportunity(
            symbol=symbol,
            isin=isin,
            buy_exchange="NSE",
            sell_exchange="BSE",
            buy_ask_price=nse_ask,
            sell_bid_price=bse_bid,
            quantity=quantity,
            gross_spread_inr=0.0,
            gross_spread_pct=0.0,
            total_friction_inr=0.0,
            net_profit_inr=0.0,
            net_return_pct=0.0,
            is_actionable=False,
            status="INSUFFICIENT_LIQUIDITY",
            notes=[f"Available sell book depth ({available_sell_depth_qty}) is less than required execution quantity ({quantity})."],
        )

    # Opportunity 1: Buy BSE, Sell NSE
    spread_bse_buy = nse_bid - bse_ask

    # Opportunity 2: Buy NSE, Sell BSE
    spread_nse_buy = bse_bid - nse_ask

    if spread_bse_buy >= spread_nse_buy:
        buy_ex: Literal["NSE", "BSE"] = "BSE"
        sell_ex: Literal["NSE", "BSE"] = "NSE"
        buy_price = bse_ask
        sell_price = nse_bid
        gross_spread = spread_bse_buy
    else:
        buy_ex = "NSE"
        sell_ex = "BSE"
        buy_price = nse_ask
        sell_price = bse_bid
        gross_spread = spread_nse_buy

    gross_pct = (gross_spread / buy_price) * 100.0 if buy_price > 0 else 0.0

    # Calculate statutory friction for Leg 1 (Buy)
    buy_cost = calculate_intraday_costs(price=buy_price, quantity=quantity, side="BUY", exchange=buy_ex)

    # Calculate statutory friction for Leg 2 (Sell)
    sell_cost = calculate_intraday_costs(price=sell_price, quantity=quantity, side="SELL", exchange=sell_ex)

    total_friction = round(buy_cost.total_friction_inr + sell_cost.total_friction_inr, 2)
    gross_profit = round(gross_spread * quantity, 2)
    net_profit = round(gross_profit - total_friction, 2)

    capital_invested = buy_price * quantity
    net_return_pct = round((net_profit / capital_invested) * 100.0, 3) if capital_invested > 0 else 0.0

    notes: list[str] = [
        f"Leg 1 (Buy {buy_ex}): ₹{buy_price:.2f} (Friction: ₹{buy_cost.total_friction_inr:.2f})",
        f"Leg 2 (Sell {sell_ex}): ₹{sell_price:.2f} (Friction: ₹{sell_cost.total_friction_inr:.2f})",
        f"Gross Spread: ₹{gross_spread:.2f} ({gross_pct:.2f}%)",
        f"Total Two-Leg Friction: ₹{total_friction:.2f} (Net Return: {net_return_pct:.2f}%)",
    ]

    if net_return_pct >= min_net_profit_margin_pct and net_profit > 0:
        status: Literal["ACTIONABLE_OPPORTUNITY", "UNECONOMIC_SPREAD", "INSUFFICIENT_LIQUIDITY"] = "ACTIONABLE_OPPORTUNITY"
        is_actionable = True
        notes.append("Passes net profit threshold after all statutory charges and slippage")
    else:
        status = "UNECONOMIC_SPREAD"
        is_actionable = False
        notes.append(
            f"Friction (₹{total_friction:.2f}) consumes gross spread (₹{gross_profit:.2f}); uneconomic to execute"
        )

    return CrossExchangeOpportunity(
        symbol=symbol,
        isin=isin,
        buy_exchange=buy_ex,
        sell_exchange=sell_ex,
        buy_ask_price=buy_price,
        sell_bid_price=sell_price,
        quantity=quantity,
        gross_spread_inr=round(gross_profit, 2),
        gross_spread_pct=round(gross_pct, 3),
        total_friction_inr=total_friction,
        net_profit_inr=net_profit,
        net_return_pct=net_return_pct,
        is_actionable=is_actionable,
        status=status,
        notes=notes,
    )
