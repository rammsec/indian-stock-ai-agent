"""Indian Equity Statutory Transaction Cost Calculator for Intraday (MIS) Trading.

Follows official SEBI, NSE, and Ministry of Finance statutory rates:
- Brokerage: 0.03% or ₹20 per executed order (whichever is lower)
- STT (Securities Transaction Tax): 0.025% on Sell side only
- NSE Transaction Charges: 0.00297% per leg (BSE: 0.00375%)
- SEBI Turnover Fees: 0.0001% (₹10 / Crore) per leg
- Stamp Duty: 0.003% on Buy side only
- GST: 18% on (Brokerage + Exchange Charges + SEBI Fees)
- Slippage: Dynamic bid-ask spread and market impact penalty (default 5.0 bps/leg)
All parameters are configurable via CostConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import CostConfig, DEFAULT_COST_CONFIG


@dataclass(frozen=True)
class IndianIntradayCost:
    """Detailed statutory friction breakdown in INR and basis points."""

    turnover_inr: float
    side: Literal["BUY", "SELL"]
    exchange: Literal["NSE", "BSE"]
    brokerage_inr: float
    stt_inr: float
    exchange_charges_inr: float
    sebi_fee_inr: float
    stamp_duty_inr: float
    gst_inr: float
    slippage_inr: float
    total_friction_inr: float
    effective_friction_bps: float


def calculate_intraday_costs(
    price: float,
    quantity: int,
    side: Literal["BUY", "SELL"],
    exchange: Literal["NSE", "BSE"] = "NSE",
    slippage_bps: float | None = None,
    cost_config: CostConfig | None = None,
) -> IndianIntradayCost:
    """Compute exact statutory transaction taxes, charges, and slippage for an intraday leg.

    Args:
        price: Traded share price.
        quantity: Number of shares.
        side: "BUY" or "SELL".
        exchange: "NSE" or "BSE".
        slippage_bps: Optional override for modeled slippage in basis points.
        cost_config: Configurable cost rates; defaults to official Indian statutory rates.

    Returns:
        IndianIntradayCost: Itemized cost object.
    """
    cfg = cost_config or DEFAULT_COST_CONFIG
    turnover = round(price * quantity, 2)
    side_upper = side.upper()
    ex_upper = exchange.upper()
    effective_slippage = cfg.base_slippage_bps if slippage_bps is None else slippage_bps

    # 1. Brokerage: 0.03% capped at ₹20 per order
    raw_brokerage = turnover * cfg.brokerage_pct
    brokerage = min(raw_brokerage, cfg.max_brokerage_per_order_inr)

    # 2. STT: 0.025% on Sell side only
    stt = turnover * cfg.stt_sell_pct if side_upper == "SELL" else 0.0

    # 3. Exchange transaction charges
    ex_rate = cfg.nse_transaction_charge_pct if ex_upper == "NSE" else cfg.bse_transaction_charge_pct
    exchange_charges = turnover * ex_rate

    # 4. SEBI turnover fee (₹10 / crore)
    sebi_fee = turnover * cfg.sebi_turnover_fee_pct

    # 5. Stamp duty: 0.003% on Buy side only
    stamp_duty = turnover * cfg.stamp_duty_buy_pct if side_upper == "BUY" else 0.0

    # 6. GST: 18% on (Brokerage + Exchange Charge + SEBI Fee)
    gst_taxable = brokerage + exchange_charges + sebi_fee
    gst = gst_taxable * cfg.gst_rate_pct

    # 7. Slippage: basis points converted to INR
    slippage = turnover * (effective_slippage / 10_000.0)

    # Total friction
    total_friction = round(brokerage + stt + exchange_charges + sebi_fee + stamp_duty + gst + slippage, 4)
    effective_bps = round((total_friction / turnover) * 10_000.0, 2) if turnover > 0 else 0.0

    return IndianIntradayCost(
        turnover_inr=turnover,
        side=side_upper,  # type: ignore[arg-type]
        exchange=ex_upper,  # type: ignore[arg-type]
        brokerage_inr=round(brokerage, 4),
        stt_inr=round(stt, 4),
        exchange_charges_inr=round(exchange_charges, 4),
        sebi_fee_inr=round(sebi_fee, 4),
        stamp_duty_inr=round(stamp_duty, 4),
        gst_inr=round(gst, 4),
        slippage_inr=round(slippage, 4),
        total_friction_inr=total_friction,
        effective_friction_bps=effective_bps,
    )
