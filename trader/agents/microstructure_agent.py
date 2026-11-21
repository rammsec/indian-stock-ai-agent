"""Agent 4: Market Microstructure Agent.

Analyzes executable bid/ask spreads, order-book depth, liquidity, and execution feasibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from ..config import IST_TZ
from .contracts import MicrostructureMessage
from .router import ModelRouter


class MicrostructureAgent:
    """Evaluates market liquidity, bid-ask spreads, and execution impact."""

    def __init__(self, router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter()

    def analyze_microstructure(
        self,
        symbol: str,
        exchange: Literal["NSE", "BSE"],
        best_bid: float,
        best_ask: float,
        traded_volume_20m: float,
        timestamp: datetime | None = None,
    ) -> MicrostructureMessage:
        """Calculate effective spread in bps and evaluate execution feasibility."""
        if timestamp is None:
            timestamp = datetime.now(tz=IST_TZ)

        mid_price = (best_bid + best_ask) / 2.0
        spread_abs = best_ask - best_bid
        spread_bps = (spread_abs / mid_price) * 10_000.0 if mid_price > 0 else 100.0

        # Liquidity status based on spread and volume
        if spread_bps <= 5.0 and traded_volume_20m > 10_000:
            liquidity_status: Literal["HIGH", "MODERATE", "ILLIQUID"] = "HIGH"
            estimated_slippage_bps = 4.0
            feasibility: Literal["PASS", "CAUTION", "REJECT"] = "PASS"
        elif spread_bps <= 15.0:
            liquidity_status = "MODERATE"
            estimated_slippage_bps = 8.0
            feasibility = "PASS"
        elif spread_bps <= 35.0:
            liquidity_status = "MODERATE"
            estimated_slippage_bps = 18.0
            feasibility = "CAUTION"
        else:
            liquidity_status = "ILLIQUID"
            estimated_slippage_bps = 35.0
            feasibility = "REJECT"

        notes = [
            f"Bid ₹{best_bid:.2f} | Ask ₹{best_ask:.2f} (Spread: {spread_bps:.1f} bps)",
            f"Liquidity Status: {liquidity_status}",
        ]

        return MicrostructureMessage(
            symbol=symbol,
            exchange=exchange,
            timestamp=timestamp,
            bid_ask_spread_bps=round(spread_bps, 2),
            liquidity_status=liquidity_status,
            estimated_slippage_bps=round(estimated_slippage_bps, 2),
            execution_feasibility=feasibility,
            notes=notes,
        )
