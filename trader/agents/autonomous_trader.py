"""Agent 8: Autonomous Intraday Trading Agent (Layer B).

Operates continuously on 5-minute candles, manages orders in the simulated portfolio,
and acts independently under the Central Council's approved market posture.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from ..config import IST_TZ
from ..portfolio.ledger import IntradayPaperPortfolio, OrderSide
from ..risk.engine import DeterministicRiskEngine
from .contracts import (
    MarketPosture,
    CouncilPostureMessage,
    TechnicalAnalysisMessage,
    MicrostructureMessage,
    TradingAgentActionMessage,
)
from .router import ModelRouter


class AutonomousTradingAgent:
    """Layer B operational agent executing independent decisions on 5-minute candles."""

    def __init__(
        self,
        portfolio: IntradayPaperPortfolio,
        risk_engine: DeterministicRiskEngine,
        router: ModelRouter | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.risk_engine = risk_engine
        self.router = router or ModelRouter()
        self.current_council_posture: MarketPosture = MarketPosture.NEUTRAL
        self.reassessment_alert_received: bool = False

    def on_council_posture_update(self, update: CouncilPostureMessage) -> None:
        """Receive asynchronous event notification when the Central Council updates posture."""
        self.current_council_posture = update.posture
        self.reassessment_alert_received = True

        # If market changes to BEARISH or HIGH_RISK, reassess open long positions
        if update.posture in (MarketPosture.SELL_BEARISH, MarketPosture.HOLD_HIGH_RISK):
            self._tighten_or_exit_positions_on_risk(update.timestamp)

    def _tighten_or_exit_positions_on_risk(self, timestamp: datetime) -> None:
        """Reassess open positions when market posture deteriorates."""
        for symbol, pos in list(self.portfolio.positions.items()):
            # Move stop-loss to breakeven if in profit
            if pos.current_price > pos.entry_price:
                pos.stop_loss = max(pos.stop_loss or 0.0, pos.entry_price)

    def process_5m_candle_setup(
        self,
        symbol: str,
        tech_signal: TechnicalAnalysisMessage,
        microstructure: MicrostructureMessage,
        strategy_id: str = "STRAT-001-VWAP-PULLBACK",
    ) -> TradingAgentActionMessage:
        """Evaluate a closed 5m candle for entry, exit, or position management."""
        current_time = tech_signal.timestamp
        close = tech_signal.current_price
        rationale: list[str] = []

        # 1. Check existing position in this stock
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            # Update MTM in portfolio
            self.portfolio.update_market_price(symbol, close, timestamp=current_time)

            # Check if Technical Signal turned strongly bearish -> Exit early
            if tech_signal.signal == "BEARISH" and tech_signal.confidence >= 0.70:
                self.portfolio.submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    price=close,
                    exchange=tech_signal.exchange,
                    timestamp=current_time,
                )
                rationale.append("Early exit triggered: Technical signal reversed to BEARISH (>= 70% confidence)")
                return TradingAgentActionMessage(
                    symbol=symbol,
                    timestamp=current_time,
                    action="EXIT",
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    rationale=rationale,
                )

            # Trailing stop update if in profit by 1.5x ATR
            if pos.highest_price - pos.entry_price > 1.5 * tech_signal.atr_14:
                new_sl = round(pos.highest_price - 1.0 * tech_signal.atr_14, 2)
                if pos.stop_loss is None or new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
                    rationale.append(f"Trailing stop ratcheted up to ₹{new_sl:.2f}")

            return TradingAgentActionMessage(
                symbol=symbol,
                timestamp=current_time,
                action="HOLD",
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                stop_loss=pos.stop_loss,
                target=pos.target,
                rationale=rationale or ["Holding position; stops and targets active"],
            )

        # 2. No open position: Check if Entry Conditions are satisfied

        # Condition A: Central Council Posture must allow Longs
        if self.current_council_posture in (MarketPosture.SELL_BEARISH, MarketPosture.HOLD_HIGH_RISK):
            rationale.append(f"Entry blocked by Central Council posture: {self.current_council_posture.value}")
            return TradingAgentActionMessage(
                symbol=symbol,
                timestamp=current_time,
                action="WAIT",
                rationale=rationale,
            )

        # Condition B: Technical Signal must be Bullish
        if tech_signal.signal != "BULLISH" or tech_signal.confidence < 0.65:
            rationale.append(f"Technical setup not qualified: signal={tech_signal.signal}, conf={tech_signal.confidence}")
            return TradingAgentActionMessage(
                symbol=symbol,
                timestamp=current_time,
                action="WAIT",
                rationale=rationale,
            )

        # Condition C: Microstructure must be executable
        if microstructure.execution_feasibility == "REJECT":
            rationale.append(f"Microstructure rejected: spread {microstructure.bid_ask_spread_bps:.1f} bps, status {microstructure.liquidity_status}")
            return TradingAgentActionMessage(
                symbol=symbol,
                timestamp=current_time,
                action="WAIT",
                rationale=rationale,
            )

        # Condition D: Calculate Setup Stop-Loss & Target
        # Stop-loss: 1.0x ATR below entry or lowest recent support
        stop_loss = round(close - (1.0 * tech_signal.atr_14), 2)
        target = round(close + (2.0 * tech_signal.atr_14), 2)  # 2:1 Risk/Reward

        # Condition E: Risk Engine Gatekeeper Check
        desired_qty = 100  # Initial baseline request
        risk_res = self.risk_engine.evaluate_order(
            symbol=symbol,
            side="BUY",
            requested_quantity=desired_qty,
            entry_price=close,
            stop_loss_price=stop_loss,
            portfolio=self.portfolio,
            current_time=current_time,
        )

        if not risk_res.approved or risk_res.approved_quantity <= 0:
            rationale.extend(risk_res.rejection_reasons)
            return TradingAgentActionMessage(
                symbol=symbol,
                timestamp=current_time,
                action="WAIT",
                stop_loss=stop_loss,
                target=target,
                rationale=rationale,
                risk_assessment_status=risk_res.status,
            )

        # Execute Approved Simulated Order
        order = self.portfolio.submit_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=risk_res.approved_quantity,
            price=close,
            exchange=tech_signal.exchange,
            stop_loss=stop_loss,
            target=target,
            timestamp=current_time,
        )

        rationale.append(
            f"Entered Long {risk_res.approved_quantity} shares @ ₹{close:.2f} "
            f"(SL: ₹{stop_loss:.2f}, TP: ₹{target:.2f}, R:R=2.0)"
        )
        if risk_res.status == "SIZED_DOWN":
            rationale.append(f"Order sized down to {risk_res.approved_quantity} by Risk Engine")

        return TradingAgentActionMessage(
            symbol=symbol,
            timestamp=current_time,
            action="ENTER_LONG",
            quantity=risk_res.approved_quantity,
            entry_price=close,
            stop_loss=stop_loss,
            target=target,
            strategy_id=strategy_id,
            rationale=rationale,
            risk_assessment_status=risk_res.status,
        )
