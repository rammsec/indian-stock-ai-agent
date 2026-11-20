"""Agent 7: Central Decision and Debate Council (Layer A).

Conducts structured multi-agent debate and emits the official Market Posture:
1. BUY_BULLISH
2. SELL_BEARISH
3. NEUTRAL
4. HOLD_HIGH_RISK
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable
import pandas as pd

from ..config import IST_TZ
from .contracts import (
    MarketPosture,
    MarketIntelligenceMessage,
    NewsSentimentMessage,
    CouncilPostureMessage,
)
from .router import ModelRouter


class CentralMarketCouncil:
    """Layer A: Central intelligence committee evaluating macro regime and market posture."""

    def __init__(self, router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter()
        self.current_posture: MarketPosture = MarketPosture.NEUTRAL
        self.last_assessment: CouncilPostureMessage | None = None
        self._event_listeners: list[Callable[[CouncilPostureMessage], None]] = []

    def subscribe(self, callback: Callable[[CouncilPostureMessage], None]) -> None:
        """Register an event listener (e.g. Autonomous Trading Agent) for posture shifts."""
        self._event_listeners.append(callback)

    def _broadcast_shift(self, new_report: CouncilPostureMessage) -> None:
        for listener in self._event_listeners:
            try:
                listener(new_report)
            except Exception:
                pass

    def evaluate_council(
        self,
        market_intel: MarketIntelligenceMessage,
        news_intel: list[NewsSentimentMessage] | None = None,
        session_id: str = "council_session_001",
    ) -> CouncilPostureMessage:
        """Debate bullish vs bearish arguments and produce consolidated market posture."""
        timestamp = datetime.now(tz=IST_TZ)
        bull_arguments: list[str] = []
        bear_arguments: list[str] = []

        # 1. Macro & Breadth Arguments
        if market_intel.nifty_trend == "BULLISH":
            bull_arguments.append(f"Nifty 50 upward momentum supported by breadth ratio {market_intel.advance_decline_ratio:.2f}")
        elif market_intel.nifty_trend == "BEARISH":
            bear_arguments.append(f"Nifty 50 downward trend with poor market breadth ({market_intel.advance_decline_ratio:.2f})")

        if market_intel.bank_nifty_trend == "BULLISH":
            bull_arguments.append("Bank Nifty leading market with strong relative strength")
        elif market_intel.bank_nifty_trend == "BEARISH":
            bear_arguments.append("Banking sector dragging index lower")

        # 2. Volatility Guard
        if market_intel.vix_regime == "HIGH_VOLATILITY":
            bear_arguments.append(f"India VIX elevated at {market_intel.india_vix:.2f}; defensive risk posture mandated")

        # 3. News Sentiment Integration
        if news_intel:
            for n in news_intel:
                if n.sentiment == "BULLISH" and n.materiality_score > 0.6:
                    bull_arguments.append(f"High-materiality positive catalysts in {n.symbol}")
                elif n.sentiment == "BEARISH" and n.materiality_score > 0.6:
                    bear_arguments.append(f"High-materiality negative news risk in {n.symbol}")

        # 4. Posture Synthesis
        if market_intel.vix_regime == "HIGH_VOLATILITY" or market_intel.india_vix > 23.0:
            posture = MarketPosture.HOLD_HIGH_RISK
            confidence = 0.85
            summary = "Extreme market volatility or binary risk event; trading frozen or aggressively de-risked."
        elif len(bull_arguments) >= len(bear_arguments) + 2 and market_intel.market_direction == "BULLISH":
            posture = MarketPosture.BUY_BULLISH
            confidence = min(0.90, 0.65 + (len(bull_arguments) * 0.05))
            summary = "Broader market breadth and index momentum favor long opportunities in leading sectors."
        elif len(bear_arguments) >= len(bull_arguments) + 2 and market_intel.market_direction == "BEARISH":
            posture = MarketPosture.SELL_BEARISH
            confidence = min(0.90, 0.65 + (len(bear_arguments) * 0.05))
            summary = "Market breadth deterioration and index weakness warrant capital preservation; avoid new longs."
        else:
            posture = MarketPosture.NEUTRAL
            confidence = 0.60
            summary = "Mixed signals across indices and breadth; sideways consolidation regime."

        favored = market_intel.leading_sectors
        avoid = market_intel.lagging_sectors

        restrictions = []
        if posture == MarketPosture.HOLD_HIGH_RISK:
            restrictions.append("NO_NEW_LONG_POSITIONS")
            restrictions.append("TIGHTEN_TRAILING_STOPS")
        elif posture == MarketPosture.SELL_BEARISH:
            restrictions.append("NO_NEW_LONG_POSITIONS")

        reassessment_triggers = [
            "India VIX crossing above 22.0",
            "Nifty 50 breaking below key intraday VWAP",
            "Advance/Decline ratio shifting by more than 0.50",
            "Breaking macroeconomic or RBI policy announcement",
        ]

        report = CouncilPostureMessage(
            council_session_id=session_id,
            timestamp=timestamp,
            posture=posture,
            confidence=round(confidence, 2),
            summary_verdict=summary,
            favored_sectors=favored,
            avoid_sectors=avoid,
            trading_restrictions=restrictions,
            bull_arguments=bull_arguments,
            bear_arguments=bear_arguments,
            reassessment_triggers=reassessment_triggers,
        )

        # Track usage if LLM is invoked
        model_name = self.router.select_model(task_level=2, agent_name="central_council")
        if "gemini" in model_name:
            self.router.budget_manager.record_usage(
                agent_name="central_council",
                model_name=model_name,
                input_tokens=850,
                output_tokens=250,
            )

        # Check if posture changed and broadcast
        if posture != self.current_posture:
            self.current_posture = posture
            self._broadcast_shift(report)

        self.last_assessment = report
        return report
