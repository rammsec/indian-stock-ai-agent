"""Agent 1: Market Intelligence Agent.

Analyzes broader Indian market indices (Nifty 50, Bank Nifty), India VIX,
market breadth (Advance/Decline), and FII/DII institutional activity.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import pandas as pd

from ..config import IST_TZ
from .contracts import MarketIntelligenceMessage
from .router import ModelRouter


class MarketIntelligenceAgent:
    """Evaluates macro regime and sectoral breadth."""

    def __init__(self, router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter()

    def analyze_market(
        self,
        nifty_change_pct: float,
        bank_nifty_change_pct: float,
        india_vix: float,
        advances: int,
        declines: int,
        fii_net_crores: float = 0.0,
        dii_net_crores: float = 0.0,
        timestamp: datetime | None = None,
    ) -> MarketIntelligenceMessage:
        """Perform deterministic calculations and synthesize market intelligence."""
        if timestamp is None:
            timestamp = datetime.now(tz=IST_TZ)

        # 1. Market Breadth Ratio
        ad_ratio = round(advances / max(1, declines), 2)

        # 2. VIX Regime Classification
        if india_vix < 13.0:
            vix_regime = "LOW_VOLATILITY"
        elif india_vix <= 18.0:
            vix_regime = "NORMAL"
        elif india_vix <= 24.0:
            vix_regime = "ELEVATED"
        else:
            vix_regime = "HIGH_VOLATILITY"

        # 3. Index Trends
        nifty_trend = "BULLISH" if nifty_change_pct > 0.25 else ("BEARISH" if nifty_change_pct < -0.25 else "SIDEWAYS")
        bank_nifty_trend = (
            "BULLISH" if bank_nifty_change_pct > 0.3 else ("BEARISH" if bank_nifty_change_pct < -0.3 else "SIDEWAYS")
        )

        # 4. Institutional Bias
        net_inst = fii_net_crores + dii_net_crores
        fii_bias = "BUYING" if net_inst > 500.0 else ("SELLING" if net_inst < -500.0 else "NEUTRAL")

        # 5. Composite Direction & Scoring
        bull_score = 0
        bear_score = 0

        if nifty_trend == "BULLISH":
            bull_score += 2
        elif nifty_trend == "BEARISH":
            bear_score += 2

        if bank_nifty_trend == "BULLISH":
            bull_score += 2
        elif bank_nifty_trend == "BEARISH":
            bear_score += 2

        if ad_ratio > 1.2:
            bull_score += 2
        elif ad_ratio < 0.8:
            bear_score += 2

        if fii_bias == "BUYING":
            bull_score += 1
        elif fii_bias == "SELLING":
            bear_score += 1

        if vix_regime == "HIGH_VOLATILITY":
            bear_score += 1

        total_pts = bull_score + bear_score
        confidence = round(max(bull_score, bear_score) / max(1, total_pts), 2) if total_pts > 0 else 0.5

        if bull_score > bear_score + 1:
            market_dir = "BULLISH"
        elif bear_score > bull_score + 1:
            market_dir = "BEARISH"
        else:
            market_dir = "NEUTRAL"

        evidence = [
            f"Nifty 50: {nifty_change_pct:+.2f}% ({nifty_trend})",
            f"Bank Nifty: {bank_nifty_change_pct:+.2f}% ({bank_nifty_trend})",
            f"Advance/Decline Ratio: {ad_ratio:.2f} ({advances} Up vs {declines} Down)",
            f"India VIX: {india_vix:.2f} ({vix_regime})",
            f"Net Institutional Flow: ₹{net_inst:+.1f} Cr ({fii_bias})",
        ]

        leading = ["Banking / Financials"] if bank_nifty_change_pct > 0.5 else []
        lagging = ["IT"] if nifty_change_pct < -0.5 else []

        # Record usage if router invokes LLM
        model_name = self.router.select_model(task_level=1, agent_name="market_intelligence")
        if "gemini" in model_name:
            self.router.budget_manager.record_usage(
                agent_name="market_intelligence",
                model_name=model_name,
                input_tokens=450,
                output_tokens=120,
            )

        return MarketIntelligenceMessage(
            timestamp=timestamp,
            nifty_trend=nifty_trend,  # type: ignore[arg-type]
            bank_nifty_trend=bank_nifty_trend,  # type: ignore[arg-type]
            india_vix=india_vix,
            vix_regime=vix_regime,  # type: ignore[arg-type]
            advance_decline_ratio=ad_ratio,
            fii_dii_bias=fii_bias,  # type: ignore[arg-type]
            leading_sectors=leading,
            lagging_sectors=lagging,
            market_direction=market_dir,  # type: ignore[arg-type]
            confidence=confidence,
            supporting_evidence=evidence,
        )
