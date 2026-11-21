"""Pydantic v2 typed message contracts for inter-agent communication."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketPosture(str, Enum):
    """Four canonical market posture instructions emitted by the Central Council."""

    BUY_BULLISH = "BUY_BULLISH"       # Risk appetite normal; long setups favored in leading sectors
    SELL_BEARISH = "SELL_BEARISH"     # Capital preservation; avoid new longs, tighten stops, selective shorts
    NEUTRAL = "NEUTRAL"               # Choppy/sideways; tight mean-reversion setups with smaller sizing
    HOLD_HIGH_RISK = "HOLD_HIGH_RISK" # High volatility (VIX spike / binary event); halt trading or de-risk


class MarketIntelligenceMessage(_StrictMessage):
    """Output from Agent 1: Market Intelligence Agent."""

    agent: str = "market_intelligence"
    timestamp: datetime
    nifty_trend: Literal["BULLISH", "BEARISH", "SIDEWAYS"]
    bank_nifty_trend: Literal["BULLISH", "BEARISH", "SIDEWAYS"]
    india_vix: float
    vix_regime: Literal["LOW_VOLATILITY", "NORMAL", "ELEVATED", "HIGH_VOLATILITY"]
    advance_decline_ratio: float
    fii_dii_bias: Literal["BUYING", "SELLING", "NEUTRAL"]
    leading_sectors: list[str] = Field(default_factory=list)
    lagging_sectors: list[str] = Field(default_factory=list)
    market_direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)


class TechnicalAnalysisMessage(_StrictMessage):
    """Output from Agent 2: Technical Analysis Agent."""

    agent: str = "technical_analysis"
    symbol: str
    exchange: Literal["NSE", "BSE"]
    timeframe: str = "5m"
    timestamp: datetime
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    current_price: float
    vwap: float
    above_vwap: bool
    ema_9: float
    ema_21: float
    ema_50: float
    rsi_14: float
    macd_signal: Literal["BULLISH_CROSS", "BEARISH_CROSS", "NEUTRAL"]
    atr_14: float
    volume_ratio: float
    detected_patterns: list[str] = Field(default_factory=list)
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class NewsSentimentMessage(_StrictMessage):
    """Output from Agent 3: News and Sentiment Agent."""

    agent: str = "news_sentiment"
    symbol: str
    timestamp: datetime
    data_cutoff_timestamp: datetime
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    materiality_score: float = Field(ge=0.0, le=1.0)
    headlines: list[str] = Field(default_factory=list)
    corporate_actions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class MicrostructureMessage(_StrictMessage):
    """Output from Agent 4: Market Microstructure Agent."""

    agent: str = "microstructure"
    symbol: str
    exchange: Literal["NSE", "BSE"]
    timestamp: datetime
    bid_ask_spread_bps: float
    liquidity_status: Literal["HIGH", "MODERATE", "ILLIQUID"]
    estimated_slippage_bps: float
    execution_feasibility: Literal["PASS", "CAUTION", "REJECT"]
    notes: list[str] = Field(default_factory=list)


class CouncilPostureMessage(_StrictMessage):
    """Consensus output from Agent 7: Central Market Intelligence Council."""

    council_session_id: str
    timestamp: datetime
    posture: MarketPosture
    confidence: float = Field(ge=0.0, le=1.0)
    summary_verdict: str
    favored_sectors: list[str] = Field(default_factory=list)
    avoid_sectors: list[str] = Field(default_factory=list)
    trading_restrictions: list[str] = Field(default_factory=list)
    bull_arguments: list[str] = Field(default_factory=list)
    bear_arguments: list[str] = Field(default_factory=list)
    reassessment_triggers: list[str] = Field(default_factory=list)
    model_used: str = "deterministic_python"


class TradingAgentActionMessage(_StrictMessage):
    """Action decision from Agent 8: Autonomous Intraday Trading Agent."""

    agent: str = "autonomous_trading_agent"
    symbol: str
    timestamp: datetime
    action: Literal["ENTER_LONG", "EXIT", "TRAIL_STOP", "HOLD", "WAIT"]
    quantity: int = 0
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    strategy_id: str | None = None
    rationale: list[str] = Field(default_factory=list)
    risk_assessment_status: str | None = None
