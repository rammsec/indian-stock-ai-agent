"""Agent 2: Technical Analysis Agent.

Processes 5-minute OHLCV candles, technical indicators, price action, and volume spikes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
import pandas as pd

from ..config import IST_TZ
from ..data.schema import Candle5M
from ..features.indicators import compute_all_features, compute_cpr_pivots
from .contracts import TechnicalAnalysisMessage
from .router import ModelRouter


class TechnicalAnalysisAgent:
    """Evaluates 5-minute candlestick charts and multi-indicator confluence."""

    def __init__(self, router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter()

    def analyze_candles(
        self,
        candles: list[Candle5M],
        prev_session_ohlc: tuple[float, float, float] | None = None,
    ) -> TechnicalAnalysisMessage:
        """Run technical calculations on closed 5m candles and generate setup analysis."""
        if not candles:
            raise ValueError("No candles provided to Technical Analysis Agent")

        symbol = candles[-1].symbol
        exchange = candles[-1].exchange
        timestamp = candles[-1].timestamp

        df = compute_all_features(candles)
        last_bar = df.iloc[-1]
        close = float(last_bar["close"])
        vwap = float(last_bar["vwap"])
        ema_9 = float(last_bar["ema_9"])
        ema_21 = float(last_bar["ema_21"])
        ema_50 = float(last_bar["ema_50"])
        rsi_14 = float(last_bar["rsi_14"])
        macd_line = float(last_bar["macd"])
        macd_sig = float(last_bar["macd_signal"])
        atr_14 = float(last_bar["atr_14"])
        vol_ratio = float(last_bar["volume_ratio"]) if not pd.isna(last_bar["volume_ratio"]) else 1.0

        # MACD cross determination
        if macd_line > macd_sig and len(df) > 1 and df.iloc[-2]["macd"] <= df.iloc[-2]["macd_signal"]:
            macd_signal = "BULLISH_CROSS"
        elif macd_line < macd_sig and len(df) > 1 and df.iloc[-2]["macd"] >= df.iloc[-2]["macd_signal"]:
            macd_signal = "BEARISH_CROSS"
        else:
            macd_signal = "NEUTRAL"

        # Pivot levels if prior day OHLC available
        support_levels = []
        resistance_levels = []
        if prev_session_ohlc:
            pH, pL, pC = prev_session_ohlc
            pivots = compute_cpr_pivots(pH, pL, pC)
            support_levels = [pivots["s1"], pivots["s2"]]
            resistance_levels = [pivots["r1"], pivots["r2"]]

        # Bullish / Bearish evidence scoring
        evidence: list[str] = []
        risks: list[str] = []
        bull_score = 0
        bear_score = 0

        # VWAP Rule
        if close > vwap:
            bull_score += 2
            evidence.append(f"Price (₹{close:.2f}) above Intraday VWAP (₹{vwap:.2f})")
        else:
            bear_score += 2
            evidence.append(f"Price (₹{close:.2f}) below Intraday VWAP (₹{vwap:.2f})")

        # EMA Trend Alignment
        if ema_9 > ema_21 > ema_50:
            bull_score += 2
            evidence.append("Bullish EMA Stack: EMA 9 > EMA 21 > EMA 50")
        elif ema_9 < ema_21 < ema_50:
            bear_score += 2
            evidence.append("Bearish EMA Stack: EMA 9 < EMA 21 < EMA 50")

        # RSI Momentum
        if 55.0 <= rsi_14 <= 70.0:
            bull_score += 1
            evidence.append(f"RSI-14 ({rsi_14:.1f}) in bullish momentum zone")
        elif rsi_14 > 75.0:
            risks.append(f"RSI-14 ({rsi_14:.1f}) overbought; pullback risk")
        elif 30.0 <= rsi_14 <= 45.0:
            bear_score += 1
            evidence.append(f"RSI-14 ({rsi_14:.1f}) in bearish momentum zone")
        elif rsi_14 < 25.0:
            risks.append(f"RSI-14 ({rsi_14:.1f}) oversold")

        # Volume confirmation
        if vol_ratio > 1.5:
            evidence.append(f"Volume surge: {vol_ratio:.1f}x of 20-bar SMA")

        # Composite Signal
        total_pts = bull_score + bear_score
        confidence = round(max(bull_score, bear_score) / max(1, total_pts), 2) if total_pts > 0 else 0.5

        if bull_score > bear_score + 1:
            signal: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "BULLISH"
        elif bear_score > bull_score + 1:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        # Model usage tracking
        model_name = self.router.select_model(task_level=1, agent_name="technical_analysis")
        if "gemini" in model_name:
            self.router.budget_manager.record_usage(
                agent_name="technical_analysis",
                model_name=model_name,
                input_tokens=600,
                output_tokens=150,
            )

        return TechnicalAnalysisMessage(
            symbol=symbol,
            exchange=exchange,  # type: ignore[arg-type]
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            current_price=close,
            vwap=round(vwap, 2),
            above_vwap=close > vwap,
            ema_9=round(ema_9, 2),
            ema_21=round(ema_21, 2),
            ema_50=round(ema_50, 2),
            rsi_14=round(rsi_14, 2),
            macd_signal=macd_signal,
            atr_14=round(atr_14, 2),
            volume_ratio=round(vol_ratio, 2),
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            evidence=evidence,
            risks=risks,
        )
