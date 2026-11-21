"""Strategy 7: Multi-Indicator Confirmation Confluence Strategy."""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class MultiIndicatorConfluenceStrategy(BaseStrategy):
    """Requires confluence across VWAP, EMA 9/21/50 stack, RSI, MACD, and volume."""

    def __init__(self, atr_sl_mult: float = 1.0, rr_ratio: float = 2.0) -> None:
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id="STRAT-007-MULTI-INDICATOR",
            name="Multi-Indicator Confluence",
            origin_source="Multi-Factor Ensemble",
            source_url="internal://multi-indicator-5m",
            license_type="MIT",
            required_indicators=["vwap", "ema_9", "ema_21", "ema_50", "rsi_14", "macd", "macd_signal", "volume_ratio", "atr_14"],
            validation_status="APPROVED_PAPER",
        )

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        ts = features_df.iloc[current_bar_idx]["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if current_bar_idx < 30 or current_bar_idx >= len(features_df):
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="NO_SIGNAL",
                confidence=0.0,
                rationale=["Warming up ensemble indicators"],
            )

        curr = features_df.iloc[current_bar_idx]
        close = float(curr["close"])
        vwap = float(curr["vwap"])
        ema9 = float(curr["ema_9"])
        ema21 = float(curr["ema_21"])
        ema50 = float(curr["ema_50"])
        rsi = float(curr["rsi_14"])
        macd = float(curr["macd"])
        sig = float(curr["macd_signal"])
        vol_ratio = float(curr.get("volume_ratio", 1.0))
        atr = max(float(curr["atr_14"]), close * 0.002)

        score = 0
        reasons = []
        if close > vwap:
            score += 1
            reasons.append("Close>VWAP")
        if ema9 > ema21 > ema50:
            score += 1
            reasons.append("EMA9>21>50")
        if 52.0 <= rsi <= 72.0:
            score += 1
            reasons.append(f"RSI={rsi:.1f}")
        if macd > sig:
            score += 1
            reasons.append("MACD>Signal")
        if vol_ratio >= 1.1:
            score += 1
            reasons.append(f"Vol={vol_ratio:.1f}x")

        if score >= 4:
            sl = round(close - (self.atr_sl_mult * atr), 2)
            tp = round(close + (self.atr_sl_mult * self.rr_ratio * atr), 2)
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="ENTER_LONG",
                entry_price=close,
                stop_loss=sl,
                target=tp,
                confidence=round(0.60 + score * 0.06, 2),
                rationale=[f"Confluence {score}/5: {', '.join(reasons)}"],
            )

        if score <= 1 and close < vwap:
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="EXIT",
                confidence=0.72,
                rationale=[f"Confluence deteriorated ({score}/5) below VWAP"],
            )

        return StrategySignal(
            strategy_id=self.metadata.strategy_id,
            symbol=symbol,
            timestamp=ts,
            action="HOLD",
            confidence=0.5,
            rationale=[f"Confluence {score}/5 (<4 required)"],
        )
