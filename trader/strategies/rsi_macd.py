"""Strategy 5: RSI & MACD Momentum Confirmation Intraday Strategy."""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class RSIMACDStrategy(BaseStrategy):
    """Enters long when MACD crosses above signal with RSI momentum between 45 and 70 above VWAP."""

    def __init__(self, atr_sl_mult: float = 1.2, rr_ratio: float = 1.8) -> None:
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id="STRAT-005-RSI-MACD",
            name="RSI & MACD Momentum",
            origin_source="Quantitative Momentum Research",
            source_url="internal://rsi-macd-5m",
            license_type="MIT",
            required_indicators=["vwap", "rsi_14", "macd", "macd_signal", "atr_14"],
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

        if current_bar_idx < 26 or current_bar_idx >= len(features_df):
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="NO_SIGNAL",
                confidence=0.0,
                rationale=["Warming up MACD/RSI"],
            )

        curr = features_df.iloc[current_bar_idx]
        prev = features_df.iloc[current_bar_idx - 1]
        close = float(curr["close"])
        vwap = float(curr["vwap"])
        rsi = float(curr["rsi_14"])
        macd = float(curr["macd"])
        sig = float(curr["macd_signal"])
        prev_macd = float(prev["macd"])
        prev_sig = float(prev["macd_signal"])
        atr = max(float(curr["atr_14"]), close * 0.002)

        bull_cross = prev_macd <= prev_sig and macd > sig
        if bull_cross and 45.0 <= rsi <= 70.0 and close >= vwap:
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
                confidence=0.74,
                rationale=[f"MACD bullish cross ({macd:.2f}>{sig:.2f}) with RSI={rsi:.1f} above VWAP"],
            )

        if macd < sig and rsi < 42.0:
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="EXIT",
                confidence=0.70,
                rationale=[f"MACD bearish cross with RSI weakening ({rsi:.1f})"],
            )

        return StrategySignal(
            strategy_id=self.metadata.strategy_id,
            symbol=symbol,
            timestamp=ts,
            action="HOLD",
            confidence=0.5,
            rationale=["No RSI/MACD trigger"],
        )
