"""Strategy STRAT-003: VectorBT Fast 9/21 EMA Cross with 50-EMA Trend Filter."""

from __future__ import annotations

from datetime import datetime
import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class EMACrossStrategy(BaseStrategy):
    """Buys when 9-period EMA crosses above 21-period EMA while price is above 50-period EMA."""

    def __init__(self) -> None:
        meta = StrategyMetadata(
            strategy_id="STRAT-003-EMA-CROSS",
            name="VectorBT 9/21 EMA Dynamic Trend Filter",
            origin_source="VectorBT Open-Source Quantitative Library",
            source_url="https://github.com/polakowo/vectorbt",
            license_type="Apache-2.0",
            timeframe="5m",
            required_indicators=["EMA 9", "EMA 21", "EMA 50", "ATR 14"],
            known_limitations=["Lagging indicator cross in rapid mean-reverting ranges"],
            validation_status="IN_VALIDATION",
        )
        super().__init__(meta)

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        if current_bar_idx < 2 or current_bar_idx >= len(features_df):
            return StrategySignal(self.metadata.strategy_id, symbol, datetime.now(), "NO_SIGNAL")

        curr = features_df.iloc[current_bar_idx]
        prev = features_df.iloc[current_bar_idx - 1]
        ts = curr["timestamp"].to_pydatetime() if hasattr(curr["timestamp"], "to_pydatetime") else curr["timestamp"]

        curr_e9 = float(curr["ema_9"])
        curr_e21 = float(curr["ema_21"])
        prev_e9 = float(prev["ema_9"])
        prev_e21 = float(prev["ema_21"])
        e50 = float(curr["ema_50"])
        close = float(curr["close"])
        atr = float(curr["atr_14"])

        # Bullish cross: EMA 9 crosses above EMA 21 on current bar
        bullish_cross = (prev_e9 <= prev_e21) and (curr_e9 > curr_e21)
        trend_filter = close > e50 and curr_e21 > e50

        if bullish_cross and trend_filter:
            stop_loss = round(curr_e21 - (0.8 * atr), 2)
            risk = close - stop_loss
            target = round(close + (2.0 * risk), 2)

            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="ENTER_LONG",
                entry_price=close,
                stop_loss=stop_loss,
                target=target,
                confidence=0.70,
                rationale=[
                    f"EMA 9 ({curr_e9:.1f}) crossed above EMA 21 ({curr_e21:.1f})",
                    f"Trend confirmed above 50-EMA ({e50:.1f})",
                ],
            )

        # Bearish cross: Exit signal
        bearish_cross = (prev_e9 >= prev_e21) and (curr_e9 < curr_e21)
        if bearish_cross:
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="EXIT",
                entry_price=close,
                confidence=0.70,
                rationale=["EMA 9 crossed below EMA 21; exit signal"],
            )

        return StrategySignal(self.metadata.strategy_id, symbol, ts, "NO_SIGNAL")
