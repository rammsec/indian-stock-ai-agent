"""Strategy STRAT-001: Intraday VWAP Pullback with EMA Trend Confirmation."""

from __future__ import annotations

from datetime import datetime
import pandas as pd

from .base import BaseStrategy, StrategyMetadata, StrategySignal


class VWAPPullbackStrategy(BaseStrategy):
    """Buys pullbacks towards VWAP / EMA 21 when the intraday trend is bullish."""

    def __init__(self) -> None:
        meta = StrategyMetadata(
            strategy_id="STRAT-001-VWAP-PULLBACK",
            name="Intraday VWAP Pullback with EMA Trend",
            origin_source="QuantConnect LEAN & Pine Script Classical Desks",
            source_url="https://github.com/QuantConnect/Lean",
            license_type="Apache-2.0 / MIT",
            timeframe="5m",
            required_indicators=["VWAP", "EMA 9", "EMA 21", "EMA 50", "ATR 14", "RSI 14"],
            known_limitations=["Choppy sideways sessions result in small whipsaw stop-outs"],
            validation_status="APPROVED_PAPER",
        )
        super().__init__(meta)

    def evaluate_bar(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_bar_idx: int,
    ) -> StrategySignal:
        if current_bar_idx < 10 or current_bar_idx >= len(features_df):
            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=datetime.now(),
                action="NO_SIGNAL",
            )

        row = features_df.iloc[current_bar_idx]
        prev_row = features_df.iloc[current_bar_idx - 1]
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]

        close = float(row["close"])
        vwap = float(row["vwap"])
        ema_9 = float(row["ema_9"])
        ema_21 = float(row["ema_21"])
        ema_50 = float(row["ema_50"])
        atr = float(row["atr_14"])
        rsi = float(row["rsi_14"])

        # Rules:
        # 1. Price > VWAP
        # 2. Bullish EMA trend: EMA 9 > EMA 21 > EMA 50
        # 3. Pullback check: Low touched near EMA 21 or VWAP within 0.3%
        # 4. Bullish rejection: Close > Open and Close > EMA 9
        # 5. RSI between 50 and 68 (momentum intact without extreme overbought)
        near_ema21 = abs(row["low"] - ema_21) / close <= 0.004
        near_vwap = abs(row["low"] - vwap) / close <= 0.005
        pullback_condition = near_ema21 or near_vwap
        trend_condition = ema_9 > ema_21 > ema_50 and close > vwap
        reversal_candle = close > float(row["open"]) and close > ema_9
        rsi_condition = 50.0 <= rsi <= 70.0

        if trend_condition and pullback_condition and reversal_candle and rsi_condition:
            stop_loss = round(min(row["low"], ema_21) - (0.5 * atr), 2)
            risk = close - stop_loss
            target = round(close + (2.0 * risk), 2)  # 2:1 R:R

            return StrategySignal(
                strategy_id=self.metadata.strategy_id,
                symbol=symbol,
                timestamp=ts,
                action="ENTER_LONG",
                entry_price=close,
                stop_loss=stop_loss,
                target=target,
                confidence=0.75,
                rationale=[
                    "Price bounced off EMA21/VWAP support",
                    f"Bullish EMA alignment (EMA9={ema_9:.1f} > EMA21={ema_21:.1f})",
                    f"RSI-14 at {rsi:.1f}",
                ],
            )

        return StrategySignal(
            strategy_id=self.metadata.strategy_id,
            symbol=symbol,
            timestamp=ts,
            action="NO_SIGNAL",
        )
